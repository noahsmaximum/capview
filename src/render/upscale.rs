//! Optional AI upscaling between convert and the scale pass. Today that's NVIDIA RTX
//! Video Super Resolution (Windows + GeForce RTX, `--features rtx-video`); everywhere
//! else the option reports why it's unavailable and the normal scaler runs.
//!
//! Per new frame: a prep pass copies the cropped, gamma-encoded frame into `input`
//! and an empty pass "touches" `output` (so wgpu leaves both as render targets), the
//! frame encoder is submitted, then [`Upscale::dispatch`] runs the upscale on the same
//! queue. The scale pass then samples `output` (~1:1 in "Match window").

use super::{Geometry, SourceBinding, fullscreen_pipeline, tex_entry, ubo_entry};
use crate::settings::{Settings, UpscaleTarget, Upscaler};
use bytemuck::{Pod, Zeroable};
use std::time::{Duration, Instant};

#[cfg(all(windows, feature = "rtx-video"))]
use crate::rtx::Vsr;

/// Largest upscale output we'll allocate (matches the device's texture limit).
const MAX_OUT: u32 = 8192;
/// Hold the old output while the window is being resized, so we don't rebuild
/// 4K textures every frame of a drag.
const RESIZE_SETTLE: Duration = Duration::from_millis(150);

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct PrepParams {
    origin: [u32; 2],
    linear: u32,
    _pad: u32,
}

#[cfg_attr(not(all(windows, feature = "rtx-video")), allow(dead_code))]
struct Targets {
    input: wgpu::Texture,
    input_view: wgpu::TextureView,
    output: wgpu::Texture,
    output_view: wgpu::TextureView,
    in_size: [u32; 2],
    out_size: [u32; 2],
    origin: [u32; 2],
    src_generation: u64,
    from_sharp: bool,
    prep_bg: wgpu::BindGroup,
    /// Prep reading frame generation's output instead, keyed by its texture id.
    prep_bg_gen: Option<(u64, wgpu::BindGroup)>,
    /// Scale-pass group 0 sampling `output`.
    scale_bg: wgpu::BindGroup,
    /// `output` holds a finished upscale.
    valid: bool,
}

pub struct Upscale {
    /// UAV-capable texture format for the upscaler (10-bit when the GPU allows).
    format: Option<wgpu::TextureFormat>,
    prep_bgl: wgpu::BindGroupLayout,
    prep_pipeline: Option<wgpu::RenderPipeline>,
    prep_ubo: wgpu::Buffer,
    #[cfg(all(windows, feature = "rtx-video"))]
    backend: Option<Result<Vsr, String>>,
    targets: Option<Targets>,
    resize_since: Option<([u32; 2], Instant)>,
    last_seq: u64,
    last_quality: u8,
    /// Quality of the upscale queued by `encode`, run by `dispatch`.
    pending: Option<u8>,
    /// This frame's scale pass reads the upscaled texture.
    active: bool,
    /// Why it isn't running (shown in the UI), when the user turned it on.
    reason: Option<String>,
    #[cfg_attr(not(all(windows, feature = "rtx-video")), allow(dead_code))]
    warned: bool,
}

impl Upscale {
    pub fn new(device: &wgpu::Device, format: Option<wgpu::TextureFormat>) -> Self {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("upscale_prep"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/upscale_prep.wgsl").into()),
        });
        let prep_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("upscale_prep_bgl"),
            entries: &[
                ubo_entry(0),
                tex_entry(1, wgpu::TextureSampleType::Float { filterable: false }),
            ],
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("upscale_prep"),
            bind_group_layouts: &[Some(&prep_bgl)],
            immediate_size: 0,
        });
        let prep_pipeline = format.map(|f| {
            fullscreen_pipeline(device, "upscale_prep", &module, &layout, "fs_main", f)
        });
        let prep_ubo = super::ubo(device, "upscale_prep", std::mem::size_of::<PrepParams>());
        Self {
            format,
            prep_bgl,
            prep_pipeline,
            prep_ubo,
            #[cfg(all(windows, feature = "rtx-video"))]
            backend: None,
            targets: None,
            resize_since: None,
            last_seq: u64::MAX,
            last_quality: 0,
            pending: None,
            active: false,
            reason: None,
            warned: false,
        }
    }

    /// The upscaled texture and its size, when this frame should draw from it.
    pub fn scale_source(&self) -> Option<(&wgpu::BindGroup, [u32; 2])> {
        let t = self.targets.as_ref()?;
        (self.active && (t.valid || self.pending.is_some())).then_some((&t.scale_bg, t.out_size))
    }

    /// One line for the UI: what it's doing (true), or why it isn't (false).
    pub fn status(&self) -> Option<(String, bool)> {
        if let Some(r) = &self.reason {
            return Some((r.clone(), false));
        }
        let t = self.targets.as_ref().filter(|_| self.active)?;
        #[allow(unused_mut)]
        let mut line = format!(
            "{}×{} → {}×{}",
            t.in_size[0], t.in_size[1], t.out_size[0], t.out_size[1]
        );
        #[cfg(all(windows, feature = "rtx-video"))]
        if let Some(Ok(v)) = &self.backend
            && let Some(ms) = v.gpu_ms
        {
            line.push_str(&format!(" · {ms:.1} ms GPU"));
        }
        Some((line, true))
    }

    /// Bring up the upscaler on first use. Err = user-facing reason it can't run.
    #[allow(unused_variables)]
    fn ensure_backend(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) -> Result<(), String> {
        if self.format.is_none() {
            return Err("GPU has no writable 10/8-bit format for it".into());
        }
        #[cfg(all(windows, feature = "rtx-video"))]
        {
            let b = self.backend.get_or_insert_with(|| {
                let r = Vsr::new(device, queue);
                match &r {
                    Ok(_) => log::info!("RTX Video Super Resolution ready"),
                    Err(e) => log::warn!("RTX VSR unavailable: {e}"),
                }
                r
            });
            b.as_ref().map(|_| ()).map_err(|e| e.clone())
        }
        #[cfg(all(windows, not(feature = "rtx-video")))]
        {
            Err("this build has no RTX Video support".into())
        }
        #[cfg(not(windows))]
        {
            Err("Windows + NVIDIA RTX only".into())
        }
    }

    fn wait_idle(&self) {
        #[cfg(all(windows, feature = "rtx-video"))]
        if let Some(Ok(v)) = &self.backend {
            v.wait_idle();
        }
    }

    fn drop_targets(&mut self) {
        if self.targets.is_some() {
            // Our upscale submissions are invisible to wgpu: make sure none still use
            // these textures before wgpu frees them.
            self.wait_idle();
            self.targets = None;
        }
        self.resize_since = None;
    }

    /// Size the upscaler should render at, or None when it wouldn't enlarge anything.
    fn desired_out(geo: &Geometry, target: UpscaleTarget) -> Option<[u32; 2]> {
        let (cw, ch) = (geo.crop_size[0], geo.crop_size[1]);
        let (w, h) = match target {
            UpscaleTarget::Window => (cw * geo.scale[0], ch * geo.scale[1]),
            UpscaleTarget::P1440 => (cw * 1440.0 / ch, 1440.0),
            UpscaleTarget::P2160 => (cw * 2160.0 / ch, 2160.0),
            UpscaleTarget::X2 => (cw * 2.0, ch * 2.0),
            UpscaleTarget::X3 => (cw * 3.0, ch * 3.0),
            UpscaleTarget::X4 => (cw * 4.0, ch * 4.0),
        };
        // Keep the aspect when clamping to the texture limit.
        let k = (MAX_OUT as f32 / w.max(h)).min(1.0);
        let out = [(w * k).round() as u32, (h * k).round() as u32];
        let enlarges = out[0] >= cw as u32 && out[1] >= ch as u32 && out != [cw as u32, ch as u32];
        enlarges.then_some(out)
    }

    /// Encode the prep pass when there's a new frame to upscale. Returns true when
    /// the caller must submit `encoder` and then call [`Upscale::dispatch`].
    #[allow(clippy::too_many_arguments)]
    pub fn encode(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        src: Option<&SourceBinding>,
        frame_seq: u64,
        geo: Option<&Geometry>,
        s: &Settings,
        scale: (&wgpu::BindGroupLayout, &wgpu::Buffer, &wgpu::Sampler),
        generated: Option<(&wgpu::TextureView, u64)>,
    ) -> bool {
        self.active = false;
        self.pending = None;
        if s.upscaler == Upscaler::Off {
            self.reason = None;
            self.drop_targets();
            return false;
        }
        let (Some(src), Some(geo)) = (src, geo) else {
            self.reason = None;
            return false;
        };
        if let Err(e) = self.ensure_backend(device, queue) {
            self.reason = Some(e);
            self.drop_targets();
            return false;
        }
        let Some(want) = Self::desired_out(geo, s.upscale_target) else {
            self.reason = Some("off: the video isn't enlarged at this size".into());
            return false;
        };
        self.reason = None;
        let in_size = [geo.crop_size[0] as u32, geo.crop_size[1] as u32];
        let origin = [geo.crop_origin[0] as u32, geo.crop_origin[1] as u32];
        let from_sharp = s.sharpen > 0.001;

        // Rebuild for a new source/crop right away; for a new output size only once
        // the window has stopped changing (meanwhile the old output is rescaled).
        let rebuild = match &self.targets {
            None => true,
            Some(t) if t.src_generation != src.generation
                || t.in_size != in_size
                || t.origin != origin
                || t.from_sharp != from_sharp =>
            {
                true
            }
            Some(t) if t.out_size != want => {
                let now = Instant::now();
                match self.resize_since {
                    Some((w, since)) if w == want => now - since >= RESIZE_SETTLE,
                    _ => {
                        self.resize_since = Some((want, now));
                        false
                    }
                }
            }
            Some(_) => {
                self.resize_since = None;
                false
            }
        };
        if rebuild {
            self.drop_targets();
            self.targets = Some(self.build_targets(device, src, in_size, want, origin, from_sharp, scale));
        }

        let quality = s.vsr_quality.clamp(1, 4);
        if let Some((view, id)) = generated {
            let t = self.targets.as_mut().expect("targets");
            if t.prep_bg_gen.as_ref().is_none_or(|(i, _)| *i != id) {
                let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("upscale_prep_generated"),
                    layout: &self.prep_bgl,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: self.prep_ubo.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::TextureView(view),
                        },
                    ],
                });
                t.prep_bg_gen = Some((id, bg));
            }
        }
        let t = self.targets.as_ref().expect("targets");
        self.active = true;
        if t.valid && frame_seq == self.last_seq && quality == self.last_quality {
            return false; // nothing new: keep showing the last upscale
        }

        queue.write_buffer(
            &self.prep_ubo,
            0,
            bytemuck::bytes_of(&PrepParams {
                origin,
                linear: s.linear_light as u32,
                _pad: 0,
            }),
        );
        {
            let mut pass = super::begin_pass(encoder, "upscale_prep", &t.input_view);
            pass.set_pipeline(self.prep_pipeline.as_ref().expect("prep pipeline"));
            let bg = match (&generated, &t.prep_bg_gen) {
                (Some(_), Some((_, bg))) => bg,
                _ => &t.prep_bg,
            };
            pass.set_bind_group(0, bg, &[]);
            pass.draw(0..3, 0..1);
        }
        // Empty pass: leaves `output` as a render target in wgpu's tracking, which is
        // the state the upscale expects to find it in (see rtx.rs).
        drop(encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("upscale_touch"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &t.output_view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        }));
        self.last_seq = frame_seq;
        self.last_quality = quality;
        self.pending = Some(quality);
        true
    }

    /// Run the upscale queued by `encode`. Call right after submitting its encoder.
    pub fn dispatch(&mut self) {
        let Some(quality) = self.pending else { return };
        #[cfg(all(windows, feature = "rtx-video"))]
        if let (Some(Ok(v)), Some(t)) = (&mut self.backend, &mut self.targets) {
            let [iw, ih] = t.in_size;
            let [ow, oh] = t.out_size;
            match v.evaluate(&t.input, [0, 0, iw, ih], &t.output, [0, 0, ow, oh], quality as i32) {
                Ok(()) => {
                    t.valid = true;
                    self.pending = None;
                    return;
                }
                Err(e) => {
                    if !self.warned {
                        log::warn!("{e}");
                        self.warned = true;
                    }
                    self.reason = Some(e);
                }
            }
        }
        let _ = quality;
        // Failed (or no backend): fall back to the normal scaler this frame.
        self.pending = None;
        self.active = false;
        if let Some(t) = &mut self.targets {
            t.valid = false;
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn build_targets(
        &self,
        device: &wgpu::Device,
        src: &SourceBinding,
        in_size: [u32; 2],
        out_size: [u32; 2],
        origin: [u32; 2],
        from_sharp: bool,
        (scale_bgl, scale_ubo, sampler): (&wgpu::BindGroupLayout, &wgpu::Buffer, &wgpu::Sampler),
    ) -> Targets {
        let format = self.format.expect("format");
        let mk = |label: &str, size: [u32; 2], storage: bool| {
            let mut usage =
                wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING;
            if storage {
                // The upscaler writes it as a UAV (ALLOW_UNORDERED_ACCESS on D3D12).
                usage |= wgpu::TextureUsages::STORAGE_BINDING;
            }
            device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d {
                    width: size[0].max(1),
                    height: size[1].max(1),
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage,
                view_formats: &[],
            })
        };
        let input = mk("upscale_in", in_size, false);
        let output = mk("upscale_out", out_size, true);
        let input_view = input.create_view(&Default::default());
        let output_view = output.create_view(&Default::default());
        let src_tex = if from_sharp { &src.sharp_tex } else { &src.rgb_tex };
        let src_view = src_tex.create_view(&Default::default());
        let prep_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("upscale_prep"),
            layout: &self.prep_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.prep_ubo.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&src_view),
                },
            ],
        });
        let scale_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("scale_bg_upscaled"),
            layout: scale_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: scale_ubo.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&output_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(sampler),
                },
            ],
        });
        Targets {
            input,
            input_view,
            output,
            output_view,
            in_size,
            out_size,
            origin,
            src_generation: src.generation,
            from_sharp,
            prep_bg,
            prep_bg_gen: None,
            scale_bg,
            valid: false,
        }
    }
}

impl Drop for Upscale {
    fn drop(&mut self) {
        self.drop_targets();
    }
}
