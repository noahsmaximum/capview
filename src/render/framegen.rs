//! Optional 2× frame generation: the GPU's optical-flow engine measures motion
//! between the previous source frame and the one that just arrived (both
//! directions), and framegen_warp.wgsl builds the frame halfway between them. That
//! frame is shown at once and the real one half a source interval later, so motion
//! is sampled twice as often, at the cost of showing the real frame ~half a frame
//! late.
//!
//! Per new source frame, in the frame encoder: luma of the frame → `luma[cur]`
//! (flow input), the frame itself → `colour[cur]`, and an empty pass "touches" the
//! two flow textures. The caller submits it and calls [`FrameGen::dispatch`] (flow
//! on the engine, ordered on the queue by a fence), then [`FrameGen::encode_warp`]
//! in the next encoder renders the in-between frame into `shown`.

#![cfg_attr(not(windows), allow(dead_code, unused_imports, unused_variables))]

use super::{INTERMEDIATE, SourceBinding, fullscreen_pipeline, tex_entry, ubo_entry};
use crate::settings::Settings;
use bytemuck::{Pod, Zeroable};
use std::time::{Duration, Instant};

#[cfg(windows)]
use crate::nvof::{FlowSet, OpticalFlow};

/// Longer gaps than this (pauses, signal loss) aren't interpolated across.
const MAX_GAP: f64 = 0.1;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct LumaParams {
    linear: u32,
    _pad: [u32; 3],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct WarpParams {
    size: [f32; 2],
    grid: f32,
    t: f32,
    radius: f32,
    guard: u32,
    softness: f32,
    _pad: f32,
}

struct Targets {
    luma: [wgpu::Texture; 2],
    luma_view: [wgpu::TextureView; 2],
    colour: [wgpu::Texture; 2],
    forward: wgpu::Texture,
    forward_view: wgpu::TextureView,
    backward: wgpu::Texture,
    backward_view: wgpu::TextureView,
    shown_view: wgpu::TextureView,
    /// `colour[cur]` / `luma[cur]` hold the newest frame, the other slot the one before.
    cur: usize,
    /// Slots that hold a real frame (so the first frame after a rebuild isn't
    /// interpolated against garbage).
    filled: u8,
    size: [u32; 2],
    grid: u32,
    src_generation: u64,
    from_sharp: bool,
    luma_bg: wgpu::BindGroup,
    /// Warp inputs for cur = 0 / 1.
    warp_bg: [wgpu::BindGroup; 2],
    /// Scale-pass group 0 sampling `shown`.
    scale_bg: wgpu::BindGroup,
    /// Bumped on rebuild, so dependents can re-bind `shown_view`.
    id: u64,
}

pub struct FrameGen {
    luma_bgl: wgpu::BindGroupLayout,
    luma_pipeline: wgpu::RenderPipeline,
    luma_ubo: wgpu::Buffer,
    warp_bgl: wgpu::BindGroupLayout,
    warp_pipeline: wgpu::RenderPipeline,
    warp_ubo: wgpu::Buffer,
    sampler: wgpu::Sampler,
    #[cfg(windows)]
    flow: Option<Result<OpticalFlow, String>>,
    targets: Option<Targets>,
    next_id: u64,
    last_seq: u64,
    prev_arrival: Option<Instant>,
    /// Smoothed source frame interval, seconds.
    interval: f64,
    /// Queued by `encode` for `dispatch`: the new frame's arrival.
    pending: Option<Instant>,
    /// Flow is queued; the warp pass still has to run.
    warp_pending: bool,
    showing_generated: bool,
    real_due: Option<Instant>,
    reason: Option<String>,
}

impl FrameGen {
    pub fn new(device: &wgpu::Device) -> Self {
        let nf = wgpu::TextureSampleType::Float { filterable: false };
        let layout = |bgl: &wgpu::BindGroupLayout| {
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[Some(bgl)],
                immediate_size: 0,
            })
        };
        let luma_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("framegen_luma"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/framegen_luma.wgsl").into()),
        });
        let luma_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("framegen_luma_bgl"),
            entries: &[ubo_entry(0), tex_entry(1, nf)],
        });
        let luma_pipeline = fullscreen_pipeline(
            device,
            "framegen_luma",
            &luma_module,
            &layout(&luma_bgl),
            "fs_main",
            wgpu::TextureFormat::R8Unorm,
        );
        let warp_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("framegen_warp"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/framegen_warp.wgsl").into()),
        });
        let filt = wgpu::TextureSampleType::Float { filterable: true };
        let sint = wgpu::TextureSampleType::Sint;
        let warp_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("framegen_warp_bgl"),
            entries: &[
                ubo_entry(0),
                tex_entry(1, filt),
                tex_entry(2, filt),
                tex_entry(3, sint),
                tex_entry(4, sint),
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let warp_pipeline = fullscreen_pipeline(
            device,
            "framegen_warp",
            &warp_module,
            &layout(&warp_bgl),
            "fs_main",
            INTERMEDIATE,
        );
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("framegen"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        Self {
            luma_bgl,
            luma_pipeline,
            luma_ubo: super::ubo(device, "framegen_luma", std::mem::size_of::<LumaParams>()),
            warp_bgl,
            warp_pipeline,
            warp_ubo: super::ubo(device, "framegen_warp", std::mem::size_of::<WarpParams>()),
            sampler,
            #[cfg(windows)]
            flow: None,
            targets: None,
            next_id: 1,
            last_seq: u64::MAX,
            prev_arrival: None,
            interval: 1.0 / 60.0,
            pending: None,
            warp_pending: false,
            showing_generated: false,
            real_due: None,
            reason: None,
        }
    }

    /// The generated frame, while it's the one on screen: scale-pass group 0, plus
    /// its view and an id for the AI upscaler.
    pub fn showing(&self) -> Option<(&wgpu::BindGroup, &wgpu::TextureView, u64)> {
        let t = self.targets.as_ref()?;
        self.showing_generated
            .then_some((&t.scale_bg, &t.shown_view, t.id))
    }

    /// When the real frame should replace the generated one on screen.
    pub fn due(&self) -> Option<Instant> {
        self.real_due
    }

    pub fn status(&self) -> Option<(String, bool)> {
        if let Some(r) = &self.reason {
            return Some((r.clone(), false));
        }
        let t = self.targets.as_ref()?;
        let fps = 1.0 / self.interval.max(1e-3);
        Some((
            format!(
                "{fps:.0} → {:.0} fps · +{:.1} ms · flow {}px",
                fps * 2.0,
                self.interval * 500.0,
                t.grid
            ),
            true,
        ))
    }

    fn wait_idle(&mut self) {
        #[cfg(windows)]
        if let Some(Ok(f)) = &mut self.flow {
            // Registrations point at textures we're about to drop.
            f.unregister_all();
        }
    }

    fn reset(&mut self) {
        if self.targets.is_some() {
            self.wait_idle();
        }
        self.targets = None;
        #[cfg(windows)]
        {
            self.flow = None;
        }
        self.showing_generated = false;
        self.real_due = None;
        self.prev_arrival = None;
        self.last_seq = u64::MAX;
        self.warp_pending = false;
    }

    /// Per render: flip to the real frame when it's due, and stage a new source
    /// frame. Returns true when the caller must submit `encoder` and call `dispatch`.
    #[allow(clippy::too_many_arguments)]
    pub fn encode(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        src: Option<&SourceBinding>,
        frame_seq: u64,
        arrival: Option<Instant>,
        s: &Settings,
        scale: (&wgpu::BindGroupLayout, &wgpu::Buffer, &wgpu::Sampler),
    ) -> bool {
        self.pending = None;
        if !s.frame_gen {
            self.reason = None;
            self.reset();
            return false;
        }
        #[cfg(not(windows))]
        {
            self.reason = Some("Windows + NVIDIA (Turing or newer) only".into());
            false
        }
        #[cfg(windows)]
        {
            let now = Instant::now();
            if self.real_due.is_some_and(|t| now >= t) {
                self.showing_generated = false;
                self.real_due = None;
            }
            let Some(src) = src else {
                self.showing_generated = false;
                return false;
            };
            if frame_seq == self.last_seq || frame_seq == u64::MAX {
                return false;
            }
            self.last_seq = frame_seq;

            let size = [src.width.max(1), src.height.max(1)];
            let from_sharp = s.sharpen > 0.001;
            let rebuild = self.targets.as_ref().is_none_or(|t| {
                t.src_generation != src.generation || t.size != size || t.from_sharp != from_sharp
            });
            if rebuild {
                self.wait_idle();
                self.targets = None;
                if self
                    .flow
                    .as_ref()
                    .is_some_and(|f| f.as_ref().map_or(true, |f| f.size != size))
                {
                    self.flow = None;
                }
                self.prev_arrival = None;
                self.showing_generated = false;
                self.real_due = None;
            }
            if self.flow.is_none() {
                let r = OpticalFlow::new(device, queue, size[0], size[1]);
                match &r {
                    Ok(f) => log::info!(
                        "frame generation ready ({}x{}, {}px flow)",
                        size[0],
                        size[1],
                        f.grid
                    ),
                    Err(e) => log::warn!("frame generation unavailable: {e}"),
                }
                self.flow = Some(r);
            }
            let grid = match &self.flow {
                Some(Ok(f)) => f.grid,
                Some(Err(e)) => {
                    self.reason = Some(e.clone());
                    return false;
                }
                None => return false,
            };
            self.reason = None;
            if self.targets.is_none() {
                self.targets = Some(self.build_targets(device, src, size, grid, from_sharp, scale));
            }

            let arrival = arrival.unwrap_or(now);
            let gap = self
                .prev_arrival
                .map(|p| arrival.saturating_duration_since(p).as_secs_f64());
            let interpolate = gap.is_some_and(|g| g > 0.0 && g < MAX_GAP);
            if let Some(g) = gap.filter(|_| interpolate) {
                self.interval = self.interval * 0.9 + g * 0.1;
            }
            self.prev_arrival = Some(arrival);

            let t = self.targets.as_mut().expect("targets");
            t.cur ^= 1;
            t.filled = (t.filled + 1).min(2);
            let t = self.targets.as_ref().expect("targets");
            queue.write_buffer(
                &self.luma_ubo,
                0,
                bytemuck::bytes_of(&LumaParams {
                    linear: s.linear_light as u32,
                    _pad: [0; 3],
                }),
            );
            {
                let mut pass = super::begin_pass(encoder, "framegen_luma", &t.luma_view[t.cur]);
                pass.set_pipeline(&self.luma_pipeline);
                pass.set_bind_group(0, &t.luma_bg, &[]);
                pass.draw(0..3, 0..1);
            }
            let src_tex = if from_sharp { &src.sharp_tex } else { &src.rgb_tex };
            encoder.copy_texture_to_texture(
                src_tex.as_image_copy(),
                t.colour[t.cur].as_image_copy(),
                wgpu::Extent3d {
                    width: size[0],
                    height: size[1],
                    depth_or_array_layers: 1,
                },
            );
            if !(interpolate && t.filled >= 2) {
                self.showing_generated = false;
                self.real_due = None;
                return false;
            }
            // Leave both flow textures as render targets in wgpu's tracking: the
            // state the flow engine's lists expect to find them in (see nvof.rs).
            for view in [&t.forward_view, &t.backward_view] {
                drop(encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("framegen_touch"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view,
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
            }
            self.pending = Some(arrival);
            true
        }
    }

    /// Queue the optical flow for the frame staged by `encode`. Call right after
    /// submitting that encoder.
    pub fn dispatch(&mut self) {
        #[cfg(windows)]
        {
            let Some(arrival) = self.pending.take() else { return };
            let (Some(Ok(flow)), Some(t)) = (&mut self.flow, &self.targets) else {
                return;
            };
            let set = FlowSet {
                prev_luma: &t.luma[t.cur ^ 1],
                next_luma: &t.luma[t.cur],
                forward: &t.forward,
                backward: &t.backward,
            };
            match flow.execute(set) {
                Ok(()) => {
                    self.warp_pending = true;
                    self.showing_generated = true;
                    self.real_due =
                        Some(arrival + Duration::from_secs_f64(self.interval * 0.5));
                }
                Err(e) => {
                    log::warn!("{e}");
                    self.reason = Some(e);
                    self.showing_generated = false;
                    self.real_due = None;
                }
            }
        }
    }

    /// Render the in-between frame (after `dispatch`, before anything reads
    /// `showing`).
    pub fn encode_warp(
        &mut self,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        s: &Settings,
    ) {
        if !std::mem::take(&mut self.warp_pending) {
            return;
        }
        let Some(t) = &self.targets else { return };
        let strength = s.frame_gen_guard.clamp(0.0, 1.0);
        let radius = (32.0 + (4.0 - 32.0) * strength) * t.size[1] as f32 / 1080.0;
        queue.write_buffer(
            &self.warp_ubo,
            0,
            bytemuck::bytes_of(&WarpParams {
                size: [t.size[0] as f32, t.size[1] as f32],
                grid: t.grid as f32,
                t: 0.5,
                radius,
                guard: (strength > 0.001) as u32,
                softness: 0.04,
                _pad: 0.0,
            }),
        );
        let mut pass = super::begin_pass(encoder, "framegen_warp", &t.shown_view);
        pass.set_pipeline(&self.warp_pipeline);
        pass.set_bind_group(0, &t.warp_bg[t.cur], &[]);
        pass.draw(0..3, 0..1);
    }

    fn build_targets(
        &mut self,
        device: &wgpu::Device,
        src: &SourceBinding,
        size: [u32; 2],
        grid: u32,
        from_sharp: bool,
        (scale_bgl, scale_ubo, sampler): (&wgpu::BindGroupLayout, &wgpu::Buffer, &wgpu::Sampler),
    ) -> Targets {
        let mk = |label: &str, w: u32, h: u32, format: wgpu::TextureFormat, usage| {
            device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d {
                    width: w,
                    height: h,
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
        use wgpu::TextureUsages as U;
        let rt_tex = U::RENDER_ATTACHMENT | U::TEXTURE_BINDING;
        let (w, h) = (size[0], size[1]);
        let luma = [
            mk("framegen_luma0", w, h, wgpu::TextureFormat::R8Unorm, rt_tex),
            mk("framegen_luma1", w, h, wgpu::TextureFormat::R8Unorm, rt_tex),
        ];
        let colour = [
            mk("framegen_colour0", w, h, INTERMEDIATE, U::COPY_DST | U::TEXTURE_BINDING),
            mk("framegen_colour1", w, h, INTERMEDIATE, U::COPY_DST | U::TEXTURE_BINDING),
        ];
        let (fw, fh) = (w.div_ceil(grid), h.div_ceil(grid));
        let forward = mk("framegen_fwd", fw, fh, wgpu::TextureFormat::Rg16Sint, rt_tex);
        let backward = mk("framegen_bwd", fw, fh, wgpu::TextureFormat::Rg16Sint, rt_tex);
        let shown = mk("framegen_shown", w, h, INTERMEDIATE, rt_tex);
        let view = |t: &wgpu::Texture| t.create_view(&Default::default());
        let luma_view = [view(&luma[0]), view(&luma[1])];
        let colour_view = [view(&colour[0]), view(&colour[1])];
        let forward_view = view(&forward);
        let backward_view = view(&backward);
        let shown_view = view(&shown);
        let src_view = view(if from_sharp { &src.sharp_tex } else { &src.rgb_tex });

        fn tv(binding: u32, v: &wgpu::TextureView) -> wgpu::BindGroupEntry<'_> {
            wgpu::BindGroupEntry {
                binding,
                resource: wgpu::BindingResource::TextureView(v),
            }
        }
        let luma_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("framegen_luma"),
            layout: &self.luma_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.luma_ubo.as_entire_binding(),
                },
                tv(1, &src_view),
            ],
        });
        let warp_bg = |cur: usize| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("framegen_warp"),
                layout: &self.warp_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.warp_ubo.as_entire_binding(),
                    },
                    tv(1, &colour_view[cur ^ 1]),
                    tv(2, &colour_view[cur]),
                    tv(3, &forward_view),
                    tv(4, &backward_view),
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: wgpu::BindingResource::Sampler(&self.sampler),
                    },
                ],
            })
        };
        let warp_bg = [warp_bg(0), warp_bg(1)];
        let scale_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("scale_bg_generated"),
            layout: scale_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: scale_ubo.as_entire_binding(),
                },
                tv(1, &shown_view),
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(sampler),
                },
            ],
        });
        let id = self.next_id;
        self.next_id += 1;
        Targets {
            luma,
            luma_view,
            colour,
            forward,
            forward_view,
            backward,
            backward_view,
            shown_view,
            cur: 0,
            filled: 0,
            size,
            grid,
            src_generation: src.generation,
            from_sharp,
            luma_bg,
            warp_bg,
            scale_bg,
            id,
        }
    }
}

impl Drop for FrameGen {
    fn drop(&mut self) {
        if self.targets.is_some() {
            self.wait_idle();
        }
    }
}
