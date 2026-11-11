//! GPU pipeline: source planes -> convert (YUV->RGB, Rgba16Float @ source res)
//! -> optional CAS sharpen -> scale/letterbox into the swapchain -> egui overlay.

use crate::capture::{PixelFormat, SourceTextures};
use crate::settings::*;
use bytemuck::{Pod, Zeroable};

mod framegen;
mod upscale;

pub const INTERMEDIATE: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ConvertParams {
    size: [u32; 2],
    fmt: u32,
    flip_y: u32,
    row_r: [f32; 4],
    row_g: [f32; 4],
    row_b: [f32; 4],
    code_scale: f32,
    gamma: f32,
    linearize: u32,
    chroma_filter: u32,
    cosited: u32,
    _pad: [u32; 3],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct SharpenParams {
    sharpness: f32,
    linear: u32,
    _pad: [u32; 2],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ScaleParams {
    dst_origin: [f32; 2],
    dst_size: [f32; 2],
    crop_origin: [f32; 2],
    crop_size: [f32; 2],
    background: [f32; 4],
    scale: [f32; 2],
    filter: u32,
    rotation: u32,
    flip_h: u32,
    flip_v: u32,
    linear: u32,
    dither_levels: f32,
    antiring: f32,
    frame: u32,
    bg_mode: u32,
    bg_fit: u32,
    target: [f32; 2],
    bg_size: [f32; 2],
    bg_dim: f32,
    amb_style: u32,
    amb_strength: f32,
    amb_blur: f32,
    amb_reach: f32,
    amb_flow: f32,
    time: f32,
    amb_max_lod: f32,
    amb_crop: [f32; 4],
    amb_k: f32,
    amb_linear: u32,
    _pad0: u32,
    _pad1: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct AmbientLevel {
    out_size: [f32; 2],
    _pad: [f32; 2],
}

/// Blur chain of the current frame for the Ambient background.
struct AmbientChain {
    /// One pass per mip: render into `target`, reading the level above via `input`.
    passes: Vec<(wgpu::TextureView, wgpu::BindGroup)>,
    /// Scale-pass group 2 (all mips + trilinear sampler).
    group: wgpu::BindGroup,
    max_lod: f32,
}

/// Resolved colour interpretation for display in the UI.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ResolvedColor {
    pub matrix: u32,
    pub full_range: bool,
    pub cosited: bool,
}

struct SourceBinding {
    generation: u64,
    format: PixelFormat,
    width: u32,
    height: u32,
    rgb_tex: wgpu::Texture,
    sharp_tex: wgpu::Texture,
    convert_bgs: Vec<wgpu::BindGroup>,
    sharpen_bg: wgpu::BindGroup,
    scale_bg_rgb: wgpu::BindGroup,
    scale_bg_sharp: wgpu::BindGroup,
    ambient: AmbientChain,
}

pub struct VideoRenderer {
    convert_packed: wgpu::RenderPipeline,
    convert_planar: wgpu::RenderPipeline,
    convert_rgb: wgpu::RenderPipeline,
    bgl_packed: wgpu::BindGroupLayout,
    bgl_planar: wgpu::BindGroupLayout,
    bgl_uniform: wgpu::BindGroupLayout,
    bgl_float_tex: wgpu::BindGroupLayout,
    convert_ubo: wgpu::Buffer,

    sharpen_pipeline: wgpu::RenderPipeline,
    sharpen_bgl: wgpu::BindGroupLayout,
    sharpen_ubo: wgpu::Buffer,

    scale_module: wgpu::ShaderModule,
    scale_layout: wgpu::PipelineLayout,
    scale_pipeline: wgpu::RenderPipeline,
    scale_target: wgpu::TextureFormat,
    scale_bgl: wgpu::BindGroupLayout,
    scale_ubo: wgpu::Buffer,
    sampler: wgpu::Sampler,
    /// Group 1: background image + trilinear sampler (1x1 dummy when unused).
    bg_bgl: wgpu::BindGroupLayout,
    bg_sampler: wgpu::Sampler,
    bg_group: wgpu::BindGroup,
    bg_size: [f32; 2],
    bg_loaded: bool,
    /// Bound when no source is open, so the background still draws.
    dummy_scale_group: wgpu::BindGroup,
    ambient_first: wgpu::RenderPipeline,
    ambient_down: wgpu::RenderPipeline,
    ambient_bgl: wgpu::BindGroupLayout,
    /// Group 2 when there's no blur chain to bind.
    dummy_ambient_group: wgpu::BindGroup,
    /// frame_seq the blur chain was last built from.
    ambient_seq: Option<u64>,
    started: std::time::Instant,

    src: Option<SourceBinding>,
    upscale: upscale::Upscale,
    framegen: framegen::FrameGen,
    frame_counter: u32,
    last_converted_seq: u64,
    last_convert_params: Option<[u8; std::mem::size_of::<ConvertParams>()]>,
    pub resolved: ResolvedColor,
    /// Last computed dest rect (for UI / mouse mapping).
    pub dest_rect: [f32; 4],
}

fn tex_entry(binding: u32, sample: wgpu::TextureSampleType) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Texture {
            sample_type: sample,
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    }
}

fn ubo_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn fullscreen_pipeline(
    device: &wgpu::Device,
    label: &str,
    module: &wgpu::ShaderModule,
    layout: &wgpu::PipelineLayout,
    fs: &str,
    format: wgpu::TextureFormat,
) -> wgpu::RenderPipeline {
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(label),
        layout: Some(layout),
        vertex: wgpu::VertexState {
            module,
            entry_point: Some("vs_main"),
            buffers: &[],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module,
            entry_point: Some(fs),
            targets: &[Some(wgpu::ColorTargetState {
                format,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: Default::default(),
        }),
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview_mask: None,
        cache: None,
    })
}

fn ubo(device: &wgpu::Device, label: &str, size: usize) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: size as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

impl VideoRenderer {
    /// `upscale_format`: UAV-capable format for the AI upscaler, if any.
    pub fn new(
        device: &wgpu::Device,
        target: wgpu::TextureFormat,
        upscale_format: Option<wgpu::TextureFormat>,
    ) -> Self {
        let convert_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("convert"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/convert.wgsl").into()),
        });
        let sharpen_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("sharpen"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/sharpen.wgsl").into()),
        });
        let scale_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("scale"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/scale.wgsl").into()),
        });

        let uint = wgpu::TextureSampleType::Uint;
        let float = wgpu::TextureSampleType::Float { filterable: true };
        let bgl_packed = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("bgl_packed"),
            entries: &[ubo_entry(0), tex_entry(1, uint)],
        });
        let bgl_planar = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("bgl_planar"),
            entries: &[ubo_entry(0), tex_entry(1, uint), tex_entry(2, uint)],
        });
        let bgl_uniform = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("bgl_uniform"),
            entries: &[ubo_entry(0)],
        });
        let bgl_float_tex = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("bgl_float_tex"),
            entries: &[tex_entry(0, float)],
        });
        let layout = |bgls: &[&wgpu::BindGroupLayout]| {
            let bgls: Vec<Option<&wgpu::BindGroupLayout>> = bgls.iter().map(|b| Some(*b)).collect();
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &bgls,
                immediate_size: 0,
            })
        };
        let convert_packed = fullscreen_pipeline(
            device,
            "convert_packed",
            &convert_module,
            &layout(&[&bgl_packed]),
            "fs_packed",
            INTERMEDIATE,
        );
        let convert_planar = fullscreen_pipeline(
            device,
            "convert_planar",
            &convert_module,
            &layout(&[&bgl_planar]),
            "fs_planar",
            INTERMEDIATE,
        );
        let convert_rgb = fullscreen_pipeline(
            device,
            "convert_rgb",
            &convert_module,
            &layout(&[&bgl_uniform, &bgl_float_tex]),
            "fs_rgb",
            INTERMEDIATE,
        );

        let sharpen_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("sharpen_bgl"),
            entries: &[ubo_entry(0), tex_entry(1, float)],
        });
        let sharpen_pipeline = fullscreen_pipeline(
            device,
            "sharpen",
            &sharpen_module,
            &layout(&[&sharpen_bgl]),
            "fs_main",
            INTERMEDIATE,
        );

        let scale_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("scale_bgl"),
            entries: &[
                ubo_entry(0),
                tex_entry(1, float),
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let bg_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("bg_bgl"),
            entries: &[
                tex_entry(0, float),
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let scale_layout = layout(&[&scale_bgl, &bg_bgl, &bg_bgl]);
        let ambient_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("ambient"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/ambient.wgsl").into()),
        });
        let ambient_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("ambient_bgl"),
            entries: &[
                ubo_entry(0),
                tex_entry(1, float),
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let ambient_layout = layout(&[&ambient_bgl]);
        let ambient_first = fullscreen_pipeline(
            device,
            "ambient_first",
            &ambient_module,
            &ambient_layout,
            "fs_first",
            INTERMEDIATE,
        );
        let ambient_down = fullscreen_pipeline(
            device,
            "ambient_down",
            &ambient_module,
            &ambient_layout,
            "fs_down",
            INTERMEDIATE,
        );
        let scale_pipeline = fullscreen_pipeline(
            device,
            "scale",
            &scale_module,
            &scale_layout,
            "fs_main",
            target,
        );

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("bilinear"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        let bg_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("bg_trilinear"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            ..Default::default()
        });
        let dummy = |format| {
            device
                .create_texture(&wgpu::TextureDescriptor {
                    label: Some("dummy"),
                    size: wgpu::Extent3d {
                        width: 1,
                        height: 1,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format,
                    usage: wgpu::TextureUsages::TEXTURE_BINDING,
                    view_formats: &[],
                })
                .create_view(&Default::default())
        };
        let scale_ubo = ubo(device, "scale_ubo", std::mem::size_of::<ScaleParams>());
        let bg_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bg_group_dummy"),
            layout: &bg_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&dummy(
                        wgpu::TextureFormat::Rgba8Unorm,
                    )),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&bg_sampler),
                },
            ],
        });
        let dummy_ambient_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ambient_group_dummy"),
            layout: &bg_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&dummy(INTERMEDIATE)),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&bg_sampler),
                },
            ],
        });
        let dummy_scale_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("scale_dummy"),
            layout: &scale_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: scale_ubo.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&dummy(INTERMEDIATE)),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
        });

        Self {
            bg_bgl,
            bg_sampler,
            bg_group,
            bg_size: [1.0, 1.0],
            bg_loaded: false,
            dummy_scale_group,
            ambient_first,
            ambient_down,
            ambient_bgl,
            dummy_ambient_group,
            ambient_seq: None,
            started: std::time::Instant::now(),
            convert_packed,
            convert_planar,
            convert_rgb,
            bgl_packed,
            bgl_planar,
            bgl_uniform,
            bgl_float_tex,
            convert_ubo: ubo(device, "convert_ubo", std::mem::size_of::<ConvertParams>()),
            sharpen_pipeline,
            sharpen_bgl,
            sharpen_ubo: ubo(device, "sharpen_ubo", std::mem::size_of::<SharpenParams>()),
            scale_module,
            scale_layout,
            scale_pipeline,
            scale_target: target,
            scale_bgl,
            scale_ubo,
            sampler,
            src: None,
            upscale: upscale::Upscale::new(device, upscale_format),
            framegen: framegen::FrameGen::new(device),
            frame_counter: 0,
            last_converted_seq: u64::MAX,
            last_convert_params: None,
            resolved: ResolvedColor::default(),
            dest_rect: [0.0; 4],
        }
    }

    pub fn set_target_format(&mut self, device: &wgpu::Device, target: wgpu::TextureFormat) {
        if target != self.scale_target {
            self.scale_pipeline = fullscreen_pipeline(
                device,
                "scale",
                &self.scale_module,
                &self.scale_layout,
                "fs_main",
                target,
            );
            self.scale_target = target;
        }
    }

    pub fn clear_source(&mut self) {
        self.src = None;
    }

    /// Rebuild per-source resources when the capture thread swaps textures.
    pub fn sync_source(&mut self, device: &wgpu::Device, st: &SourceTextures) {
        if self
            .src
            .as_ref()
            .is_some_and(|s| s.generation == st.generation)
        {
            return;
        }
        let (w, h) = (st.width.max(1), st.height.max(1));
        let mk = |label: &str| {
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
                format: INTERMEDIATE,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::COPY_SRC,
                view_formats: &[],
            })
        };
        let rgb_tex = mk("rgb");
        let sharp_tex = mk("sharp");
        let rgb_view = rgb_tex.create_view(&Default::default());
        let sharp_view = sharp_tex.create_view(&Default::default());
        let views: Vec<wgpu::TextureView> = st
            .planes
            .iter()
            .map(|t| t.create_view(&Default::default()))
            .collect();
        let ubo_binding = wgpu::BindGroupEntry {
            binding: 0,
            resource: self.convert_ubo.as_entire_binding(),
        };

        let convert_bgs = match st.format {
            PixelFormat::Yuy2 | PixelFormat::Uyvy | PixelFormat::Mjpeg => {
                vec![device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("convert_packed_bg"),
                    layout: &self.bgl_packed,
                    entries: &[
                        ubo_binding,
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::TextureView(&views[0]),
                        },
                    ],
                })]
            }
            PixelFormat::Nv12 | PixelFormat::P010 => {
                vec![device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("convert_planar_bg"),
                    layout: &self.bgl_planar,
                    entries: &[
                        ubo_binding,
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::TextureView(&views[0]),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: wgpu::BindingResource::TextureView(&views[1]),
                        },
                    ],
                })]
            }
            PixelFormat::Bgra => vec![
                device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("convert_rgb_bg0"),
                    layout: &self.bgl_uniform,
                    entries: &[ubo_binding],
                }),
                device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("convert_rgb_bg1"),
                    layout: &self.bgl_float_tex,
                    entries: &[wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&views[0]),
                    }],
                }),
            ],
        };
        let sharpen_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("sharpen_bg"),
            layout: &self.sharpen_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.sharpen_ubo.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&rgb_view),
                },
            ],
        });
        let scale_bg = |view: &wgpu::TextureView| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("scale_bg"),
                layout: &self.scale_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.scale_ubo.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Sampler(&self.sampler),
                    },
                ],
            })
        };
        let ambient = self.build_ambient(device, w, h, &rgb_view);
        self.src = Some(SourceBinding {
            ambient,
            generation: st.generation,
            format: st.format,
            width: st.width,
            height: st.height,
            scale_bg_rgb: scale_bg(&rgb_view),
            scale_bg_sharp: scale_bg(&sharp_view),
            rgb_tex,
            sharp_tex,
            convert_bgs,
            sharpen_bg,
        });
        self.last_converted_seq = u64::MAX;
        self.last_convert_params = None;
        self.ambient_seq = None;
    }

    /// Quarter-res blur texture with a full mip chain, plus one pass per level.
    fn build_ambient(
        &self,
        device: &wgpu::Device,
        w: u32,
        h: u32,
        rgb_view: &wgpu::TextureView,
    ) -> AmbientChain {
        use wgpu::util::DeviceExt;
        let (bw, bh) = (w.div_ceil(4).max(1), h.div_ceil(4).max(1));
        let levels = 32 - bw.max(bh).leading_zeros();
        let tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("ambient"),
            size: wgpu::Extent3d {
                width: bw,
                height: bh,
                depth_or_array_layers: 1,
            },
            mip_level_count: levels,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: INTERMEDIATE,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let level_view = |l: u32| {
            tex.create_view(&wgpu::TextureViewDescriptor {
                base_mip_level: l,
                mip_level_count: Some(1),
                ..Default::default()
            })
        };
        let mut passes = Vec::with_capacity(levels as usize);
        for l in 0..levels {
            let size = [(bw >> l).max(1) as f32, (bh >> l).max(1) as f32];
            let ubo = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("ambient_level"),
                contents: bytemuck::bytes_of(&AmbientLevel {
                    out_size: size,
                    _pad: [0.0; 2],
                }),
                usage: wgpu::BufferUsages::UNIFORM,
            });
            let input = if l == 0 { rgb_view.clone() } else { level_view(l - 1) };
            let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("ambient_pass"),
                layout: &self.ambient_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: ubo.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&input),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Sampler(&self.sampler),
                    },
                ],
            });
            passes.push((level_view(l), group));
        }
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ambient_group"),
            layout: &self.bg_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(
                        &tex.create_view(&Default::default()),
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.bg_sampler),
                },
            ],
        });
        AmbientChain {
            passes,
            group,
            max_lod: (levels - 1) as f32,
        }
    }

    /// Rebuild the ambient blur chain from the converted frame (once per new frame,
    /// only while the Ambient background is on).
    pub fn encode_ambient(&mut self, encoder: &mut wgpu::CommandEncoder, s: &Settings) {
        if s.background_mode != BackgroundMode::Ambient {
            self.ambient_seq = None;
            return;
        }
        let Some(src) = &self.src else { return };
        if self.last_converted_seq == u64::MAX || self.ambient_seq == Some(self.last_converted_seq)
        {
            return;
        }
        for (i, (target, group)) in src.ambient.passes.iter().enumerate() {
            let mut pass = begin_pass(encoder, "ambient", target);
            pass.set_pipeline(if i == 0 {
                &self.ambient_first
            } else {
                &self.ambient_down
            });
            pass.set_bind_group(0, group, &[]);
            pass.draw(0..3, 0..1);
        }
        self.ambient_seq = Some(self.last_converted_seq);
    }

    /// AI upscale prep for this frame (after convert). Returns true when the caller
    /// must submit `encoder` now and then call [`Self::dispatch_upscale`].
    pub fn encode_upscale(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        target_size: [u32; 2],
        s: &Settings,
    ) -> bool {
        let geo = self
            .src
            .as_ref()
            .map(|src| compute_geometry(src.width, src.height, target_size, s));
        self.framegen.encode_warp(queue, encoder, s);
        let seq = self.last_converted_seq;
        let src = if seq == u64::MAX { None } else { self.src.as_ref() };
        // Re-upscale whenever the frame on screen changes: a new source frame, or the
        // switch between a generated frame and the real one.
        let generated = self.framegen.showing().map(|(_, view, id)| (view, id));
        let display_seq = seq.wrapping_mul(2) | generated.is_some() as u64;
        self.upscale.encode(
            device,
            queue,
            encoder,
            src,
            display_seq,
            geo.as_ref(),
            s,
            (&self.scale_bgl, &self.scale_ubo, &self.sampler),
            generated,
        )
    }

    /// Frame generation for this render (after convert, before the upscaler).
    /// Returns true when the caller must submit `encoder` now and then call
    /// [`Self::dispatch_framegen`].
    pub fn encode_framegen(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        arrival: Option<std::time::Instant>,
        s: &Settings,
    ) -> bool {
        let seq = self.last_converted_seq;
        let src = if seq == u64::MAX { None } else { self.src.as_ref() };
        self.framegen.encode(
            device,
            queue,
            encoder,
            src,
            seq,
            arrival,
            s,
            (&self.scale_bgl, &self.scale_ubo, &self.sampler),
        )
    }

    pub fn dispatch_framegen(&mut self) {
        self.framegen.dispatch();
    }

    /// When the next render is due to swap a generated frame for the real one.
    pub fn framegen_due(&self) -> Option<std::time::Instant> {
        self.framegen.due()
    }

    pub fn framegen_status(&self) -> Option<(String, bool)> {
        self.framegen.status()
    }

    pub fn dispatch_upscale(&mut self) {
        self.upscale.dispatch();
    }

    /// What the AI upscaler is doing (or why it can't), when it's switched on.
    pub fn upscale_status(&self) -> Option<(String, bool)> {
        self.upscale.status()
    }

    /// Encode convert (+ sharpen) when there's a new frame or params changed.
    pub fn encode_convert(
        &mut self,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        st: &SourceTextures,
        s: &Settings,
        frame_seq: u64,
    ) {
        let Some(src) = &self.src else { return };
        let (params, resolved) = convert_params(st, s);
        self.resolved = resolved;
        let bytes: [u8; std::mem::size_of::<ConvertParams>()] = bytemuck::cast(params);
        let params_changed = self.last_convert_params != Some(bytes);
        if !params_changed && frame_seq == self.last_converted_seq {
            return;
        }
        if params_changed {
            queue.write_buffer(&self.convert_ubo, 0, &bytes);
            self.last_convert_params = Some(bytes);
            queue.write_buffer(
                &self.sharpen_ubo,
                0,
                bytemuck::bytes_of(&SharpenParams {
                    sharpness: s.sharpen,
                    linear: s.linear_light as u32,
                    _pad: [0; 2],
                }),
            );
        }
        self.last_converted_seq = frame_seq;

        let rgb_view = src.rgb_tex.create_view(&Default::default());
        {
            let mut pass = begin_pass(encoder, "convert", &rgb_view);
            let pipeline = match src.format {
                PixelFormat::Yuy2 | PixelFormat::Uyvy | PixelFormat::Mjpeg => &self.convert_packed,
                PixelFormat::Nv12 | PixelFormat::P010 => &self.convert_planar,
                PixelFormat::Bgra => &self.convert_rgb,
            };
            pass.set_pipeline(pipeline);
            for (i, bg) in src.convert_bgs.iter().enumerate() {
                pass.set_bind_group(i as u32, bg, &[]);
            }
            pass.draw(0..3, 0..1);
        }
        if s.sharpen > 0.001 {
            let sharp_view = src.sharp_tex.create_view(&Default::default());
            let mut pass = begin_pass(encoder, "sharpen", &sharp_view);
            pass.set_pipeline(&self.sharpen_pipeline);
            pass.set_bind_group(0, &src.sharpen_bg, &[]);
            pass.draw(0..3, 0..1);
        }
    }

    /// Draw the video into an already-begun render pass covering the whole target.
    pub fn draw_scale(
        &mut self,
        queue: &wgpu::Queue,
        pass: &mut wgpu::RenderPass<'_>,
        target_size: [u32; 2],
        s: &Settings,
        ten_bit: bool,
    ) {
        self.frame_counter = self.frame_counter.wrapping_add(1);
        let geo = match &self.src {
            Some(src) => compute_geometry(src.width, src.height, target_size, s),
            // No source: park the video rect off-screen so only the background draws.
            None => Geometry {
                dst_origin: [-10.0, -10.0],
                dst_size: [1.0, 1.0],
                crop_origin: [0.0, 0.0],
                crop_size: [1.0, 1.0],
                scale: [1.0, 1.0],
            },
        };
        self.dest_rect = [
            geo.dst_origin[0],
            geo.dst_origin[1],
            geo.dst_size[0],
            geo.dst_size[1],
        ];
        // Ambient always reads the blur chain of the full converted frame.
        let src_dims = self
            .src
            .as_ref()
            .map_or([1.0, 1.0], |s| [s.width.max(1) as f32, s.height.max(1) as f32]);
        let amb_crop = [
            geo.crop_origin[0] / src_dims[0],
            geo.crop_origin[1] / src_dims[1],
            geo.crop_size[0] / src_dims[0],
            geo.crop_size[1] / src_dims[1],
        ];
        let amb_k = ((geo.crop_size[0] * geo.crop_size[1])
            / (geo.dst_size[0] * geo.dst_size[1]).max(1.0))
        .sqrt();
        // With the AI upscaler on, the video comes from its (gamma-encoded) output.
        let upscaled = self.upscale.scale_source().map(|(_, out)| out);
        let (crop_origin, crop_size, scale, video_linear) = match upscaled {
            Some([ow, oh]) => {
                let (ow, oh) = (ow as f32, oh as f32);
                (
                    [0.0, 0.0],
                    [ow, oh],
                    [
                        geo.scale[0] * geo.crop_size[0] / ow,
                        geo.scale[1] * geo.crop_size[1] / oh,
                    ],
                    false,
                )
            }
            None => (geo.crop_origin, geo.crop_size, geo.scale, s.linear_light),
        };
        let params = ScaleParams {
            dst_origin: geo.dst_origin,
            dst_size: geo.dst_size,
            crop_origin,
            crop_size,
            background: [s.background[0], s.background[1], s.background[2], 1.0],
            scale,
            filter: s.scale_filter.id(),
            rotation: s.rotation as u32,
            flip_h: s.flip_h as u32,
            flip_v: s.flip_v as u32,
            linear: video_linear as u32,
            dither_levels: if s.dither {
                if ten_bit { 1023.0 } else { 255.0 }
            } else {
                0.0
            },
            antiring: match s.scale_filter {
                ScaleFilter::CatmullRom | ScaleFilter::Lanczos2 | ScaleFilter::Lanczos3 => {
                    s.antiring
                }
                _ => 0.0,
            },
            frame: self.frame_counter,
            bg_mode: match s.background_mode {
                BackgroundMode::Image if self.bg_loaded => 1,
                BackgroundMode::Ambient if self.ambient_seq.is_some() => 2,
                _ => 0,
            },
            bg_fit: s.background_fit.id(),
            target: [target_size[0] as f32, target_size[1] as f32],
            bg_size: self.bg_size,
            bg_dim: s.background_dim.clamp(0.0, 1.0),
            amb_style: s.ambient_style.id(),
            amb_strength: s.ambient_strength.clamp(0.0, 1.5),
            amb_blur: s.ambient_blur.clamp(0.0, 1.0),
            amb_reach: s.ambient_reach.clamp(0.0, 1.0),
            amb_flow: s.ambient_flow.clamp(0.0, 1.0),
            time: (self.started.elapsed().as_secs_f64() % 3600.0) as f32,
            amb_max_lod: self.src.as_ref().map_or(0.0, |s| s.ambient.max_lod),
            amb_crop,
            amb_k,
            amb_linear: s.linear_light as u32,
            _pad0: 0,
            _pad1: 0,
        };
        queue.write_buffer(&self.scale_ubo, 0, bytemuck::bytes_of(&params));
        pass.set_pipeline(&self.scale_pipeline);
        let generated = self.framegen.showing().map(|(bg, _, _)| bg);
        let group = match (&self.src, self.upscale.scale_source(), generated) {
            (_, Some((bg, _)), _) => bg,
            (Some(_), None, Some(bg)) => bg,
            (Some(src), None, None) if s.sharpen > 0.001 => &src.scale_bg_sharp,
            (Some(src), None, None) => &src.scale_bg_rgb,
            (None, None, _) => &self.dummy_scale_group,
        };
        pass.set_bind_group(0, group, &[]);
        pass.set_bind_group(1, &self.bg_group, &[]);
        let ambient = match &self.src {
            Some(src) if self.ambient_seq.is_some() => &src.ambient.group,
            _ => &self.dummy_ambient_group,
        };
        pass.set_bind_group(2, ambient, &[]);
        pass.draw(0..3, 0..1);
    }

    /// Upload a decoded background image with a full mip chain (trilinear sampling
    /// keeps large photos from shimmering when shrunk to the window).
    pub fn set_background_image(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        img: Option<&image::RgbaImage>,
    ) {
        let Some(img) = img else {
            self.bg_loaded = false;
            return;
        };
        let (w, h) = img.dimensions();
        let levels = 32 - w.max(h).max(1).leading_zeros();
        let tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("background"),
            size: wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
            mip_level_count: levels,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            // Stored sRGB-encoded and sampled raw: values go straight to the
            // (non-sRGB) swapchain, like the video.
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let mut level_img = img.clone();
        for level in 0..levels {
            let (lw, lh) = level_img.dimensions();
            queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &tex,
                    mip_level: level,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                &level_img,
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(4 * lw),
                    rows_per_image: Some(lh),
                },
                wgpu::Extent3d {
                    width: lw,
                    height: lh,
                    depth_or_array_layers: 1,
                },
            );
            if level + 1 < levels {
                level_img = image::imageops::resize(
                    &level_img,
                    (lw / 2).max(1),
                    (lh / 2).max(1),
                    image::imageops::FilterType::Triangle,
                );
            }
        }
        let view = tex.create_view(&Default::default());
        self.bg_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bg_group"),
            layout: &self.bg_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.bg_sampler),
                },
            ],
        });
        self.bg_size = [w as f32, h as f32];
        self.bg_loaded = true;
    }

    /// Copy the processed source-resolution image into a mappable buffer.
    pub fn encode_screenshot(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        s: &Settings,
    ) -> Option<PendingShot> {
        let src = self.src.as_ref()?;
        let tex = if s.sharpen > 0.001 {
            &src.sharp_tex
        } else {
            &src.rgb_tex
        };
        let (w, h) = (src.width, src.height);
        let bpr = (w * 8).div_ceil(256) * 256;
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("screenshot"),
            size: bpr as u64 * h as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(bpr),
                    rows_per_image: Some(h),
                },
            },
            wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
        );
        Some(PendingShot {
            buffer,
            width: w,
            height: h,
            bpr,
            linear: s.linear_light,
        })
    }
}

fn begin_pass<'a>(
    encoder: &'a mut wgpu::CommandEncoder,
    label: &str,
    view: &'a wgpu::TextureView,
) -> wgpu::RenderPass<'a> {
    encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some(label),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view,
            depth_slice: None,
            resolve_target: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                store: wgpu::StoreOp::Store,
            },
        })],
        depth_stencil_attachment: None,
        timestamp_writes: None,
        occlusion_query_set: None,
        multiview_mask: None,
    })
}

pub struct PendingShot {
    pub buffer: wgpu::Buffer,
    pub width: u32,
    pub height: u32,
    pub bpr: u32,
    pub linear: bool,
}

impl PendingShot {
    /// Call after the buffer has been mapped. Returns tightly packed RGB8.
    pub fn read_rgb8(&self) -> Vec<u8> {
        let data = self.buffer.slice(..).get_mapped_range().expect("mapped");
        let mut out = Vec::with_capacity((self.width * self.height * 3) as usize);
        for y in 0..self.height as usize {
            let row = &data[y * self.bpr as usize..];
            for x in 0..self.width as usize {
                for c in 0..3 {
                    let i = x * 8 + c * 2;
                    let mut v = f16_to_f32(u16::from_le_bytes([row[i], row[i + 1]]));
                    if self.linear {
                        v = v.max(0.0).powf(1.0 / 2.2);
                    }
                    out.push((v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8);
                }
            }
        }
        drop(data);
        self.buffer.unmap();
        out
    }
}

fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) & 1) as u32;
    let exp = ((h >> 10) & 0x1f) as u32;
    let frac = (h & 0x3ff) as u32;
    let bits = if exp == 0 {
        if frac == 0 {
            sign << 31
        } else {
            // subnormal
            let mut e = 127 - 15 + 1;
            let mut f = frac;
            while f & 0x400 == 0 {
                f <<= 1;
                e -= 1;
            }
            (sign << 31) | ((e as u32) << 23) | ((f & 0x3ff) << 13)
        }
    } else if exp == 31 {
        (sign << 31) | 0x7f80_0000 | (frac << 13)
    } else {
        (sign << 31) | ((exp + 127 - 15) << 23) | (frac << 13)
    };
    f32::from_bits(bits)
}

// ---------------------------------------------------------------------------------
// Colour math

type Affine = [[f32; 4]; 3]; // rows: out = M.xyz . in + M.w

fn mul(a: &Affine, b: &Affine) -> Affine {
    // (a ∘ b)(x) = a(b(x))
    let mut o = [[0.0f32; 4]; 3];
    for r in 0..3 {
        for c in 0..3 {
            o[r][c] = (0..3).map(|k| a[r][k] * b[k][c]).sum();
        }
        o[r][3] = (0..3).map(|k| a[r][k] * b[k][3]).sum::<f32>() + a[r][3];
    }
    o
}

fn kr_kb(matrix: u32) -> (f32, f32) {
    match matrix {
        601 => (0.299, 0.114),
        2020 => (0.2627, 0.0593),
        _ => (0.2126, 0.0722),
    }
}

/// Y'PbPr (Y in 0..1, Pb/Pr in -0.5..0.5) -> R'G'B'
fn ypbpr_to_rgb(matrix: u32) -> Affine {
    let (kr, kb) = kr_kb(matrix);
    let kg = 1.0 - kr - kb;
    [
        [1.0, 0.0, 2.0 * (1.0 - kr), 0.0],
        [
            1.0,
            -2.0 * kb * (1.0 - kb) / kg,
            -2.0 * kr * (1.0 - kr) / kg,
            0.0,
        ],
        [1.0, 2.0 * (1.0 - kb), 0.0, 0.0],
    ]
}

fn rgb_to_ypbpr(matrix: u32) -> Affine {
    let (kr, kb) = kr_kb(matrix);
    let kg = 1.0 - kr - kb;
    [
        [kr, kg, kb, 0.0],
        [-0.5 * kr / (1.0 - kb), -0.5 * kg / (1.0 - kb), 0.5, 0.0],
        [0.5, -0.5 * kg / (1.0 - kr), -0.5 * kb / (1.0 - kr), 0.0],
    ]
}

fn adjust(s: &Settings) -> Affine {
    let c = s.contrast;
    let sat = s.saturation * c;
    let (sn, cs) = s.hue.to_radians().sin_cos();
    [
        [c, 0.0, 0.0, 0.5 - 0.5 * c + s.brightness],
        [0.0, sat * cs, -sat * sn, 0.0],
        [0.0, sat * sn, sat * cs, 0.0],
    ]
}

fn convert_params(st: &SourceTextures, s: &Settings) -> (ConvertParams, ResolvedColor) {
    let hints = st.hints;
    let matrix = match s.matrix {
        ColorMatrix::Bt601 => 601,
        ColorMatrix::Bt709 => 709,
        ColorMatrix::Bt2020 => 2020,
        ColorMatrix::Auto => hints
            .matrix
            .unwrap_or(if st.height >= 720 { 709 } else { 601 }),
    };
    let full_range = match s.range {
        ColorRange::Full => true,
        ColorRange::Limited => false,
        ColorRange::Auto => hints.full_range.unwrap_or(st.format.is_rgb()),
    };
    let cosited = match s.chroma_siting {
        ChromaSiting::Left => true,
        ChromaSiting::Center => false,
        ChromaSiting::Auto => hints.cosited.unwrap_or(true),
    };
    let ten_bit = st.format == PixelFormat::P010;

    let m = if st.format.is_rgb() {
        // RGB source: range expansion, then adjust in Y'PbPr space and back.
        let range: Affine = if full_range {
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
            ]
        } else {
            let k = 255.0 / 219.0;
            let o = -16.0 / 219.0;
            [[k, 0.0, 0.0, o], [0.0, k, 0.0, o], [0.0, 0.0, k, o]]
        };
        let to = rgb_to_ypbpr(709);
        // Y'PbPr with Y offset back to 0..1 for adjust (which centers luma at 0.5).
        mul(&ypbpr_to_rgb(709), &mul(&adjust(s), &mul(&to, &range)))
    } else {
        // Normalised codes n (8-bit-equivalent / 255) -> Y'PbPr
        let decode: Affine = if full_range {
            let off = if ten_bit {
                512.0 / 1023.0
            } else {
                128.0 / 255.0
            };
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, -off],
                [0.0, 0.0, 1.0, -off],
            ]
        } else {
            let ky = 255.0 / 219.0;
            let kc = 255.0 / 224.0;
            [
                [ky, 0.0, 0.0, -16.0 / 219.0],
                [0.0, kc, 0.0, -128.0 / 224.0],
                [0.0, 0.0, kc, -128.0 / 224.0],
            ]
        };
        mul(&ypbpr_to_rgb(matrix), &mul(&adjust(s), &decode))
    };

    let code_scale = match (ten_bit, full_range) {
        (true, false) => 1.0 / (64.0 * 1020.0),
        (true, true) => 1.0 / (64.0 * 1023.0),
        _ => 1.0 / 255.0,
    };

    (
        ConvertParams {
            size: [st.width, st.height],
            fmt: st.format.shader_id(),
            flip_y: st.flip_y as u32,
            row_r: m[0],
            row_g: m[1],
            row_b: m[2],
            code_scale,
            gamma: s.gamma,
            linearize: s.linear_light as u32,
            chroma_filter: match s.chroma_filter {
                ChromaFilter::Nearest => 0,
                ChromaFilter::Linear => 1,
                ChromaFilter::Cubic => 2,
            },
            cosited: cosited as u32,
            _pad: [0; 3],
        },
        ResolvedColor {
            matrix,
            full_range,
            cosited,
        },
    )
}

// ---------------------------------------------------------------------------------
// Geometry

pub struct Geometry {
    pub dst_origin: [f32; 2],
    pub dst_size: [f32; 2],
    pub crop_origin: [f32; 2],
    pub crop_size: [f32; 2],
    pub scale: [f32; 2],
}

pub fn compute_geometry(sw: u32, sh: u32, target: [u32; 2], s: &Settings) -> Geometry {
    // Leave room for the frame (and any user inset) so decorations are never clipped.
    let (full_w, full_h) = (target[0].max(1) as f32, target[1].max(1) as f32);
    let margin = (s.video_inset.max(0.0) + s.frame_space())
        .round()
        .min(full_w * 0.45)
        .min(full_h * 0.45);
    let (tw, th) = (
        (full_w - 2.0 * margin).max(16.0),
        (full_h - 2.0 * margin).max(16.0),
    );
    let cl = s.crop_l.min(sw.saturating_sub(1));
    let ct = s.crop_t.min(sh.saturating_sub(1));
    let cw = sw.saturating_sub(cl + s.crop_r).max(1) as f32;
    let ch = sh.saturating_sub(ct + s.crop_b).max(1) as f32;

    // Display aspect of the cropped content (before rotation).
    let mut ratio = s.aspect_override.ratio().unwrap_or(cw / ch);
    // "Native" size in output pixels: keep source height, width from aspect.
    let mut nat = [ch * ratio, ch];
    let rotated = matches!(s.rotation, Rotation::R90 | Rotation::R270);
    if rotated {
        ratio = 1.0 / ratio;
        nat = [nat[1], nat[0]];
    }

    let (mut w, mut h) = match s.aspect_mode {
        AspectMode::Stretch => (tw, th),
        AspectMode::Fit => {
            if tw / th > ratio {
                (th * ratio, th)
            } else {
                (tw, tw / ratio)
            }
        }
        AspectMode::Fill => {
            if tw / th > ratio {
                (tw, tw / ratio)
            } else {
                (th * ratio, th)
            }
        }
        AspectMode::Integer => {
            let k = (tw / nat[0]).min(th / nat[1]).floor().max(1.0);
            (nat[0] * k, nat[1] * k)
        }
        AspectMode::Native => (nat[0], nat[1]),
    };
    w *= s.zoom;
    h *= s.zoom;
    let mut x = margin + (tw - w) * 0.5;
    let mut y = margin + (th - h) * 0.5;
    // The AI upscaler renders at the exact on-screen size: keep the rect on whole
    // pixels so its output maps 1:1.
    if matches!(s.aspect_mode, AspectMode::Integer | AspectMode::Native)
        || s.scale_filter == ScaleFilter::Nearest
        || s.upscaler != Upscaler::Off
    {
        x = x.round();
        y = y.round();
        w = w.round();
        h = h.round();
    }
    let scale = if rotated {
        [h / cw, w / ch]
    } else {
        [w / cw, h / ch]
    };
    Geometry {
        dst_origin: [x, y],
        dst_size: [w.max(1.0), h.max(1.0)],
        crop_origin: [cl as f32, ct as f32],
        crop_size: [cw, ch],
        scale,
    }
}

#[cfg(test)]
mod shader_tests {
    /// Every pipeline shader must validate and translate to Metal (macOS), not just
    /// to HLSL (the Windows path that runs day to day).
    #[test]
    fn shaders_translate_to_msl() {
        for (name, src) in [
            ("convert", include_str!("shaders/convert.wgsl")),
            ("sharpen", include_str!("shaders/sharpen.wgsl")),
            ("scale", include_str!("shaders/scale.wgsl")),
            ("upscale_prep", include_str!("shaders/upscale_prep.wgsl")),
            ("framegen_luma", include_str!("shaders/framegen_luma.wgsl")),
            ("framegen_warp", include_str!("shaders/framegen_warp.wgsl")),
            ("ambient", include_str!("shaders/ambient.wgsl")),
        ] {
            let module = naga::front::wgsl::parse_str(src)
                .unwrap_or_else(|e| panic!("{name}: {}", e.emit_to_string(src)));
            let info = naga::valid::Validator::new(
                naga::valid::ValidationFlags::all(),
                naga::valid::Capabilities::all(),
            )
            .validate(&module)
            .unwrap_or_else(|e| panic!("{name}: {e:?}"));
            let options = naga::back::msl::Options {
                lang_version: (2, 1),
                ..Default::default()
            };
            for ep in &module.entry_points {
                let pipeline = naga::back::msl::PipelineOptions {
                    entry_point: Some((ep.stage, ep.name.clone())),
                    ..Default::default()
                };
                naga::back::msl::write_string(&module, &info, &options, &pipeline)
                    .unwrap_or_else(|e| panic!("{name}/{}: {e:?}", ep.name));
            }
        }
    }
}
