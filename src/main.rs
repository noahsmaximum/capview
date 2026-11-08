#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]
//! capview — low-latency, high-quality USB capture card viewer.

mod assets;
mod audio;
mod capture;
mod dsp;
mod render;
mod settings;
mod theme;
mod ui;

use capture::{Capture, CaptureCmd, CaptureState, DeviceInfo, FormatInfo};
use render::{PendingShot, VideoRenderer};
use settings::{AspectMode, OutputDepth, PresentMode, Settings};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};
use winit::application::ApplicationHandler;
use winit::event::{ElementState, MouseButton, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop, EventLoopProxy};
use winit::keyboard::{Key, KeyCode, ModifiersState, NamedKey, PhysicalKey};
use winit::window::{Fullscreen, Window, WindowId, WindowLevel};

#[derive(Debug, Clone, Copy)]
enum UserEvent {
    Wake,
}

struct Gfx {
    _adapter: wgpu::Adapter,
    device: wgpu::Device,
    queue: wgpu::Queue,
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,
    ten_bit_ok: bool,
    present_modes: Vec<PresentMode>,
    egui_renderer: egui_wgpu::Renderer,
    video: VideoRenderer,
    adapter_name: String,
}

#[derive(Clone, PartialEq, Default)]
struct AppliedSurface {
    present: Option<PresentMode>,
    latency: u32,
    depth: Option<OutputDepth>,
    size: [u32; 2],
}

struct App {
    proxy: EventLoopProxy<UserEvent>,
    wake_pending: Arc<AtomicBool>,
    settings: Settings,
    saved: Settings,
    last_change: Instant,
    window: Option<Arc<Window>>,
    gfx: Option<Gfx>,
    egui_ctx: egui::Context,
    egui_state: Option<egui_winit::State>,
    capture: Option<Capture>,
    opened: Option<(String, Option<String>)>,
    applied_surface: AppliedSurface,
    applied_on_top: Option<bool>,
    applied_ui_scale: f32,

    audio: Option<audio::AudioPassthrough>,
    audio_key: Option<(bool, bool, Option<String>, Option<String>, Option<String>)>,
    audio_error: Option<String>,
    audio_inputs: Vec<String>,
    audio_outputs: Vec<String>,

    show_ui: bool,
    modifiers: ModifiersState,
    last_mouse_move: Instant,
    cursor_hidden: bool,
    last_click: Option<Instant>,

    stats: ui::Stats,
    stat_t0: Instant,
    stat_frames0: u64,
    stat_presents: u32,
    latency_max_window: f32,
    last_shown_seq: u64,
    last_new_frame_present: Option<Instant>,

    screenshot_requested: bool,
    pending_shot: Option<(PendingShot, Arc<AtomicBool>)>,
    toast: Option<(String, Instant)>,
    next_ui_repaint: Option<Instant>,
    last_idle_redraw: Instant,
    last_render: Instant,
    /// CAPVIEW_SMOKE=<secs>: log stats, screenshot at 3 s, exit after N s.
    smoke: Option<(Instant, f32, bool, Instant)>,
    /// Settings as of the previous frame: auto-save waits for 1 s without changes.
    last_seen: Settings,
    assets: assets::Assets,
    presets: Vec<settings::PresetEntry>,
    /// Last preset loaded or saved, with its contents cached for the "modified" tag.
    active_preset: Option<(settings::PresetEntry, Settings)>,
    /// The presets folder is re-scanned while the panel is open (external renames etc.).
    presets_scanned: Instant,
    preset_name: String,
}

impl App {
    fn new(proxy: EventLoopProxy<UserEvent>) -> Self {
        let mut settings = Settings::load();
        let smoke_ui = std::env::var_os("CAPVIEW_SMOKE_UI").is_some();
        if smoke_ui {
            // CAPVIEW_SMOKE_UI=panel shows only the settings card (README screenshots).
            settings.show_stats = std::env::var("CAPVIEW_SMOKE_UI").as_deref() != Ok("panel");
        }
        let proxy_for_assets = proxy.clone();
        let last_seen = settings.clone();
        Self {
            proxy,
            wake_pending: Arc::new(AtomicBool::new(false)),
            saved: settings.clone(),
            settings,
            last_change: Instant::now(),
            window: None,
            gfx: None,
            egui_ctx: egui::Context::default(),
            egui_state: None,
            capture: None,
            opened: None,
            applied_surface: AppliedSurface::default(),
            applied_on_top: None,
            applied_ui_scale: 0.0,
            audio: None,
            audio_key: None,
            audio_error: None,
            audio_inputs: audio::list_inputs(),
            audio_outputs: audio::list_outputs(),
            show_ui: smoke_ui,
            modifiers: ModifiersState::default(),
            last_mouse_move: Instant::now(),
            cursor_hidden: false,
            last_click: None,
            stats: ui::Stats::default(),
            stat_t0: Instant::now(),
            stat_frames0: 0,
            stat_presents: 0,
            latency_max_window: 0.0,
            last_shown_seq: 0,
            last_new_frame_present: None,
            screenshot_requested: false,
            pending_shot: None,
            toast: None,
            next_ui_repaint: None,
            last_idle_redraw: Instant::now(),
            last_render: Instant::now(),
            last_seen,
            assets: assets::Assets::new({
                let p = proxy_for_assets;
                Arc::new(move || {
                    let _ = p.send_event(UserEvent::Wake);
                })
            }),
            presets: settings::list_presets(),
            active_preset: None,
            presets_scanned: Instant::now(),
            preset_name: String::new(),
            smoke: std::env::var("CAPVIEW_SMOKE")
                .ok()
                .and_then(|v| v.parse::<f32>().ok())
                .map(|secs| (Instant::now(), secs, false, Instant::now())),
        }
    }

    fn init_gfx(&mut self, window: Arc<Window>) {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::from_env().unwrap_or(native_backend()),
            ..wgpu::InstanceDescriptor::new_without_display_handle()
        });
        let surface = instance
            .create_surface(window.clone())
            .expect("create surface");
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
            apply_limit_buckets: false,
        }))
        .expect("no GPU adapter");
        let info = adapter.get_info();
        let adapter_name = format!("{} ({:?})", info.name, info.backend);
        log::info!("adapter: {adapter_name}");
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("capview"),
            required_limits: wgpu::Limits {
                max_texture_dimension_2d: 8192,
                ..wgpu::Limits::default()
            },
            ..Default::default()
        }))
        .expect("request device");

        let caps = surface.get_capabilities(&adapter);
        let ten_bit_ok = caps.formats.contains(&wgpu::TextureFormat::Rgb10a2Unorm);
        let mut present_modes = Vec::new();
        for (m, w) in [
            (PresentMode::Immediate, wgpu::PresentMode::Immediate),
            (PresentMode::Mailbox, wgpu::PresentMode::Mailbox),
            (PresentMode::Fifo, wgpu::PresentMode::Fifo),
        ] {
            if caps.present_modes.contains(&w) {
                present_modes.push(m);
            }
        }
        let size = window.inner_size();
        let config = surface
            .get_default_config(&adapter, size.width.max(1), size.height.max(1))
            .expect("surface unsupported");
        let format = wgpu::TextureFormat::Bgra8Unorm;
        let video = VideoRenderer::new(&device, format);
        let egui_renderer =
            egui_wgpu::Renderer::new(&device, format, egui_wgpu::RendererOptions::default());

        // Capture thread uploads straight into GPU textures.
        let proxy = self.proxy.clone();
        let pending = self.wake_pending.clone();
        let notify = Box::new(move || {
            if !pending.swap(true, Ordering::AcqRel) {
                let _ = proxy.send_event(UserEvent::Wake);
            }
        });
        self.capture = Some(Capture::start(device.clone(), queue.clone(), notify));

        self.gfx = Some(Gfx {
            _adapter: adapter,
            device,
            queue,
            surface,
            config,
            ten_bit_ok,
            present_modes,
            egui_renderer,
            video,
            adapter_name,
        });
        self.applied_surface = AppliedSurface::default();
        self.configure_surface();
    }

    fn reset_egui(&mut self) {
        self.assets.invalidate_egui();
        let Some(window) = &self.window else { return };
        self.egui_ctx = egui::Context::default();
        self.applied_ui_scale = 0.0;
        self.egui_state = Some(egui_winit::State::new(
            self.egui_ctx.clone(),
            egui::ViewportId::ROOT,
            window,
            Some(window.scale_factor() as f32),
            None,
            Some(8192),
        ));
    }

    fn configure_surface(&mut self) {
        let Some(window) = &self.window else { return };
        let Some(g) = &mut self.gfx else { return };
        let size = window.inner_size();
        if size.width == 0 || size.height == 0 {
            return;
        }
        let s = &self.settings;
        let want = AppliedSurface {
            present: Some(s.present_mode),
            latency: s.frame_latency,
            depth: Some(s.output_depth),
            size: [size.width, size.height],
        };
        if want == self.applied_surface {
            return;
        }
        let present = if g.present_modes.contains(&s.present_mode) {
            s.present_mode
        } else {
            PresentMode::Fifo
        };
        let format = if s.output_depth == OutputDepth::Bits10 && g.ten_bit_ok {
            wgpu::TextureFormat::Rgb10a2Unorm
        } else {
            wgpu::TextureFormat::Bgra8Unorm
        };
        let format_changed = g.config.format != format;
        g.config.format = format;
        g.config.view_formats = vec![format];
        g.config.width = size.width;
        g.config.height = size.height;
        g.config.present_mode = match present {
            PresentMode::Immediate => wgpu::PresentMode::Immediate,
            PresentMode::Mailbox => wgpu::PresentMode::Mailbox,
            PresentMode::Fifo => wgpu::PresentMode::Fifo,
        };
        g.config.desired_maximum_frame_latency = s.frame_latency.clamp(1, 3);
        g.config.alpha_mode = wgpu::CompositeAlphaMode::Opaque;
        g.surface.configure(&g.device, &g.config);
        if format_changed {
            g.video.set_target_format(&g.device, format);
            g.egui_renderer =
                egui_wgpu::Renderer::new(&g.device, format, egui_wgpu::RendererOptions::default());
            self.applied_surface = want;
            // New renderer has no textures: restart egui so it re-uploads the font atlas.
            self.reset_egui();
            return;
        }
        self.applied_surface = want;
    }

    /// Pick / reopen the capture device according to settings.
    fn sync_capture(&mut self) {
        // Test hook: CAPVIEW_NO_SOURCE=1 leaves every device closed (UI screenshots).
        if std::env::var_os("CAPVIEW_NO_SOURCE").is_some() {
            return;
        }
        let Some(cap) = &self.capture else { return };
        let devices = cap.shared.status.lock().devices.clone();
        if devices.is_empty() {
            return;
        }
        // Test override (smoke runs): CAPVIEW_DEVICE=<name substring>.
        if let Ok(want) = std::env::var("CAPVIEW_DEVICE") {
            let want = want.to_lowercase();
            if let Some(d) = devices
                .iter()
                .find(|d| d.name.to_lowercase().contains(&want))
            {
                if self.settings.device_link.as_ref() != Some(&d.link) {
                    self.settings.device_link = Some(d.link.clone());
                    self.settings.device_name = Some(d.name.clone());
                    self.settings.format_key = std::env::var("CAPVIEW_FORMAT").ok();
                }
            }
        }
        // Resolve the preferred device: saved link, then saved name, then heuristic.
        let link_ok = self
            .settings
            .device_link
            .as_ref()
            .is_some_and(|l| devices.iter().any(|d| &d.link == l));
        if !link_ok {
            let by_name = self
                .settings
                .device_name
                .as_ref()
                .and_then(|n| devices.iter().find(|d| &d.name == n));
            let pick = by_name.cloned().or_else(|| auto_pick(&devices));
            if let Some(d) = pick {
                self.settings.device_link = Some(d.link);
                self.settings.device_name = Some(d.name);
            }
        }
        let Some(link) = self.settings.device_link.clone() else {
            return;
        };
        let want = (link.clone(), self.settings.format_key.clone());
        if self.opened.as_ref() != Some(&want) {
            cap.send(CaptureCmd::Open {
                link,
                format_key: want.1.clone(),
            });
            self.opened = Some(want);
        }
    }

    fn sync_audio(&mut self) {
        let s = &self.settings;
        let video_name = self.settings.device_name.clone();
        let key = (
            s.audio_enabled,
            s.audio_ms2109_fix,
            s.audio_input.clone(),
            s.audio_output.clone(),
            video_name.clone(),
        );
        if self.audio_key.as_ref() != Some(&key) {
            self.audio_key = Some(key);
            self.audio = None;
            self.audio_error = None;
            if s.audio_enabled {
                let input = s
                    .audio_input
                    .clone()
                    .or_else(|| video_name.as_deref().and_then(audio::guess_input_for));
                match input {
                    Some(name) => {
                        match audio::start(&name, s.audio_output.as_deref(), s.audio_ms2109_fix) {
                            Ok(a) => self.audio = Some(a),
                            Err(e) => self.audio_error = Some(format!("Audio: {e}")),
                        }
                    }
                    None => {
                        self.audio_error = Some("No matching audio input — pick one above.".into())
                    }
                }
            }
        }
        if let Some(a) = &self.audio {
            a.shared.set_volume(self.settings.volume);
            a.shared.set_muted(self.settings.muted);
            a.shared.set_target_ms(self.settings.audio_buffer_ms);
            a.shared
                .dsp
                .dc_block
                .store(self.settings.audio_dc_block, Ordering::Relaxed);
            a.shared
                .dsp
                .gate
                .store(self.settings.audio_gate, Ordering::Relaxed);
            a.shared.dsp.set_gate_db(self.settings.audio_gate_db);
            a.shared
                .dsp
                .nr
                .store(self.settings.audio_nr, Ordering::Relaxed);
            a.shared
                .dsp
                .set_nr_strength(self.settings.audio_nr_strength);
            if let Some(e) = a.shared.error.lock().take() {
                self.audio_error = Some(e);
            }
        }
    }

    fn apply_settings(&mut self) {
        self.configure_surface();
        self.sync_capture();
        self.sync_audio();
        if let Some(w) = &self.window {
            if self.applied_on_top != Some(self.settings.always_on_top) {
                w.set_window_level(if self.settings.always_on_top {
                    WindowLevel::AlwaysOnTop
                } else {
                    WindowLevel::Normal
                });
                self.applied_on_top = Some(self.settings.always_on_top);
            }
        }
        if self.applied_ui_scale != self.settings.ui_scale {
            ui::apply_style(&self.egui_ctx, self.settings.ui_scale);
            self.applied_ui_scale = self.settings.ui_scale;
        }
    }

    fn toggle_fullscreen(&mut self) {
        if let Some(w) = &self.window {
            if w.fullscreen().is_some() {
                w.set_fullscreen(None);
            } else {
                w.set_fullscreen(Some(Fullscreen::Borderless(None)));
            }
        }
    }

    fn save_preset(&mut self, name: &str) {
        match settings::save_preset(name, &self.settings) {
            Ok(entry) => {
                self.toast(format!("Saved preset {}", entry.name));
                self.active_preset = Some((entry, self.settings.as_preset()));
                self.refresh_presets();
            }
            Err(e) => self.toast(format!("Couldn't save preset: {e}")),
        }
    }

    fn update_preset(&mut self) {
        let Some((entry, _)) = self.active_preset.clone() else {
            return;
        };
        match settings::update_preset(&entry, &self.settings) {
            Ok(()) => {
                self.toast(format!("Updated preset {}", entry.name));
                self.active_preset = Some((entry, self.settings.as_preset()));
            }
            Err(e) => self.toast(format!("Couldn't update preset: {e}")),
        }
    }

    fn load_preset(&mut self, path: &std::path::Path) {
        let entry = self.presets.iter().find(|e| e.path == path).cloned();
        match (entry, settings::load_preset(path)) {
            (Some(entry), Some(p)) => {
                self.settings.apply_preset(&p);
                self.toast(format!("Preset: {}", entry.name));
                self.active_preset = Some((entry, p));
            }
            _ => {
                self.toast("Preset not found (renamed or deleted?)");
                self.refresh_presets();
            }
        }
    }

    /// Re-read the presets folder. The active preset follows its file; if the file is
    /// gone (renamed/deleted outside the app) it's matched by contents, else cleared.
    fn refresh_presets(&mut self) {
        self.presets_scanned = Instant::now();
        self.presets = settings::list_presets();
        if let Some((entry, cached)) = self.active_preset.take() {
            let by_path = self.presets.iter().find(|e| e.path == entry.path).cloned();
            self.active_preset = match by_path {
                Some(e) => settings::load_preset(&e.path).map(|p| (e, p)),
                None => self
                    .presets
                    .iter()
                    .find(|e| settings::load_preset(&e.path).as_ref() == Some(&cached))
                    .map(|e| (e.clone(), cached)),
            };
        }
    }

    fn toast(&mut self, msg: impl Into<String>) {
        self.toast = Some((msg.into(), Instant::now()));
    }

    fn update_stats(&mut self) {
        let now = Instant::now();
        let dt = now.duration_since(self.stat_t0).as_secs_f32();
        if dt < 0.5 {
            return;
        }
        if let Some(cap) = &self.capture {
            let st = &cap.shared.stats;
            let frames = st.frames.load(Ordering::Relaxed);
            self.stats.capture_fps = frames.saturating_sub(self.stat_frames0) as f32 / dt;
            self.stat_frames0 = frames;
            self.stats.interval_ms = st.interval_us.load(Ordering::Relaxed) as f32 / 1000.0;
            self.stats.jitter_ms = st.jitter_us.load(Ordering::Relaxed) as f32 / 1000.0;
            self.stats.dropped = st.dropped.load(Ordering::Relaxed);
            self.stats.upload_ms = st.upload_us.load(Ordering::Relaxed) as f32 / 1000.0;
            let dl = st.device_latency_us.load(Ordering::Relaxed);
            self.stats.device_latency_ms = (dl > 0).then(|| dl as f32 / 1000.0);
        }
        self.stats.present_fps = self.stat_presents as f32 / dt;
        self.stat_presents = 0;
        self.stats.latency_max_ms = self.latency_max_window;
        self.latency_max_window = 0.0;
        if let Some(a) = &self.audio {
            self.stats.audio_fill_ms = Some(a.shared.fill());
            self.stats.audio_underruns = a.shared.underruns.load(Ordering::Relaxed);
            self.stats.audio_rates = Some((
                a.shared.in_rate.load(Ordering::Relaxed),
                a.shared.out_rate.load(Ordering::Relaxed),
            ));
        } else {
            self.stats.audio_fill_ms = None;
        }
        self.stat_t0 = now;
    }

    fn poll_screenshot(&mut self) {
        let Some(g) = &self.gfx else { return };
        if self.pending_shot.is_none() {
            return;
        }
        let _ = g.device.poll(wgpu::PollType::Poll);
        let ready = self
            .pending_shot
            .as_ref()
            .is_some_and(|(_, f)| f.load(Ordering::Acquire));
        if !ready {
            return;
        }
        let (shot, _) = self.pending_shot.take().unwrap();
        let rgb = shot.read_rgb8();
        let (w, h) = (shot.width, shot.height);
        let dir = self.settings.screenshot_dir();
        let path = match std::env::var("CAPVIEW_SHOT_PATH") {
            Ok(p) => std::path::PathBuf::from(p),
            Err(_) => dir.join(format!("capview_{}.png", timestamp())),
        };
        log::info!("screenshot -> {}", path.display());
        self.toast(format!("Saved {}", path.display()));
        std::thread::spawn(move || {
            let _ = std::fs::create_dir_all(&dir);
            if let Err(e) = write_png(&path, w, h, &rgb) {
                log::error!("screenshot failed: {e}");
            }
        });
    }

    fn handle_key(
        &mut self,
        event_loop: &ActiveEventLoop,
        key: &Key,
        code: Option<KeyCode>,
    ) -> bool {
        let ctrl = self.modifiers.control_key();
        let alt = self.modifiers.alt_key();
        match (key, code) {
            (Key::Named(NamedKey::F1), _) | (Key::Named(NamedKey::Tab), _) => {
                self.show_ui = !self.show_ui
            }
            (Key::Named(NamedKey::F2), _) => self.settings.show_stats = !self.settings.show_stats,
            (Key::Named(NamedKey::F11), _) => self.toggle_fullscreen(),
            (Key::Named(NamedKey::Enter), _) if alt => self.toggle_fullscreen(),
            (Key::Named(NamedKey::F12), _) => self.screenshot_requested = true,
            (_, Some(KeyCode::KeyS)) if ctrl => self.screenshot_requested = true,
            (_, Some(KeyCode::KeyQ)) if ctrl => event_loop.exit(),
            (Key::Named(NamedKey::Escape), _) => {
                if self
                    .window
                    .as_ref()
                    .is_some_and(|w| w.fullscreen().is_some())
                {
                    self.toggle_fullscreen();
                } else if self.show_ui {
                    self.show_ui = false;
                }
            }
            (_, Some(KeyCode::KeyM)) if !ctrl => {
                self.settings.muted = !self.settings.muted;
                self.toast(if self.settings.muted {
                    "Muted"
                } else {
                    "Unmuted"
                });
            }
            (_, Some(KeyCode::KeyP)) if !ctrl => {
                self.settings.always_on_top = !self.settings.always_on_top;
                self.toast(if self.settings.always_on_top {
                    "Pinned on top"
                } else {
                    "Unpinned"
                });
            }
            (_, Some(KeyCode::KeyI)) if !ctrl => {
                self.settings.aspect_mode = if self.settings.aspect_mode == AspectMode::Integer {
                    AspectMode::Fit
                } else {
                    AspectMode::Integer
                };
                self.toast(format!("{:?}", self.settings.aspect_mode));
            }
            (_, Some(code)) if ctrl && digit(code).is_some() => {
                let i = digit(code).unwrap();
                self.refresh_presets();
                match self.presets.get(i).map(|e| e.path.clone()) {
                    Some(path) => self.load_preset(&path),
                    None => self.toast(format!("No preset in slot {}", i + 1)),
                }
            }
            (_, Some(KeyCode::KeyR)) if !ctrl => {
                let d = Settings::default();
                let s = &mut self.settings;
                s.brightness = d.brightness;
                s.contrast = d.contrast;
                s.saturation = d.saturation;
                s.hue = d.hue;
                s.gamma = d.gamma;
                s.sharpen = d.sharpen;
                self.toast("Picture reset");
            }
            _ => return false,
        }
        true
    }

    fn render(&mut self, event_loop: &ActiveEventLoop) {
        let t0 = Instant::now();
        self.last_render = t0;
        self.wake_pending.store(false, Ordering::Release);
        self.apply_settings();
        self.update_stats();
        self.poll_screenshot();

        let Some(window) = self.window.clone() else {
            return;
        };
        let Some(cap) = &self.capture else { return };
        let frame_seq = cap.shared.frame_seq.load(Ordering::Acquire);
        let textures = cap.shared.textures.lock().clone();
        let (devices, formats, active_device, active_format, state): (
            Vec<DeviceInfo>,
            Vec<FormatInfo>,
            Option<DeviceInfo>,
            Option<FormatInfo>,
            CaptureState,
        ) = {
            let st = cap.shared.status.lock();
            (
                st.devices.clone(),
                st.formats.clone(),
                st.active_device.clone(),
                st.active_format.clone(),
                st.state.clone(),
            )
        };
        let last_arrival_ns = cap.shared.stats.last_arrival_ns.load(Ordering::Relaxed);
        let epoch = cap.shared.epoch;

        if self.show_ui && self.presets_scanned.elapsed() >= Duration::from_secs(2) {
            self.refresh_presets();
        }
        let Some(g) = &mut self.gfx else { return };
        match &textures {
            Some(t) => g.video.sync_source(&g.device, t),
            None => g.video.clear_source(),
        }

        for msg in self.assets.update(
            &mut self.settings,
            &mut g.video,
            &g.device,
            &g.queue,
            &self.egui_ctx,
        ) {
            self.toast = Some((msg, Instant::now()));
        }
        // Video rect for this frame (pixels), so the frame overlay never lags a resize.
        let video_rect = textures.as_ref().map(|t| {
            let geo = render::compute_geometry(
                t.width,
                t.height,
                [g.config.width, g.config.height],
                &self.settings,
            );
            [
                geo.dst_origin[0],
                geo.dst_origin[1],
                geo.dst_size[0],
                geo.dst_size[1],
            ]
        });

        // ---- egui ----
        let mut actions = Vec::new();
        let output_format = Some(format!("{:?}", g.config.format));
        let audio_active = self
            .audio
            .as_ref()
            .map(|a| (a.input_name.as_str(), a.output_name.as_str()));
        if let Some(a) = &self.audio {
            self.stats.audio_peaks = a.shared.peaks();
        }
        let raw_input = self.egui_state.as_mut().unwrap().take_egui_input(&window);
        let show_ui = self.show_ui;
        let has_source = textures.is_some();
        let toast = self
            .toast
            .as_ref()
            .filter(|(_, t)| t.elapsed() < Duration::from_millis(2500))
            .map(|(m, _)| m.clone());
        let fullscreen = window.fullscreen().is_some();
        let mut full_output = {
            let mut view = ui::UiView {
                settings: &mut self.settings,
                devices: &devices,
                formats: &formats,
                active_device: active_device.as_ref(),
                active_format: active_format.as_ref(),
                state: &state,
                resolved: g.video.resolved,
                output_format: output_format.clone(),
                audio_inputs: &self.audio_inputs,
                audio_outputs: &self.audio_outputs,
                audio_active,
                audio_error: self.audio_error.clone(),
                audio_floor_db: self.audio.as_ref().and_then(|a| a.shared.dsp.floor_db()),
                audio_gate_open: self
                    .audio
                    .as_ref()
                    .map(|a| a.shared.dsp.gate_open.load(Ordering::Relaxed)),
                audio_nr_ready: self
                    .audio
                    .as_ref()
                    .map(|a| a.shared.dsp.nr_ready.load(Ordering::Relaxed)),
                adapter: &g.adapter_name,
                present_modes: &g.present_modes,
                ten_bit_available: g.ten_bit_ok,
                fullscreen,
                stats: &self.stats,
                presets: &self.presets,
                active_preset: self.active_preset.as_ref().map(|(e, _)| e),
                preset_modified: false,
                preset_name: &mut self.preset_name,
                frame_loaded: self.assets.frame.is_some(),
                actions: &mut actions,
            };
            if let Some((_, cached)) = &self.active_preset {
                view.preset_modified = !view.settings.matches_preset(cached);
            }
            let frame_tex = self.assets.frame.as_ref();
            self.egui_ctx.run_ui(raw_input, |ui| {
                let ctx = ui.ctx().clone();
                if let Some(r) = video_rect {
                    ui::draw_frame(&ctx, r, view.settings, frame_tex);
                }
                if show_ui {
                    ui::settings_panel(ui, &mut view);
                }
                if view.settings.show_stats {
                    ui::stats_overlay(&ctx, &view);
                }
                if !has_source {
                    let msg = match view.state {
                        CaptureState::Error(e) => e.clone(),
                        CaptureState::Opening => "Opening device…".into(),
                        _ if view.devices.is_empty() => "No capture devices found".into(),
                        _ => "No signal".into(),
                    };
                    ui::no_signal(&ctx, &msg);
                }
                if let Some(t) = &toast {
                    ui::toast(&ctx, t);
                }
            })
        };
        self.egui_state
            .as_mut()
            .unwrap()
            .handle_platform_output(&window, full_output.platform_output);
        let ppp = full_output.pixels_per_point;
        let paint_jobs = self.egui_ctx.tessellate(full_output.shapes, ppp);
        let repaint_delay = full_output
            .viewport_output
            .get(&egui::ViewportId::ROOT)
            .map(|v| v.repaint_delay)
            .unwrap_or(Duration::MAX);

        // ---- GPU ----
        let size = [g.config.width, g.config.height];
        let screen = egui_wgpu::ScreenDescriptor {
            size_in_pixels: size,
            pixels_per_point: ppp,
        };
        let mut textures_delta = std::mem::take(&mut full_output.textures_delta);
        for (id, deltas) in &textures_delta.set {
            for delta in deltas.iter() {
                g.egui_renderer
                    .update_texture(&g.device, &g.queue, *id, delta);
            }
        }
        let to_free: Vec<egui::TextureId> = textures_delta.free.iter().copied().collect();
        textures_delta.clear();
        let mut encoder = g
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("frame"),
            });
        if let Some(t) = &textures {
            g.video
                .encode_convert(&g.queue, &mut encoder, t, &self.settings, frame_seq);
            g.video.encode_ambient(&mut encoder, &self.settings);
        }
        let mut shot = None;
        if self.screenshot_requested && self.pending_shot.is_none() {
            self.screenshot_requested = false;
            shot = g
                .video
                .encode_screenshot(&g.device, &mut encoder, &self.settings);
            if shot.is_none() {
                self.toast = Some(("No source to capture".into(), Instant::now()));
            }
        }
        let egui_cmds =
            g.egui_renderer
                .update_buffers(&g.device, &g.queue, &mut encoder, &paint_jobs, &screen);

        let frame = match g.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(f) => Some(f),
            wgpu::CurrentSurfaceTexture::Suboptimal(f) => {
                self.applied_surface = AppliedSurface::default();
                Some(f)
            }
            wgpu::CurrentSurfaceTexture::Outdated | wgpu::CurrentSurfaceTexture::Lost => {
                self.applied_surface = AppliedSurface::default();
                None
            }
            _ => None,
        };
        let Some(frame) = frame else {
            g.queue.submit(
                egui_cmds
                    .into_iter()
                    .chain(std::iter::once(encoder.finish())),
            );
            for id in &to_free {
                g.egui_renderer.free_texture(id);
            }
            return;
        };
        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        {
            let bg = self.settings.background;
            let mut pass = encoder
                .begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("present"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &view,
                        depth_slice: None,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color {
                                r: bg[0] as f64,
                                g: bg[1] as f64,
                                b: bg[2] as f64,
                                a: 1.0,
                            }),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                })
                .forget_lifetime();
            let ten_bit = g.config.format == wgpu::TextureFormat::Rgb10a2Unorm;
            g.video
                .draw_scale(&g.queue, &mut pass, size, &self.settings, ten_bit);
            g.egui_renderer.render(&mut pass, &paint_jobs, &screen);
        }
        g.queue.submit(
            egui_cmds
                .into_iter()
                .chain(std::iter::once(encoder.finish())),
        );
        if let Some(shot) = shot {
            let flag = Arc::new(AtomicBool::new(false));
            let f2 = flag.clone();
            shot.buffer.map_async(wgpu::MapMode::Read, .., move |r| {
                if r.is_ok() {
                    f2.store(true, Ordering::Release);
                }
            });
            self.pending_shot = Some((shot, flag));
        }
        window.pre_present_notify();
        g.queue.present(frame);
        for id in &to_free {
            g.egui_renderer.free_texture(id);
        }

        // ---- timing ----
        let now = Instant::now();
        self.stat_presents += 1;
        if frame_seq != self.last_shown_seq && last_arrival_ns > 0 {
            self.last_shown_seq = frame_seq;
            let arrival = epoch + Duration::from_nanos(last_arrival_ns);
            let lat = now.saturating_duration_since(arrival).as_secs_f32() * 1000.0;
            self.stats.latency_ms = self.stats.latency_ms * 0.9 + lat * 0.1;
            self.latency_max_window = self.latency_max_window.max(lat);
            if let Some(prev) = self.last_new_frame_present {
                let ms = now.duration_since(prev).as_secs_f32() * 1000.0;
                self.stats.frame_times.push_back(ms);
                if self.stats.frame_times.len() > 240 {
                    self.stats.frame_times.pop_front();
                }
            }
            self.last_new_frame_present = Some(now);
        }
        self.stats.cpu_frame_ms =
            self.stats.cpu_frame_ms * 0.9 + t0.elapsed().as_secs_f32() * 1000.0 * 0.1;

        // ---- actions ----
        for a in actions {
            match a {
                ui::UiAction::RefreshDevices => {
                    if let Some(c) = &self.capture {
                        c.send(CaptureCmd::RefreshDevices);
                    }
                    self.audio_inputs = audio::list_inputs();
                    self.audio_outputs = audio::list_outputs();
                }
                ui::UiAction::Screenshot => self.screenshot_requested = true,
                ui::UiAction::ToggleFullscreen => self.toggle_fullscreen(),
                ui::UiAction::RestartAudio => {
                    self.audio_key = None;
                    self.audio_inputs = audio::list_inputs();
                    self.audio_outputs = audio::list_outputs();
                }
                ui::UiAction::OpenScreenshotFolder => {
                    let dir = self.settings.screenshot_dir();
                    let _ = std::fs::create_dir_all(&dir);
                    open_folder(&dir);
                }
                ui::UiAction::Close => self.show_ui = false,
                ui::UiAction::PickBackground => self.assets.pick(assets::Kind::Background),
                ui::UiAction::PickFrame => self.assets.pick(assets::Kind::Frame),
                ui::UiAction::SavePreset(name) => self.save_preset(&name),
                ui::UiAction::UpdatePreset => self.update_preset(),
                ui::UiAction::LoadPreset(path) => self.load_preset(&path),
                ui::UiAction::DeletePreset(path) => {
                    match settings::delete_preset(&path) {
                        Ok(()) => self.toast("Deleted preset"),
                        Err(e) => self.toast(format!("Couldn't delete: {e}")),
                    }
                    if self
                        .active_preset
                        .as_ref()
                        .is_some_and(|(e, _)| e.path == path)
                    {
                        self.active_preset = None;
                    }
                    self.refresh_presets();
                }
                ui::UiAction::OpenPresetsFolder => {
                    let dir = settings::presets_dir();
                    let _ = std::fs::create_dir_all(&dir);
                    open_folder(&dir);
                }
            }
        }
        // Auto-save 1 s after the *last* change (previously reset every frame, so it
        // never fired while video was playing).
        if self.settings != self.last_seen {
            self.last_change = Instant::now();
            self.last_seen = self.settings.clone();
        }
        let _ = event_loop;
        if repaint_delay.is_zero() || self.screenshot_requested {
            window.request_redraw();
        } else if repaint_delay < Duration::from_secs(1) {
            self.next_ui_repaint = Some(Instant::now() + repaint_delay);
        }
    }
}

// Separate field for egui repaint scheduling (kept out of the big init for clarity).
impl App {
    fn schedule(&self) -> Option<Instant> {
        let mut next: Option<Instant> = self.next_ui_repaint;
        let mut push = |t: Instant| next = Some(next.map_or(t, |n| n.min(t)));
        if !self.cursor_hidden && !self.show_ui {
            push(self.last_mouse_move + Duration::from_secs_f32(self.settings.hide_cursor_secs));
        }
        if self.settings.show_stats {
            push(Instant::now() + Duration::from_millis(500));
        }
        if self
            .toast
            .as_ref()
            .is_some_and(|(_, t)| t.elapsed() < Duration::from_millis(2600))
        {
            push(Instant::now() + Duration::from_millis(100));
        }
        if self.settings != self.saved {
            push(self.last_change + Duration::from_secs(1));
        }
        if self.pending_shot.is_some() {
            push(Instant::now() + Duration::from_millis(2));
        }
        next
    }
}

/// Ctrl+1..9 → preset slot 0..8.
fn digit(code: KeyCode) -> Option<usize> {
    Some(match code {
        KeyCode::Digit1 => 0,
        KeyCode::Digit2 => 1,
        KeyCode::Digit3 => 2,
        KeyCode::Digit4 => 3,
        KeyCode::Digit5 => 4,
        KeyCode::Digit6 => 5,
        KeyCode::Digit7 => 6,
        KeyCode::Digit8 => 7,
        KeyCode::Digit9 => 8,
        _ => return None,
    })
}

fn auto_pick(devices: &[DeviceInfo]) -> Option<DeviceInfo> {
    let score = |n: &str| -> i32 {
        let l = n.to_lowercase();
        let mut s = 0;
        for good in [
            "capture",
            "hdmi",
            "usb3 video",
            "cam link",
            "elgato",
            "avermedia",
            "magewell",
            "video",
        ] {
            if l.contains(good) {
                s += 10;
            }
        }
        for bad in [
            "virtual",
            "obs",
            "quest",
            "webcam",
            "facetime",
            "integrated",
            "ir camera",
        ] {
            if l.contains(bad) {
                s -= 20;
            }
        }
        s
    };
    devices.iter().max_by_key(|d| score(&d.name)).cloned()
}

/// The GPU API each OS is best at; override with WGPU_BACKEND.
fn native_backend() -> wgpu::Backends {
    if cfg!(target_os = "windows") {
        wgpu::Backends::DX12
    } else if cfg!(target_os = "macos") {
        wgpu::Backends::METAL
    } else {
        wgpu::Backends::VULKAN
    }
}

/// Show a folder in the system file manager.
fn open_folder(dir: &std::path::Path) {
    let cmd = if cfg!(target_os = "windows") {
        "explorer"
    } else if cfg!(target_os = "macos") {
        "open"
    } else {
        "xdg-open"
    };
    if let Err(e) = std::process::Command::new(cmd).arg(dir).spawn() {
        log::warn!("couldn't open {}: {e}", dir.display());
    }
}

/// Local time as `YYYYMMDD_HHMMSS_mmm` for screenshot names.
#[cfg(windows)]
fn timestamp() -> String {
    use windows::Win32::System::SystemInformation::GetLocalTime;
    let t = unsafe { GetLocalTime() };
    format!(
        "{:04}{:02}{:02}_{:02}{:02}{:02}_{:03}",
        t.wYear, t.wMonth, t.wDay, t.wHour, t.wMinute, t.wSecond, t.wMilliseconds
    )
}

#[cfg(unix)]
fn timestamp() -> String {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    let secs = now.as_secs() as libc::time_t;
    // SAFETY: localtime_r only writes the tm we hand it.
    let t = unsafe {
        let mut t: libc::tm = std::mem::zeroed();
        libc::localtime_r(&secs, &mut t);
        t
    };
    format!(
        "{:04}{:02}{:02}_{:02}{:02}{:02}_{:03}",
        t.tm_year + 1900,
        t.tm_mon + 1,
        t.tm_mday,
        t.tm_hour,
        t.tm_min,
        t.tm_sec,
        now.subsec_millis()
    )
}

fn write_png(path: &std::path::Path, w: u32, h: u32, rgb: &[u8]) -> Result<(), String> {
    let file = std::fs::File::create(path).map_err(|e| e.to_string())?;
    let mut enc = png::Encoder::new(std::io::BufWriter::new(file), w, h);
    enc.set_color(png::ColorType::Rgb);
    enc.set_depth(png::BitDepth::Eight);
    enc.set_compression(png::Compression::Fast);
    let mut wr = enc.write_header().map_err(|e| e.to_string())?;
    wr.write_image_data(rgb).map_err(|e| e.to_string())
}

impl ApplicationHandler<UserEvent> for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }
        let size = self.settings.window_size.unwrap_or([1280, 720]);
        let attrs = Window::default_attributes()
            .with_title("capview")
            .with_inner_size(winit::dpi::PhysicalSize::new(
                size[0].max(320),
                size[1].max(180),
            ))
            .with_min_inner_size(winit::dpi::PhysicalSize::new(320, 180));
        let window = Arc::new(event_loop.create_window(attrs).expect("create window"));
        self.window = Some(window.clone());
        self.reset_egui();
        self.init_gfx(window.clone());
        if self.settings.start_fullscreen {
            window.set_fullscreen(Some(Fullscreen::Borderless(None)));
        }
        window.request_redraw();
    }

    fn user_event(&mut self, _event_loop: &ActiveEventLoop, _event: UserEvent) {
        if let Some(w) = &self.window {
            w.request_redraw();
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        let Some(window) = self.window.clone() else {
            return;
        };
        // Never feed RedrawRequested to egui: it answers "repaint" and we'd spin forever.
        let response = if matches!(event, WindowEvent::RedrawRequested) {
            None
        } else {
            self.egui_state
                .as_mut()
                .map(|s| s.on_window_event(&window, &event))
        };
        let consumed = response.as_ref().is_some_and(|r| r.consumed) && self.show_ui;
        // With the UI closed, pointer motion needs no repaint (it would re-present
        // the same video frame hundreds of times a second).
        let pointer_only = matches!(event, WindowEvent::CursorMoved { .. });
        if response.as_ref().is_some_and(|r| r.repaint) && (self.show_ui || !pointer_only) {
            // Pointer hover over the UI: cap re-presents at ~240 Hz (a 1 kHz mouse would
            // otherwise re-present the same video frame a thousand times a second).
            let next = self.last_render + Duration::from_micros(4167);
            if pointer_only && Instant::now() < next {
                self.next_ui_repaint = Some(self.next_ui_repaint.map_or(next, |t| t.min(next)));
            } else {
                window.request_redraw();
            }
        }
        match event {
            WindowEvent::CloseRequested => {
                if window.fullscreen().is_none() {
                    let s = window.inner_size();
                    self.settings.window_size = Some([s.width, s.height]);
                }
                self.settings.save();
                event_loop.exit();
            }
            WindowEvent::Resized(_) => {
                self.configure_surface();
                window.request_redraw();
            }
            WindowEvent::ScaleFactorChanged { .. } => {
                self.applied_ui_scale = 0.0;
                window.request_redraw();
            }
            WindowEvent::RedrawRequested => self.render(event_loop),
            WindowEvent::ModifiersChanged(m) => self.modifiers = m.state(),
            WindowEvent::CursorMoved { .. } => {
                self.last_mouse_move = Instant::now();
                if self.cursor_hidden {
                    window.set_cursor_visible(true);
                    self.cursor_hidden = false;
                }
            }
            WindowEvent::MouseInput {
                state: ElementState::Pressed,
                button,
                ..
            } => {
                if consumed || self.egui_ctx.egui_wants_pointer_input() {
                    return;
                }
                match button {
                    MouseButton::Right => {
                        self.show_ui = !self.show_ui;
                        window.request_redraw();
                    }
                    MouseButton::Left => {
                        let now = Instant::now();
                        if self
                            .last_click
                            .is_some_and(|t| now.duration_since(t) < Duration::from_millis(350))
                        {
                            self.toggle_fullscreen();
                            self.last_click = None;
                        } else {
                            self.last_click = Some(now);
                        }
                    }
                    _ => {}
                }
            }
            WindowEvent::KeyboardInput { event, .. } => {
                if event.state != ElementState::Pressed || event.repeat {
                    return;
                }
                if self.show_ui && self.egui_ctx.egui_wants_keyboard_input() {
                    return;
                }
                let code = match event.physical_key {
                    PhysicalKey::Code(c) => Some(c),
                    _ => None,
                };
                if self.handle_key(event_loop, &event.logical_key, code) {
                    window.request_redraw();
                }
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        let now = Instant::now();
        // Debounced settings save.
        if self.settings != self.saved
            && now.duration_since(self.last_change) > Duration::from_secs(1)
        {
            self.settings.save();
            self.saved = self.settings.clone();
        }
        // Cursor auto-hide over video.
        if let Some(w) = &self.window {
            if !self.cursor_hidden
                && !self.show_ui
                && now.duration_since(self.last_mouse_move).as_secs_f32()
                    > self.settings.hide_cursor_secs
            {
                w.set_cursor_visible(false);
                self.cursor_hidden = true;
            }
        }
        if let Some(t) = self.next_ui_repaint {
            if now >= t {
                self.next_ui_repaint = None;
                if let Some(w) = &self.window {
                    w.request_redraw();
                }
            }
        }
        // With no frames arriving, refresh the stats HUD twice a second (not every loop pass).
        let running = self
            .capture
            .as_ref()
            .is_some_and(|c| c.shared.running.load(Ordering::Relaxed));
        if self.settings.show_stats
            && !running
            && now.duration_since(self.last_idle_redraw) >= Duration::from_millis(500)
        {
            self.last_idle_redraw = now;
            if let Some(w) = &self.window {
                w.request_redraw();
            }
        }
        if self
            .toast
            .as_ref()
            .is_some_and(|(_, t)| t.elapsed() > Duration::from_millis(2600))
        {
            self.toast = None;
            if let Some(w) = &self.window {
                w.request_redraw();
            }
        }
        if self.pending_shot.is_some() {
            self.poll_screenshot();
        }
        if let Some((start, secs, shot_done, last_log)) = self.smoke {
            let el = now.duration_since(start).as_secs_f32();
            let mut sm = (start, secs, shot_done, last_log);
            if now.duration_since(last_log) >= Duration::from_secs(1) {
                sm.3 = now;
                let st = &self.stats;
                log::info!(
                    "SMOKE t={el:.1}s cap={:.2}fps present={:.2}fps interval={:.2}ms jitter={:.2}ms drops={} lat={:.2}ms max={:.2}ms dev={:?} upload={:.2}ms cpu={:.2}ms audio={:?} xruns={} color={:?}",
                    st.capture_fps,
                    st.present_fps,
                    st.interval_ms,
                    st.jitter_ms,
                    st.dropped,
                    st.latency_ms,
                    st.latency_max_ms,
                    st.device_latency_ms,
                    st.upload_ms,
                    st.cpu_frame_ms,
                    st.audio_fill_ms,
                    st.audio_underruns,
                    self.gfx.as_ref().map(|g| g.video.resolved)
                );
                if let Some(e) = &self.audio_error {
                    log::info!("SMOKE audio_error={e}");
                }
            }
            if !shot_done && el > 3.0 {
                sm.2 = true;
                self.screenshot_requested = true;
                if let Some(w) = &self.window {
                    w.request_redraw();
                }
            }
            self.smoke = Some(sm);
            if el > secs {
                self.settings.save();
                event_loop.exit();
                return;
            }
            event_loop.set_control_flow(ControlFlow::WaitUntil(now + Duration::from_millis(100)));
            return;
        }
        match self.schedule() {
            Some(t) => event_loop.set_control_flow(ControlFlow::WaitUntil(
                t.max(now + Duration::from_millis(1)),
            )),
            None => event_loop.set_control_flow(ControlFlow::Wait),
        }
    }

    fn exiting(&mut self, _event_loop: &ActiveEventLoop) {
        self.settings.save();
        self.audio = None;
        self.capture = None;
    }
}

fn main() {
    env_logger::Builder::from_env(
        env_logger::Env::default().default_filter_or("info,wgpu_core=warn,wgpu_hal=warn"),
    )
    .init();
    log::info!("capview v{}", env!("CARGO_PKG_VERSION"));
    let event_loop = EventLoop::<UserEvent>::with_user_event()
        .build()
        .expect("event loop");
    event_loop.set_control_flow(ControlFlow::Wait);
    let mut app = App::new(event_loop.create_proxy());
    event_loop.run_app(&mut app).expect("run");
}
