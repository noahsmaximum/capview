//! egui overlay: floating settings card, stats HUD, no-signal card, toasts.
//! Visual language lives in `theme.rs`.

use crate::capture::{CaptureState, DeviceInfo, FormatInfo};
use crate::render::ResolvedColor;
use crate::settings::*;
use crate::theme::{self, BONE, Btn, COBALT, INK, INK_SOFT, PINE, SKY, TAN, VERMILLION};
use egui::{Color32, RichText};

#[derive(Default)]
pub struct Stats {
    pub capture_fps: f32,
    pub present_fps: f32,
    pub interval_ms: f32,
    pub jitter_ms: f32,
    pub dropped: u64,
    pub latency_ms: f32,
    pub latency_max_ms: f32,
    pub device_latency_ms: Option<f32>,
    pub upload_ms: f32,
    pub cpu_frame_ms: f32,
    pub frame_times: std::collections::VecDeque<f32>,
    pub audio_fill_ms: Option<f32>,
    pub audio_underruns: u64,
    pub audio_rates: Option<(u32, u32)>,
    pub audio_peaks: (f32, f32),
}

pub enum UiAction {
    RefreshDevices,
    Screenshot,
    ToggleFullscreen,
    RestartAudio,
    OpenScreenshotFolder,
    Close,
}

pub struct UiView<'a> {
    pub settings: &'a mut Settings,
    pub devices: &'a [DeviceInfo],
    pub formats: &'a [FormatInfo],
    pub active_device: Option<&'a DeviceInfo>,
    pub active_format: Option<&'a FormatInfo>,
    pub state: &'a CaptureState,
    pub resolved: ResolvedColor,
    pub output_format: Option<String>,
    pub audio_inputs: &'a [String],
    pub audio_outputs: &'a [String],
    pub audio_active: Option<(&'a str, &'a str)>,
    pub audio_error: Option<String>,
    pub adapter: &'a str,
    pub present_modes: &'a [PresentMode],
    pub ten_bit_available: bool,
    pub fullscreen: bool,
    pub stats: &'a Stats,
    pub actions: &'a mut Vec<UiAction>,
}

pub fn apply_style(ctx: &egui::Context, scale: f32) {
    theme::apply(ctx, scale);
}

const PANEL_W: f32 = 372.0;
const LABEL_W: f32 = 104.0;
const EDGE: f32 = 18.0;

fn combo<T: PartialEq + Copy>(ui: &mut egui::Ui, id: &str, value: &mut T, options: &[(T, &str)]) {
    let cur = options
        .iter()
        .find(|(v, _)| *v == *value)
        .map(|(_, l)| *l)
        .unwrap_or("?");
    egui::ComboBox::from_id_salt(id)
        .selected_text(cur)
        .width(200.0)
        .show_ui(ui, |ui| {
            for (v, l) in options {
                ui.selectable_value(value, *v, *l);
            }
        });
}

fn row(ui: &mut egui::Ui, label: &str, add: impl FnOnce(&mut egui::Ui)) {
    ui.horizontal(|ui| {
        ui.allocate_ui_with_layout(
            egui::vec2(LABEL_W, 22.0),
            egui::Layout::left_to_right(egui::Align::Center),
            |ui| {
                ui.set_min_width(LABEL_W);
                ui.label(RichText::new(label).color(INK_SOFT));
            },
        );
        add(ui);
    });
}

fn ellipsize(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        s.to_owned()
    } else {
        format!("{}…", s.chars().take(max - 1).collect::<String>())
    }
}

fn note(ui: &mut egui::Ui, text: &str) {
    ui.label(RichText::new(text).size(12.0).color(INK_SOFT));
}

fn state_tag(ui: &mut egui::Ui, state: &CaptureState) {
    match state {
        CaptureState::Running => theme::tag(ui, "● live", PINE, BONE),
        CaptureState::Opening => theme::tag(ui, "opening", COBALT, BONE),
        CaptureState::NoSignal => theme::tag(ui, "no signal", VERMILLION, BONE),
        CaptureState::Error(_) => theme::tag(ui, "error", VERMILLION, BONE),
        CaptureState::Idle => theme::tag(ui, "idle", TAN, INK),
    };
}

pub fn settings_panel(ui: &mut egui::Ui, v: &mut UiView) {
    let screen = ui.max_rect();
    egui::Area::new(egui::Id::unique("settings_card"))
        .anchor(egui::Align2::RIGHT_TOP, egui::vec2(-EDGE - 6.0, EDGE))
        .movable(false)
        .order(egui::Order::Foreground)
        .show(ui.ctx(), |ui| {
            // Over video an ink shadow disappears; cobalt keeps the hard-shadow read.
            theme::card(BONE)
                .shadow(theme::hard_shadow(6, COBALT))
                .show(ui, |ui| {
                    ui.set_width(PANEL_W);
                    // Header: wordmark, marginalia, live tag, close.
                    ui.horizontal(|ui| {
                        theme::wordmark(ui, 28.0);
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            if theme::button(ui, "esc", Btn::Secondary)
                                .on_hover_text("Close (F1 / Tab / Esc)")
                                .clicked()
                            {
                                v.actions.push(UiAction::Close);
                            }
                        });
                    });
                    ui.horizontal(|ui| {
                        ui.label(theme::margin_text("low-latency capture viewer // v0.1"));
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            state_tag(ui, v.state)
                        });
                    });
                    ui.add_space(4.0);
                    theme::dashed_rule(ui);
                    ui.add_space(2.0);

                    let max_h = (screen.height() - 2.0 * EDGE - 210.0).max(160.0);
                    egui::ScrollArea::vertical()
                        .max_height(max_h)
                        .auto_shrink([false, true])
                        .show(ui, |ui| {
                            ui.set_width(PANEL_W - 10.0);
                            device_section(ui, v);
                            picture_section(ui, v);
                            scaling_section(ui, v);
                            display_section(ui, v);
                            audio_section(ui, v);
                            overlay_section(ui, v);
                        });
                    ui.add_space(4.0);
                    theme::dashed_rule(ui);
                    ui.label(theme::margin_text(
                        "F1 menu · F2 stats · F11 full · ^S shot · M mute · P pin",
                    ));
                });
        });
}

fn device_section(ui: &mut egui::Ui, v: &mut UiView) {
    theme::section(ui, 1, "Source", true, |ui| {
        if let CaptureState::Error(e) = v.state {
            ui.label(RichText::new(e.as_str()).color(VERMILLION).size(12.5));
        }
        row(ui, "Device", |ui| {
            let cur = v
                .active_device
                .map(|d| d.name.clone())
                .or(v.settings.device_name.clone());
            egui::ComboBox::from_id_salt("dev")
                .selected_text(ellipsize(&cur.unwrap_or_else(|| "None".into()), 18))
                .width(150.0)
                .show_ui(ui, |ui| {
                    for d in v.devices {
                        let sel = v.settings.device_link.as_deref() == Some(d.link.as_str());
                        if ui.selectable_label(sel, &d.name).clicked() && !sel {
                            v.settings.device_link = Some(d.link.clone());
                            v.settings.device_name = Some(d.name.clone());
                            v.settings.format_key = None;
                        }
                    }
                });
            if theme::button(ui, "scan", Btn::Secondary)
                .on_hover_text("Rescan devices")
                .clicked()
            {
                v.actions.push(UiAction::RefreshDevices);
            }
        });
        row(ui, "Format", |ui| {
            let cur = v
                .active_format
                .map(|f| f.short_label())
                .unwrap_or_else(|| "—".into());
            egui::ComboBox::from_id_salt("fmt")
                .selected_text(cur)
                .width(200.0)
                .height(420.0)
                .show_ui(ui, |ui| {
                    let auto = v.settings.format_key.is_none();
                    if ui.selectable_label(auto, "Auto (best quality)").clicked() {
                        v.settings.format_key = None;
                    }
                    for f in v.formats {
                        let key = f.key();
                        let sel = v.settings.format_key.as_deref() == Some(key.as_str());
                        if ui.selectable_label(sel, f.label()).clicked() {
                            v.settings.format_key = Some(key);
                        }
                    }
                });
        });
        if let Some(f) = v.active_format {
            ui.horizontal(|ui| {
                ui.add_space(LABEL_W + 8.0);
                if f.uncompressed() {
                    theme::tag(ui, "uncompressed", PINE, BONE);
                } else if f.windows_decoded {
                    theme::tag(ui, "windows 4:2:0", TAN, INK);
                } else {
                    theme::tag(ui, "mjpeg 4:2:2", COBALT, BONE);
                }
            });
            let text = if f.uncompressed() {
                "Uncompressed — best quality, lowest latency."
            } else if f.windows_decoded {
                "MJPEG decoded by Windows to 4:2:0 — pick the MJPEG mode for 4:2:2 colour."
            } else {
                "MJPEG (lossy), decoded here to 4:2:2. Uncompressed modes are limited by USB bandwidth."
            };
            note(ui, text);
            if let Some(hint) = slow_mode_hint(f, v.stats.capture_fps) {
                ui.label(RichText::new(hint).size(12.0).color(VERMILLION));
            }
        }
    });
}

/// Cheap cards (e.g. MS2109) advertise 1080p60 but deliver ~30 fps.
pub fn slow_mode_hint(f: &FormatInfo, measured: f32) -> Option<String> {
    let nominal = f.fps() as f32;
    if measured > 1.0 && nominal > 0.0 && measured < nominal * 0.8 {
        Some(format!(
            "Device is only delivering ~{measured:.0} of {nominal:.0} fps in this mode (USB bandwidth / card limit). \
             A lower resolution may give true {nominal:.0} fps."
        ))
    } else {
        None
    }
}

fn picture_section(ui: &mut egui::Ui, v: &mut UiView) {
    theme::section(ui, 2, "Picture", true, |ui| {
        let s = &mut *v.settings;
        let r = v.resolved;
        let m_auto = if r.matrix == 0 {
            "Auto".to_owned()
        } else {
            format!("Auto (BT.{})", r.matrix)
        };
        row(ui, "Matrix", |ui| {
            combo(
                ui,
                "matrix",
                &mut s.matrix,
                &[
                    (ColorMatrix::Auto, m_auto.as_str()),
                    (ColorMatrix::Bt601, "BT.601 (SD)"),
                    (ColorMatrix::Bt709, "BT.709 (HD)"),
                    (ColorMatrix::Bt2020, "BT.2020 (UHD)"),
                ],
            );
        });
        let r_auto = format!("Auto ({})", if r.full_range { "full" } else { "limited" });
        row(ui, "Range", |ui| {
            combo(
                ui,
                "range",
                &mut s.range,
                &[
                    (ColorRange::Auto, r_auto.as_str()),
                    (ColorRange::Limited, "Limited (16–235)"),
                    (ColorRange::Full, "Full (0–255)"),
                ],
            );
        });
        row(ui, "Chroma filter", |ui| {
            combo(
                ui,
                "chroma",
                &mut s.chroma_filter,
                &[
                    (ChromaFilter::Cubic, "Bicubic (sharpest)"),
                    (ChromaFilter::Linear, "Bilinear"),
                    (ChromaFilter::Nearest, "Nearest"),
                ],
            );
        });
        let c_auto = format!("Auto ({})", if r.cosited { "left" } else { "center" });
        row(ui, "Chroma siting", |ui| {
            combo(
                ui,
                "siting",
                &mut s.chroma_siting,
                &[
                    (ChromaSiting::Auto, c_auto.as_str()),
                    (ChromaSiting::Left, "Left / co-sited"),
                    (ChromaSiting::Center, "Center"),
                ],
            );
        });
        ui.add_space(2.0);
        row(ui, "Brightness", |ui| {
            ui.add(egui::Slider::new(&mut s.brightness, -0.5..=0.5).fixed_decimals(3));
        });
        row(ui, "Contrast", |ui| {
            ui.add(egui::Slider::new(&mut s.contrast, 0.0..=2.0).fixed_decimals(3));
        });
        row(ui, "Saturation", |ui| {
            ui.add(egui::Slider::new(&mut s.saturation, 0.0..=2.0).fixed_decimals(3));
        });
        row(ui, "Hue", |ui| {
            ui.add(
                egui::Slider::new(&mut s.hue, -180.0..=180.0)
                    .suffix("°")
                    .fixed_decimals(1),
            );
        });
        row(ui, "Gamma", |ui| {
            ui.add(egui::Slider::new(&mut s.gamma, 0.5..=2.0).fixed_decimals(3));
        });
        row(ui, "Sharpen (CAS)", |ui| {
            ui.add(egui::Slider::new(&mut s.sharpen, 0.0..=1.0).fixed_decimals(2));
        });
        ui.add_space(2.0);
        if theme::button(ui, "reset picture", Btn::Secondary).clicked() {
            let d = Settings::default();
            s.brightness = d.brightness;
            s.contrast = d.contrast;
            s.saturation = d.saturation;
            s.hue = d.hue;
            s.gamma = d.gamma;
            s.sharpen = d.sharpen;
        }
    });
}

fn scaling_section(ui: &mut egui::Ui, v: &mut UiView) {
    theme::section(ui, 3, "Scaling", true, |ui| {
        let s = &mut *v.settings;
        row(ui, "Filter", |ui| {
            let opts: Vec<(ScaleFilter, &str)> =
                ScaleFilter::ALL.iter().map(|f| (*f, f.label())).collect();
            combo(ui, "filter", &mut s.scale_filter, &opts);
        });
        if matches!(
            s.scale_filter,
            ScaleFilter::CatmullRom | ScaleFilter::Lanczos2 | ScaleFilter::Lanczos3
        ) {
            row(ui, "Anti-ringing", |ui| {
                ui.add(egui::Slider::new(&mut s.antiring, 0.0..=1.0).fixed_decimals(2));
            });
        }
        row(ui, "", |ui| {
            theme::check(ui, &mut s.linear_light, "Scale in linear light");
        });
        row(ui, "Size", |ui| {
            combo(
                ui,
                "aspect",
                &mut s.aspect_mode,
                &[
                    (AspectMode::Fit, "Fit (letterbox)"),
                    (AspectMode::Fill, "Fill (crop)"),
                    (AspectMode::Stretch, "Stretch"),
                    (AspectMode::Integer, "Integer scale"),
                    (AspectMode::Native, "1:1 native"),
                ],
            );
        });
        row(ui, "Aspect ratio", |ui| {
            combo(
                ui,
                "ar",
                &mut s.aspect_override,
                &[
                    (AspectOverride::Source, "Source (square pixels)"),
                    (AspectOverride::R16_9, "16:9"),
                    (AspectOverride::R4_3, "4:3"),
                    (AspectOverride::R16_10, "16:10"),
                    (AspectOverride::R21_9, "21:9"),
                    (AspectOverride::R1_1, "1:1"),
                ],
            );
        });
        row(ui, "Zoom", |ui| {
            ui.add(
                egui::Slider::new(&mut s.zoom, 0.25..=4.0)
                    .logarithmic(true)
                    .fixed_decimals(2),
            );
        });
        row(ui, "Crop L / R", |ui| {
            ui.add(
                egui::DragValue::new(&mut s.crop_l)
                    .range(0..=2000)
                    .suffix(" px"),
            );
            ui.add(
                egui::DragValue::new(&mut s.crop_r)
                    .range(0..=2000)
                    .suffix(" px"),
            );
        });
        row(ui, "Crop T / B", |ui| {
            ui.add(
                egui::DragValue::new(&mut s.crop_t)
                    .range(0..=2000)
                    .suffix(" px"),
            );
            ui.add(
                egui::DragValue::new(&mut s.crop_b)
                    .range(0..=2000)
                    .suffix(" px"),
            );
        });
        row(ui, "Rotate", |ui| {
            combo(
                ui,
                "rot",
                &mut s.rotation,
                &[
                    (Rotation::R0, "0°"),
                    (Rotation::R90, "90°"),
                    (Rotation::R180, "180°"),
                    (Rotation::R270, "270°"),
                ],
            );
        });
        row(ui, "Flip", |ui| {
            theme::check(ui, &mut s.flip_h, "Horizontal");
            theme::check(ui, &mut s.flip_v, "Vertical");
        });
        row(ui, "Letterbox", |ui| {
            ui.color_edit_button_rgb(&mut s.background);
        });
        ui.add_space(2.0);
        if theme::button(ui, "reset geometry", Btn::Secondary).clicked() {
            let d = Settings::default();
            s.zoom = d.zoom;
            s.crop_l = 0;
            s.crop_r = 0;
            s.crop_t = 0;
            s.crop_b = 0;
            s.flip_h = false;
            s.flip_v = false;
            s.rotation = Rotation::R0;
            s.aspect_override = AspectOverride::Source;
        }
    });
}

fn display_section(ui: &mut egui::Ui, v: &mut UiView) {
    theme::section(ui, 4, "Display & latency", false, |ui| {
        let modes = v.present_modes.to_vec();
        let s = &mut *v.settings;
        row(ui, "Present", |ui| {
            let all = [
                (PresentMode::Immediate, "Immediate (lowest latency)"),
                (PresentMode::Mailbox, "Mailbox (no tearing)"),
                (PresentMode::Fifo, "V-Sync (smoothest)"),
            ];
            let opts: Vec<_> = all.into_iter().filter(|(m, _)| modes.contains(m)).collect();
            combo(ui, "present", &mut s.present_mode, &opts);
        });
        row(ui, "Frame queue", |ui| {
            ui.add(egui::Slider::new(&mut s.frame_latency, 1..=3));
        });
        row(ui, "Output depth", |ui| {
            ui.add_enabled_ui(v.ten_bit_available, |ui| {
                combo(
                    ui,
                    "depth",
                    &mut s.output_depth,
                    &[
                        (OutputDepth::Bits10, "10-bit"),
                        (OutputDepth::Bits8, "8-bit"),
                    ],
                );
            });
        });
        row(ui, "", |ui| {
            theme::check(ui, &mut s.dither, "Dither output");
        });
        row(ui, "", |ui| {
            theme::check(ui, &mut s.always_on_top, "Always on top");
        });
        row(ui, "", |ui| {
            theme::check(ui, &mut s.start_fullscreen, "Start fullscreen");
        });
        row(ui, "Hide cursor", |ui| {
            ui.add(
                egui::Slider::new(&mut s.hide_cursor_secs, 0.5..=10.0)
                    .suffix(" s")
                    .fixed_decimals(1),
            );
        });
        ui.add_space(2.0);
        let label = if v.fullscreen {
            "exit fullscreen"
        } else {
            "fullscreen"
        };
        if theme::button(ui, label, Btn::Primary).clicked() {
            v.actions.push(UiAction::ToggleFullscreen);
        }
        note(
            ui,
            "Lowest latency: Immediate + 1 frame in flight + fullscreen (the frame skips the compositor).",
        );
        if let Some(f) = &v.output_format {
            ui.label(theme::margin_text(&format!(
                "{} // swapchain {f}",
                v.adapter
            )));
        }
    });
}

fn audio_section(ui: &mut egui::Ui, v: &mut UiView) {
    theme::section(ui, 5, "Audio", false, |ui| {
        let s = &mut *v.settings;
        row(ui, "", |ui| {
            theme::check(ui, &mut s.audio_enabled, "Play capture audio");
        });
        row(ui, "Input", |ui| {
            let cur = s
                .audio_input
                .clone()
                .unwrap_or_else(|| "Auto (match video)".into());
            egui::ComboBox::from_id_salt("ain")
                .selected_text(ellipsize(&cur, 26))
                .width(200.0)
                .show_ui(ui, |ui| {
                    ui.selectable_value(&mut s.audio_input, None, "Auto (match video)");
                    for n in v.audio_inputs {
                        ui.selectable_value(&mut s.audio_input, Some(n.clone()), n);
                    }
                });
        });
        row(ui, "Output", |ui| {
            let cur = s
                .audio_output
                .clone()
                .unwrap_or_else(|| "System default".into());
            egui::ComboBox::from_id_salt("aout")
                .selected_text(ellipsize(&cur, 26))
                .width(200.0)
                .show_ui(ui, |ui| {
                    ui.selectable_value(&mut s.audio_output, None, "System default");
                    for n in v.audio_outputs {
                        ui.selectable_value(&mut s.audio_output, Some(n.clone()), n);
                    }
                });
        });
        row(ui, "Volume", |ui| {
            ui.add(
                egui::Slider::new(&mut s.volume, 0.0..=2.0)
                    .custom_formatter(|x, _| format!("{:.0}%", x * 100.0)),
            );
        });
        row(ui, "", |ui| {
            theme::check(ui, &mut s.muted, "Mute");
        });
        row(ui, "Buffer / delay", |ui| {
            ui.add(
                egui::Slider::new(&mut s.audio_buffer_ms, 5.0..=300.0)
                    .suffix(" ms")
                    .fixed_decimals(0),
            );
        });
        row(ui, "", |ui| {
            theme::check(ui, &mut s.audio_ms2109_fix, "MS2109 fix (96k mono → 48k stereo)")
                .on_hover_text("Cheap MacroSilicon cards mislabel their audio. Only applies when the input reports 96 kHz mono.");
        });
        if let Some((i, o)) = v.audio_active {
            ui.label(theme::margin_text(&format!("{i} → {o}")));
            let (l, r) = v.stats.audio_peaks;
            meter(ui, "L", l);
            meter(ui, "R", r);
        }
        if let Some(e) = &v.audio_error {
            ui.label(RichText::new(e).color(VERMILLION).size(12.0));
        }
        ui.add_space(2.0);
        if theme::button(ui, "restart audio", Btn::Secondary).clicked() {
            v.actions.push(UiAction::RestartAudio);
        }
    });
}

/// Segmented pixel meter: pine blocks, vermillion near clipping.
fn meter(ui: &mut egui::Ui, ch: &str, level: f32) {
    let db = 20.0 * level.max(1e-5).log10();
    let t = ((db + 60.0) / 60.0).clamp(0.0, 1.0);
    ui.horizontal(|ui| {
        ui.label(RichText::new(ch).font(theme::mono(16.0)).color(INK_SOFT));
        let w = ui.available_width().min(300.0);
        let (rect, _) = ui.allocate_exact_size(egui::vec2(w, 10.0), egui::Sense::hover());
        let p = ui.painter();
        p.rect_filled(rect, 0.0, theme::PARCHMENT);
        p.rect_stroke(
            rect,
            0.0,
            egui::Stroke::new(1.5, INK),
            egui::StrokeKind::Inside,
        );
        let segs = 30;
        let seg_w = (rect.width() - 4.0) / segs as f32;
        let lit = (t * segs as f32).round() as i32;
        for i in 0..lit {
            let x = rect.min.x + 2.0 + i as f32 * seg_w;
            let col = if i >= segs - 3 { VERMILLION } else { PINE };
            p.rect_filled(
                egui::Rect::from_min_max(
                    egui::pos2(x, rect.min.y + 2.0),
                    egui::pos2(x + seg_w - 1.5, rect.max.y - 2.0),
                ),
                0.0,
                col,
            );
        }
    });
}

fn overlay_section(ui: &mut egui::Ui, v: &mut UiView) {
    theme::section(ui, 6, "Overlay & capture", false, |ui| {
        let s = &mut *v.settings;
        row(ui, "", |ui| {
            theme::check(ui, &mut s.show_stats, "Show stats HUD (F2)");
        });
        row(ui, "HUD corner", |ui| {
            combo(
                ui,
                "corner",
                &mut s.stats_corner,
                &[
                    (0u8, "Top left"),
                    (1u8, "Top right"),
                    (2u8, "Bottom left"),
                    (3u8, "Bottom right"),
                ],
            );
        });
        row(ui, "UI scale", |ui| {
            ui.add(egui::Slider::new(&mut s.ui_scale, 0.75..=2.0).fixed_decimals(2));
        });
        ui.add_space(2.0);
        ui.horizontal(|ui| {
            if theme::button(ui, "screenshot", Btn::Primary)
                .on_hover_text("Ctrl+S / F12")
                .clicked()
            {
                v.actions.push(UiAction::Screenshot);
            }
            if theme::button(ui, "open folder", Btn::Secondary).clicked() {
                v.actions.push(UiAction::OpenScreenshotFolder);
            }
        });
        ui.label(theme::margin_text(
            &s.screenshot_dir().display().to_string(),
        ));
    });
}

// ---------------------------------------------------------------------------------
// HUD

pub fn stats_overlay(ctx: &egui::Context, v: &UiView) {
    let st = v.stats;
    let (anchor, off) = match v.settings.stats_corner {
        1 => (egui::Align2::RIGHT_TOP, egui::vec2(-EDGE - 6.0, EDGE)),
        2 => (egui::Align2::LEFT_BOTTOM, egui::vec2(EDGE, -EDGE - 6.0)),
        3 => (
            egui::Align2::RIGHT_BOTTOM,
            egui::vec2(-EDGE - 6.0, -EDGE - 6.0),
        ),
        _ => (egui::Align2::LEFT_TOP, egui::vec2(EDGE, EDGE)),
    };
    egui::Area::new(egui::Id::unique("stats"))
        .anchor(anchor, off)
        .interactable(false)
        .order(egui::Order::Middle)
        .show(ctx, |ui| {
            egui::Frame::new()
                .fill(Color32::from_rgba_unmultiplied(0x16, 0x13, 0x10, 238))
                .stroke(egui::Stroke::new(2.0, INK))
                .shadow(theme::hard_shadow(5, COBALT))
                .inner_margin(egui::Margin::same(12))
                .show(ui, |ui| {
                    ui.spacing_mut().item_spacing.y = 1.0;
                    ui.horizontal(|ui| {
                        ui.label(RichText::new("SIGNAL").font(theme::pixel(15.0)).color(BONE));
                        ui.add_space(6.0);
                        state_tag(ui, v.state);
                    });
                    ui.add_space(4.0);
                    let kv = |ui: &mut egui::Ui, k: &str, val: String, col: Color32| {
                        ui.horizontal(|ui| {
                            ui.spacing_mut().item_spacing.x = 0.0;
                            // VT323 is monospaced: pad the key for a clean left-aligned column.
                            ui.label(
                                RichText::new(format!("{k:<9}"))
                                    .font(theme::mono(17.0))
                                    .color(TAN),
                            );
                            ui.label(RichText::new(val).font(theme::mono(17.0)).color(col));
                        });
                    };
                    if let Some(f) = v.active_format {
                        kv(
                            ui,
                            "SOURCE",
                            format!(
                                "{} {}x{} @ {:.2}",
                                f.format.name(),
                                f.width,
                                f.height,
                                f.fps()
                            ),
                            BONE,
                        );
                        if slow_mode_hint(f, st.capture_fps).is_some() {
                            kv(
                                ui,
                                "WARN",
                                format!("device only delivering {:.0} fps", st.capture_fps),
                                VERMILLION,
                            );
                        }
                    } else {
                        kv(ui, "SOURCE", "none".into(), BONE);
                    }
                    let r = v.resolved;
                    if r.matrix == 0 {
                        kv(ui, "COLOUR", "—".into(), BONE);
                    } else {
                        kv(
                            ui,
                            "COLOUR",
                            format!(
                                "BT.{} {} · chroma {}",
                                r.matrix,
                                if r.full_range { "full" } else { "limited" },
                                if r.cosited { "left" } else { "center" }
                            ),
                            BONE,
                        );
                    }
                    kv(
                        ui,
                        "FPS",
                        format!(
                            "capture {:.2} · present {:.2}",
                            st.capture_fps, st.present_fps
                        ),
                        BONE,
                    );
                    kv(
                        ui,
                        "PACING",
                        format!(
                            "{:.2} ms ± {:.2} · drops {}",
                            st.interval_ms, st.jitter_ms, st.dropped
                        ),
                        BONE,
                    );
                    let dev = st
                        .device_latency_ms
                        .map(|d| format!(" · device {d:.1}"))
                        .unwrap_or_default();
                    kv(
                        ui,
                        "LATENCY",
                        format!(
                            "{:.2} ms (max {:.2}){dev}",
                            st.latency_ms, st.latency_max_ms
                        ),
                        SKY,
                    );
                    kv(
                        ui,
                        "GPU/CPU",
                        format!("upload {:.2} · cpu {:.2} ms", st.upload_ms, st.cpu_frame_ms),
                        BONE,
                    );
                    if let Some(fill) = st.audio_fill_ms {
                        let rates = st
                            .audio_rates
                            .map(|(i, o)| format!(" · {i}→{o}"))
                            .unwrap_or_default();
                        kv(
                            ui,
                            "AUDIO",
                            format!("{fill:.1} ms · xruns {}{rates}", st.audio_underruns),
                            BONE,
                        );
                    }
                    kv(
                        ui,
                        "OUTPUT",
                        format!(
                            "{:?} · {}",
                            v.settings.present_mode,
                            v.output_format.as_deref().unwrap_or("?")
                        ),
                        BONE,
                    );
                    ui.add_space(6.0);
                    frame_graph(ui, st);
                });
        });
}

fn frame_graph(ui: &mut egui::Ui, st: &Stats) {
    let (rect, _) = ui.allocate_exact_size(egui::vec2(320.0, 46.0), egui::Sense::hover());
    let p = ui.painter();
    p.rect_stroke(
        rect,
        0.0,
        egui::Stroke::new(1.0, Color32::from_rgb(0x4A, 0x43, 0x3A)),
        egui::StrokeKind::Inside,
    );
    let max_ms = 40.0f32;
    let target = if st.interval_ms > 0.0 {
        st.interval_ms
    } else {
        16.67
    };
    let ty = rect.bottom() - (target / max_ms).min(1.0) * rect.height();
    p.line_segment(
        [egui::pos2(rect.left(), ty), egui::pos2(rect.right(), ty)],
        egui::Stroke::new(1.0, COBALT),
    );
    // Bars, pixel-style: one column per frame.
    let n = st.frame_times.len();
    if n == 0 {
        return;
    }
    let bw = rect.width() / 240.0;
    for (i, &ms) in st.frame_times.iter().enumerate() {
        let h = (ms / max_ms).min(1.0) * (rect.height() - 2.0);
        let x = rect.right() - (n - i) as f32 * bw;
        let late = ms > target * 1.5;
        p.rect_filled(
            egui::Rect::from_min_max(
                egui::pos2(x, rect.bottom() - 1.0 - h),
                egui::pos2(x + bw.max(1.0), rect.bottom() - 1.0),
            ),
            0.0,
            if late { VERMILLION } else { PINE },
        );
    }
}

pub fn no_signal(ctx: &egui::Context, text: &str) {
    let sub = if text == "No signal" {
        "Waiting for an HDMI source"
    } else {
        text
    };
    egui::Area::new(egui::Id::unique("nosignal"))
        .anchor(egui::Align2::CENTER_CENTER, egui::vec2(-3.0, -3.0))
        .interactable(false)
        .order(egui::Order::Background)
        .show(ctx, |ui| {
            theme::card(BONE)
                .shadow(theme::hard_shadow(6, COBALT))
                .inner_margin(egui::Margin::symmetric(34, 22))
                .show(ui, |ui| {
                    ui.set_width(330.0);
                    ui.vertical_centered(|ui| {
                        let mut title = egui::text::LayoutJob::default();
                        let fmt = |c| egui::TextFormat {
                            font_id: theme::pixel(40.0),
                            color: c,
                            ..Default::default()
                        };
                        title.append("NO SIGNAL", 0.0, fmt(INK));
                        title.append(".", 0.0, fmt(VERMILLION));
                        ui.label(title);
                        ui.label(RichText::new(sub).font(theme::mono(20.0)).color(INK_SOFT));
                        ui.add_space(8.0);
                        theme::dashed_rule(ui);
                        ui.label(theme::margin_text("F1 · tab · right-click — settings"));
                    });
                });
        });
}

pub fn toast(ctx: &egui::Context, text: &str) {
    egui::Area::new(egui::Id::unique("toast"))
        .anchor(egui::Align2::CENTER_BOTTOM, egui::vec2(-2.0, -30.0))
        .interactable(false)
        .order(egui::Order::Tooltip)
        .show(ctx, |ui| {
            egui::Frame::new()
                .fill(BONE)
                .stroke(egui::Stroke::new(2.0, INK))
                .shadow(theme::hard_shadow(4, INK))
                .inner_margin(egui::Margin::symmetric(14, 6))
                .show(ui, |ui| {
                    ui.horizontal(|ui| {
                        let (r, _) =
                            ui.allocate_exact_size(egui::vec2(10.0, 10.0), egui::Sense::hover());
                        ui.painter().rect_filled(r, 0.0, VERMILLION);
                        ui.label(RichText::new(text).font(theme::mono(19.0)).color(INK));
                    });
                });
        });
}
