//! Persistent user settings (JSON in %APPDATA%\capview\settings.json).

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ColorMatrix {
    Auto,
    Bt601,
    Bt709,
    Bt2020,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ColorRange {
    Auto,
    Limited,
    Full,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ChromaFilter {
    Nearest,
    Linear,
    Cubic,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ChromaSiting {
    /// Use what the device/format reports.
    Auto,
    /// Co-sited with the left luma sample (MPEG-2 / HDMI 4:2:2 default).
    Left,
    /// Centered between luma samples (MPEG-1 / JPEG).
    Center,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ScaleFilter {
    Nearest,
    Bilinear,
    SharpBilinear,
    CatmullRom,
    Mitchell,
    Lanczos2,
    Lanczos3,
}

impl ScaleFilter {
    pub const ALL: [ScaleFilter; 7] = [
        ScaleFilter::Nearest,
        ScaleFilter::Bilinear,
        ScaleFilter::SharpBilinear,
        ScaleFilter::CatmullRom,
        ScaleFilter::Mitchell,
        ScaleFilter::Lanczos2,
        ScaleFilter::Lanczos3,
    ];
    pub fn label(self) -> &'static str {
        match self {
            ScaleFilter::Nearest => "Nearest (pixel exact)",
            ScaleFilter::Bilinear => "Bilinear",
            ScaleFilter::SharpBilinear => "Sharp bilinear (pixel art)",
            ScaleFilter::CatmullRom => "Bicubic Catmull-Rom",
            ScaleFilter::Mitchell => "Bicubic Mitchell",
            ScaleFilter::Lanczos2 => "Lanczos 2",
            ScaleFilter::Lanczos3 => "Lanczos 3",
        }
    }
    pub fn id(self) -> u32 {
        self as u32
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum AspectMode {
    /// Fit inside the window, keep source aspect.
    Fit,
    /// Fill the window, crop overflow.
    Fill,
    /// Stretch to window.
    Stretch,
    /// Largest integer multiple that fits.
    Integer,
    /// 1:1 pixels, centered.
    Native,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum AspectOverride {
    Source,
    R16_9,
    R4_3,
    R21_9,
    R16_10,
    R1_1,
}

impl AspectOverride {
    pub fn ratio(self) -> Option<f32> {
        match self {
            AspectOverride::Source => None,
            AspectOverride::R16_9 => Some(16.0 / 9.0),
            AspectOverride::R4_3 => Some(4.0 / 3.0),
            AspectOverride::R21_9 => Some(64.0 / 27.0),
            AspectOverride::R16_10 => Some(16.0 / 10.0),
            AspectOverride::R1_1 => Some(1.0),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum PresentMode {
    /// Tearing allowed, lowest latency (best in fullscreen).
    Immediate,
    /// Latest-frame-wins without tearing.
    Mailbox,
    /// Classic vsync.
    Fifo,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum OutputDepth {
    Bits8,
    Bits10,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Rotation {
    R0,
    R90,
    R180,
    R270,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Settings {
    // --- device ---
    /// Symbolic link of the preferred video device (stable across reboots).
    pub device_link: Option<String>,
    pub device_name: Option<String>,
    /// Preferred format, as "SUBTYPE WxH@num/den". None = best available.
    pub format_key: Option<String>,

    // --- picture ---
    pub matrix: ColorMatrix,
    pub range: ColorRange,
    pub chroma_filter: ChromaFilter,
    pub chroma_siting: ChromaSiting,
    pub brightness: f32,
    pub contrast: f32,
    pub saturation: f32,
    pub hue: f32,
    pub gamma: f32,
    pub sharpen: f32,

    // --- geometry ---
    pub scale_filter: ScaleFilter,
    pub antiring: f32,
    pub linear_light: bool,
    pub aspect_mode: AspectMode,
    pub aspect_override: AspectOverride,
    pub zoom: f32,
    pub crop_l: u32,
    pub crop_r: u32,
    pub crop_t: u32,
    pub crop_b: u32,
    pub flip_h: bool,
    pub flip_v: bool,
    pub rotation: Rotation,
    pub background: [f32; 3],

    // --- display ---
    pub present_mode: PresentMode,
    pub frame_latency: u32,
    pub output_depth: OutputDepth,
    pub dither: bool,
    pub always_on_top: bool,
    pub start_fullscreen: bool,
    pub hide_cursor_secs: f32,
    pub window_size: Option<[u32; 2]>,

    // --- audio ---
    pub audio_enabled: bool,
    pub audio_input: Option<String>,
    pub audio_output: Option<String>,
    pub volume: f32,
    pub muted: bool,
    pub audio_buffer_ms: f32,
    /// Re-interpret MS2109 "96 kHz mono" audio as 48 kHz stereo.
    pub audio_ms2109_fix: bool,

    // --- overlay ---
    pub show_stats: bool,
    pub stats_corner: u8,
    pub ui_scale: f32,
    pub screenshot_dir: Option<String>,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            device_link: None,
            device_name: None,
            format_key: None,
            matrix: ColorMatrix::Auto,
            range: ColorRange::Auto,
            chroma_filter: ChromaFilter::Cubic,
            chroma_siting: ChromaSiting::Auto,
            brightness: 0.0,
            contrast: 1.0,
            saturation: 1.0,
            hue: 0.0,
            gamma: 1.0,
            sharpen: 0.0,
            scale_filter: ScaleFilter::Lanczos3,
            antiring: 0.6,
            linear_light: false,
            aspect_mode: AspectMode::Fit,
            aspect_override: AspectOverride::Source,
            zoom: 1.0,
            crop_l: 0,
            crop_r: 0,
            crop_t: 0,
            crop_b: 0,
            flip_h: false,
            flip_v: false,
            rotation: Rotation::R0,
            background: [0.0, 0.0, 0.0],
            present_mode: PresentMode::Immediate,
            frame_latency: 1,
            output_depth: OutputDepth::Bits10,
            dither: true,
            always_on_top: false,
            start_fullscreen: false,
            hide_cursor_secs: 1.5,
            window_size: None,
            audio_enabled: true,
            audio_input: None,
            audio_output: None,
            volume: 1.0,
            muted: false,
            audio_buffer_ms: 30.0,
            audio_ms2109_fix: true,
            show_stats: false,
            stats_corner: 0,
            ui_scale: 1.0,
            screenshot_dir: None,
        }
    }
}

pub fn config_dir() -> PathBuf {
    dirs::config_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("capview")
}

fn settings_path() -> PathBuf {
    config_dir().join("settings.json")
}

impl Settings {
    pub fn load() -> Self {
        std::fs::read_to_string(settings_path())
            .ok()
            .and_then(|s| serde_json::from_str(&s).ok())
            .unwrap_or_default()
    }

    pub fn save(&self) {
        if std::env::var_os("CAPVIEW_SMOKE").is_some() {
            return; // test runs never touch the user's settings
        }
        let dir = config_dir();
        let _ = std::fs::create_dir_all(&dir);
        if let Ok(json) = serde_json::to_string_pretty(self) {
            let tmp = dir.join("settings.json.tmp");
            if std::fs::write(&tmp, json).is_ok() {
                let _ = std::fs::rename(&tmp, settings_path());
            }
        }
    }

    pub fn screenshot_dir(&self) -> PathBuf {
        if let Some(d) = &self.screenshot_dir {
            return PathBuf::from(d);
        }
        dirs::picture_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join("capview")
    }
}
