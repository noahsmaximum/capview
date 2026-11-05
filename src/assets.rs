//! Background / frame images: native file picker and decoding on worker threads,
//! uploaded on the render thread when ready (the video never stalls on a big PNG).

use crate::render::VideoRenderer;
use crate::settings::{self, BackgroundMode, FrameStyle, Settings};
use std::path::PathBuf;
use std::sync::mpsc::{Receiver, Sender};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Background,
    Frame,
}

enum Msg {
    Picked(Kind, Option<PathBuf>),
    Decoded(Kind, String, Result<image::RgbaImage, String>),
}

pub struct FrameTexture {
    pub tex: egui::TextureHandle,
    pub size: [usize; 2],
}

pub struct Assets {
    tx: Sender<Msg>,
    rx: Receiver<Msg>,
    notify: std::sync::Arc<dyn Fn() + Send + Sync>,
    bg_requested: Option<String>,
    frame_requested: Option<String>,
    pub frame: Option<FrameTexture>,
}

const MAX_BG: u32 = 4096;
const MAX_FRAME: u32 = 2048;

fn decode(path: &str, max: u32) -> Result<image::RgbaImage, String> {
    let img = image::open(path).map_err(|e| format!("{}: {e}", short(path)))?;
    let mut rgba = img.to_rgba8();
    let (w, h) = rgba.dimensions();
    if w > max || h > max {
        let k = max as f32 / w.max(h) as f32;
        rgba = image::imageops::resize(
            &rgba,
            ((w as f32 * k).round() as u32).max(1),
            ((h as f32 * k).round() as u32).max(1),
            image::imageops::FilterType::Lanczos3,
        );
    }
    Ok(rgba)
}

pub fn short(path: &str) -> String {
    std::path::Path::new(path)
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| path.to_owned())
}

impl Assets {
    pub fn new(notify: std::sync::Arc<dyn Fn() + Send + Sync>) -> Self {
        let (tx, rx) = std::sync::mpsc::channel();
        Self { tx, rx, notify, bg_requested: None, frame_requested: None, frame: None }
    }

    /// Open the native picker without blocking the render loop.
    pub fn pick(&self, kind: Kind) {
        let tx = self.tx.clone();
        let notify = self.notify.clone();
        std::thread::spawn(move || {
            let title = match kind {
                Kind::Background => "Choose a background image",
                Kind::Frame => "Choose a frame image (PNG with a transparent centre)",
            };
            let path = rfd::FileDialog::new()
                .set_title(title)
                .add_filter("Images", &["png", "jpg", "jpeg", "webp", "bmp", "gif"])
                .pick_file();
            let _ = tx.send(Msg::Picked(kind, path));
            notify();
        });
    }

    /// The egui context was recreated: its textures are gone, reload the frame.
    pub fn invalidate_egui(&mut self) {
        self.frame = None;
        self.frame_requested = None;
    }

    /// Apply pick results, finished decodes and settings changes. Returns messages to toast.
    pub fn update(
        &mut self,
        settings: &mut Settings,
        video: &mut VideoRenderer,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        ctx: &egui::Context,
    ) -> Vec<String> {
        let mut toasts = Vec::new();
        while let Ok(msg) = self.rx.try_recv() {
            match msg {
                Msg::Picked(_, None) => {}
                Msg::Picked(kind, Some(path)) => match settings::import_image(&path) {
                    Ok(stored) => match kind {
                        Kind::Background => {
                            settings.background_image = Some(stored);
                            settings.background_mode = BackgroundMode::Image;
                        }
                        Kind::Frame => {
                            settings.frame_image = Some(stored);
                            settings.frame_style = FrameStyle::Image;
                        }
                    },
                    Err(e) => toasts.push(format!("Couldn't import image: {e}")),
                },
                Msg::Decoded(kind, path, result) => {
                    let wanted = match kind {
                        Kind::Background => self.bg_requested.as_deref() == Some(path.as_str()),
                        Kind::Frame => self.frame_requested.as_deref() == Some(path.as_str()),
                    };
                    if !wanted {
                        continue; // superseded by a newer choice
                    }
                    match (kind, result) {
                        (Kind::Background, Ok(img)) => video.set_background_image(device, queue, Some(&img)),
                        (Kind::Frame, Ok(img)) => {
                            let size = [img.width() as usize, img.height() as usize];
                            let color = egui::ColorImage::from_rgba_unmultiplied(size, img.as_raw());
                            let tex = ctx.load_texture("frame", color, egui::TextureOptions::LINEAR);
                            self.frame = Some(FrameTexture { tex, size });
                        }
                        (_, Err(e)) => toasts.push(format!("Couldn't load image: {e}")),
                    }
                }
            }
        }

        let want_bg = (settings.background_mode == BackgroundMode::Image)
            .then(|| settings.background_image.clone())
            .flatten();
        if want_bg != self.bg_requested {
            self.bg_requested = want_bg.clone();
            match want_bg {
                Some(path) => self.spawn_decode(Kind::Background, path, MAX_BG),
                None => video.set_background_image(device, queue, None),
            }
        }
        let want_frame = (settings.frame_style == FrameStyle::Image)
            .then(|| settings.frame_image.clone())
            .flatten();
        if want_frame != self.frame_requested {
            self.frame_requested = want_frame.clone();
            self.frame = None;
            if let Some(path) = want_frame {
                self.spawn_decode(Kind::Frame, path, MAX_FRAME);
            }
        }
        toasts
    }

    fn spawn_decode(&self, kind: Kind, path: String, max: u32) {
        let tx = self.tx.clone();
        let notify = self.notify.clone();
        std::thread::spawn(move || {
            let result = decode(&path, max);
            let _ = tx.send(Msg::Decoded(kind, path, result));
            notify();
        });
    }
}
