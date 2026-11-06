//! Capture thread, shared by every platform backend.
//!
//! The worker owns the platform capture state (Media Foundation on Windows,
//! AVFoundation on macOS) and uploads each frame straight into GPU textures with
//! `queue.write_texture` from this thread. wgpu applies pending writes at the start of
//! the next `submit`, so the render thread always draws the newest frame without an
//! extra CPU copy or a hand-off queue.
//!
//! Backends only enumerate devices/formats, open a stream, and turn their native
//! deliveries into [`Frame`]s. Session handling (retry, no-signal, stall watchdog),
//! stats and the GPU upload live here.

use parking_lot::Mutex;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{Receiver, Sender};
use std::time::{Duration, Instant};

#[cfg(target_os = "macos")]
mod avf;
#[cfg(target_os = "windows")]
mod mf;
#[cfg(not(any(target_os = "windows", target_os = "macos")))]
mod unsupported;

#[cfg(target_os = "macos")]
type Platform = avf::Avf;
#[cfg(target_os = "windows")]
type Platform = mf::Mf;
#[cfg(not(any(target_os = "windows", target_os = "macos")))]
type Platform = unsupported::Unsupported;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PixelFormat {
    Yuy2,
    Uyvy,
    Nv12,
    P010,
    Bgra,
    Mjpeg,
}

impl PixelFormat {
    pub fn name(self) -> &'static str {
        match self {
            PixelFormat::Yuy2 => "YUY2",
            PixelFormat::Uyvy => "UYVY",
            PixelFormat::Nv12 => "NV12",
            PixelFormat::P010 => "P010",
            PixelFormat::Bgra => "RGB32",
            PixelFormat::Mjpeg => "MJPEG",
        }
    }
    pub fn uncompressed(self) -> bool {
        self != PixelFormat::Mjpeg
    }
    /// Shader id for the convert pass.
    pub fn shader_id(self) -> u32 {
        match self {
            PixelFormat::Yuy2 => 0,
            PixelFormat::Uyvy => 1,
            PixelFormat::Nv12 => 2,
            PixelFormat::P010 => 3,
            PixelFormat::Bgra => 4,
            PixelFormat::Mjpeg => 0,
        }
    }
    pub fn is_rgb(self) -> bool {
        self == PixelFormat::Bgra
    }
}

/// Colour hints read from the media type (None = not reported).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ColorHints {
    /// 601 / 709 / 2020
    pub matrix: Option<u32>,
    pub full_range: Option<bool>,
    /// Some(true) = horizontally co-sited.
    pub cosited: Option<bool>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct FormatInfo {
    /// Backend-specific index used to reopen this exact mode.
    pub native_index: u32,
    pub format: PixelFormat,
    pub width: u32,
    pub height: u32,
    pub fps_num: u32,
    pub fps_den: u32,
    pub hints: ColorHints,
    /// NV12 synthesised by the Windows camera Frame Server from an MJPEG mode
    /// (4:2:0, decoded out of process) — not real uncompressed capture.
    pub windows_decoded: bool,
}

impl FormatInfo {
    /// Truly uncompressed on the wire.
    pub fn uncompressed(&self) -> bool {
        self.format.uncompressed() && !self.windows_decoded
    }
    pub fn fps(&self) -> f64 {
        if self.fps_den == 0 {
            0.0
        } else {
            self.fps_num as f64 / self.fps_den as f64
        }
    }
    pub fn key(&self) -> String {
        format!(
            "{} {}x{}@{}/{}",
            self.format.name(),
            self.width,
            self.height,
            self.fps_num,
            self.fps_den
        )
    }
    pub fn label(&self) -> String {
        format!(
            "{}x{} @ {:.2} — {}{}",
            self.width,
            self.height,
            self.fps(),
            self.format.name(),
            if self.windows_decoded {
                " (MJPEG decoded by Windows, 4:2:0)"
            } else if self.format.uncompressed() {
                " (uncompressed)"
            } else {
                " (compressed, decoded 4:2:2)"
            }
        )
    }
    /// Compact form for the closed format combo: "1920×1080 · 60 · YUY2".
    pub fn short_label(&self) -> String {
        let fps = self.fps();
        let fps = if (fps - fps.round()).abs() < 0.01 {
            format!("{fps:.0}")
        } else {
            format!("{fps:.2}")
        };
        let fmt = if self.windows_decoded {
            "NV12 (Win)"
        } else {
            self.format.name()
        };
        format!("{}×{} · {} · {}", self.width, self.height, fps, fmt)
    }
    /// Default preference: fps tier (>=50, >=29, other), then resolution, then
    /// source quality (uncompressed > own MJPEG decode (4:2:2) > Windows NV12 (4:2:0)),
    /// then exact fps. Avoids e.g. 2304x1536 @ 2 fps on webcams.
    fn score(&self) -> (u8, u64, u8, u64) {
        let fps = self.fps();
        let quality = if self.uncompressed() {
            2
        } else if self.windows_decoded {
            0
        } else {
            1
        };
        (
            if fps >= 49.0 {
                2
            } else if fps >= 29.0 {
                1
            } else {
                0
            },
            self.width as u64 * self.height as u64,
            quality,
            (fps * 1000.0) as u64,
        )
    }
}

/// Dedupe by key (keep first), sort best-first and pick `format_key` or the best.
/// Returns None when there is nothing usable.
fn rank_formats(formats: &mut Vec<FormatInfo>, format_key: Option<&str>) -> Option<FormatInfo> {
    let mut seen = std::collections::HashSet::new();
    formats.retain(|f| seen.insert(f.key()));
    formats.sort_by(|a, b| b.score().cmp(&a.score()));
    for f in formats.iter() {
        log::debug!("format: {}  [{}]", f.label(), f.key());
    }
    format_key
        .and_then(|k| formats.iter().find(|f| f.key() == k))
        .or(formats.first())
        .cloned()
}

#[derive(Clone, Debug, PartialEq)]
pub struct DeviceInfo {
    pub name: String,
    /// Stable backend identifier (MF symbolic link, AVFoundation uniqueID, …).
    pub link: String,
}

/// The texture set that holds the most recent frame. Recreated on format change.
#[derive(Clone)]
pub struct SourceTextures {
    pub generation: u64,
    /// Decoded/output format actually in the textures (MJPEG becomes YUY2/UYVY/NV12).
    pub format: PixelFormat,
    pub width: u32,
    pub height: u32,
    pub planes: Vec<wgpu::Texture>,
    pub hints: ColorHints,
    pub flip_y: bool,
}

#[derive(Clone, Debug, Default)]
pub enum CaptureState {
    #[default]
    Idle,
    Opening,
    Running,
    NoSignal,
    Error(String),
}

#[derive(Default)]
pub struct CaptureStatus {
    pub state: CaptureState,
    pub devices: Vec<DeviceInfo>,
    pub formats: Vec<FormatInfo>,
    pub active_device: Option<DeviceInfo>,
    pub active_format: Option<FormatInfo>,
}

#[derive(Default)]
pub struct CaptureStats {
    pub frames: AtomicU64,
    pub dropped: AtomicU64,
    /// Last frame arrival as nanoseconds since `epoch`.
    pub last_arrival_ns: AtomicU64,
    /// Device capture -> sample delivered to us, microseconds (0 = unknown).
    pub device_latency_us: AtomicU64,
    /// Time spent locking + uploading the last frame, microseconds.
    pub upload_us: AtomicU64,
    /// Frame interval stats (microseconds), updated by the capture thread.
    pub interval_us: AtomicU64,
    pub jitter_us: AtomicU64,
}

pub struct CaptureShared {
    pub epoch: Instant,
    pub status: Mutex<CaptureStatus>,
    pub textures: Mutex<Option<SourceTextures>>,
    pub stats: CaptureStats,
    pub frame_seq: AtomicU64,
    pub running: AtomicBool,
}

pub enum CaptureCmd {
    RefreshDevices,
    Open {
        link: String,
        format_key: Option<String>,
    },
    #[allow(dead_code)]
    Close,
    Shutdown,
}

// ---------------------------------------------------------------------------
// Backend interface
// ---------------------------------------------------------------------------

/// One platform capture API. Lives on the capture thread.
trait Backend: Sized {
    /// What the platform callback hands us (sent across threads to the worker).
    type Raw: Send + 'static;
    type Stream: Stream<Raw = Self::Raw>;
    /// Per-thread setup (COM / MF startup, scheduling priority).
    fn init() -> Result<Self, String>;
    fn devices(&mut self) -> Vec<DeviceInfo>;
    /// Open `link`, choosing `format_key` when it exists (else the best mode).
    /// Returns the stream and every mode the device offers, best first.
    fn open(
        &mut self,
        link: &str,
        format_key: Option<&str>,
        sink: Sink<Self::Raw>,
    ) -> Result<(Self::Stream, Vec<FormatInfo>), String>;
}

trait Stream {
    type Raw;
    fn selected(&self) -> &FormatInfo;
    fn hints(&self) -> ColorHints;
    /// Begin delivery.
    fn start(&mut self) -> Result<(), String>;
    /// Turn one platform delivery into frames / drop reports. An `Err` ends the
    /// session and reopens the device.
    fn handle(&mut self, raw: Self::Raw, emit: &mut dyn FnMut(Event<'_>)) -> Result<(), String>;
}

enum Event<'a> {
    Frame(Frame<'a>),
    Dropped(u64),
}

/// A frame whose pixels are mapped for the duration of [`Stream::handle`].
struct Frame<'a> {
    /// Format of the pixels in `planes` (never MJPEG — backends decode first).
    format: PixelFormat,
    width: u32,
    height: u32,
    /// Packed formats: one plane. NV12 / P010: luma, then interleaved chroma.
    planes: &'a [Plane<'a>],
    bottom_up: bool,
    /// Device timestamp in 100 ns units (drop accounting).
    ts_100ns: Option<i64>,
    /// Device capture -> delivery, when the platform can tell.
    device_latency_us: Option<u64>,
}

struct Plane<'a> {
    data: &'a [u8],
    pitch: usize,
}

enum Msg<R> {
    Cmd(CaptureCmd),
    Raw {
        session: u64,
        raw: R,
        arrival: Instant,
    },
}

/// Hands platform deliveries to the worker, tagged with the session that opened them
/// so late deliveries from a closed stream are ignored.
struct Sink<R> {
    session: u64,
    tx: Sender<Msg<R>>,
}

impl<R> Clone for Sink<R> {
    fn clone(&self) -> Self {
        Sink {
            session: self.session,
            tx: self.tx.clone(),
        }
    }
}

impl<R> Sink<R> {
    fn send(&self, raw: R) {
        let _ = self.tx.send(Msg::Raw {
            session: self.session,
            raw,
            arrival: Instant::now(),
        });
    }
}

// ---------------------------------------------------------------------------
// Public handle
// ---------------------------------------------------------------------------

type RawOf = <Platform as Backend>::Raw;

pub struct Capture {
    pub shared: Arc<CaptureShared>,
    tx: Sender<Msg<RawOf>>,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl Capture {
    pub fn start(device: wgpu::Device, queue: wgpu::Queue, notify: Box<dyn Fn() + Send>) -> Self {
        let shared = Arc::new(CaptureShared {
            epoch: Instant::now(),
            status: Mutex::new(CaptureStatus::default()),
            textures: Mutex::new(None),
            stats: CaptureStats::default(),
            frame_seq: AtomicU64::new(0),
            running: AtomicBool::new(false),
        });
        let (tx, rx) = std::sync::mpsc::channel();
        let sh = shared.clone();
        let cb_tx = tx.clone();
        let thread = std::thread::Builder::new()
            .name("capture".into())
            .spawn(move || {
                let backend = match Platform::init() {
                    Ok(b) => b,
                    Err(e) => {
                        sh.status.lock().state = CaptureState::Error(e);
                        notify();
                        return;
                    }
                };
                let mut w = Worker {
                    shared: sh,
                    device,
                    queue,
                    notify,
                    rx,
                    cb_tx,
                    session: 0,
                    generation: 0,
                    backend,
                };
                w.run();
            })
            .expect("spawn capture thread");
        let _ = tx.send(Msg::Cmd(CaptureCmd::RefreshDevices));
        Capture {
            shared,
            tx,
            thread: Some(thread),
        }
    }

    pub fn send(&self, cmd: CaptureCmd) {
        let _ = self.tx.send(Msg::Cmd(cmd));
    }
}

impl Drop for Capture {
    fn drop(&mut self) {
        let _ = self.tx.send(Msg::Cmd(CaptureCmd::Shutdown));
        if let Some(t) = self.thread.take() {
            // Never hang the UI on a wedged driver: give it a moment, then detach.
            let deadline = Instant::now() + Duration::from_millis(1500);
            while !t.is_finished() && Instant::now() < deadline {
                std::thread::sleep(Duration::from_millis(10));
            }
            if t.is_finished() {
                let _ = t.join();
            } else {
                log::warn!("capture thread did not stop in time; detaching");
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Worker
// ---------------------------------------------------------------------------

struct Worker<B: Backend> {
    shared: Arc<CaptureShared>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    notify: Box<dyn Fn() + Send>,
    rx: Receiver<Msg<B::Raw>>,
    cb_tx: Sender<Msg<B::Raw>>,
    session: u64,
    generation: u64,
    backend: B,
}

/// No frame for this long -> report "No signal" (keep waiting).
const NO_SIGNAL_AFTER: Duration = Duration::from_millis(1500);
/// Frames stopped after streaming was working -> tear down and reopen.
const STALL_REOPEN_AFTER: Duration = Duration::from_millis(3000);

enum ReadExit {
    Cmd(CaptureCmd),
    Error(String),
}

/// Per-session timing state.
struct Timing {
    nominal_us: f64,
    last_ts: Option<i64>,
    last_arrival: Option<Instant>,
    ema_interval: f64,
    ema_jitter: f64,
}

impl<B: Backend> Worker<B> {
    fn set_state(&self, s: CaptureState) {
        self.shared.status.lock().state = s;
        (self.notify)();
    }

    fn refresh_devices(&mut self) {
        let devices = self.backend.devices();
        self.shared.status.lock().devices = devices;
        (self.notify)();
    }

    fn run(&mut self) {
        let mut pending: Option<CaptureCmd> = None;
        loop {
            let cmd = match pending.take() {
                Some(c) => c,
                None => match self.rx.recv() {
                    Ok(Msg::Cmd(c)) => c,
                    Ok(Msg::Raw { .. }) => continue, // stale delivery from a closed session
                    Err(_) => break,
                },
            };
            match cmd {
                CaptureCmd::Shutdown => break,
                CaptureCmd::RefreshDevices => self.refresh_devices(),
                CaptureCmd::Close => {
                    self.shared.running.store(false, Ordering::Relaxed);
                    let mut st = self.shared.status.lock();
                    st.active_device = None;
                    st.active_format = None;
                    st.state = CaptureState::Idle;
                    drop(st);
                    *self.shared.textures.lock() = None;
                    (self.notify)();
                }
                CaptureCmd::Open { link, format_key } => {
                    pending = self.capture_session(&link, format_key.as_deref());
                }
            }
        }
    }

    /// Runs one capture session; returns a command that interrupted it (if any).
    fn capture_session(&mut self, link: &str, format_key: Option<&str>) -> Option<CaptureCmd> {
        self.set_state(CaptureState::Opening);
        let device = self
            .shared
            .status
            .lock()
            .devices
            .iter()
            .find(|d| d.link == link)
            .cloned()
            .unwrap_or(DeviceInfo {
                name: "Unknown device".into(),
                link: link.to_string(),
            });

        let mut retry_delay = Duration::from_millis(250);
        loop {
            self.session += 1;
            let sink = Sink {
                session: self.session,
                tx: self.cb_tx.clone(),
            };
            match self.backend.open(link, format_key, sink) {
                Ok((stream, formats)) => {
                    {
                        let mut st = self.shared.status.lock();
                        st.formats = formats;
                        st.active_device = Some(device.clone());
                        st.active_format = Some(stream.selected().clone());
                        st.state = CaptureState::Opening;
                    }
                    (self.notify)();
                    let result = self.read_loop(stream);
                    self.shared.running.store(false, Ordering::Relaxed);
                    match result {
                        ReadExit::Cmd(c) => return Some(c),
                        ReadExit::Error(e) => {
                            log::warn!("capture error: {e}; reopening");
                            self.set_state(CaptureState::Error(format!("{e} — reconnecting…")));
                        }
                    }
                }
                Err(e) => {
                    log::warn!("open failed: {e}");
                    self.set_state(CaptureState::Error(format!("Open failed: {e}")));
                }
            }
            // Wait (interruptible) before retrying — handles unplug/replug.
            let deadline = Instant::now() + retry_delay;
            loop {
                let left = deadline.saturating_duration_since(Instant::now());
                match self.rx.recv_timeout(left) {
                    Ok(Msg::Cmd(CaptureCmd::RefreshDevices)) => self.refresh_devices(),
                    Ok(Msg::Cmd(c)) => return Some(c),
                    Ok(Msg::Raw { .. }) => continue,
                    Err(std::sync::mpsc::RecvTimeoutError::Timeout) => break,
                    Err(_) => return Some(CaptureCmd::Shutdown),
                }
            }
            retry_delay = (retry_delay * 2).min(Duration::from_secs(2));
            self.set_state(CaptureState::Opening);
        }
    }

    fn read_loop(&mut self, mut stream: B::Stream) -> ReadExit {
        let shared = self.shared.clone();
        shared.stats.dropped.store(0, Ordering::Relaxed);
        let fps = stream.selected().fps();
        let nominal_us = if fps > 0.0 { 1e6 / fps } else { 16_667.0 };
        let mut timing = Timing {
            nominal_us,
            last_ts: None,
            last_arrival: None,
            ema_interval: nominal_us,
            ema_jitter: 0.0,
        };
        let mut textures: Option<SourceTextures> = None;
        let opened_at = Instant::now();
        let mut signal = false;
        let mut reported_no_signal = false;

        if let Err(e) = stream.start() {
            return ReadExit::Error(e);
        }

        loop {
            let msg = match self.rx.recv_timeout(Duration::from_millis(250)) {
                Ok(m) => m,
                Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                    let since = timing.last_arrival.unwrap_or(opened_at).elapsed();
                    if timing.last_arrival.is_some() && since > STALL_REOPEN_AFTER {
                        return ReadExit::Error("stream stalled".into());
                    }
                    if since > NO_SIGNAL_AFTER && !reported_no_signal {
                        signal = false;
                        reported_no_signal = true;
                        self.shared.running.store(false, Ordering::Relaxed);
                        self.set_state(CaptureState::NoSignal);
                    }
                    continue;
                }
                Err(_) => return ReadExit::Cmd(CaptureCmd::Shutdown),
            };
            let (raw, arrival) = match msg {
                Msg::Cmd(CaptureCmd::RefreshDevices) => {
                    self.refresh_devices();
                    continue;
                }
                Msg::Cmd(c) => return ReadExit::Cmd(c),
                Msg::Raw { session, .. } if session != self.session => continue,
                Msg::Raw { raw, arrival, .. } => (raw, arrival),
            };

            let hints = stream.hints();
            let mut got_frame = false;
            let res = stream.handle(raw, &mut |ev| match ev {
                Event::Dropped(n) => {
                    shared.stats.dropped.fetch_add(n, Ordering::Relaxed);
                }
                Event::Frame(f) => {
                    got_frame = true;
                    self.on_frame(&f, arrival, hints, &mut timing, &mut textures);
                }
            });
            if let Err(e) = res {
                return ReadExit::Error(e);
            }
            if got_frame {
                if !signal {
                    signal = true;
                    reported_no_signal = false;
                    self.shared.running.store(true, Ordering::Relaxed);
                    self.set_state(CaptureState::Running);
                }
                self.shared.frame_seq.fetch_add(1, Ordering::Release);
                (self.notify)();
            }
        }
    }

    fn on_frame(
        &mut self,
        f: &Frame<'_>,
        arrival: Instant,
        hints: ColorHints,
        t: &mut Timing,
        textures: &mut Option<SourceTextures>,
    ) {
        let shared = self.shared.clone();
        let stats = &shared.stats;
        // Drop / interval accounting from device timestamps (100 ns units).
        if let Some(ts) = f.ts_100ns {
            if let Some(prev) = t.last_ts {
                let d_us = (ts - prev) as f64 / 10.0;
                if d_us > t.nominal_us * 1.5 {
                    let missed = (d_us / t.nominal_us).round() as u64;
                    stats
                        .dropped
                        .fetch_add(missed.saturating_sub(1), Ordering::Relaxed);
                }
            }
            t.last_ts = Some(ts);
        }
        if let Some(prev) = t.last_arrival {
            let d = arrival.duration_since(prev).as_secs_f64() * 1e6;
            t.ema_interval = t.ema_interval * 0.9 + d * 0.1;
            t.ema_jitter = t.ema_jitter * 0.9 + (d - t.ema_interval).abs() * 0.1;
            stats
                .interval_us
                .store(t.ema_interval as u64, Ordering::Relaxed);
            stats
                .jitter_us
                .store(t.ema_jitter as u64, Ordering::Relaxed);
        }
        t.last_arrival = Some(arrival);
        if let Some(lat) = f.device_latency_us.filter(|&l| l < 2_000_000) {
            stats.device_latency_us.store(lat, Ordering::Relaxed);
        }

        let t0 = Instant::now();
        self.upload(f, hints, textures);
        stats
            .upload_us
            .store(t0.elapsed().as_micros() as u64, Ordering::Relaxed);
        stats.frames.fetch_add(1, Ordering::Relaxed);
        stats.last_arrival_ns.store(
            arrival.duration_since(shared.epoch).as_nanos() as u64,
            Ordering::Relaxed,
        );
    }

    fn upload(&mut self, f: &Frame<'_>, hints: ColorHints, textures: &mut Option<SourceTextures>) {
        let (fmt, w, h) = (f.format, f.width, f.height);
        let need_new = match textures {
            Some(t) => {
                t.width != w
                    || t.height != h
                    || t.format != fmt
                    || t.flip_y != f.bottom_up
                    || t.hints != hints
            }
            None => true,
        };
        if need_new {
            self.generation += 1;
            let t = SourceTextures {
                generation: self.generation,
                format: fmt,
                width: w,
                height: h,
                planes: create_planes(&self.device, fmt, w, h),
                hints,
                flip_y: f.bottom_up,
            };
            *textures = Some(t.clone());
            *self.shared.textures.lock() = Some(t);
        }
        let t = textures.as_ref().unwrap();

        let write = |tex: &wgpu::Texture, plane: &Plane<'_>, rows: u32, tw: u32| {
            if rows == 0 || tw == 0 {
                return;
            }
            let texel = tex.format().block_copy_size(None).unwrap_or(1) as usize;
            let need = plane.pitch * (rows as usize - 1) + texel * tw as usize;
            if need > plane.data.len() {
                return;
            }
            self.queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: tex,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                plane.data,
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(plane.pitch as u32),
                    rows_per_image: Some(rows),
                },
                wgpu::Extent3d {
                    width: tw,
                    height: rows,
                    depth_or_array_layers: 1,
                },
            );
        };

        match (fmt, f.planes) {
            (PixelFormat::Yuy2 | PixelFormat::Uyvy | PixelFormat::Mjpeg, [p, ..]) => {
                write(&t.planes[0], p, h, w / 2)
            }
            (PixelFormat::Bgra, [p, ..]) => write(&t.planes[0], p, h, w),
            (PixelFormat::Nv12 | PixelFormat::P010, [y, uv, ..]) => {
                write(&t.planes[0], y, h, w);
                write(&t.planes[1], uv, h / 2, w / 2);
            }
            _ => log::warn!("frame has too few planes for {fmt:?}"),
        }
    }
}

fn create_planes(device: &wgpu::Device, fmt: PixelFormat, w: u32, h: u32) -> Vec<wgpu::Texture> {
    let mk = |label: &str, format: wgpu::TextureFormat, tw: u32, th: u32| {
        device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d {
                width: tw.max(1),
                height: th.max(1),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        })
    };
    match fmt {
        PixelFormat::Yuy2 | PixelFormat::Uyvy | PixelFormat::Mjpeg => {
            vec![mk(
                "src_packed422",
                wgpu::TextureFormat::Rgba8Uint,
                w / 2,
                h,
            )]
        }
        PixelFormat::Bgra => vec![mk("src_bgra", wgpu::TextureFormat::Bgra8Unorm, w, h)],
        PixelFormat::Nv12 => vec![
            mk("src_y", wgpu::TextureFormat::R8Uint, w, h),
            mk("src_uv", wgpu::TextureFormat::Rg8Uint, w / 2, h / 2),
        ],
        PixelFormat::P010 => vec![
            mk("src_y16", wgpu::TextureFormat::R16Uint, w, h),
            mk("src_uv16", wgpu::TextureFormat::Rg16Uint, w / 2, h / 2),
        ],
    }
}
