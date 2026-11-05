//! Media Foundation capture thread.
//!
//! Owns all MF/COM state. Reads samples synchronously as fast as the device delivers
//! them (MF_LOW_LATENCY, no converters for native formats) and uploads each frame
//! straight into GPU textures with `queue.write_texture` from this thread. wgpu applies
//! pending writes at the start of the next `submit`, so the render thread always draws
//! the newest frame without an extra CPU copy or a hand-off queue.

use parking_lot::Mutex;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{Receiver, Sender};
use std::time::{Duration, Instant};

use windows::Win32::Media::MediaFoundation::*;
use windows::Win32::System::Com::{COINIT_MULTITHREADED, CoInitializeEx, CoTaskMemFree};
use windows::Win32::System::Performance::{QueryPerformanceCounter, QueryPerformanceFrequency};
use windows::Win32::System::Threading::AvSetMmThreadCharacteristicsW;
use windows::core::{GUID, HSTRING, PWSTR};

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
    fn from_guid(g: &GUID) -> Option<Self> {
        Some(match *g {
            x if x == MFVideoFormat_YUY2 => PixelFormat::Yuy2,
            x if x == MFVideoFormat_UYVY => PixelFormat::Uyvy,
            x if x == MFVideoFormat_NV12 => PixelFormat::Nv12,
            x if x == MFVideoFormat_P010 => PixelFormat::P010,
            x if x == MFVideoFormat_RGB32 || x == MFVideoFormat_ARGB32 => PixelFormat::Bgra,
            x if x == MFVideoFormat_MJPG => PixelFormat::Mjpeg,
            _ => return None,
        })
    }
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

#[derive(Clone, Debug, PartialEq)]
pub struct DeviceInfo {
    pub name: String,
    pub link: String,
}

/// The texture set that holds the most recent frame. Recreated on format change.
#[derive(Clone)]
pub struct SourceTextures {
    pub generation: u64,
    /// Decoded/output format actually in the textures (MJPEG becomes YUY2/NV12).
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

/// A sample delivered by the async source-reader callback.
struct SampleMsg {
    session: u64,
    hr: windows::core::HRESULT,
    flags: u32,
    ts: i64,
    sample: Option<IMFSample>,
    arrival: Instant,
}
// MF samples are free-threaded.
unsafe impl Send for SampleMsg {}

enum Msg {
    Cmd(CaptureCmd),
    Sample(SampleMsg),
}

#[windows::core::implement(IMFSourceReaderCallback)]
struct ReaderCallback {
    session: u64,
    tx: Mutex<Sender<Msg>>,
}

impl IMFSourceReaderCallback_Impl for ReaderCallback_Impl {
    fn OnReadSample(
        &self,
        hrstatus: windows::core::HRESULT,
        _stream: u32,
        flags: u32,
        ts: i64,
        sample: windows::core::Ref<IMFSample>,
    ) -> windows::core::Result<()> {
        let arrival = Instant::now();
        let _ = self.tx.lock().send(Msg::Sample(SampleMsg {
            session: self.session,
            hr: hrstatus,
            flags,
            ts,
            sample: sample.cloned(),
            arrival,
        }));
        Ok(())
    }
    fn OnFlush(&self, _stream: u32) -> windows::core::Result<()> {
        Ok(())
    }
    fn OnEvent(
        &self,
        _stream: u32,
        _event: windows::core::Ref<IMFMediaEvent>,
    ) -> windows::core::Result<()> {
        Ok(())
    }
}

pub struct Capture {
    pub shared: Arc<CaptureShared>,
    tx: Sender<Msg>,
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
                let mut w = Worker {
                    shared: sh,
                    device,
                    queue,
                    notify,
                    rx,
                    cb_tx,
                    session: 0,
                    generation: 0,
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

struct Worker {
    shared: Arc<CaptureShared>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    notify: Box<dyn Fn() + Send>,
    rx: Receiver<Msg>,
    cb_tx: Sender<Msg>,
    session: u64,
    generation: u64,
}

struct OpenReader {
    source: IMFMediaSource,
    reader: IMFSourceReader,
    out_format: PixelFormat,
    width: u32,
    height: u32,
    default_stride: Option<i32>,
    selected: FormatInfo,
    hints: ColorHints,
}

struct SendSource(IMFMediaSource);
// MF media sources are free-threaded.
unsafe impl Send for SendSource {}

impl Drop for OpenReader {
    /// Shutdown can block forever on a wedged driver; do it on a helper thread and
    /// wait at most 1 s so the capture thread always stays responsive.
    fn drop(&mut self) {
        let src = SendSource(self.source.clone());
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let s = src;
            unsafe {
                let _ = s.0.Shutdown();
            }
            let _ = tx.send(());
        });
        if rx.recv_timeout(Duration::from_secs(1)).is_err() {
            log::warn!("media source shutdown is hanging (driver wedged?); abandoning it");
        }
    }
}

const VIDEO_STREAM: u32 = MF_SOURCE_READER_FIRST_VIDEO_STREAM.0 as u32;
/// No frame for this long -> report "No signal" (keep waiting).
const NO_SIGNAL_AFTER: Duration = Duration::from_millis(1500);
/// Frames stopped after streaming was working -> tear down and reopen.
const STALL_REOPEN_AFTER: Duration = Duration::from_millis(3000);

impl Worker {
    fn set_state(&self, s: CaptureState) {
        self.shared.status.lock().state = s;
        (self.notify)();
    }

    fn refresh_devices(&self) {
        let devices = unsafe { enumerate_devices() }.unwrap_or_default();
        self.shared.status.lock().devices = devices;
        (self.notify)();
    }

    fn run(&mut self) {
        unsafe {
            let _ = CoInitializeEx(None, COINIT_MULTITHREADED);
            if let Err(e) = MFStartup(MF_VERSION, MFSTARTUP_FULL) {
                self.set_state(CaptureState::Error(format!("MFStartup failed: {e}")));
                return;
            }
            // Ask MMCSS to schedule us like a capture thread.
            let mut task = 0u32;
            let _ = AvSetMmThreadCharacteristicsW(&HSTRING::from("Capture"), &mut task);
        }

        let mut pending: Option<CaptureCmd> = None;
        loop {
            let cmd = match pending.take() {
                Some(c) => c,
                None => match self.rx.recv() {
                    Ok(Msg::Cmd(c)) => c,
                    Ok(Msg::Sample(_)) => continue, // stale sample from a closed session
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
        unsafe {
            let _ = MFShutdown();
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
            let callback: IMFSourceReaderCallback = ReaderCallback {
                session: self.session,
                tx: Mutex::new(self.cb_tx.clone()),
            }
            .into();
            let open = unsafe { open_reader(link, format_key, &callback) };
            match open {
                Ok((open, formats)) => {
                    {
                        let mut st = self.shared.status.lock();
                        st.formats = formats;
                        st.active_device = Some(device.clone());
                        st.active_format = Some(open.selected.clone());
                        st.state = CaptureState::Opening;
                    }
                    (self.notify)();
                    let result = self.read_loop(open);
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
                    Ok(Msg::Sample(_)) => continue,
                    Err(std::sync::mpsc::RecvTimeoutError::Timeout) => break,
                    Err(_) => return Some(CaptureCmd::Shutdown),
                }
            }
            retry_delay = (retry_delay * 2).min(Duration::from_secs(2));
            self.set_state(CaptureState::Opening);
        }
    }

    fn read_loop(&mut self, mut open: OpenReader) -> ReadExit {
        let shared = self.shared.clone();
        let stats = &shared.stats;
        stats.dropped.store(0, Ordering::Relaxed);
        let nominal_us = if open.selected.fps() > 0.0 {
            1e6 / open.selected.fps()
        } else {
            16_667.0
        };
        let mut last_ts: Option<i64> = None;
        let mut last_arrival: Option<Instant> = None;
        let mut ema_interval = nominal_us;
        let mut ema_jitter = 0.0f64;
        let qpc_freq = unsafe {
            let mut f = 0i64;
            let _ = QueryPerformanceFrequency(&mut f);
            f.max(1)
        };
        let mut textures: Option<SourceTextures> = None;
        let opened_at = Instant::now();
        let mut signal = false;
        let mut reported_no_signal = false;

        if let Err(e) = unsafe {
            open.reader
                .ReadSample(VIDEO_STREAM, 0, None, None, None, None)
        } {
            return ReadExit::Error(format!("ReadSample: {e}"));
        }

        loop {
            let msg = match self.rx.recv_timeout(Duration::from_millis(250)) {
                Ok(m) => m,
                Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                    let since = last_arrival.unwrap_or(opened_at).elapsed();
                    if last_arrival.is_some() && since > STALL_REOPEN_AFTER {
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
            let m = match msg {
                Msg::Cmd(CaptureCmd::RefreshDevices) => {
                    self.refresh_devices();
                    continue;
                }
                Msg::Cmd(c) => return ReadExit::Cmd(c),
                Msg::Sample(m) if m.session != self.session => continue,
                Msg::Sample(m) => m,
            };
            if let Err(e) = m.hr.ok() {
                return ReadExit::Error(format!("ReadSample: {e}"));
            }
            let flags = m.flags;
            if flags & (MF_SOURCE_READERF_ERROR.0 as u32 | MF_SOURCE_READERF_ENDOFSTREAM.0 as u32)
                != 0
            {
                return ReadExit::Error("stream ended (device removed?)".into());
            }
            // Keep exactly one request outstanding; issue the next before we upload.
            if let Err(e) = unsafe {
                open.reader
                    .ReadSample(VIDEO_STREAM, 0, None, None, None, None)
            } {
                return ReadExit::Error(format!("ReadSample: {e}"));
            }
            if flags & MF_SOURCE_READERF_CURRENTMEDIATYPECHANGED.0 as u32 != 0 {
                if let Ok(t) = unsafe { open.reader.GetCurrentMediaType(VIDEO_STREAM) } {
                    if let Some((f, w, h, stride)) = unsafe { describe_output(&t) } {
                        open.out_format = f;
                        open.width = w;
                        open.height = h;
                        open.default_stride = stride;
                    }
                }
                textures = None;
            }
            if flags & MF_SOURCE_READERF_STREAMTICK.0 as u32 != 0 {
                stats.dropped.fetch_add(1, Ordering::Relaxed);
            }
            let Some(sample) = m.sample else { continue };
            let (ts, arrival) = (m.ts, m.arrival);

            if !signal {
                signal = true;
                reported_no_signal = false;
                self.shared.running.store(true, Ordering::Relaxed);
                self.set_state(CaptureState::Running);
            }

            // Drop / interval accounting from device timestamps (100 ns units).
            if let Some(prev) = last_ts {
                let d_us = (ts - prev) as f64 / 10.0;
                if d_us > nominal_us * 1.5 {
                    let missed = (d_us / nominal_us).round() as u64;
                    stats
                        .dropped
                        .fetch_add(missed.saturating_sub(1), Ordering::Relaxed);
                }
            }
            last_ts = Some(ts);
            if let Some(prev) = last_arrival {
                let d = arrival.duration_since(prev).as_secs_f64() * 1e6;
                ema_interval = ema_interval * 0.9 + d * 0.1;
                ema_jitter = ema_jitter * 0.9 + (d - ema_interval).abs() * 0.1;
                stats
                    .interval_us
                    .store(ema_interval as u64, Ordering::Relaxed);
                stats.jitter_us.store(ema_jitter as u64, Ordering::Relaxed);
            }
            last_arrival = Some(arrival);

            // True device latency when the driver stamps the capture QPC time.
            unsafe {
                if let Ok(dev_time) = sample.GetUINT64(&MFSampleExtension_DeviceReferenceSystemTime)
                {
                    let mut now = 0i64;
                    let _ = QueryPerformanceCounter(&mut now);
                    let now_100ns = (now as i128 * 10_000_000 / qpc_freq as i128) as i64;
                    let lat = (now_100ns - dev_time as i64) / 10;
                    if (0..2_000_000).contains(&lat) {
                        stats.device_latency_us.store(lat as u64, Ordering::Relaxed);
                    }
                }
            }

            let t0 = Instant::now();
            if let Err(e) = unsafe { self.upload(&open, &sample, &mut textures) } {
                log::warn!("upload failed: {e}");
                continue;
            }
            stats
                .upload_us
                .store(t0.elapsed().as_micros() as u64, Ordering::Relaxed);
            stats.frames.fetch_add(1, Ordering::Relaxed);
            stats.last_arrival_ns.store(
                arrival.duration_since(self.shared.epoch).as_nanos() as u64,
                Ordering::Relaxed,
            );
            self.shared.frame_seq.fetch_add(1, Ordering::Release);
            (self.notify)();
        }
    }

    unsafe fn upload(
        &mut self,
        open: &OpenReader,
        sample: &IMFSample,
        textures: &mut Option<SourceTextures>,
    ) -> windows::core::Result<()> {
        unsafe {
            let buffer = if sample.GetBufferCount()? > 1 {
                sample.ConvertToContiguousBuffer()?
            } else {
                sample.GetBufferByIndex(0)?
            };
            let locked = LockedBuffer::lock(&buffer, open)?;

            let fmt = open.out_format;
            let (w, h) = (open.width, open.height);
            let flip_y = locked.bottom_up;

            let need_new = match textures {
                Some(t) => t.width != w || t.height != h || t.format != fmt || t.flip_y != flip_y,
                None => true,
            };
            if need_new {
                self.generation += 1;
                let planes = create_planes(&self.device, fmt, w, h);
                let t = SourceTextures {
                    generation: self.generation,
                    format: fmt,
                    width: w,
                    height: h,
                    planes,
                    hints: open.hints,
                    flip_y,
                };
                *textures = Some(t.clone());
                *self.shared.textures.lock() = Some(t);
            }
            let t = textures.as_ref().unwrap();
            let pitch = locked.pitch;
            let data = locked.data();

            let write = |tex: &wgpu::Texture, offset: usize, bpr: usize, rows: u32, tw: u32| {
                let need = bpr * (rows as usize - 1)
                    + (tex.format().block_copy_size(None).unwrap_or(1) * tw) as usize;
                if offset + need > data.len() {
                    return;
                }
                self.queue.write_texture(
                    wgpu::TexelCopyTextureInfo {
                        texture: tex,
                        mip_level: 0,
                        origin: wgpu::Origin3d::ZERO,
                        aspect: wgpu::TextureAspect::All,
                    },
                    &data[offset..],
                    wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(bpr as u32),
                        rows_per_image: Some(rows),
                    },
                    wgpu::Extent3d {
                        width: tw,
                        height: rows,
                        depth_or_array_layers: 1,
                    },
                );
            };

            match fmt {
                PixelFormat::Yuy2 | PixelFormat::Uyvy | PixelFormat::Mjpeg => {
                    write(&t.planes[0], 0, pitch, h, w / 2)
                }
                PixelFormat::Bgra => write(&t.planes[0], 0, pitch, h, w),
                PixelFormat::Nv12 | PixelFormat::P010 => {
                    // Luma rows may be padded (e.g. 1088 for 1080p): derive from length.
                    let total_rows = data.len() / pitch;
                    let luma_rows = ((total_rows * 2) / 3).max(h as usize);
                    write(&t.planes[0], 0, pitch, h, w);
                    write(&t.planes[1], luma_rows * pitch, pitch, h / 2, w / 2);
                }
            }
            drop(locked);
            Ok(())
        }
    }
}

enum ReadExit {
    Cmd(CaptureCmd),
    Error(String),
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

struct LockedBuffer {
    buf2d: Option<IMF2DBuffer>,
    buf: Option<IMFMediaBuffer>,
    start: *const u8,
    len: usize,
    pitch: usize,
    bottom_up: bool,
}

impl LockedBuffer {
    unsafe fn lock(buffer: &IMFMediaBuffer, open: &OpenReader) -> windows::core::Result<Self> {
        unsafe {
            use windows::core::Interface;
            if let Ok(b2) = buffer.cast::<IMF2DBuffer2>() {
                let mut scan0 = std::ptr::null_mut();
                let mut pitch = 0i32;
                let mut start = std::ptr::null_mut();
                let mut len = 0u32;
                if b2
                    .Lock2DSize(
                        MF2DBuffer_LockFlags_Read,
                        &mut scan0,
                        &mut pitch,
                        &mut start,
                        &mut len,
                    )
                    .is_ok()
                {
                    return Ok(LockedBuffer {
                        buf2d: Some(b2.cast()?),
                        buf: None,
                        start,
                        len: len as usize,
                        pitch: pitch.unsigned_abs() as usize,
                        bottom_up: pitch < 0,
                    });
                }
            }
            // Contiguous fallback.
            let mut p = std::ptr::null_mut();
            let mut cur = 0u32;
            buffer.Lock(&mut p, None, Some(&mut cur))?;
            let stride = open.default_stride.unwrap_or_else(|| {
                let bpp = match open.out_format {
                    PixelFormat::Bgra => 4,
                    PixelFormat::Nv12 => 1,
                    PixelFormat::P010 => 2,
                    _ => 2,
                };
                (open.width * bpp) as i32
            });
            Ok(LockedBuffer {
                buf2d: None,
                buf: Some(buffer.clone()),
                start: p,
                len: cur as usize,
                pitch: stride.unsigned_abs() as usize,
                bottom_up: stride < 0,
            })
        }
    }
    fn data(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.start, self.len) }
    }
}

impl Drop for LockedBuffer {
    fn drop(&mut self) {
        unsafe {
            if let Some(b) = &self.buf2d {
                let _ = b.Unlock2D();
            }
            if let Some(b) = &self.buf {
                let _ = b.Unlock();
            }
        }
    }
}

unsafe fn get_string(a: &IMFActivate, key: &GUID) -> Option<String> {
    unsafe {
        let mut p = PWSTR::null();
        let mut n = 0u32;
        a.GetAllocatedString(key, &mut p, &mut n).ok()?;
        let s = p.to_string().ok();
        CoTaskMemFree(Some(p.0 as _));
        s
    }
}

pub unsafe fn enumerate_devices() -> windows::core::Result<Vec<DeviceInfo>> {
    unsafe {
        let mut attrs = None;
        MFCreateAttributes(&mut attrs, 1)?;
        let attrs = attrs.unwrap();
        attrs.SetGUID(
            &MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE,
            &MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE_VIDCAP_GUID,
        )?;
        let mut arr: *mut Option<IMFActivate> = std::ptr::null_mut();
        let mut count = 0u32;
        MFEnumDeviceSources(&attrs, &mut arr, &mut count)?;
        let mut out = Vec::new();
        for i in 0..count as usize {
            if let Some(a) = (*arr.add(i)).take() {
                let name =
                    get_string(&a, &MF_DEVSOURCE_ATTRIBUTE_FRIENDLY_NAME).unwrap_or_default();
                let link = get_string(&a, &MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE_VIDCAP_SYMBOLIC_LINK)
                    .unwrap_or_default();
                out.push(DeviceInfo { name, link });
            }
        }
        if !arr.is_null() {
            CoTaskMemFree(Some(arr as _));
        }
        Ok(out)
    }
}

fn split_u64(v: u64) -> (u32, u32) {
    ((v >> 32) as u32, v as u32)
}

unsafe fn read_hints(t: &IMFMediaType) -> ColorHints {
    unsafe {
        let matrix = t.GetUINT32(&MF_MT_YUV_MATRIX).ok().and_then(|m| match m {
            1 => Some(709),
            2 => Some(601),
            4 | 5 => Some(2020),
            _ => None,
        });
        let full_range = t
            .GetUINT32(&MF_MT_VIDEO_NOMINAL_RANGE)
            .ok()
            .and_then(|r| match r {
                1 => Some(true),
                2 => Some(false),
                _ => None,
            });
        let cosited = t
            .GetUINT32(&MF_MT_VIDEO_CHROMA_SITING)
            .ok()
            .filter(|&s| s != 0)
            .map(|s| s & 0x4 != 0); // MFVideoChromaSubsampling_Horizontally_Cosited
        ColorHints {
            matrix,
            full_range,
            cosited,
        }
    }
}

unsafe fn describe_native(t: &IMFMediaType, index: u32) -> Option<FormatInfo> {
    unsafe {
        let sub = t.GetGUID(&MF_MT_SUBTYPE).ok()?;
        let format = PixelFormat::from_guid(&sub)?;
        let (width, height) = split_u64(t.GetUINT64(&MF_MT_FRAME_SIZE).ok()?);
        let (fps_num, fps_den) = t
            .GetUINT64(&MF_MT_FRAME_RATE)
            .map(split_u64)
            .unwrap_or((0, 1));
        Some(FormatInfo {
            native_index: index,
            format,
            width,
            height,
            fps_num,
            fps_den,
            hints: read_hints(t),
            windows_decoded: false,
        })
    }
}

unsafe fn describe_output(t: &IMFMediaType) -> Option<(PixelFormat, u32, u32, Option<i32>)> {
    unsafe {
        let sub = t.GetGUID(&MF_MT_SUBTYPE).ok()?;
        let f = PixelFormat::from_guid(&sub)?;
        let (w, h) = split_u64(t.GetUINT64(&MF_MT_FRAME_SIZE).ok()?);
        let stride = t.GetUINT32(&MF_MT_DEFAULT_STRIDE).ok().map(|s| s as i32);
        Some((f, w, h, stride))
    }
}

unsafe fn open_reader(
    link: &str,
    format_key: Option<&str>,
    callback: &IMFSourceReaderCallback,
) -> windows::core::Result<(OpenReader, Vec<FormatInfo>)> {
    unsafe {
        let mut attrs = None;
        MFCreateAttributes(&mut attrs, 2)?;
        let attrs = attrs.unwrap();
        attrs.SetGUID(
            &MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE,
            &MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE_VIDCAP_GUID,
        )?;
        attrs.SetString(
            &MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE_VIDCAP_SYMBOLIC_LINK,
            &HSTRING::from(link),
        )?;
        let source = MFCreateDeviceSource(&attrs)?;
        // Shut the source down if anything below fails.
        struct Guard(Option<IMFMediaSource>);
        impl Drop for Guard {
            fn drop(&mut self) {
                if let Some(s) = &self.0 {
                    unsafe {
                        let _ = s.Shutdown();
                    }
                }
            }
        }
        let mut guard = Guard(Some(source.clone()));

        let mut rattrs = None;
        MFCreateAttributes(&mut rattrs, 3)?;
        let rattrs = rattrs.unwrap();
        rattrs.SetUINT32(&MF_LOW_LATENCY, 1)?;
        rattrs.SetUINT32(&MF_SOURCE_READER_ENABLE_VIDEO_PROCESSING, 0)?;
        rattrs.SetUnknown(&MF_SOURCE_READER_ASYNC_CALLBACK, callback)?;
        let reader = MFCreateSourceReaderFromMediaSource(&source, &rattrs)?;
        reader.SetStreamSelection(MF_SOURCE_READER_ALL_STREAMS.0 as u32, false)?;
        reader.SetStreamSelection(VIDEO_STREAM, true)?;

        // Enumerate native types (dedupe by key, keep first).
        let mut formats: Vec<FormatInfo> = Vec::new();
        let mut i = 0u32;
        while let Ok(t) = reader.GetNativeMediaType(VIDEO_STREAM, i) {
            if let Some(f) = describe_native(&t, i) {
                if !formats.iter().any(|x| x.key() == f.key()) {
                    formats.push(f);
                }
            }
            i += 1;
        }
        if formats.is_empty() {
            return Err(windows::core::Error::new(
                MF_E_NO_MORE_TYPES,
                "no supported video formats",
            ));
        }
        // NV12 modes that exactly mirror an MJPEG mode come from the Frame Server.
        let mjpeg: Vec<(u32, u32, u32, u32)> = formats
            .iter()
            .filter(|f| f.format == PixelFormat::Mjpeg)
            .map(|f| (f.width, f.height, f.fps_num, f.fps_den))
            .collect();
        for f in formats.iter_mut() {
            if f.format == PixelFormat::Nv12
                && mjpeg.contains(&(f.width, f.height, f.fps_num, f.fps_den))
            {
                f.windows_decoded = true;
            }
        }
        formats.sort_by(|a, b| b.score().cmp(&a.score()));
        for f in &formats {
            log::debug!("format: {}  [{}]", f.label(), f.key());
        }

        let pick = format_key
            .and_then(|k| formats.iter().find(|f| f.key() == k))
            .unwrap_or(&formats[0])
            .clone();

        let native = reader.GetNativeMediaType(VIDEO_STREAM, pick.native_index)?;
        reader.SetCurrentMediaType(VIDEO_STREAM, None, &native)?;

        if pick.format == PixelFormat::Mjpeg {
            // Let the reader insert the MJPEG decoder; prefer 4:2:2 output.
            let mut ok = false;
            for sub in [MFVideoFormat_YUY2, MFVideoFormat_NV12] {
                let out = MFCreateMediaType()?;
                native.CopyAllItems(&out)?;
                out.SetGUID(&MF_MT_SUBTYPE, &sub)?;
                let _ = out.DeleteItem(&MF_MT_DEFAULT_STRIDE);
                let _ = out.DeleteItem(&MF_MT_SAMPLE_SIZE);
                if reader.SetCurrentMediaType(VIDEO_STREAM, None, &out).is_ok() {
                    ok = true;
                    break;
                }
            }
            if !ok {
                return Err(windows::core::Error::new(
                    MF_E_NO_MORE_TYPES,
                    "no MJPEG decoder available",
                ));
            }
        }

        let cur = reader.GetCurrentMediaType(VIDEO_STREAM)?;
        let (out_format, width, height, default_stride) =
            describe_output(&cur).ok_or_else(|| {
                windows::core::Error::new(MF_E_NO_MORE_TYPES, "unsupported output type")
            })?;
        let mut hints = pick.hints;
        if pick.format == PixelFormat::Mjpeg {
            // JPEG is full range unless the decoder says otherwise.
            let h = read_hints(&cur);
            hints.full_range = h.full_range.or(hints.full_range).or(Some(true));
            hints.matrix = h.matrix.or(hints.matrix);
            hints.cosited = h.cosited.or(hints.cosited).or(Some(false));
        }

        log::info!(
            "opened {} -> output {:?} {}x{}",
            pick.label(),
            out_format,
            width,
            height
        );
        guard.0 = None;
        Ok((
            OpenReader {
                source,
                reader,
                out_format,
                width,
                height,
                default_stride,
                selected: pick,
                hints,
            },
            formats,
        ))
    }
}
