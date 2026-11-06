//! Windows backend: Media Foundation async source reader.
//!
//! MF_LOW_LATENCY, no video processing for native formats; MJPEG modes get the
//! reader's own decoder inserted (4:2:2 YUY2 preferred). One ReadSample request is
//! kept outstanding; the async callback forwards each sample to the worker.

use std::time::Duration;

use windows::Win32::Media::MediaFoundation::*;
use windows::Win32::System::Com::{COINIT_MULTITHREADED, CoInitializeEx, CoTaskMemFree};
use windows::Win32::System::Performance::{QueryPerformanceCounter, QueryPerformanceFrequency};
use windows::Win32::System::Threading::AvSetMmThreadCharacteristicsW;
use windows::core::{GUID, HSTRING, PWSTR};

use super::{
    Backend, ColorHints, DeviceInfo, Event, FormatInfo, Frame, PixelFormat, Plane, Sink, Stream,
    rank_formats,
};

fn pixel_format(g: &GUID) -> Option<PixelFormat> {
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

pub struct Mf;

impl Backend for Mf {
    type Raw = SampleMsg;
    type Stream = MfStream;

    fn init() -> Result<Self, String> {
        unsafe {
            let _ = CoInitializeEx(None, COINIT_MULTITHREADED);
            MFStartup(MF_VERSION, MFSTARTUP_FULL).map_err(|e| format!("MFStartup failed: {e}"))?;
            // Ask MMCSS to schedule us like a capture thread.
            let mut task = 0u32;
            let _ = AvSetMmThreadCharacteristicsW(&HSTRING::from("Capture"), &mut task);
        }
        Ok(Mf)
    }

    fn devices(&mut self) -> Vec<DeviceInfo> {
        unsafe { enumerate_devices() }.unwrap_or_default()
    }

    fn open(
        &mut self,
        link: &str,
        format_key: Option<&str>,
        sink: Sink<SampleMsg>,
    ) -> Result<(MfStream, Vec<FormatInfo>), String> {
        let callback: IMFSourceReaderCallback = ReaderCallback { sink }.into();
        unsafe { open_reader(link, format_key, &callback) }.map_err(|e| e.message().to_string())
    }
}

impl Drop for Mf {
    fn drop(&mut self) {
        unsafe {
            let _ = MFShutdown();
        }
    }
}

/// A sample delivered by the async source-reader callback.
pub struct SampleMsg {
    hr: windows::core::HRESULT,
    flags: u32,
    ts: i64,
    sample: Option<IMFSample>,
}
// MF samples are free-threaded.
unsafe impl Send for SampleMsg {}

#[windows::core::implement(IMFSourceReaderCallback)]
struct ReaderCallback {
    sink: Sink<SampleMsg>,
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
        self.sink.send(SampleMsg {
            hr: hrstatus,
            flags,
            ts,
            sample: sample.cloned(),
        });
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

pub struct MfStream {
    source: IMFMediaSource,
    reader: IMFSourceReader,
    out_format: PixelFormat,
    width: u32,
    height: u32,
    default_stride: Option<i32>,
    selected: FormatInfo,
    hints: ColorHints,
    qpc_freq: i64,
}

struct SendSource(IMFMediaSource);
// MF media sources are free-threaded.
unsafe impl Send for SendSource {}

impl Drop for MfStream {
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

impl MfStream {
    fn request(&self) -> Result<(), String> {
        unsafe {
            self.reader
                .ReadSample(VIDEO_STREAM, 0, None, None, None, None)
        }
        .map_err(|e| format!("ReadSample: {e}"))
    }
}

impl Stream for MfStream {
    type Raw = SampleMsg;

    fn selected(&self) -> &FormatInfo {
        &self.selected
    }
    fn hints(&self) -> ColorHints {
        self.hints
    }
    fn start(&mut self) -> Result<(), String> {
        self.request()
    }

    fn handle(&mut self, m: SampleMsg, emit: &mut dyn FnMut(Event<'_>)) -> Result<(), String> {
        m.hr.ok().map_err(|e| format!("ReadSample: {e}"))?;
        let flags = m.flags;
        if flags & (MF_SOURCE_READERF_ERROR.0 as u32 | MF_SOURCE_READERF_ENDOFSTREAM.0 as u32) != 0
        {
            return Err("stream ended (device removed?)".into());
        }
        // Keep exactly one request outstanding; issue the next before we upload.
        self.request()?;
        if flags & MF_SOURCE_READERF_CURRENTMEDIATYPECHANGED.0 as u32 != 0 {
            if let Ok(t) = unsafe { self.reader.GetCurrentMediaType(VIDEO_STREAM) } {
                if let Some((f, w, h, stride)) = unsafe { describe_output(&t) } {
                    self.out_format = f;
                    self.width = w;
                    self.height = h;
                    self.default_stride = stride;
                }
            }
        }
        if flags & MF_SOURCE_READERF_STREAMTICK.0 as u32 != 0 {
            emit(Event::Dropped(1));
        }
        let Some(sample) = m.sample else {
            return Ok(());
        };

        // True device latency when the driver stamps the capture QPC time.
        let device_latency_us = unsafe {
            sample
                .GetUINT64(&MFSampleExtension_DeviceReferenceSystemTime)
                .ok()
                .and_then(|dev_time| {
                    let mut now = 0i64;
                    let _ = QueryPerformanceCounter(&mut now);
                    let now_100ns = (now as i128 * 10_000_000 / self.qpc_freq as i128) as i64;
                    u64::try_from((now_100ns - dev_time as i64) / 10).ok()
                })
        };

        let locked = match unsafe { lock_sample(&sample, self) } {
            Ok(l) => l,
            Err(e) => {
                log::warn!("lock failed: {e}");
                return Ok(());
            }
        };
        let data = locked.data();
        let pitch = locked.pitch;
        let (w, h) = (self.width, self.height);
        let planes_buf;
        let planes: &[Plane<'_>] = match self.out_format {
            PixelFormat::Nv12 | PixelFormat::P010 => {
                // Luma rows may be padded (e.g. 1088 for 1080p): derive from length.
                let total_rows = data.len() / pitch.max(1);
                let luma_rows = ((total_rows * 2) / 3).max(h as usize);
                let split = (luma_rows * pitch).min(data.len());
                planes_buf = [
                    Plane {
                        data: &data[..split],
                        pitch,
                    },
                    Plane {
                        data: &data[split..],
                        pitch,
                    },
                ];
                &planes_buf
            }
            _ => {
                planes_buf = [
                    Plane { data, pitch },
                    Plane {
                        data: &[],
                        pitch: 0,
                    },
                ];
                &planes_buf[..1]
            }
        };
        emit(Event::Frame(Frame {
            format: self.out_format,
            width: w,
            height: h,
            planes,
            bottom_up: locked.bottom_up,
            ts_100ns: Some(m.ts),
            device_latency_us,
        }));
        Ok(())
    }
}

unsafe fn lock_sample(sample: &IMFSample, open: &MfStream) -> windows::core::Result<LockedBuffer> {
    unsafe {
        let buffer = if sample.GetBufferCount()? > 1 {
            sample.ConvertToContiguousBuffer()?
        } else {
            sample.GetBufferByIndex(0)?
        };
        LockedBuffer::lock(&buffer, open)
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
    unsafe fn lock(buffer: &IMFMediaBuffer, open: &MfStream) -> windows::core::Result<Self> {
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

unsafe fn enumerate_devices() -> windows::core::Result<Vec<DeviceInfo>> {
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
        let format = pixel_format(&sub)?;
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
        let f = pixel_format(&sub)?;
        let (w, h) = split_u64(t.GetUINT64(&MF_MT_FRAME_SIZE).ok()?);
        let stride = t.GetUINT32(&MF_MT_DEFAULT_STRIDE).ok().map(|s| s as i32);
        Some((f, w, h, stride))
    }
}

unsafe fn open_reader(
    link: &str,
    format_key: Option<&str>,
    callback: &IMFSourceReaderCallback,
) -> windows::core::Result<(MfStream, Vec<FormatInfo>)> {
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

        // Enumerate native types.
        let mut formats: Vec<FormatInfo> = Vec::new();
        let mut i = 0u32;
        while let Ok(t) = reader.GetNativeMediaType(VIDEO_STREAM, i) {
            if let Some(f) = describe_native(&t, i) {
                formats.push(f);
            }
            i += 1;
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
        let pick = rank_formats(&mut formats, format_key).ok_or_else(|| {
            windows::core::Error::new(MF_E_NO_MORE_TYPES, "no supported video formats")
        })?;

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
        let mut qpc_freq = 0i64;
        let _ = QueryPerformanceFrequency(&mut qpc_freq);
        Ok((
            MfStream {
                source,
                reader,
                out_format,
                width,
                height,
                default_stride,
                selected: pick,
                hints,
                qpc_freq: qpc_freq.max(1),
            },
            formats,
        ))
    }
}
