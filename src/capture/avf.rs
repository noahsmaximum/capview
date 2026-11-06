//! macOS backend: AVFoundation (AVCaptureSession + AVCaptureVideoDataOutput).
//!
//! The device's native pixel format is requested as the output format so no
//! conversion runs in between; MJPEG modes are decoded by AVFoundation to '2vuy'
//! (UYVY 4:2:2). Late frames are discarded by the output rather than queued, and the
//! delegate runs on a user-interactive serial queue that only retains the
//! CVPixelBuffer and forwards it to the worker, which uploads it.

use std::ffi::c_void;
use std::sync::atomic::{AtomicBool, Ordering};

use block2::RcBlock;
use dispatch2::{
    DispatchQoS, DispatchQueue, DispatchQueueAttr, DispatchRetained, GlobalQueueIdentifier,
};
use objc2::rc::{Retained, autoreleasepool};
use objc2::runtime::{AnyObject, Bool, ProtocolObject};
use objc2::{AllocAnyThread, DefinedClass, define_class, msg_send};
use objc2_av_foundation::{
    AVAuthorizationStatus, AVCaptureConnection, AVCaptureDevice, AVCaptureDeviceFormat,
    AVCaptureDeviceInput, AVCaptureOutput, AVCaptureSession, AVCaptureVideoDataOutput,
    AVCaptureVideoDataOutputSampleBufferDelegate, AVMediaTypeVideo,
};
use objc2_core_foundation::{CFRetained, CFString};
use objc2_core_media::{
    CMClock, CMFormatDescription, CMSampleBuffer, CMTime, CMVideoFormatDescriptionGetDimensions,
    kCMFormatDescriptionExtension_YCbCrMatrix, kCMFormatDescriptionYCbCrMatrix_ITU_R_601_4,
    kCMFormatDescriptionYCbCrMatrix_ITU_R_709_2, kCMFormatDescriptionYCbCrMatrix_ITU_R_2020,
    kCMVideoCodecType_JPEG, kCMVideoCodecType_JPEG_OpenDML,
};
use objc2_core_video::{
    CVPixelBuffer, CVPixelBufferGetBaseAddress, CVPixelBufferGetBaseAddressOfPlane,
    CVPixelBufferGetBytesPerRow, CVPixelBufferGetBytesPerRowOfPlane, CVPixelBufferGetHeight,
    CVPixelBufferGetHeightOfPlane, CVPixelBufferGetPixelFormatType, CVPixelBufferGetPlaneCount,
    CVPixelBufferGetWidth, CVPixelBufferIsPlanar, CVPixelBufferLockBaseAddress,
    CVPixelBufferLockFlags, CVPixelBufferUnlockBaseAddress, kCVPixelBufferPixelFormatTypeKey,
    kCVPixelFormatType_32BGRA, kCVPixelFormatType_420YpCbCr8BiPlanarFullRange,
    kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange,
    kCVPixelFormatType_420YpCbCr10BiPlanarVideoRange, kCVPixelFormatType_422YpCbCr8,
    kCVPixelFormatType_422YpCbCr8_yuvs,
};
use objc2_foundation::{NSDictionary, NSNumber, NSObject, NSObjectProtocol, NSString};

use super::{
    Backend, ColorHints, DeviceInfo, Event, FormatInfo, Frame, PixelFormat, Plane, Sink, Stream,
    rank_formats,
};

/// CoreVideo pixel format → our format (what lands in the CVPixelBuffer).
fn pixel_format(fourcc: u32) -> Option<PixelFormat> {
    #[allow(non_upper_case_globals)]
    Some(match fourcc {
        kCVPixelFormatType_422YpCbCr8_yuvs => PixelFormat::Yuy2,
        kCVPixelFormatType_422YpCbCr8 => PixelFormat::Uyvy,
        kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange
        | kCVPixelFormatType_420YpCbCr8BiPlanarFullRange => PixelFormat::Nv12,
        // 10 bits in the high bits of 16 — the same layout as P010.
        kCVPixelFormatType_420YpCbCr10BiPlanarVideoRange => PixelFormat::P010,
        kCVPixelFormatType_32BGRA => PixelFormat::Bgra,
        kCMVideoCodecType_JPEG_OpenDML | kCMVideoCodecType_JPEG => PixelFormat::Mjpeg,
        _ => return None,
    })
}

/// The CVPixelBuffer format to ask the output for. Native formats pass straight
/// through; MJPEG is decoded by AVFoundation to UYVY 4:2:2 (keeps full chroma).
fn output_fourcc(native: u32) -> u32 {
    match pixel_format(native) {
        Some(PixelFormat::Mjpeg) | None => kCVPixelFormatType_422YpCbCr8,
        Some(_) => native,
    }
}

fn video_type() -> &'static objc2_av_foundation::AVMediaType {
    // SAFETY: framework constant, always present.
    unsafe { AVMediaTypeVideo }.expect("AVMediaTypeVideo")
}

fn cf_str_as_ns(s: &CFString) -> &NSString {
    // SAFETY: CFString and NSString are toll-free bridged.
    unsafe { &*(s as *const CFString).cast::<NSString>() }
}

pub struct Avf {
    /// We asked for camera permission this run (avoid re-prompting every retry).
    asked: bool,
}

impl Backend for Avf {
    type Raw = Delivery;
    type Stream = AvfStream;

    fn init() -> Result<Self, String> {
        // SAFETY: plain libc call on the current thread.
        unsafe {
            libc::pthread_set_qos_class_self_np(libc::qos_class_t::QOS_CLASS_USER_INTERACTIVE, 0);
        }
        Ok(Avf { asked: false })
    }

    fn devices(&mut self) -> Vec<DeviceInfo> {
        autoreleasepool(|_| {
            #[allow(deprecated)]
            // SAFETY: plain AVFoundation query.
            let list = unsafe { AVCaptureDevice::devicesWithMediaType(video_type()) };
            list.iter()
                .map(|d| unsafe {
                    DeviceInfo {
                        name: d.localizedName().to_string(),
                        link: d.uniqueID().to_string(),
                    }
                })
                .collect()
        })
    }

    fn open(
        &mut self,
        link: &str,
        format_key: Option<&str>,
        sink: Sink<Delivery>,
    ) -> Result<(AvfStream, Vec<FormatInfo>), String> {
        autoreleasepool(|_| unsafe { self.open_inner(link, format_key, sink) })
    }
}

impl Avf {
    unsafe fn open_inner(
        &mut self,
        link: &str,
        format_key: Option<&str>,
        sink: Sink<Delivery>,
    ) -> Result<(AvfStream, Vec<FormatInfo>), String> {
        unsafe {
            match AVCaptureDevice::authorizationStatusForMediaType(video_type()) {
                AVAuthorizationStatus::Authorized => {}
                AVAuthorizationStatus::NotDetermined => {
                    if !self.asked {
                        self.asked = true;
                        let block = RcBlock::new(|granted: Bool| {
                            log::info!("camera access granted: {}", granted.as_bool());
                        });
                        AVCaptureDevice::requestAccessForMediaType_completionHandler(
                            video_type(),
                            &block,
                        );
                    }
                    return Err("waiting for camera access — allow capview in the prompt".into());
                }
                _ => {
                    return Err(
                        "camera access is off — enable capview in System Settings › \
                                Privacy & Security › Camera"
                            .into(),
                    );
                }
            }

            let device = AVCaptureDevice::deviceWithUniqueID(&NSString::from_str(link))
                .ok_or("device not found (unplugged?)")?;

            // Every (format, frame-rate range) pair is one selectable mode.
            let native: Vec<Retained<AVCaptureDeviceFormat>> = device.formats().to_vec();
            let mut formats = Vec::new();
            for (fi, f) in native.iter().enumerate() {
                let desc = f.formatDescription();
                let fourcc = desc.media_sub_type();
                let Some(format) = pixel_format(fourcc) else {
                    continue;
                };
                let dims = CMVideoFormatDescriptionGetDimensions(&desc);
                let hints = read_hints(&desc, fourcc);
                for (ri, r) in f.videoSupportedFrameRateRanges().iter().enumerate() {
                    let d = r.minFrameDuration();
                    let (num, den) = if d.value > 0 && d.timescale > 0 {
                        reduce(d.timescale as u64, d.value as u64)
                    } else {
                        ((r.maxFrameRate() * 1000.0).round() as u32, 1000)
                    };
                    formats.push(FormatInfo {
                        native_index: ((fi as u32) << 8) | ri as u32,
                        format,
                        width: dims.width.max(0) as u32,
                        height: dims.height.max(0) as u32,
                        fps_num: num,
                        fps_den: den,
                        hints,
                        windows_decoded: false,
                    });
                }
            }
            let pick =
                rank_formats(&mut formats, format_key).ok_or("no supported video formats")?;
            let fmt = &native[(pick.native_index >> 8) as usize];
            let range = fmt
                .videoSupportedFrameRateRanges()
                .to_vec()
                .into_iter()
                .nth((pick.native_index & 0xff) as usize)
                .ok_or("frame rate range vanished")?;
            let native_fourcc = fmt.formatDescription().media_sub_type();

            let session = AVCaptureSession::new();
            let input = AVCaptureDeviceInput::deviceInputWithDevice_error(&device)
                .map_err(|e| e.localizedDescription().to_string())?;
            session.beginConfiguration();
            if !session.canAddInput(&input) {
                session.commitConfiguration();
                return Err("device is busy (in use by another app?)".into());
            }
            session.addInput(&input);

            let output = AVCaptureVideoDataOutput::new();
            output.setAlwaysDiscardsLateVideoFrames(true);
            let fourcc = NSNumber::new_u32(output_fourcc(native_fourcc));
            let settings = NSDictionary::<NSString, AnyObject>::from_slices(
                &[cf_str_as_ns(kCVPixelBufferPixelFormatTypeKey)],
                &[&*fourcc as &AnyObject],
            );
            output.setVideoSettings(Some(&settings));

            let delegate = FrameDelegate::new(sink);
            let target = DispatchQueue::global_queue(GlobalQueueIdentifier::QualityOfService(
                DispatchQoS::UserInteractive,
            ));
            let queue = DispatchQueue::new_with_target(
                "capview.capture",
                DispatchQueueAttr::SERIAL,
                Some(&target),
            );
            output.setSampleBufferDelegate_queue(
                Some(ProtocolObject::from_ref(&*delegate)),
                Some(&queue),
            );
            if !session.canAddOutput(&output) {
                session.commitConfiguration();
                return Err("can't attach video output".into());
            }
            session.addOutput(&output);
            session.commitConfiguration();

            // Set the mode after the input is attached (attaching resets it to the
            // session preset), and hold the lock until the session is running so the
            // preset can't override it.
            device
                .lockForConfiguration()
                .map_err(|e| e.localizedDescription().to_string())?;
            device.setActiveFormat(fmt);
            let dur = range.minFrameDuration();
            device.setActiveVideoMinFrameDuration(dur);
            device.setActiveVideoMaxFrameDuration(dur);

            let mut hints = pick.hints;
            if pick.format == PixelFormat::Mjpeg {
                // AVFoundation's decoder writes video-range 4:2:2, cosited chroma.
                hints.full_range = Some(false);
                hints.cosited = hints.cosited.or(Some(true));
            }
            log::info!(
                "opened {} -> output {}",
                pick.label(),
                fourcc_str(output_fourcc(native_fourcc))
            );
            Ok((
                AvfStream {
                    session,
                    output,
                    device,
                    _queue: queue,
                    _delegate: delegate,
                    config_locked: AtomicBool::new(true),
                    selected: pick,
                    hints,
                },
                formats,
            ))
        }
    }
}

fn reduce(mut num: u64, mut den: u64) -> (u32, u32) {
    let (mut a, mut b) = (num, den);
    while b != 0 {
        (a, b) = (b, a % b);
    }
    if a > 1 {
        num /= a;
        den /= a;
    }
    (
        num.min(u32::MAX as u64) as u32,
        den.min(u32::MAX as u64) as u32,
    )
}

fn fourcc_str(f: u32) -> String {
    f.to_be_bytes().iter().map(|&b| b as char).collect()
}

unsafe fn read_hints(desc: &CMFormatDescription, fourcc: u32) -> ColorHints {
    unsafe {
        let matrix = desc
            .extension(kCMFormatDescriptionExtension_YCbCrMatrix)
            .and_then(|v| {
                let s = v.downcast_ref::<CFString>()?;
                if s == kCMFormatDescriptionYCbCrMatrix_ITU_R_709_2 {
                    Some(709)
                } else if s == kCMFormatDescriptionYCbCrMatrix_ITU_R_601_4 {
                    Some(601)
                } else if s == kCMFormatDescriptionYCbCrMatrix_ITU_R_2020 {
                    Some(2020)
                } else {
                    None
                }
            });
        #[allow(non_upper_case_globals)]
        let full_range = match fourcc {
            kCVPixelFormatType_420YpCbCr8BiPlanarFullRange => Some(true),
            kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange
            | kCVPixelFormatType_420YpCbCr10BiPlanarVideoRange => Some(false),
            _ => None,
        };
        ColorHints {
            matrix,
            full_range,
            cosited: None,
        }
    }
}

/// What the delegate queue hands the worker.
pub enum Delivery {
    Frame {
        buffer: CFRetained<CVPixelBuffer>,
        ts_100ns: Option<i64>,
        latency_us: Option<u64>,
    },
    Dropped,
}
// CVPixelBuffers are reference counted and safe to hand between threads.
unsafe impl Send for Delivery {}

pub struct AvfStream {
    session: Retained<AVCaptureSession>,
    output: Retained<AVCaptureVideoDataOutput>,
    device: Retained<AVCaptureDevice>,
    _queue: DispatchRetained<DispatchQueue>,
    _delegate: Retained<FrameDelegate>,
    config_locked: AtomicBool,
    selected: FormatInfo,
    hints: ColorHints,
}

impl AvfStream {
    fn unlock_config(&self) {
        if self.config_locked.swap(false, Ordering::Relaxed) {
            unsafe { self.device.unlockForConfiguration() };
        }
    }
}

impl Drop for AvfStream {
    fn drop(&mut self) {
        autoreleasepool(|_| unsafe {
            self.output.setSampleBufferDelegate_queue(None, None);
            if self.session.isRunning() {
                self.session.stopRunning();
            }
            self.unlock_config();
        });
    }
}

impl Stream for AvfStream {
    type Raw = Delivery;

    fn selected(&self) -> &FormatInfo {
        &self.selected
    }
    fn hints(&self) -> ColorHints {
        self.hints
    }

    fn start(&mut self) -> Result<(), String> {
        autoreleasepool(|_| unsafe {
            self.session.startRunning();
            self.unlock_config();
            if self.session.isRunning() {
                Ok(())
            } else {
                Err("capture session failed to start".into())
            }
        })
    }

    fn handle(&mut self, raw: Delivery, emit: &mut dyn FnMut(Event<'_>)) -> Result<(), String> {
        let (pb, ts_100ns, latency_us) = match raw {
            Delivery::Dropped => {
                emit(Event::Dropped(1));
                return Ok(());
            }
            Delivery::Frame {
                buffer,
                ts_100ns,
                latency_us,
            } => (buffer, ts_100ns, latency_us),
        };
        let pb: &CVPixelBuffer = &pb;
        let Some(format) = pixel_format(CVPixelBufferGetPixelFormatType(pb)) else {
            return Ok(());
        };
        // SAFETY: read-only lock; every slice below is dropped before the unlock.
        unsafe {
            if CVPixelBufferLockBaseAddress(pb, CVPixelBufferLockFlags::ReadOnly) != 0 {
                return Ok(());
            }
            let plane = |base: *mut c_void, pitch: usize, rows: usize| Plane {
                data: if base.is_null() {
                    &[]
                } else {
                    std::slice::from_raw_parts(base as *const u8, pitch * rows)
                },
                pitch,
            };
            let planes: Vec<Plane<'_>> = if CVPixelBufferIsPlanar(pb) {
                (0..CVPixelBufferGetPlaneCount(pb).min(2))
                    .map(|i| {
                        plane(
                            CVPixelBufferGetBaseAddressOfPlane(pb, i),
                            CVPixelBufferGetBytesPerRowOfPlane(pb, i),
                            CVPixelBufferGetHeightOfPlane(pb, i),
                        )
                    })
                    .collect()
            } else {
                vec![plane(
                    CVPixelBufferGetBaseAddress(pb),
                    CVPixelBufferGetBytesPerRow(pb),
                    CVPixelBufferGetHeight(pb),
                )]
            };
            emit(Event::Frame(Frame {
                format,
                width: CVPixelBufferGetWidth(pb) as u32,
                height: CVPixelBufferGetHeight(pb) as u32,
                planes: &planes,
                bottom_up: false,
                ts_100ns,
                device_latency_us: latency_us,
            }));
            drop(planes);
            CVPixelBufferUnlockBaseAddress(pb, CVPixelBufferLockFlags::ReadOnly);
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Sample-buffer delegate
// ---------------------------------------------------------------------------

pub struct DelegateIvars {
    sink: Sink<Delivery>,
}

define_class!(
    // SAFETY: NSObject has no subclassing requirements; we don't implement Drop.
    #[unsafe(super(NSObject))]
    #[thread_kind = AllocAnyThread]
    #[name = "CapviewFrameDelegate"]
    #[ivars = DelegateIvars]
    pub struct FrameDelegate;

    unsafe impl NSObjectProtocol for FrameDelegate {}

    unsafe impl AVCaptureVideoDataOutputSampleBufferDelegate for FrameDelegate {
        #[unsafe(method(captureOutput:didOutputSampleBuffer:fromConnection:))]
        fn did_output(
            &self,
            _output: &AVCaptureOutput,
            sample: &CMSampleBuffer,
            _connection: &AVCaptureConnection,
        ) {
            // SAFETY: CoreMedia getters on a valid sample buffer.
            unsafe {
                let Some(buffer) = sample.image_buffer() else {
                    return;
                };
                // Capture timestamps are on the host clock, so "now - pts" is the
                // device-to-app latency.
                let pts = sample.presentation_time_stamp();
                let (ts_100ns, latency_us) = if pts.timescale > 0 {
                    let now = CMClock::host_time_clock().time();
                    (
                        Some((pts.value as i128 * 10_000_000 / pts.timescale as i128) as i64),
                        latency_us(pts, now),
                    )
                } else {
                    (None, None)
                };
                self.ivars().sink.send(Delivery::Frame {
                    buffer,
                    ts_100ns,
                    latency_us,
                });
            }
        }

        #[unsafe(method(captureOutput:didDropSampleBuffer:fromConnection:))]
        fn did_drop(
            &self,
            _output: &AVCaptureOutput,
            _sample: &CMSampleBuffer,
            _connection: &AVCaptureConnection,
        ) {
            self.ivars().sink.send(Delivery::Dropped);
        }
    }
);

impl FrameDelegate {
    fn new(sink: Sink<Delivery>) -> Retained<Self> {
        let this = Self::alloc().set_ivars(DelegateIvars { sink });
        // SAFETY: NSObject's designated initialiser.
        unsafe { msg_send![super(this), init] }
    }
}

fn latency_us(pts: CMTime, now: CMTime) -> Option<u64> {
    if now.timescale <= 0 {
        return None;
    }
    let pts_s = pts.value as f64 / pts.timescale as f64;
    let now_s = now.value as f64 / now.timescale as f64;
    let us = (now_s - pts_s) * 1e6;
    (0.0..2e6).contains(&us).then_some(us as u64)
}
