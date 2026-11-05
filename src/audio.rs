//! Low-latency audio passthrough: capture-card input -> speakers.
//!
//! Input and output run on separate WASAPI clocks, so the output side resamples
//! (linear) with a small proportional correction that holds the ring-buffer fill at
//! the target delay. This also handles 48k -> 44.1k etc. Underruns re-prime to target;
//! large overruns hard-resync instead of slowly drifting back.

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{FromSample, Sample, SizedSample};
use ringbuf::HeapRb;
use ringbuf::traits::{Consumer, Observer, Producer, Split};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};

pub struct AudioShared {
    volume: AtomicU32,
    muted: AtomicBool,
    target_ms: AtomicU32,
    pub fill_ms: AtomicU32,
    pub underruns: AtomicU64,
    pub peak_l: AtomicU32,
    pub peak_r: AtomicU32,
    pub in_rate: AtomicU32,
    pub out_rate: AtomicU32,
    pub error: parking_lot::Mutex<Option<String>>,
}

impl AudioShared {
    fn new() -> Self {
        Self {
            volume: AtomicU32::new(1.0f32.to_bits()),
            muted: AtomicBool::new(false),
            target_ms: AtomicU32::new(30.0f32.to_bits()),
            fill_ms: AtomicU32::new(0),
            underruns: AtomicU64::new(0),
            peak_l: AtomicU32::new(0),
            peak_r: AtomicU32::new(0),
            in_rate: AtomicU32::new(0),
            out_rate: AtomicU32::new(0),
            error: parking_lot::Mutex::new(None),
        }
    }
    pub fn set_volume(&self, v: f32) {
        self.volume.store(v.to_bits(), Ordering::Relaxed);
    }
    pub fn set_muted(&self, m: bool) {
        self.muted.store(m, Ordering::Relaxed);
    }
    pub fn set_target_ms(&self, ms: f32) {
        self.target_ms.store(ms.to_bits(), Ordering::Relaxed);
    }
    pub fn fill(&self) -> f32 {
        f32::from_bits(self.fill_ms.load(Ordering::Relaxed))
    }
    pub fn peaks(&self) -> (f32, f32) {
        (
            f32::from_bits(self.peak_l.load(Ordering::Relaxed)),
            f32::from_bits(self.peak_r.load(Ordering::Relaxed)),
        )
    }
}

pub struct AudioPassthrough {
    _input: cpal::Stream,
    _output: cpal::Stream,
    pub shared: Arc<AudioShared>,
    pub input_name: String,
    pub output_name: String,
}

fn dev_name(d: &cpal::Device) -> String {
    d.description()
        .map(|d| d.name().to_string())
        .unwrap_or_else(|_| "?".into())
}

pub fn list_inputs() -> Vec<String> {
    let host = cpal::default_host();
    host.input_devices()
        .map(|it| it.map(|d| dev_name(&d)).collect())
        .unwrap_or_default()
}

pub fn list_outputs() -> Vec<String> {
    let host = cpal::default_host();
    host.output_devices()
        .map(|it| it.map(|d| dev_name(&d)).collect())
        .unwrap_or_default()
}

/// Guess the capture card's audio interface from the video device name
/// (e.g. "USB3 Video" -> "Digital Audio Interface (USB3 Digital Audio)").
pub fn guess_input_for(video_name: &str) -> Option<String> {
    let inputs = list_inputs();
    // MacroSilicon-style pairs: "<X> Video" <-> "<X> Digital Audio" / "<X> Audio".
    let lower = video_name.to_lowercase();
    if let Some(prefix) = lower.strip_suffix("video").map(str::trim) {
        for suffix in ["digital audio", "audio"] {
            let want = format!("({prefix} {suffix})");
            let want_bare = format!("{prefix} {suffix}");
            if let Some(n) = inputs.iter().find(|n| {
                let l = n.to_lowercase();
                l.contains(&want) || l == want_bare
            }) {
                return Some(n.clone());
            }
        }
    }
    const GENERIC: [&str; 9] = [
        "video",
        "audio",
        "digital",
        "interface",
        "capture",
        "device",
        "usb",
        "hd",
        "the",
    ];
    let tokens: Vec<String> = video_name
        .split(|c: char| !c.is_alphanumeric())
        .map(|s| s.to_lowercase())
        .filter(|s| s.len() > 1 && !GENERIC.contains(&s.as_str()))
        .collect();
    if tokens.is_empty() {
        return None;
    }
    inputs
        .into_iter()
        .map(|n| {
            let l = n.to_lowercase();
            (tokens.iter().filter(|t| l.contains(t.as_str())).count(), n)
        })
        .filter(|(s, _)| *s > 0)
        .max_by_key(|(s, _)| *s)
        .map(|(_, n)| n)
}

pub fn start(
    input: &str,
    output: Option<&str>,
    fix_ms2109: bool,
) -> Result<AudioPassthrough, String> {
    let host = cpal::default_host();
    let in_dev = host
        .input_devices()
        .map_err(|e| e.to_string())?
        .find(|d| dev_name(d) == input)
        .ok_or_else(|| format!("input '{input}' not found"))?;
    let out_dev = match output {
        Some(name) => host
            .output_devices()
            .map_err(|e| e.to_string())?
            .find(|d| dev_name(d) == name)
            .ok_or_else(|| format!("output '{name}' not found"))?,
        None => host
            .default_output_device()
            .ok_or("no default output device")?,
    };

    let in_cfg = in_dev.default_input_config().map_err(|e| e.to_string())?;
    let out_cfg = out_dev.default_output_config().map_err(|e| e.to_string())?;
    let mut in_rate = in_cfg.sample_rate();
    let out_rate = out_cfg.sample_rate();
    let mut in_ch = in_cfg.channels() as usize;
    let out_ch = out_cfg.channels() as usize;
    // MacroSilicon MS2109 quirk: descriptors claim 96 kHz mono, the stream is really
    // 48 kHz stereo interleaved. Re-read the "mono" samples as L/R pairs at 48 kHz.
    let ms2109 = fix_ms2109
        && in_rate == 96_000
        && in_ch == 1
        && input.to_lowercase().contains("digital audio");
    if ms2109 {
        in_rate = 48_000;
        in_ch = 2;
        log::info!("audio: applying MS2109 fix (96 kHz mono -> 48 kHz stereo)");
    }

    let shared = Arc::new(AudioShared::new());
    shared.in_rate.store(in_rate, Ordering::Relaxed);
    shared.out_rate.store(out_rate, Ordering::Relaxed);

    // 1 second of stereo frames.
    let rb = HeapRb::<f32>::new(in_rate as usize * 2);
    let (prod, cons) = rb.split();

    let input_stream = match in_cfg.sample_format() {
        cpal::SampleFormat::F32 => {
            build_input::<f32>(&in_dev, in_cfg.config(), in_ch, prod, shared.clone())
        }
        cpal::SampleFormat::I16 => {
            build_input::<i16>(&in_dev, in_cfg.config(), in_ch, prod, shared.clone())
        }
        cpal::SampleFormat::I32 => {
            build_input::<i32>(&in_dev, in_cfg.config(), in_ch, prod, shared.clone())
        }
        cpal::SampleFormat::U16 => {
            build_input::<u16>(&in_dev, in_cfg.config(), in_ch, prod, shared.clone())
        }
        f => Err(format!("unsupported input format {f:?}")),
    }?;
    let state = OutState::new(cons, in_rate, out_rate, shared.clone());
    let output_stream = match out_cfg.sample_format() {
        cpal::SampleFormat::F32 => build_output::<f32>(&out_dev, out_cfg.config(), out_ch, state),
        cpal::SampleFormat::I16 => build_output::<i16>(&out_dev, out_cfg.config(), out_ch, state),
        cpal::SampleFormat::I32 => build_output::<i32>(&out_dev, out_cfg.config(), out_ch, state),
        cpal::SampleFormat::U16 => build_output::<u16>(&out_dev, out_cfg.config(), out_ch, state),
        f => Err(format!("unsupported output format {f:?}")),
    }?;
    input_stream.play().map_err(|e| e.to_string())?;
    output_stream.play().map_err(|e| e.to_string())?;
    log::info!(
        "audio: {input} ({in_rate} Hz, {in_ch}ch) -> {} ({out_rate} Hz, {out_ch}ch)",
        dev_name(&out_dev)
    );
    Ok(AudioPassthrough {
        _input: input_stream,
        _output: output_stream,
        shared,
        input_name: input.to_string(),
        output_name: dev_name(&out_dev),
    })
}

type Prod = <HeapRb<f32> as Split>::Prod;
type Cons = <HeapRb<f32> as Split>::Cons;

fn build_input<T>(
    dev: &cpal::Device,
    cfg: cpal::StreamConfig,
    ch: usize,
    mut prod: Prod,
    shared: Arc<AudioShared>,
) -> Result<cpal::Stream, String>
where
    T: SizedSample + Sample,
    f32: FromSample<T>,
{
    let sh = shared.clone();
    let mut scratch: Vec<f32> = Vec::with_capacity(4096);
    dev.build_input_stream::<T, _, _>(
        cfg,
        move |data: &[T], _| {
            scratch.clear();
            for frame in data.chunks(ch.max(1)) {
                let l = f32::from_sample(frame[0]);
                let r = if ch > 1 {
                    f32::from_sample(frame[1])
                } else {
                    l
                };
                scratch.push(l);
                scratch.push(r);
            }
            prod.push_slice(&scratch);
        },
        move |e| {
            let msg = e.to_string();
            if msg.contains("underrun") || msg.contains("overrun") {
                // Transient glitch (WASAPI discontinuity) — count it, don't surface as an error.
                sh.underruns.fetch_add(1, Ordering::Relaxed);
            } else {
                *sh.error.lock() = Some(format!("input: {msg}"));
            }
        },
        None,
    )
    .map_err(|e| e.to_string())
}

struct OutState {
    cons: Cons,
    in_rate: f64,
    out_rate: f64,
    pos: f64,
    prev: [f32; 2],
    cur: [f32; 2],
    priming: bool,
    shared: Arc<AudioShared>,
    peak: [f32; 2],
    peak_frames: usize,
}

impl OutState {
    fn new(cons: Cons, in_rate: u32, out_rate: u32, shared: Arc<AudioShared>) -> Self {
        Self {
            cons,
            in_rate: in_rate as f64,
            out_rate: out_rate as f64,
            pos: 0.0,
            prev: [0.0; 2],
            cur: [0.0; 2],
            priming: true,
            shared,
            peak: [0.0; 2],
            peak_frames: 0,
        }
    }

    fn pop_frame(&mut self) -> Option<[f32; 2]> {
        if self.cons.occupied_len() < 2 {
            return None;
        }
        let l = self.cons.try_pop()?;
        let r = self.cons.try_pop().unwrap_or(l);
        Some([l, r])
    }

    /// Produce `frames` stereo frames into `out`.
    fn render(&mut self, out: &mut [[f32; 2]]) {
        let target_ms = f32::from_bits(self.shared.target_ms.load(Ordering::Relaxed)) as f64;
        let fill_frames = (self.cons.occupied_len() / 2) as f64;
        let fill_ms = fill_frames * 1000.0 / self.in_rate;
        self.shared
            .fill_ms
            .store((fill_ms as f32).to_bits(), Ordering::Relaxed);

        let target_frames = target_ms * self.in_rate / 1000.0;
        if self.priming {
            if fill_frames < target_frames {
                out.fill([0.0; 2]);
                return;
            }
            self.priming = false;
        }
        // Hard resync on large overrun (e.g. after a stall).
        let max_ms = target_ms + 80.0f64.max(target_ms);
        if fill_ms > max_ms {
            let drop = ((fill_frames - target_frames) as usize) * 2;
            self.cons.skip(drop);
        }

        let err = (fill_ms - target_ms) / target_ms.max(5.0);
        let adj = (err * 0.01).clamp(-0.008, 0.008);
        let ratio = self.in_rate / self.out_rate * (1.0 + adj);

        let vol = if self.shared.muted.load(Ordering::Relaxed) {
            0.0
        } else {
            f32::from_bits(self.shared.volume.load(Ordering::Relaxed))
        };

        for o in out.iter_mut() {
            self.pos += ratio;
            while self.pos >= 1.0 {
                self.pos -= 1.0;
                self.prev = self.cur;
                match self.pop_frame() {
                    Some(f) => self.cur = f,
                    None => {
                        self.shared.underruns.fetch_add(1, Ordering::Relaxed);
                        self.priming = true;
                        self.cur = [0.0; 2];
                    }
                }
            }
            let t = self.pos as f32;
            let l = (self.prev[0] + (self.cur[0] - self.prev[0]) * t) * vol;
            let r = (self.prev[1] + (self.cur[1] - self.prev[1]) * t) * vol;
            self.peak[0] = self.peak[0].max(l.abs());
            self.peak[1] = self.peak[1].max(r.abs());
            *o = [l.clamp(-1.0, 1.0), r.clamp(-1.0, 1.0)];
        }
        self.peak_frames += out.len();
        if self.peak_frames as f64 > self.out_rate * 0.05 {
            self.shared
                .peak_l
                .store(self.peak[0].to_bits(), Ordering::Relaxed);
            self.shared
                .peak_r
                .store(self.peak[1].to_bits(), Ordering::Relaxed);
            self.peak = [0.0; 2];
            self.peak_frames = 0;
        }
    }
}

fn build_output<T>(
    dev: &cpal::Device,
    cfg: cpal::StreamConfig,
    ch: usize,
    mut state: OutState,
) -> Result<cpal::Stream, String>
where
    T: SizedSample + Sample + FromSample<f32>,
{
    let sh = state.shared.clone();
    let mut scratch: Vec<[f32; 2]> = Vec::with_capacity(4096);
    dev.build_output_stream::<T, _, _>(
        cfg,
        move |data: &mut [T], _| {
            let frames = data.len() / ch.max(1);
            scratch.resize(frames, [0.0; 2]);
            state.render(&mut scratch);
            for (frame, s) in data.chunks_mut(ch.max(1)).zip(scratch.iter()) {
                if ch == 1 {
                    frame[0] = T::from_sample((s[0] + s[1]) * 0.5);
                } else {
                    for (i, x) in frame.iter_mut().enumerate() {
                        *x = match i {
                            0 => T::from_sample(s[0]),
                            1 => T::from_sample(s[1]),
                            _ => T::EQUILIBRIUM,
                        };
                    }
                }
            }
        },
        move |e| {
            let msg = e.to_string();
            if msg.contains("underrun") || msg.contains("overrun") {
                sh.underruns.fetch_add(1, Ordering::Relaxed);
            } else {
                *sh.error.lock() = Some(format!("output: {msg}"));
            }
        },
        None,
    )
    .map_err(|e| e.to_string())
}
