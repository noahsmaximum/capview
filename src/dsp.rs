//! Input clean-up for noisy capture cards (e.g. MS2130 boards that route HDMI audio
//! through an on-board analog stage):
//!   DC blocker → spectral hiss reduction (STFT, decision-directed Wiener) → RMS gate.
//! The hiss profile is learned automatically whenever the input sits at the noise floor.

use realfft::num_complex::Complex;
use realfft::{ComplexToReal, RealFftPlanner, RealToComplex};
use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};

/// STFT size: 512 @ 48 kHz ≈ 10.7 ms of added latency when hiss reduction is on.
pub const NR_FFT: usize = 512;
const NR_HOP: usize = NR_FFT / 4;

/// Live-tunable parameters shared with the UI thread.
pub struct DspParams {
    pub dc_block: AtomicBool,
    pub gate: AtomicBool,
    gate_db: AtomicU32,
    pub nr: AtomicBool,
    /// 0..1: how hard hiss is pushed down under sound.
    nr_strength: AtomicU32,
    /// Estimated noise floor (post DC-block RMS, dBFS); NaN until measured.
    floor_db: AtomicU32,
    pub gate_open: AtomicBool,
    /// Hiss profile has been learned at least once.
    pub nr_ready: AtomicBool,
}

impl Default for DspParams {
    fn default() -> Self {
        Self {
            dc_block: AtomicBool::new(true),
            gate: AtomicBool::new(true),
            gate_db: AtomicU32::new((-50.0f32).to_bits()),
            nr: AtomicBool::new(true),
            nr_strength: AtomicU32::new(0.6f32.to_bits()),
            floor_db: AtomicU32::new(f32::NAN.to_bits()),
            gate_open: AtomicBool::new(true),
            nr_ready: AtomicBool::new(false),
        }
    }
}

impl DspParams {
    pub fn set_gate_db(&self, db: f32) {
        self.gate_db.store(db.to_bits(), Ordering::Relaxed);
    }
    pub fn gate_db(&self) -> f32 {
        f32::from_bits(self.gate_db.load(Ordering::Relaxed))
    }
    pub fn set_nr_strength(&self, s: f32) {
        self.nr_strength.store(s.clamp(0.0, 1.0).to_bits(), Ordering::Relaxed);
    }
    fn nr_strength(&self) -> f32 {
        f32::from_bits(self.nr_strength.load(Ordering::Relaxed))
    }
    /// Quietest 100 ms RMS seen over the last ~3 s: a good noise-floor estimate.
    pub fn floor_db(&self) -> Option<f32> {
        let v = f32::from_bits(self.floor_db.load(Ordering::Relaxed));
        v.is_finite().then_some(v)
    }
}

fn db_to_lin(db: f32) -> f32 {
    10f32.powf(db / 20.0)
}

/// Streaming STFT noise reducer (stereo, shared noise profile).
struct SpectralNr {
    fwd: Arc<dyn RealToComplex<f32>>,
    inv: Arc<dyn ComplexToReal<f32>>,
    window: Vec<f32>,
    hist: [Vec<f32>; 2],
    pending: [Vec<f32>; 2],
    acc: [Vec<f32>; 2],
    out: VecDeque<[f32; 2]>,
    noise: Vec<f32>,
    noise_ready: bool,
    prev_clean: [Vec<f32>; 2],
    quiet_hop: bool,
    buf: Vec<f32>,
    spec: [Vec<Complex<f32>>; 2],
    scratch_f: Vec<Complex<f32>>,
    scratch_i: Vec<Complex<f32>>,
    ola_scale: f32,
}

impl SpectralNr {
    fn new() -> Self {
        let mut planner = RealFftPlanner::<f32>::new();
        let fwd = planner.plan_fft_forward(NR_FFT);
        let inv = planner.plan_fft_inverse(NR_FFT);
        let window: Vec<f32> = (0..NR_FFT)
            .map(|i| 0.5 - 0.5 * (2.0 * std::f32::consts::PI * i as f32 / NR_FFT as f32).cos())
            .collect();
        // Hann analysis × Hann synthesis at 75% overlap sums to 1.5.
        let ola = window.iter().map(|w| w * w).sum::<f32>() / NR_HOP as f32;
        let bins = NR_FFT / 2 + 1;
        let mut out = VecDeque::with_capacity(NR_FFT * 2);
        // Prime with the algorithmic latency so output is continuous from the start.
        out.extend(std::iter::repeat_n([0.0f32; 2], NR_FFT - NR_HOP));
        Self {
            scratch_f: fwd.make_scratch_vec(),
            scratch_i: inv.make_scratch_vec(),
            spec: [fwd.make_output_vec(), fwd.make_output_vec()],
            fwd,
            inv,
            window,
            hist: [vec![0.0; NR_FFT], vec![0.0; NR_FFT]],
            pending: [Vec::with_capacity(NR_HOP), Vec::with_capacity(NR_HOP)],
            acc: [vec![0.0; NR_FFT], vec![0.0; NR_FFT]],
            out,
            noise: vec![0.0; bins],
            noise_ready: false,
            prev_clean: [vec![0.0; bins], vec![0.0; bins]],
            quiet_hop: true,
            buf: vec![0.0; NR_FFT],
            ola_scale: 1.0 / (ola * NR_FFT as f32),
        }
    }

    fn push(&mut self, f: [f32; 2], quiet: bool, strength: f32, ready_flag: &AtomicBool) -> [f32; 2] {
        self.quiet_hop &= quiet;
        self.pending[0].push(f[0]);
        self.pending[1].push(f[1]);
        if self.pending[0].len() == NR_HOP {
            self.frame(strength);
            ready_flag.store(self.noise_ready, Ordering::Relaxed);
        }
        self.out.pop_front().unwrap_or([0.0; 2])
    }

    fn frame(&mut self, strength: f32) {
        let bins = NR_FFT / 2 + 1;
        for c in 0..2 {
            self.hist[c].copy_within(NR_HOP.., 0);
            self.hist[c][NR_FFT - NR_HOP..].copy_from_slice(&self.pending[c]);
            self.pending[c].clear();
            for (b, (x, w)) in self.buf.iter_mut().zip(self.hist[c].iter().zip(&self.window)) {
                *b = x * w;
            }
            let _ = self.fwd.process_with_scratch(&mut self.buf, &mut self.spec[c], &mut self.scratch_f);
        }

        // Learn the hiss profile only from hops that were entirely at the noise floor.
        if self.quiet_hop {
            for k in 0..bins {
                let p = 0.5 * (self.spec[0][k].norm_sqr() + self.spec[1][k].norm_sqr());
                self.noise[k] = if self.noise_ready { 0.92 * self.noise[k] + 0.08 * p } else { p };
            }
            self.noise_ready = true;
        }
        self.quiet_hop = true;

        // Strength 0..1 → spectral floor -6..-30 dB and noise over-estimate 1.0..2.0.
        let g_min = db_to_lin(-6.0 - 24.0 * strength);
        let over = 1.0 + strength;
        const DD_A: f32 = 0.98;

        for c in 0..2 {
            if self.noise_ready {
                for k in 0..bins {
                    let n = (self.noise[k] * over).max(1e-20);
                    let p = self.spec[c][k].norm_sqr();
                    let gamma = p / n;
                    let xi = DD_A * self.prev_clean[c][k] / n + (1.0 - DD_A) * (gamma - 1.0).max(0.0);
                    let g = (xi / (1.0 + xi)).max(g_min);
                    self.prev_clean[c][k] = g * g * p;
                    self.spec[c][k] *= g;
                }
            }
            self.spec[c][0].im = 0.0;
            self.spec[c][bins - 1].im = 0.0;
            let _ = self.inv.process_with_scratch(&mut self.spec[c], &mut self.buf, &mut self.scratch_i);
            for i in 0..NR_FFT {
                self.acc[c][i] += self.buf[i] * self.window[i] * self.ola_scale;
            }
        }
        for i in 0..NR_HOP {
            self.out.push_back([self.acc[0][i], self.acc[1][i]]);
        }
        for c in 0..2 {
            self.acc[c].copy_within(NR_HOP.., 0);
            self.acc[c][NR_FFT - NR_HOP..].fill(0.0);
        }
    }
}

pub struct InputDsp {
    // DC blocker
    r: f32,
    x1: [f32; 2],
    y1: [f32; 2],
    // level detector (shared by gate + NR learning)
    ms: f32,
    ms_coef: f32,
    // gate
    gain: f32,
    open_coef: f32,
    close_coef: f32,
    hold: usize,
    hold_len: usize,
    open: bool,
    // hiss reduction
    nr: Option<SpectralNr>,
    // floor probe
    blk_sum: f64,
    blk_n: usize,
    blk_len: usize,
    blocks: [f32; 30],
    blk_i: usize,
    blk_filled: usize,
}

impl InputDsp {
    pub fn new(rate: u32) -> Self {
        let fs = rate.max(8000) as f32;
        let tc = |secs: f32| (-1.0 / (secs * fs)).exp();
        Self {
            r: 1.0 - 2.0 * std::f32::consts::PI * 4.0 / fs, // ~4 Hz high-pass
            x1: [0.0; 2],
            y1: [0.0; 2],
            ms: 0.0,
            ms_coef: tc(0.010),    // 10 ms RMS window
            gain: 1.0,
            open_coef: tc(0.001),  // 1 ms fade-in: don't blunt transients
            close_coef: tc(0.060), // 60 ms fade-out after the hold: no chattering tails
            hold: 0,
            hold_len: (0.150 * fs) as usize,
            open: true,
            nr: None,
            blk_sum: 0.0,
            blk_n: 0,
            blk_len: (0.1 * fs) as usize,
            blocks: [f32::INFINITY; 30],
            blk_i: 0,
            blk_filled: 0,
        }
    }

    /// Process one stereo frame in place.
    #[inline]
    pub fn process(&mut self, p: &DspParams, frame: &mut [f32; 2]) {
        if p.dc_block.load(Ordering::Relaxed) {
            for c in 0..2 {
                let x = frame[c];
                let y = x - self.x1[c] + self.r * self.y1[c];
                self.x1[c] = x;
                self.y1[c] = y;
                frame[c] = y;
            }
        }

        let e = 0.5 * (frame[0] * frame[0] + frame[1] * frame[1]);
        self.ms = self.ms_coef * self.ms + (1.0 - self.ms_coef) * e;

        // Noise-floor probe (pre-processing).
        self.blk_sum += e as f64;
        self.blk_n += 1;
        if self.blk_n >= self.blk_len {
            let rms_db = 10.0 * ((self.blk_sum / self.blk_n as f64).max(1e-20)).log10() as f32;
            self.blocks[self.blk_i] = rms_db;
            self.blk_i = (self.blk_i + 1) % self.blocks.len();
            self.blk_filled = (self.blk_filled + 1).min(self.blocks.len());
            self.blk_sum = 0.0;
            self.blk_n = 0;
            if self.blk_filled >= 10 {
                let floor = self.blocks.iter().copied().fold(f32::INFINITY, f32::min);
                p.floor_db.store(floor.to_bits(), Ordering::Relaxed);
            }
        }

        // Signal detector: 3 dB hysteresis + hold so decaying sounds don't flutter.
        let thr = db_to_lin(p.gate_db());
        let level = self.ms.sqrt();
        if level > thr {
            self.open = true;
            self.hold = self.hold_len;
        } else if level < thr * 0.708 {
            if self.hold > 0 {
                self.hold -= 1;
            } else {
                self.open = false;
            }
        }
        p.gate_open.store(self.open, Ordering::Relaxed);

        if p.nr.load(Ordering::Relaxed) {
            let nr = self.nr.get_or_insert_with(SpectralNr::new);
            *frame = nr.push(*frame, !self.open, p.nr_strength(), &p.nr_ready);
        } else if self.nr.is_some() {
            self.nr = None;
            p.nr_ready.store(false, Ordering::Relaxed);
        }

        if !p.gate.load(Ordering::Relaxed) {
            self.gain = 1.0;
            return;
        }
        // The detector runs ahead of the (delayed) NR output, so the gate opens a
        // touch early — never late — and the hold covers the tail.
        let target = if self.open { 1.0 } else { 0.0 };
        let coef = if target > self.gain { self.open_coef } else { self.close_coef };
        self.gain = target + (self.gain - target) * coef;
        frame[0] *= self.gain;
        frame[1] *= self.gain;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rms_db(x: &[f32]) -> f32 {
        let ms = x.iter().map(|v| (*v as f64).powi(2)).sum::<f64>() / x.len().max(1) as f64;
        10.0 * ms.max(1e-20).log10() as f32
    }

    fn run(dsp: &mut InputDsp, p: &DspParams, input: &[[f32; 2]]) -> Vec<f32> {
        input
            .iter()
            .map(|f| {
                let mut f = *f;
                dsp.process(p, &mut f);
                f[0]
            })
            .collect()
    }

    struct Rng(u32);
    impl Rng {
        fn next(&mut self) -> f32 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 17;
            self.0 ^= self.0 << 5;
            (self.0 as f32 / u32::MAX as f32 - 0.5) * 2.0
        }
    }

    /// Synthetic card: DC offset + -61 dBFS hiss, with a -24 dBFS tone burst in the middle.
    #[test]
    fn gate_removes_hiss_and_keeps_signal() {
        let fs = 48_000usize;
        let mut rng = Rng(12345);
        let hiss = db_to_lin(-61.0) * 3f32.sqrt(); // uniform noise rms = a/sqrt(3)
        let tone = db_to_lin(-24.0) * 2f32.sqrt();
        let input: Vec<[f32; 2]> = (0..fs * 3)
            .map(|i| {
                let mut v = -0.00173 + hiss * rng.next();
                if (fs..2 * fs).contains(&i) {
                    v += tone * (2.0 * std::f32::consts::PI * 440.0 * i as f32 / fs as f32).sin();
                }
                [v, v]
            })
            .collect();
        let p = DspParams::default();
        p.nr.store(false, Ordering::Relaxed);
        let mut dsp = InputDsp::new(fs as u32);
        let out = run(&mut dsp, &p, &input);

        let silence_out = rms_db(&out[fs / 2..fs - fs / 10]);
        let tone_out = rms_db(&out[fs + fs / 10..2 * fs - fs / 10]);
        let tail_out = rms_db(&out[2 * fs + fs / 2..3 * fs]);
        println!("gate: silence {silence_out:.1} dB; tone {tone_out:.1} dB; tail {tail_out:.1} dB");
        assert!(silence_out < -100.0, "hiss + DC should be gated out");
        assert!((tone_out - -24.0).abs() < 1.0, "tone must pass unchanged");
        assert!(tail_out < -100.0, "gate must close again after the burst");
        let floor = p.floor_db().unwrap();
        assert!((floor - -61.0).abs() < 2.0, "floor probe should find the hiss level ({floor})");
    }

    /// Hiss under sustained "music": tones must survive, hiss between them must drop.
    #[test]
    fn spectral_nr_reduces_hiss_under_signal() {
        let fs = 48_000usize;
        let mut rng = Rng(777);
        let hiss = db_to_lin(-58.0) * 3f32.sqrt();
        let tones = [(440.0f32, -26.0f32), (1000.0, -30.0), (3000.0, -34.0)];
        let make = |rng: &mut Rng, with_hiss: bool, i: usize| -> f32 {
            let t = i as f32 / fs as f32;
            let mut v = if with_hiss { hiss * rng.next() } else { 0.0 };
            if i >= fs {
                for (f, db) in tones {
                    v += db_to_lin(db) * 2f32.sqrt() * (2.0 * std::f32::consts::PI * f * t).sin();
                }
            }
            v
        };
        let noisy: Vec<[f32; 2]> = (0..fs * 4)
            .map(|i| {
                let v = make(&mut rng, true, i);
                [v, v]
            })
            .collect();
        let p = DspParams::default();
        p.gate.store(false, Ordering::Relaxed); // isolate the spectral stage
        p.dc_block.store(false, Ordering::Relaxed);
        let mut dsp = InputDsp::new(fs as u32);
        let out = run(&mut dsp, &p, &noisy);
        assert!(p.nr_ready.load(Ordering::Relaxed), "should have learned the hiss in the first second");

        // Analyse 2..4 s (tones playing): spectrum of input vs output.
        let seg = |x: &[f32]| -> Vec<f32> {
            let n = 1 << 16;
            let s = &x[2 * fs..2 * fs + n];
            let mut planner = RealFftPlanner::<f32>::new();
            let fft = planner.plan_fft_forward(n);
            let mut buf: Vec<f32> = s
                .iter()
                .enumerate()
                .map(|(i, v)| v * (0.5 - 0.5 * (2.0 * std::f32::consts::PI * i as f32 / n as f32).cos()))
                .collect();
            let mut spec = fft.make_output_vec();
            fft.process(&mut buf, &mut spec).unwrap();
            spec.iter().map(|c| c.norm_sqr()).collect()
        };
        let lat = NR_FFT - NR_HOP;
        let in_l: Vec<f32> = noisy.iter().map(|f| f[0]).collect();
        let out_aligned: Vec<f32> = out[lat..].to_vec();
        let (si, so) = (seg(&in_l), seg(&out_aligned));
        let hz = |b: usize| b as f32 * fs as f32 / (1 << 16) as f32;
        let band = |s: &[f32], lo: f32, hi: f32| -> f32 {
            let v: Vec<f32> = (0..s.len()).filter(|&b| hz(b) >= lo && hz(b) < hi).map(|b| s[b]).collect();
            10.0 * (v.iter().sum::<f32>() / v.len() as f32).log10()
        };
        let peak = |s: &[f32], f: f32| -> f32 {
            let b = (f / (fs as f32 / (1 << 16) as f32)).round() as usize;
            10.0 * s[b - 3..=b + 3].iter().sum::<f32>().log10()
        };
        let hiss_drop = band(&si, 5000.0, 15000.0) - band(&so, 5000.0, 15000.0);
        println!("spectral NR: hiss band 5-15 kHz reduced by {hiss_drop:.1} dB under sustained tones");
        for (f, _) in tones {
            let d = peak(&so, f) - peak(&si, f);
            println!("  tone {f:>5} Hz level change {d:+.2} dB");
            assert!(d.abs() < 1.0, "tone at {f} Hz must be preserved");
        }
        assert!(hiss_drop > 12.0, "hiss under signal should drop by >12 dB");
    }

    /// Optional: run against a real capture. `CAPVIEW_TEST_PCM=<s16le stereo 48k file>`
    #[test]
    fn real_capture_if_provided() {
        let Ok(path) = std::env::var("CAPVIEW_TEST_PCM") else { return };
        let bytes = std::fs::read(path).unwrap();
        let input: Vec<[f32; 2]> = bytes
            .chunks_exact(4)
            .map(|c| {
                [
                    i16::from_le_bytes([c[0], c[1]]) as f32 / 32768.0,
                    i16::from_le_bytes([c[2], c[3]]) as f32 / 32768.0,
                ]
            })
            .collect();
        let skip = 24_000;
        let measure = |nr: bool, gate: bool| {
            let p = DspParams::default();
            p.nr.store(nr, Ordering::Relaxed);
            p.gate.store(gate, Ordering::Relaxed);
            let mut dsp = InputDsp::new(48_000);
            rms_db(&run(&mut dsp, &p, &input)[skip..])
        };
        println!(
            "real capture: raw {:.1} | DC-blocked {:.1} | +NR {:.1} | +NR+gate {:.1} dB",
            rms_db(&input[skip..].iter().map(|f| f[0]).collect::<Vec<_>>()),
            measure(false, false),
            measure(true, false),
            measure(true, true),
        );
    }
}
