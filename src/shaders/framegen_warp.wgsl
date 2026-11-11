// Frame generation: build the frame at time t between `prev` and `next` from the
// optical-flow engine's forward (prev→next) and backward (next→prev) vectors.
//
// For each output pixel a handful of candidate motions are tried (the vectors at
// and around the pixel, a refined vector re-sampled at the midpoint, and "no
// motion"). A candidate v predicts prev(x - v·t) and next(x + v·(1-t)); the
// better the two agree, the more it's trusted. Candidates are blended with a soft
// minimum: where one motion clearly fits the picture is sharp, and where none do
// (occlusions, transparent effects) it eases into a crossfade (a faint ghost)
// instead of copying blocks around.

struct Warp {
    size: vec2<f32>,
    grid: f32,
    t: f32,
    radius: f32,   // artifact-guard search radius, px
    guard: u32,    // 1 = clamp to the colours found near x in prev/next
    softness: f32, // soft-minimum temperature (error units)
    _pad: f32,
};

@group(0) @binding(0) var<uniform> W: Warp;
@group(0) @binding(1) var prev_tex: texture_2d<f32>;
@group(0) @binding(2) var next_tex: texture_2d<f32>;
@group(0) @binding(3) var fwd: texture_2d<i32>;
@group(0) @binding(4) var bwd: texture_2d<i32>;
@group(0) @binding(5) var samp: sampler;

struct VsOut {
    @builtin(position) pos: vec4<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VsOut {
    var o: VsOut;
    let x = f32((vi << 1u) & 2u);
    let y = f32(vi & 2u);
    o.pos = vec4<f32>(x * 2.0 - 1.0, 1.0 - y * 2.0, 0.0, 1.0);
    return o;
}

// Flow vector (pixels) at a pixel position; S10.5 fixed point in the texture.
fn flow(t: texture_2d<i32>, p: vec2<f32>) -> vec2<f32> {
    let hi = vec2<i32>(textureDimensions(t)) - vec2<i32>(1);
    let c = clamp(vec2<i32>(floor(p / W.grid)), vec2<i32>(0), hi);
    return vec2<f32>(textureLoad(t, c, 0).xy) / 32.0;
}

fn sp(tex: texture_2d<f32>, p: vec2<f32>) -> vec3<f32> {
    return textureSampleLevel(tex, samp, p / W.size, 0.0).rgb;
}

struct Cand {
    c: vec3<f32>,
    e: f32,
};

// v: prev→next motion through x.
fn cand(x: vec2<f32>, v: vec2<f32>) -> Cand {
    let pa = x - v * W.t;
    let pb = x + v * (1.0 - W.t);
    let a = sp(prev_tex, pa);
    let b = sp(next_tex, pb);
    // Two extra taps make the match a small patch, not a single pixel.
    let o = vec2<f32>(1.5, 0.75);
    let a1 = sp(prev_tex, pa + o);
    let b1 = sp(next_tex, pb + o);
    let a2 = sp(prev_tex, pa - o.yx);
    let b2 = sp(next_tex, pb - o.yx);
    let d = abs(a - b) + 0.5 * (abs(a1 - b1) + abs(a2 - b2));
    var r: Cand;
    r.c = mix(a, b, W.t);
    r.e = (d.r + d.g + d.b) * 0.5;
    return r;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let x = in.pos.xy;
    let g2 = W.grid * 2.0;

    var cs: array<Cand, 13>;
    var n = 0;
    let f0 = flow(fwd, x);
    let b0 = -flow(bwd, x);
    cs[n] = cand(x, f0); n++;
    cs[n] = cand(x, b0); n++;
    // Re-sample each field where the object actually was / will be.
    cs[n] = cand(x, flow(fwd, x - f0 * W.t)); n++;
    cs[n] = cand(x, -flow(bwd, x + b0 * (1.0 - W.t))); n++;
    let offs = array<vec2<f32>, 4>(
        vec2<f32>(g2, 0.0), vec2<f32>(-g2, 0.0), vec2<f32>(0.0, g2), vec2<f32>(0.0, -g2)
    );
    for (var i = 0; i < 4; i++) {
        cs[n] = cand(x, flow(fwd, x + offs[i])); n++;
        cs[n] = cand(x, -flow(bwd, x + offs[i])); n++;
    }
    cs[n] = cand(x, vec2<f32>(0.0)); n++;

    var emin = 1e9;
    for (var i = 0; i < 13; i++) {
        emin = min(emin, cs[i].e);
    }
    var acc = vec3<f32>(0.0);
    var wsum = 0.0;
    for (var i = 0; i < 13; i++) {
        let w = exp(-(cs[i].e - emin) / W.softness);
        acc += cs[i].c * w;
        wsum += w;
    }
    var c = acc / wsum;

    // Nothing fits at all (occlusion / effect): fade to a plain crossfade.
    let fade = smoothstep(0.15, 0.45, emin);
    c = mix(c, mix(sp(prev_tex, x), sp(next_tex, x), W.t), fade);

    if (W.guard == 1u) {
        var mn = vec3<f32>(1e9);
        var mx = vec3<f32>(-1e9);
        let step = max(W.radius * 0.5, 2.0);
        for (var yy = -2; yy <= 2; yy++) {
            for (var xx = -2; xx <= 2; xx++) {
                let q = x + vec2<f32>(f32(xx), f32(yy)) * step;
                let a = sp(prev_tex, q);
                let b = sp(next_tex, q);
                mn = min(mn, min(a, b));
                mx = max(mx, max(a, b));
            }
        }
        let a = sp(prev_tex, x);
        let b = sp(next_tex, x);
        mn = min(mn, min(a, b));
        mx = max(mx, max(a, b));
        c = clamp(c, mn - vec3<f32>(0.012), mx + vec3<f32>(0.012));
    }
    return vec4<f32>(c, 1.0);
}
