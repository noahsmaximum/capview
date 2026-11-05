// RGB (source res) -> swapchain. Maps every output pixel back into the source
// (letterbox, crop, flip, rotate, zoom) and resamples with the selected filter.

struct Params {
    dst_origin: vec2<f32>,  // dest rect top-left, target pixels
    dst_size: vec2<f32>,    // dest rect size, target pixels
    crop_origin: vec2<f32>, // source pixels
    crop_size: vec2<f32>,   // source pixels
    background: vec4<f32>,
    // axis scale (output px per source px) along source x / y, for kernel widening
    scale: vec2<f32>,
    filt: u32,              // 0 nearest,1 bilinear,2 sharp-bilinear,3 catmull,4 mitchell,5 lanczos2,6 lanczos3
    rotation: u32,          // 0,1,2,3 = 0/90/180/270 clockwise
    flip_h: u32,
    flip_v: u32,
    linear: u32,
    dither_levels: f32,     // 0 = off, else 255 / 1023
    antiring: f32,
    frame: u32,
    bg_mode: u32,           // 0 colour, 1 image
    bg_fit: u32,            // 0 cover, 1 contain, 2 stretch, 3 center, 4 tile
    tgt_size: vec2<f32>,    // swapchain size, pixels
    bg_size: vec2<f32>,     // background image size, pixels
    bg_dim: f32,
    _q0: u32,
    _q1: u32,
    _q2: u32,
};

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var src: texture_2d<f32>;
@group(0) @binding(2) var samp: sampler;
@group(1) @binding(0) var bg_tex: texture_2d<f32>;
@group(1) @binding(1) var bg_samp: sampler;

// Letterbox fill: solid colour, or the background image mapped per fit mode.
// LOD is picked from the on-screen scale so large photos minify cleanly.
fn background(p: vec2<f32>) -> vec3<f32> {
    let col = P.background.rgb;
    if (P.bg_mode != 1u) {
        return col;
    }
    let t = P.tgt_size;
    let isz = max(P.bg_size, vec2<f32>(1.0));
    var uv = vec2<f32>(0.0);
    var px_per_texel = 1.0;
    if (P.bg_fit == 0u) {
        let k = max(t.x / isz.x, t.y / isz.y);
        uv = ((p - t * 0.5) / k + isz * 0.5) / isz;
        px_per_texel = k;
    } else if (P.bg_fit == 1u) {
        let k = min(t.x / isz.x, t.y / isz.y);
        uv = ((p - t * 0.5) / k + isz * 0.5) / isz;
        px_per_texel = k;
    } else if (P.bg_fit == 2u) {
        uv = p / t;
        px_per_texel = min(t.x / isz.x, t.y / isz.y);
    } else if (P.bg_fit == 3u) {
        uv = ((p - t * 0.5) + isz * 0.5) / isz;
    } else {
        uv = fract(p / isz);
    }
    var c = col;
    if (all(uv >= vec2<f32>(0.0)) && all(uv <= vec2<f32>(1.0))) {
        let lod = max(0.0, log2(1.0 / max(px_per_texel, 1e-4)));
        c = textureSampleLevel(bg_tex, bg_samp, uv, lod).rgb;
    }
    return c * (1.0 - P.bg_dim);
}

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

const PI: f32 = 3.14159265358979;

fn sinc(x: f32) -> f32 {
    if (abs(x) < 1e-5) { return 1.0; }
    let px = PI * x;
    return sin(px) / px;
}

fn bc_spline(x0: f32, b: f32, c: f32) -> f32 {
    let x = abs(x0);
    if (x < 1.0) {
        return ((12.0 - 9.0 * b - 6.0 * c) * x * x * x + (-18.0 + 12.0 * b + 6.0 * c) * x * x + (6.0 - 2.0 * b)) / 6.0;
    } else if (x < 2.0) {
        return ((-b - 6.0 * c) * x * x * x + (6.0 * b + 30.0 * c) * x * x + (-12.0 * b - 48.0 * c) * x + (8.0 * b + 24.0 * c)) / 6.0;
    }
    return 0.0;
}

fn kernel(x: f32) -> f32 {
    var w = 0.0;
    if (P.filt == 3u) {
        w = bc_spline(x, 0.0, 0.5);
    } else if (P.filt == 4u) {
        w = bc_spline(x, 1.0 / 3.0, 1.0 / 3.0);
    } else if (P.filt == 5u) {
        w = select(sinc(x) * sinc(x / 2.0), 0.0, abs(x) >= 2.0);
    } else {
        w = select(sinc(x) * sinc(x / 3.0), 0.0, abs(x) >= 3.0);
    }
    return w;
}

fn kernel_radius() -> f32 {
    return select(2.0, 3.0, P.filt == 6u);
}

fn load(p: vec2<i32>) -> vec3<f32> {
    let d = vec2<i32>(textureDimensions(src));
    return textureLoad(src, clamp(p, vec2<i32>(0), d - 1), 0).rgb;
}

const MAX_TAPS: i32 = 12;

fn convolve(s: vec2<f32>) -> vec3<f32> {
    let c = s - 0.5;
    let base = floor(c);
    let f = c - base;
    // Widen the kernel when downscaling (antialiasing), capped.
    let ks = min(P.scale, vec2<f32>(1.0));
    let r = kernel_radius();
    let rx = min(i32(ceil(r / ks.x)), MAX_TAPS / 2);
    let ry = min(i32(ceil(r / ks.y)), MAX_TAPS / 2);

    var wx: array<f32, 12>;
    var sx = 0.0;
    for (var i = 0; i < 2 * rx; i++) {
        let k = f32(i - rx + 1);
        let w = kernel((k - f.x) * ks.x);
        wx[i] = w;
        sx += w;
    }
    var acc = vec3<f32>(0.0);
    var sy = 0.0;
    let ib = vec2<i32>(base);
    for (var j = 0; j < 2 * ry; j++) {
        let k = f32(j - ry + 1);
        let wy = kernel((k - f.y) * ks.y);
        var row = vec3<f32>(0.0);
        for (var i = 0; i < 2 * rx; i++) {
            row += load(ib + vec2<i32>(i - rx + 1, j - ry + 1)) * wx[i];
        }
        acc += row * wy;
        sy += wy;
    }
    var o = acc / max(sx * sy, 1e-6);

    if (P.antiring > 0.0) {
        let a = load(ib);
        let b = load(ib + vec2<i32>(1, 0));
        let cc = load(ib + vec2<i32>(0, 1));
        let d = load(ib + vec2<i32>(1, 1));
        let lo = min(min(a, b), min(cc, d));
        let hi = max(max(a, b), max(cc, d));
        o = mix(o, clamp(o, lo, hi), P.antiring);
    }
    return o;
}

fn sharp_bilinear(s: vec2<f32>) -> vec3<f32> {
    let dims = vec2<f32>(textureDimensions(src));
    let pre = max(floor(P.scale), vec2<f32>(1.0));
    let region = 0.5 - 0.5 / pre;
    let dist = fract(s) - 0.5;
    let f = (dist - clamp(dist, -region, region)) * pre + 0.5;
    let t = floor(s) + f;
    return textureSampleLevel(src, samp, t / dims, 0.0).rgb;
}

fn hash(p: vec3<u32>) -> f32 {
    var v = p.x * 1664525u + p.y * 1013904223u + p.z * 2654435769u;
    v ^= v >> 16u;
    v *= 2246822519u;
    v ^= v >> 13u;
    v *= 3266489917u;
    v ^= v >> 16u;
    return f32(v) / 4294967295.0;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let uv0 = (in.pos.xy - P.dst_origin) / P.dst_size;
    if (any(uv0 < vec2<f32>(0.0)) || any(uv0 >= vec2<f32>(1.0))) {
        return vec4<f32>(background(in.pos.xy), 1.0);
    }
    // dest uv -> source uv: undo rotation (clockwise), then flips.
    var uv = uv0;
    switch P.rotation {
        case 1u: { uv = vec2<f32>(uv0.y, 1.0 - uv0.x); }
        case 2u: { uv = vec2<f32>(1.0 - uv0.x, 1.0 - uv0.y); }
        case 3u: { uv = vec2<f32>(1.0 - uv0.y, uv0.x); }
        default: {}
    }
    if (P.flip_h == 1u) { uv.x = 1.0 - uv.x; }
    if (P.flip_v == 1u) { uv.y = 1.0 - uv.y; }
    let s = P.crop_origin + uv * P.crop_size;

    var rgb: vec3<f32>;
    switch P.filt {
        case 0u: { rgb = load(vec2<i32>(floor(s))); }
        case 1u: {
            let dims = vec2<f32>(textureDimensions(src));
            rgb = textureSampleLevel(src, samp, s / dims, 0.0).rgb;
        }
        case 2u: { rgb = sharp_bilinear(s); }
        default: { rgb = convolve(s); }
    }
    rgb = max(rgb, vec3<f32>(0.0));
    if (P.linear == 1u) {
        rgb = pow(rgb, vec3<f32>(1.0 / 2.2));
    }
    if (P.dither_levels > 0.0) {
        let p = vec2<u32>(in.pos.xy);
        let n = hash(vec3<u32>(p, P.frame)) + hash(vec3<u32>(p + vec2<u32>(7919u, 104729u), P.frame)) - 1.0;
        rgb += vec3<f32>(n / P.dither_levels);
    }
    return vec4<f32>(clamp(rgb, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0);
}
