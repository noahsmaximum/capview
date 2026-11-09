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
    bg_mode: u32,           // 0 colour, 1 image, 2 ambient
    bg_fit: u32,            // 0 cover, 1 contain, 2 stretch, 3 center, 4 tile
    tgt_size: vec2<f32>,    // swapchain size, pixels
    bg_size: vec2<f32>,     // background image size, pixels
    bg_dim: f32,
    amb_style: u32,         // 0 glow, 1 streaks, 2 mirror
    amb_strength: f32,
    amb_blur: f32,
    amb_reach: f32,
    amb_flow: f32,
    time: f32,              // seconds, for the streak flow
    amb_max_lod: f32,
    // Ambient reads the blur chain of the whole converted frame even when the video
    // itself comes from the AI upscaler: crop as normalised uv origin.xy / size.zw.
    amb_crop: vec4<f32>,
    amb_k: f32,             // source px per screen px, for blur radii
    amb_linear: u32,        // blur chain holds linear light
    _pad0: u32,
    _pad1: u32,
};

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var src: texture_2d<f32>;
@group(0) @binding(2) var samp: sampler;
@group(1) @binding(0) var bg_tex: texture_2d<f32>;
@group(1) @binding(1) var bg_samp: sampler;
// Blurred mip chain of the current frame (quarter res at level 0).
@group(2) @binding(0) var amb_tex: texture_2d<f32>;
@group(2) @binding(1) var amb_samp: sampler;

// Dest-rect uv (0..1 across the on-screen video) -> normalised source uv,
// undoing rotation (clockwise), flips and crop.
fn src_uv(uv0: vec2<f32>) -> vec2<f32> {
    var uv = uv0;
    switch P.rotation {
        case 1u: { uv = vec2<f32>(uv0.y, 1.0 - uv0.x); }
        case 2u: { uv = vec2<f32>(1.0 - uv0.x, 1.0 - uv0.y); }
        case 3u: { uv = vec2<f32>(1.0 - uv0.y, uv0.x); }
        default: {}
    }
    if (P.flip_h == 1u) { uv.x = 1.0 - uv.x; }
    if (P.flip_v == 1u) { uv.y = 1.0 - uv.y; }
    return P.amb_crop.xy + uv * P.amb_crop.zw;
}

// One level of the blur chain, read with a 4-tap tent to hide texel blocks.
fn amb_level(uv: vec2<f32>, lod: f32) -> vec3<f32> {
    let l = clamp(lod, 0.0, P.amb_max_lod);
    let o = 0.5 / vec2<f32>(textureDimensions(amb_tex, u32(floor(l))));
    return 0.25 * (textureSampleLevel(amb_tex, amb_samp, uv + vec2<f32>(-o.x, -o.y), l).rgb
        + textureSampleLevel(amb_tex, amb_samp, uv + vec2<f32>(o.x, -o.y), l).rgb
        + textureSampleLevel(amb_tex, amb_samp, uv + vec2<f32>(-o.x, o.y), l).rgb
        + textureSampleLevel(amb_tex, amb_samp, uv + vec2<f32>(o.x, o.y), l).rgb);
}

// The frame at `uv`, blurred by roughly `radius` source pixels. Small radii read
// the full-res frame so the fill meets the video edge without a seam.
fn amb_blur(uv: vec2<f32>, radius: f32) -> vec3<f32> {
    let lod = log2(max(radius * 0.25, 1e-3));
    let fine = textureSampleLevel(src, samp, uv, 0.0).rgb;
    return mix(fine, amb_level(uv, max(lod, 0.0)), smoothstep(-2.0, 0.0, lod));
}

fn vnoise(x: vec2<f32>) -> f32 {
    let i = floor(x);
    let f = fract(x);
    let u = f * f * (3.0 - 2.0 * f);
    let h = vec2<u32>(vec2<i32>(i) + vec2<i32>(65536));
    let a = hash(vec3<u32>(h, 7u));
    let b = hash(vec3<u32>(h + vec2<u32>(1u, 0u), 7u));
    let c = hash(vec3<u32>(h + vec2<u32>(0u, 1u), 7u));
    let d = hash(vec3<u32>(h + vec2<u32>(1u, 1u), 7u));
    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

// Ambient fill: the live picture extended into the bars.
fn ambient(p: vec2<f32>) -> vec3<f32> {
    let lo = P.dst_origin;
    let hi = P.dst_origin + P.dst_size;
    let uv0 = (p - lo) / P.dst_size;
    // Distance outside the video rect, per axis and overall (target pixels).
    let ex = max(max(lo.x - p.x, p.x - hi.x), 0.0);
    let ey = max(max(lo.y - p.y, p.y - hi.y), 0.0);
    let d = length(vec2<f32>(ex, ey));
    // Source pixels per target pixel.
    let k = P.amb_k;
    // Sample ~1.5 source pixels in from the edge: many sources put a bright or
    // dark artefact column on the very border, which would streak outward.
    let inset = max(0.5, 1.5 / max(k, 1e-3)) / P.dst_size;
    let edge = clamp(uv0, inset, vec2<f32>(1.0) - inset);
    let b = P.amb_blur;

    var c = vec3<f32>(0.0);
    if (P.amb_style == 0u) {
        // Glow: the frame scaled to cover the window, very soft; near the edge it
        // hands over to the edge pixels so the light seems to spill out.
        let zoom = max(max(P.tgt_size.x / P.dst_size.x, P.tgt_size.y / P.dst_size.y), 1.0);
        let centre = (lo + hi) * 0.5;
        let g = clamp((p - centre) / (P.dst_size * zoom) + 0.5, inset, vec2<f32>(1.0) - inset);
        let far = amb_blur(src_uv(g), mix(30.0, 220.0, b) * k * zoom);
        let near = amb_blur(src_uv(edge), (2.0 + d * mix(0.3, 1.0, b)) * k);
        c = mix(far, near, exp(-d / mix(40.0, 160.0, b)));
    } else if (P.amb_style == 1u) {
        // Streaks: each edge pixel pulled outward forever, softening as it goes.
        c = amb_blur(src_uv(edge), (1.0 + d * mix(0.03, 0.45, b)) * k);
        if (P.amb_flow > 0.0) {
            // Ripples that run outward along each streak, like falling water.
            let along = select(p.x, p.y, ex >= ey);
            let n = vnoise(vec2<f32>(along / 4.0, d / 70.0 - P.time * 1.6));
            let n2 = vnoise(vec2<f32>(along / 23.0 + 9.0, d / 180.0 - P.time * 0.7));
            c *= mix(1.0, 0.45 + 0.8 * n + 0.35 * n2, P.amb_flow);
        }
    } else {
        // Mirror: the picture reflected across each edge (repeating), softening.
        let m = vec2<f32>(1.0) - abs(vec2<f32>(1.0) - (uv0 - 2.0 * floor(uv0 * 0.5)));
        c = amb_blur(src_uv(clamp(m, inset, vec2<f32>(1.0) - inset)), (1.0 + d * mix(0.04, 0.6, b)) * k);
    }
    if (P.amb_linear == 1u) {
        c = pow(max(c, vec3<f32>(0.0)), vec3<f32>(1.0 / 2.2));
    }
    // Full brightness where it touches the video (no seam), easing to the chosen
    // brightness further out.
    c *= mix(1.0, P.amb_strength, 1.0 - exp(-d / 60.0));
    // Reach: fade to the background colour with distance (1 = never).
    var fade = 1.0;
    if (P.amb_reach < 0.999) {
        fade = exp(-d / mix(40.0, 2500.0, P.amb_reach * P.amb_reach));
    }
    return mix(P.background.rgb, c, fade);
}

// Letterbox fill: solid colour, or the background image mapped per fit mode.
// LOD is picked from the on-screen scale so large photos minify cleanly.
fn background(p: vec2<f32>) -> vec3<f32> {
    let col = P.background.rgb;
    if (P.bg_mode == 2u) {
        return ambient(p);
    }
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
        var bg = background(in.pos.xy);
        // Gradients (ambient, photos) get the same dither as the video; flat colour
        // stays exact so black bars stay truly black.
        if (P.bg_mode != 0u && P.dither_levels > 0.0) {
            let q = vec2<u32>(in.pos.xy);
            let n = hash(vec3<u32>(q, P.frame)) + hash(vec3<u32>(q + vec2<u32>(7919u, 104729u), P.frame)) - 1.0;
            bg += vec3<f32>(n / P.dither_levels);
        }
        return vec4<f32>(clamp(bg, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0);
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
