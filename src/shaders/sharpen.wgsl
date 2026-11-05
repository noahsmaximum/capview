// Contrast-adaptive sharpening (after AMD FidelityFX CAS), at source resolution.

struct Params {
    sharpness: f32,
    linear: u32,
    _p0: u32,
    _p1: u32,
};

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var src: texture_2d<f32>;

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

fn ld(p: vec2<i32>) -> vec3<f32> {
    let d = vec2<i32>(textureDimensions(src));
    return textureLoad(src, clamp(p, vec2<i32>(0), d - 1), 0).rgb;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let p = vec2<i32>(in.pos.xy);
    let b = ld(p + vec2<i32>(0, -1));
    let d = ld(p + vec2<i32>(-1, 0));
    let e = ld(p);
    let f = ld(p + vec2<i32>(1, 0));
    let h = ld(p + vec2<i32>(0, 1));
    let mn = min(min(min(d, e), min(f, b)), h);
    let mx = max(max(max(d, e), max(f, b)), h);
    let amp = sqrt(clamp(min(mn, 1.0 - mx) / max(mx, vec3<f32>(1e-5)), vec3<f32>(0.0), vec3<f32>(1.0)));
    let peak = -1.0 / mix(8.0, 5.0, clamp(P.sharpness, 0.0, 1.0));
    let w = amp * peak;
    let rcp = 1.0 / (1.0 + 4.0 * w);
    let o = (b * w + d * w + f * w + h * w + e) * rcp;
    return vec4<f32>(clamp(o, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0);
}
