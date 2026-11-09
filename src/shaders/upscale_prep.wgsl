// AI-upscaler input: the cropped region of the converted frame, gamma-encoded
// (the upscaler expects display-referred SDR RGB), into an 8/10-bit texture.

struct Prep {
    origin: vec2<u32>,
    linear: u32,
    _pad: u32,
};

@group(0) @binding(0) var<uniform> P: Prep;
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

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let p = vec2<i32>(vec2<u32>(in.pos.xy) + P.origin);
    var c = max(textureLoad(src, p, 0).rgb, vec3<f32>(0.0));
    if (P.linear == 1u) {
        c = pow(c, vec3<f32>(1.0 / 2.2));
    }
    return vec4<f32>(clamp(c, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0);
}
