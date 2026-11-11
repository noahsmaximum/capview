// Frame-gen input for the optical-flow engine: gamma-encoded luma of the frame.

struct Luma {
    linear: u32,
    _p0: u32,
    _p1: u32,
    _p2: u32,
};

@group(0) @binding(0) var<uniform> L: Luma;
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
    var c = max(textureLoad(src, vec2<i32>(in.pos.xy), 0).rgb, vec3<f32>(0.0));
    if (L.linear == 1u) {
        c = pow(c, vec3<f32>(1.0 / 2.2));
    }
    let y = dot(c, vec3<f32>(0.299, 0.587, 0.114));
    return vec4<f32>(clamp(y, 0.0, 1.0), 0.0, 0.0, 1.0);
}
