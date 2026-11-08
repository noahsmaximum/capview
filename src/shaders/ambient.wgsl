// Ambient blur chain: the converted frame -> quarter-res level 0 -> mip chain.
// Each level is a dual-Kawase style downsample (a wide tent, not a 2x2 box), so
// sampling a high mip reads as a smooth Gaussian-ish blur rather than blocks.

struct Level {
    out_size: vec2<f32>,
    _pad: vec2<f32>,
};

@group(0) @binding(0) var<uniform> L: Level;
@group(0) @binding(1) var input: texture_2d<f32>;
@group(0) @binding(2) var samp: sampler;

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

fn tap(uv: vec2<f32>) -> vec3<f32> {
    return textureSampleLevel(input, samp, uv, 0.0).rgb;
}

// Full frame -> 1/4 size: 4 bilinear taps = an exact 4x4 box.
@fragment
fn fs_first(in: VsOut) -> @location(0) vec4<f32> {
    let uv = in.pos.xy / L.out_size;
    let t = 1.0 / vec2<f32>(textureDimensions(input));
    let c = tap(uv + vec2<f32>(-t.x, -t.y)) + tap(uv + vec2<f32>(t.x, -t.y))
          + tap(uv + vec2<f32>(-t.x, t.y)) + tap(uv + vec2<f32>(t.x, t.y));
    return vec4<f32>(c * 0.25, 1.0);
}

// Half-size downsample with a 13-tap-equivalent tent (centre + 4 diagonals + 4 axes).
@fragment
fn fs_down(in: VsOut) -> @location(0) vec4<f32> {
    let uv = in.pos.xy / L.out_size;
    let t = 1.0 / vec2<f32>(textureDimensions(input));
    var c = tap(uv) * 4.0;
    c += tap(uv + vec2<f32>(-t.x, -t.y)) + tap(uv + vec2<f32>(t.x, -t.y));
    c += tap(uv + vec2<f32>(-t.x, t.y)) + tap(uv + vec2<f32>(t.x, t.y));
    c += (tap(uv + vec2<f32>(-2.0 * t.x, 0.0)) + tap(uv + vec2<f32>(2.0 * t.x, 0.0))
        + tap(uv + vec2<f32>(0.0, -2.0 * t.y)) + tap(uv + vec2<f32>(0.0, 2.0 * t.y))) * 0.5;
    return vec4<f32>(c / 10.0, 1.0);
}
