// Source -> RGB (gamma-encoded, or linear when requested) at source resolution.
// All source fetches use textureLoad: chroma interpolation is done here with
// correct siting instead of relying on hardware filtering of subsampled planes.

struct Params {
    // luma width/height
    size: vec2<u32>,
    fmt: u32,          // 0 YUY2, 1 UYVY, 2 NV12, 3 P010, 4 BGRA
    flip_y: u32,
    row_r: vec4<f32>,  // rgb = row.xyz . (y, cb, cr) + row.w   (all range/adjust folded in)
    row_g: vec4<f32>,
    row_b: vec4<f32>,
    code_scale: f32,   // integer code -> normalized
    gamma: f32,        // output = pow(rgb, 1/gamma)
    linearize: u32,    // 1 = store linear light (pow 2.2)
    chroma_filter: u32,// 0 nearest, 1 linear, 2 cubic
    cosited: u32,      // 1 = horizontally co-sited chroma
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var tex_a_u: texture_2d<u32>;
@group(0) @binding(2) var tex_b_u: texture_2d<u32>;

@group(1) @binding(0) var tex_rgb: texture_2d<f32>;

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

fn src_y(py: i32) -> i32 {
    let h = i32(P.size.y);
    var y = clamp(py, 0, h - 1);
    if (P.flip_y == 1u) { y = h - 1 - y; }
    return y;
}

// ---- packed 4:2:2 (YUY2 / UYVY) ------------------------------------------------

fn packed_luma(x: i32, y: i32) -> f32 {
    let t = textureLoad(tex_a_u, vec2<i32>(x >> 1, y), 0);
    var v: u32;
    if (P.fmt == 0u) {
        v = select(t.x, t.z, (x & 1) == 1);
    } else {
        v = select(t.y, t.w, (x & 1) == 1);
    }
    return f32(v) * P.code_scale;
}

fn packed_chroma(i: i32, y: i32) -> vec2<f32> {
    let w = i32(P.size.x) / 2;
    let t = textureLoad(tex_a_u, vec2<i32>(clamp(i, 0, w - 1), y), 0);
    var c: vec2<u32>;
    if (P.fmt == 0u) { c = vec2<u32>(t.y, t.w); } else { c = vec2<u32>(t.x, t.z); }
    return vec2<f32>(c) * P.code_scale;
}

// ---- biplanar 4:2:0 (NV12 / P010) ----------------------------------------------

fn planar_luma(x: i32, y: i32) -> f32 {
    return f32(textureLoad(tex_a_u, vec2<i32>(x, y), 0).x) * P.code_scale;
}

fn planar_chroma(i: i32, j: i32) -> vec2<f32> {
    let cw = i32(P.size.x) / 2;
    let ch = i32(P.size.y) / 2;
    var jj = clamp(j, 0, ch - 1);
    if (P.flip_y == 1u) { jj = ch - 1 - jj; }
    let t = textureLoad(tex_b_u, vec2<i32>(clamp(i, 0, cw - 1), jj), 0);
    return vec2<f32>(t.xy) * P.code_scale;
}

// Catmull-Rom weights for fractional offset f in [0,1): taps at -1, 0, 1, 2
fn cr_weights(f: f32) -> vec4<f32> {
    let f2 = f * f;
    let f3 = f2 * f;
    return vec4<f32>(
        -0.5 * f3 + f2 - 0.5 * f,
        1.5 * f3 - 2.5 * f2 + 1.0,
        -1.5 * f3 + 2.0 * f2 + 0.5 * f,
        0.5 * f3 - 0.5 * f2,
    );
}

fn chroma_pos_x(x: i32) -> f32 {
    let fx = f32(x);
    if (P.cosited == 1u) { return fx * 0.5; }
    return (fx - 0.5) * 0.5;
}

fn packed_chroma_at(x: i32, y: i32) -> vec2<f32> {
    let c = chroma_pos_x(x);
    let i0 = i32(floor(c));
    let f = c - floor(c);
    if (P.chroma_filter == 0u) {
        return packed_chroma(select(i0, i0 + 1, f >= 0.5), y);
    }
    if (P.chroma_filter == 1u || f == 0.0) {
        if (f == 0.0) { return packed_chroma(i0, y); }
        return mix(packed_chroma(i0, y), packed_chroma(i0 + 1, y), f);
    }
    let w = cr_weights(f);
    return packed_chroma(i0 - 1, y) * w.x + packed_chroma(i0, y) * w.y
         + packed_chroma(i0 + 1, y) * w.z + packed_chroma(i0 + 2, y) * w.w;
}

fn planar_chroma_row(i0: i32, j: i32, f: f32, wx: vec4<f32>) -> vec2<f32> {
    if (P.chroma_filter == 0u) {
        return planar_chroma(select(i0, i0 + 1, f >= 0.5), j);
    }
    if (P.chroma_filter == 1u) {
        return mix(planar_chroma(i0, j), planar_chroma(i0 + 1, j), f);
    }
    return planar_chroma(i0 - 1, j) * wx.x + planar_chroma(i0, j) * wx.y
         + planar_chroma(i0 + 1, j) * wx.z + planar_chroma(i0 + 2, j) * wx.w;
}

fn planar_chroma_at(x: i32, y: i32) -> vec2<f32> {
    let cx = chroma_pos_x(x);
    // 4:2:0 vertical siting: chroma rows centered between luma rows (MPEG-2).
    let cy = (f32(y) - 0.5) * 0.5;
    let i0 = i32(floor(cx));
    let fx = cx - floor(cx);
    let j0 = i32(floor(cy));
    let fy = cy - floor(cy);
    let wx = cr_weights(fx);
    if (P.chroma_filter == 0u) {
        return planar_chroma_row(i0, select(j0, j0 + 1, fy >= 0.5), fx, wx);
    }
    if (P.chroma_filter == 1u) {
        return mix(planar_chroma_row(i0, j0, fx, wx), planar_chroma_row(i0, j0 + 1, fx, wx), fy);
    }
    let wy = cr_weights(fy);
    return planar_chroma_row(i0, j0 - 1, fx, wx) * wy.x + planar_chroma_row(i0, j0, fx, wx) * wy.y
         + planar_chroma_row(i0, j0 + 1, fx, wx) * wy.z + planar_chroma_row(i0, j0 + 2, fx, wx) * wy.w;
}

fn finish(ycc: vec3<f32>) -> vec4<f32> {
    var rgb = vec3<f32>(
        dot(P.row_r.xyz, ycc) + P.row_r.w,
        dot(P.row_g.xyz, ycc) + P.row_g.w,
        dot(P.row_b.xyz, ycc) + P.row_b.w,
    );
    rgb = clamp(rgb, vec3<f32>(0.0), vec3<f32>(1.0));
    if (P.gamma != 1.0) {
        rgb = pow(rgb, vec3<f32>(1.0 / P.gamma));
    }
    if (P.linearize == 1u) {
        rgb = pow(rgb, vec3<f32>(2.2));
    }
    return vec4<f32>(rgb, 1.0);
}

@fragment
fn fs_packed(in: VsOut) -> @location(0) vec4<f32> {
    let x = i32(in.pos.x);
    let y = src_y(i32(in.pos.y));
    let luma = packed_luma(x, y);
    let c = packed_chroma_at(x, y);
    return finish(vec3<f32>(luma, c.x, c.y));
}

@fragment
fn fs_planar(in: VsOut) -> @location(0) vec4<f32> {
    let x = i32(in.pos.x);
    let yo = i32(in.pos.y);
    let luma = planar_luma(x, src_y(yo));
    let c = planar_chroma_at(x, yo);
    return finish(vec3<f32>(luma, c.x, c.y));
}

@fragment
fn fs_rgb(in: VsOut) -> @location(0) vec4<f32> {
    let x = i32(in.pos.x);
    let y = src_y(i32(in.pos.y));
    let rgb = textureLoad(tex_rgb, vec2<i32>(x, y), 0).rgb;
    // RGB path: rows hold an RGB->RGB adjustment matrix (ycc slot = rgb).
    return finish(rgb);
}
