#import bevy_core_pipeline::fullscreen_vertex_shader::FullscreenVertexOutput

@group(0) @binding(0) var color_texture: texture_2d<f32>;
@group(0) @binding(1) var depth_texture: texture_2d<f32>;
@group(0) @binding(2) var texture_sampler: sampler;

struct FragmentOutput {
#ifdef COLOR
    @location(0) color: vec4<f32>,
#endif
    @builtin(frag_depth) depth: f32,
}

@fragment
fn fragment(in: FullscreenVertexOutput) -> FragmentOutput {
    var out: FragmentOutput;
    let color = textureSample(color_texture, texture_sampler, in.uv);
    out.depth = textureSample(depth_texture, texture_sampler, in.uv).r;
    if color.a == 0. {
        discard;
    }
#ifdef COLOR
    out.color = vec4<f32>(color.rgb / color.a, 1.);
#endif
    return out;
}
