#import bevy_core_pipeline::fullscreen_vertex_shader::FullscreenVertexOutput

@group(0) @binding(0) var color_texture: texture_2d<f32>;
@group(0) @binding(1) var depth_texture: texture_2d<f32>;
@group(0) @binding(2) var texture_sampler: sampler;

struct FragmentOutput {
    @location(0) color: vec4<f32>,
    @builtin(frag_depth) depth: f32,
}

@fragment
fn fragment(in: FullscreenVertexOutput) -> FragmentOutput {
    var out: FragmentOutput;
    out.color = textureSample(color_texture, texture_sampler, in.uv);
    // TODO: Use correct depth instead of reverse infinite Z?
    out.depth = textureSample(depth_texture, texture_sampler, in.uv).r;
    // out.depth = 500.;
    return out;
}
