#import bevy_core_pipeline::fullscreen_vertex_shader::FullscreenVertexOutput
#import "ray_marcher.wgsl"::{march_ray, MarchSettings, get_initial_settings, settings, calc_normal, get_occlusion};

@fragment
fn fragment(in: FullscreenVertexOutput) -> @location(0) vec4<f32> {
    let march = get_initial_settings(in.uv);

    let res = march_ray(march);

    if res.distance < 0.05 {
        let traveled = res.traveled + res.distance;
        let depth = (settings.far - traveled) / (settings.far - settings.near);
        return vec4<f32>(vec3<f32>(1. / depth), 1.);
    }

    return vec4<f32>(1.);
}
