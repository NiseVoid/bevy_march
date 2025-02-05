#import bevy_march::{get_individual_ray, march_ray, settings, calc_normal, get_occlusion, MarchSettings, MarchResult, depth_texture};

@group(1) @binding(2) var color_texture: texture_storage_2d<rgba16float, write>;
@group(2) @binding(2) var<storage, read> materials: array<Material>;

struct Material {
    base_color: vec3<f32>,
    emissive: vec3<f32>,
}

@compute @workgroup_size(8, 8, 1)
fn march(@builtin(global_invocation_id) invocation_id: vec3<u32>) {
    let march = get_individual_ray(invocation_id.xy);
    let res = march_ray(march);
    let color = get_color(march, res);

    textureStore(depth_texture, invocation_id.xy, vec4<f32>(settings.near / res.traveled, 0., 0., 0.));
    textureStore(color_texture, invocation_id.xy, vec4<f32>(color, 1.));
}

fn get_color(march: MarchSettings, res: MarchResult) -> vec3<f32> {
    if res.traveled >= 100. {
        return vec3<f32>(0.);
    }

    let hit = march.origin + march.direction * (res.traveled - 0.01);
    let normal = calc_normal(res.id, hit);
    var diffuse = dot(normal, -settings.light_dir);

    var material = materials[res.material];
    var albedo = material.base_color;
    var emission = material.emissive;
    let light = max(diffuse, 0.15);
    let color = max(emission, vec3<f32>(albedo * light));
    return color;
}
