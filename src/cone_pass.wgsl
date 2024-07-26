#import bevy_march::{get_initial_settings, settings, NearestSdf, get_nearest, to_instance, get_ray_dir, PLACEHOLDER_ID};

@group(1) @binding(0) var cone_texture: texture_storage_2d<r32float, write>;
@group(1) @binding(1) var<storage, read> uv_scale: vec2<f32>;

struct Material {
    base_color: vec3<f32>,
    emissive: vec3<f32>,
    reflective: f32,
}

@compute @workgroup_size(8, 8, 1)
fn march(@builtin(global_invocation_id) invocation_id: vec3<u32>) {
    var size = textureDimensions(cone_texture);
    let pixel_factor = uv_scale / vec2<f32>(size);
    let position = invocation_id.xy;

    let cluster_start = vec2<f32>(position);
    let cluster_end = vec2<f32>(position + 1);
    let cluster_center = (cluster_start + cluster_end) * 0.5;
    var cone_march = get_initial_settings(cluster_center * pixel_factor, 0.);

    let epsilon_per_dist = cone_march.scale * 0.001;
    let min_epsilon = cone_march.scale * 0.002;
    let max_epsilon = cone_march.scale * 0.02;

    let tl = get_ray_dir(cluster_start * pixel_factor);
    let tr = get_ray_dir(vec2<f32>(cluster_end.x, cluster_start.y) * pixel_factor);
    let bl = get_ray_dir(vec2<f32>(cluster_start.x, cluster_end.y) * pixel_factor);
    let br = get_ray_dir(cluster_end * pixel_factor);

    var res: NearestSdf;
    res.dist = cone_march.limit;
    var cluster_size: f32;
    var traveled = cone_march.start;
    var step = 0.;
    var i = 0u;
    for (i = 0u; i < 256u; i++) {
        var pos = cone_march.origin + cone_march.direction * traveled;

        res = get_nearest(res.dist + step, pos, cone_march.ignored);

        // TODO: this can probably be done more efficiently using the angle between corners
        let hit = traveled + res.dist;
        let center = cone_march.direction * hit;
        let tl_offset = tl * hit - center;
        let tr_offset = tr * hit - center;
        let bl_offset = bl * hit - center;
        let br_offset = br * hit - center;
        let max_sq = max(max(max(len_sq(tl_offset), len_sq(tr_offset)), len_sq(bl_offset)), len_sq(br_offset));
        cluster_size = sqrt(max_sq);

        let epsilon = clamp(traveled * epsilon_per_dist, min_epsilon, max_epsilon);
        if traveled > cone_march.limit || res.dist < cluster_size + epsilon {
            break;
        }

        step = max(res.dist - cluster_size, epsilon);
        traveled += step;
    }

    textureStore(cone_texture, position, vec4<f32>(traveled, 0., 0., 0.));
}

fn len_sq(v: vec3<f32>) -> f32 {
    return dot(v, v);
}
