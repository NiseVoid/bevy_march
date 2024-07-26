#define_import_path bevy_march

#import bevy_prototype_sdf::sdf;

struct RayMarcherSettings {
    origin: vec3<f32>,
    rotation: mat3x3<f32>,
    t: f32,
    aspect_ratio: f32,
    perspective_factor: f32,
    near: f32,
    far: f32,
    light_dir: vec3<f32>,
}
@group(0) @binding(0) var<uniform> settings: RayMarcherSettings;

@group(1) @binding(0) var depth_texture: texture_storage_2d<r32float, write>;
@group(1) @binding(1) var cone_texture: texture_storage_2d<r32float, read>;

@group(2) @binding(2) var<storage, read> nodes: array<BvhNode>;
@group(2) @binding(3) var<storage, read> instances: array<Instance>;

struct BvhNode {
    min: vec3<f32>,
    count: u32,
    max: vec3<f32>,
    index: u32,
}

struct Instance {
    shape_offset: u32,
    material: u32,
    scale: f32,
    translation: vec3<f32>,
    matrix: mat3x3<f32>, // Inverse rotation + inverse scale
}

fn get_forward_ray(screen_uv: vec2<f32>) -> vec3<f32> {
    var march_uv = 1. - screen_uv * 2.;
    march_uv.x *= -settings.aspect_ratio;

    return normalize(vec3<f32>(march_uv, -settings.perspective_factor));
}

fn get_ray_dir(screen_uv: vec2<f32>) -> vec3<f32> {
    return settings.rotation * get_forward_ray(screen_uv);
}

fn get_initial_settings(screen_uv: vec2<f32>, start: f32) -> MarchSettings {
    let dir = get_forward_ray(screen_uv);
    let inv_z_factor = 1. / -dir.z;

    var march: MarchSettings;
    march.origin = settings.origin;
    march.local_direction = dir;
    march.direction = settings.rotation * dir;
    march.start = max(settings.near * inv_z_factor, start);
    march.limit = settings.far * inv_z_factor;
    march.ignored = PLACEHOLDER_ID;
    march.scale = 1.;
    return march;
}

struct MarchSettings {
    origin: vec3<f32>,
    direction: vec3<f32>,
    local_direction: vec3<f32>,
    start: f32,
    limit: f32,
    ignored: u32,
    scale: f32,
}

struct MarchResult {
    traveled: f32,
    distance: f32,
    steps: u32,
    material: u32,
    id: u32,
}

fn march_ray(march: MarchSettings) -> MarchResult {
    let epsilon_per_dist = march.scale * 0.001;
    let min_epsilon = march.scale * 0.002;
    let max_epsilon = march.scale * 0.02;

    var traveled = march.start;
    var res: NearestSdf;
    // TODO: we might be able to narrow this further with more data from the cone pass
    res.dist = max(march.limit - march.start, 1.);
    var step = 0.;
    var i: u32;
    for (i = 0u; i < 128u; i++) {
        let pos = march.origin + march.direction * traveled;
        res = get_nearest(res.dist + step, pos, march.ignored);

        let epsilon = clamp(traveled * epsilon_per_dist, min_epsilon, max_epsilon);
        if traveled > march.limit || res.dist < epsilon {
            break;
        }

        step = max(res.dist, epsilon);
        traveled += step;
    }

    var result: MarchResult;
    result.steps = i;
    result.traveled = traveled;
    if result.distance > march.scale*0.03 || traveled > march.limit {
        result.traveled = 1e9;
    }
    result.distance = res.dist;
    result.material = res.mat;
    result.id = res.id;

    return result;
}

fn get_individual_ray(position: vec2<u32>) -> MarchSettings {
    var size = textureDimensions(depth_texture);
    var cone_size = textureDimensions(cone_texture);
    let cone_factor = vec2<u32>(ceil(vec2<f32>(size) / vec2<f32>(cone_size)));
    let pixel_factor = 1. / vec2<f32>(size);

    let uv = (vec2<f32>(position) + 0.5) * pixel_factor;

    let start = textureLoad(cone_texture, position / cone_factor).r;
    return get_initial_settings(uv, start);
}

const step_dist = 0.035;
const steps = 10;

fn get_occlusion(point: vec3<f32>, normal: vec3<f32>) -> f32 {
    var occlusion = 1.;
    for (var i = 1; i <= steps; i++) {
        let step = f32(i);
        let from_point = step * step_dist;
        let pos = point + normal * from_point;
        let dist = single_dist(pos, 999999999u, from_point + 0.02);

        occlusion -= saturate(from_point - dist) / step;
    }

    return saturate(occlusion);
}

fn calc_normal(id: u32, p: vec3<f32>) -> vec3<f32> {
    let eps = 0.0001;
    let h = vec2<f32>(eps, 0.);
    return normalize(vec3<f32>(
        to_instance(id, p+h.xyy) - to_instance(id, p - h.xyy),
        to_instance(id, p+h.yxy) - to_instance(id, p - h.yxy),
        to_instance(id, p+h.yyx) - to_instance(id, p - h.yyx),
    ));
}

fn single_dist(pos: vec3<f32>, ignore: u32, max_dist: f32) -> f32 {
    return get_nearest(max_dist, pos, ignore).dist;
}

struct NearestSdf {
    dist: f32,
    mat: u32,
    id: u32,
}

fn sdf_min(prev: NearestSdf, next_dist: f32, next_mat: u32, next_id: u32) -> NearestSdf {
    var res: NearestSdf;
    if prev.id != PLACEHOLDER_ID && prev.dist < next_dist {
        return prev;
    } else {
        res.dist = next_dist;
        res.mat = next_mat;
        res.id = next_id;
    }
    return res;
}

const PLACEHOLDER_ID: u32 = 999999999u;

fn get_nearest(limit: f32, pos: vec3<f32>, ignored: u32) -> NearestSdf {
    var res: NearestSdf;
    res.dist = limit;
    res.id = PLACEHOLDER_ID;

    var stack: array<u32, 16>;
    stack[0] = 0u;
    var stackLocation = 1u;

    while true {
        if stackLocation == 0 {
            break;
        }
        stackLocation -= 1u;
        let node = nodes[stack[stackLocation]];

        let min = res.dist+0.001;
        let c = max(min(pos, node.max), node.min);
        let dist_sq = len_sq(c - pos);
        if dist_sq > min * min {
            continue;
        }

        if node.count == 0 {
            let left = nodes[node.index];
            let right = nodes[node.index+1];
            let lc = max(min(pos, left.max), left.min);
            let rc = max(min(pos, right.max), right.min);
            let ldist = len_sq(lc - pos);
            let rdist = len_sq(rc - pos);
            let min_dist = res.dist * res.dist;

            if ldist > rdist {
                // We want the larger node to be at the end
                if rdist > min_dist {
                    // If the nearest node isn't close enough, skip both
                    continue;
                }
                if ldist <= min_dist {
                    // If the further node isn't close enough, skip it
                    stack[stackLocation] = node.index;
                    stackLocation += 1u;
                }
                stack[stackLocation] = node.index+1;
                stackLocation += 1u;
            } else {
                if ldist > min_dist {
                    // If the nearest node isn't close enough, skip both
                    continue;
                }
                if rdist <= min_dist {
                    // If the further node isn't close enough, skip it
                    stack[stackLocation] = node.index+1;
                    stackLocation += 1u;
                }
                stack[stackLocation] = node.index;
                stackLocation += 1u;
            }

            continue;
        }

        for (var i = 0u; i < node.count; i++) {
            let instance_id = node.index + i;
            if instance_id == ignored {
                continue;
            }
            let instance = instances[instance_id];
            let relative_pos = instance.matrix * (pos - instance.translation);
            let dist = sdf(relative_pos, instance.shape_offset) * instance.scale;

            res = sdf_min(res, dist, instance.material, instance_id);
        }
    }

    return res;
}

fn to_instance(instance_id: u32, pos: vec3<f32>) -> f32 {
    let instance = instances[instance_id];
    let relative_pos = instance.matrix * (pos - instance.translation);
    return sdf(relative_pos, instance.shape_offset) * instance.scale;
}

fn len_sq(v: vec3<f32>) -> f32 {
    return dot(v, v);
}
