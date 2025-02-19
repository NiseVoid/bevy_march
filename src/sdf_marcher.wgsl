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

@group(2) @binding(3) var<storage, read> nodes: array<BvhNode>;
@group(2) @binding(4) var<storage, read> instances: array<Instance>;

struct BvhNode {
    min: vec3<f32>,
    count: u32,
    max: vec3<f32>,
    index: u32,
}

struct Instance {
    order_start: u32,
    data_start: u32,
    material: u32,
    scale: f32,
    translation: vec3<f32>,
    matrix: mat3x3<f32>, // Inverse rotation + inverse scale
}

fn get_forward_ray(screen_uv: vec2<f32>) -> vec3<f32> {
    var march_uv = screen_uv * 2. - 1.; // Convert from 0 - 1 to -1 - 1

    // Scale X according to the aspect ratio, flip Y because world space and screen space Y are reversed
    march_uv *= vec2<f32>(settings.aspect_ratio, -1.);

    return normalize(vec3<f32>(march_uv, -settings.perspective_factor)); // -Z is forward
}

fn get_ray_dir(screen_uv: vec2<f32>) -> vec3<f32> {
    return settings.rotation * get_forward_ray(screen_uv);
}

fn get_ray_dir_invz(screen_uv: vec2<f32>) -> vec4<f32> {
    let dir = get_forward_ray(screen_uv);
    let inv_z = 1. / -dir.z;
    return vec4<f32>(settings.rotation * dir, inv_z);
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
    material: u32,
    id: u32,
    total_steps: u32,
}

const EPSILON_PER_DIST = 0.001;
const EPSILON_MIN = 0.002;
const EPSILON_MAX = 0.02;

fn march_ray(march: MarchSettings) -> MarchResult {
    let epsilon_per_dist = march.scale * EPSILON_PER_DIST;
    let min_epsilon = march.scale * EPSILON_MIN;
    let max_epsilon = march.scale * EPSILON_MAX;

    let dir_recip = 1. / march.direction;
    let ray_positive = sign(march.direction) == vec3<f32>(1.);

    // Output variables
    var result: MarchResult;
    result.distance = 1e9;
    result.traveled = 1e9;

    // The stack for the BVH
    var stack: array<u32, 16>;
    stack[0] = 0u;
    var stack_location = 1u;

    while true {
        if stack_location == 0 {
            break;
        }
        stack_location -= 1u;
        let node = nodes[stack[stack_location]];

        var hit = get_aabb_hit(node.min-max_epsilon, node.max+max_epsilon, march.origin, dir_recip, ray_positive);
        hit = vec2<f32>(max(hit.x, march.start), min(hit.y, result.traveled));
        if hit.x > hit.y || hit.y <= march.start {
            continue;
        }

        if node.count == 0 {
            let hit_a = get_node_min(node.index, march.origin, dir_recip, ray_positive);
            let hit_b = get_node_min(node.index+1, march.origin, dir_recip, ray_positive);

            if hit_a > hit_b {
                stack[stack_location] = node.index;
                stack[stack_location+1] = node.index+1;
                stack_location += 2u;
            } else {
                stack[stack_location] = node.index+1;
                stack[stack_location+1] = node.index;
                stack_location += 2u;
            }

            continue;
        }

        let instance_id = node.index;
        if instance_id == march.ignored {
            continue;
        }

        let instance = instances[instance_id];

        let end = (hit.y - hit.x) / instance.scale;

        let start_pos = march.origin + march.direction * hit.x;

        let relative_pos = instance.matrix * (start_pos - instance.translation);
        let relative_dir = instance.matrix * march.direction * instance.scale;

        var dist = 0.;
        var local_traveled = 0.;

        let start_epsilon = clamp(epsilon_per_dist * hit.x, min_epsilon, max_epsilon) / instance.scale;
        var epsilon = start_epsilon;
        let max_epsilon = max_epsilon / instance.scale;

        for (var i = 0u; i < 64u; i++) {
            result.total_steps += 1u;
            let pos = relative_pos + relative_dir * local_traveled;
            dist = sdf(pos, instance.order_start, instance.data_start);

            epsilon = min(start_epsilon + local_traveled * epsilon_per_dist, max_epsilon);
            if local_traveled > end || dist < epsilon {
                break;
            }

            local_traveled += max(dist, epsilon);
        }

        if dist < epsilon {
            dist *= instance.scale;

            let traveled = hit.x + local_traveled * instance.scale;

            if traveled < result.traveled {
                result.distance = dist;
                result.id = instance_id;
                result.material = instance.material;
                result.traveled = traveled;
            }
        }
    }

    return result;
}

fn get_node_min(node_index: u32, origin: vec3<f32>, dir_recip: vec3<f32>, ray_positive: vec3<bool>) -> f32 {
    let node = nodes[node_index];
    let min = select(node.max, node.min, ray_positive);

    return max3((min - origin) * dir_recip);
}

fn get_aabb_hit(aabb_min: vec3<f32>, aabb_max: vec3<f32>, origin: vec3<f32>, dir_recip: vec3<f32>, ray_positive: vec3<bool>) -> vec2<f32> {
    let min = select(aabb_max, aabb_min, ray_positive);
    let max = select(aabb_min, aabb_max, ray_positive);

    let tmin = (min - origin) * dir_recip;
    let tmax = (max - origin) * dir_recip;

    return vec2<f32>(max3(tmin), min3(tmax));
}

fn get_aabb_dist_sq(aabb_min: vec3<f32>, aabb_max: vec3<f32>, pos: vec3<f32>) -> f32 {
    let c = max(min(pos, aabb_max), aabb_min);
    return len_sq(c - pos);
}

fn max3(in: vec3<f32>) -> f32 {
    return max(max(in.x, in.y), in.z);
}

fn min3(in: vec3<f32>) -> f32 {
    return min(min(in.x, in.y), in.z);
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

const STEP_DIST = 0.04;
const STEP_COUNT = 8u;

fn get_occlusion(point: vec3<f32>, normal: vec3<f32>) -> f32 {
    let max_radius = f32(STEP_COUNT) * STEP_DIST;
    let center = point + normal * max_radius;
    let max_radius_sq = max_radius * max_radius;

    var steps: array<f32, STEP_COUNT>;

    // The stack for the BVH
    var stack: array<u32, 16>;
    stack[0] = 0u;
    var stack_location = 1u;

    while true {
        if stack_location == 0 {
            break;
        }
        stack_location -= 1u;
        let node = nodes[stack[stack_location]];

        var dist_sq = get_aabb_dist_sq(node.min, node.max, center);
        if dist_sq > max_radius_sq {
            continue;
        }

        if node.count == 0 {
            stack[stack_location] = node.index;
            stack[stack_location+1] = node.index+1;
            stack_location += 2u;
            continue;
        }

        let instance = instances[node.index];

        dist_sq = get_aabb_dist_sq(node.min, node.max, center);
        if dist_sq > max_radius_sq {
            continue;
        }

        let relative_pos = instance.matrix * (point - instance.translation);
        let relative_dir = instance.matrix * normal;

        for (var step = 0u; step < STEP_COUNT; step++) {
            let from_point = f32(step + 1) * STEP_DIST;
            let pos = relative_pos + relative_dir * from_point;
            let dist = sdf(pos, instance.order_start, instance.data_start);

            steps[step] = max(steps[step], from_point - dist);
        }
    }

    var occlusion = 1.;
    for (var step = 0u; step < STEP_COUNT; step++) {
        occlusion -= saturate(steps[step]) / f32(step + 1);
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

fn to_instance(instance_id: u32, pos: vec3<f32>) -> f32 {
    let instance = instances[instance_id];
    let relative_pos = instance.matrix * (pos - instance.translation);
    return sdf(relative_pos, instance.order_start, instance.data_start) * instance.scale;
}

fn len_sq(v: vec3<f32>) -> f32 {
    return dot(v, v);
}
