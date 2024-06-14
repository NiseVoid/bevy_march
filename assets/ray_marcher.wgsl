
struct RayMarcherSettings {
    origin: vec3<f32>,
    rotation: mat3x3<f32>,
    t: f32,
    aspect_ratio: f32,
    perspective_factor: f32,
    near: f32,
    far: f32,
}
@group(0) @binding(0) var<uniform> settings: RayMarcherSettings;

@group(1) @binding(2) var<storage, read> shape_data: array<u32>;

struct Instance {
    shape_offset: u32,
    material: u32,
    scale: f32,
    translation: vec3<f32>,
    matrix: mat3x3<f32>,
}

@group(1) @binding(4) var<storage, read> instances: array<Instance>;

struct MarchSettings {
    origin: vec3<f32>,
    direction: vec3<f32>,
    start: f32,
    limit: f32,
}

fn get_initial_settings(screen_uv: vec2<f32>) -> MarchSettings {
    var march_uv = 1. - screen_uv * 2.;
    march_uv.x *= settings.aspect_ratio;

    var march: MarchSettings;
    march.origin = settings.origin;
    march.direction = settings.rotation * normalize(vec3<f32>(march_uv, -settings.perspective_factor));
    march.start = settings.near;
    march.limit = settings.far;
    return march;
}

struct MarchResult {
    traveled: f32,
    distance: f32,
    material: u32,
}

fn march_ray(march: MarchSettings) -> MarchResult {
    var traveled = march.start;
    var tint = vec3<f32>(1.);
    var res: Sdf;
    for (var i = 0u; i < 128u; i++) {
        let pos = march.origin + march.direction * traveled;
        res = get_scene_dist(pos);
        if traveled > march.limit || res.dist < max(traveled * 0.001, 0.001) {
            break;
        }

        traveled += max(res.dist, max(traveled * 0.001, 0.01));
    }

    var result: MarchResult;
    result.traveled = traveled;
    result.distance = res.dist;
    result.material = res.mat;

    return result;
}

const step_dist = 0.035;
const steps = 10;

fn get_occlusion(point: vec3<f32>, normal: vec3<f32>) -> f32 {
    var occlusion = 1.;
    for (var i = 1; i <= steps; i++) {
        let step = f32(i);
        let from_point = step * step_dist;
        let pos = point + normal * from_point;
        let res = get_scene_dist(pos);

        occlusion -= saturate(from_point - res.dist) / step;
    }

    return saturate(occlusion);
}

fn calc_normal(p: vec3<f32>) -> vec3<f32> {
    let eps = 0.0001;
    let h = vec2<f32>(eps, 0.);
    return normalize(vec3<f32>(
        get_scene_dist(p+h.xyy).dist - get_scene_dist(p - h.xyy).dist,
        get_scene_dist(p+h.yxy).dist - get_scene_dist(p - h.yxy).dist,
        get_scene_dist(p+h.yyx).dist - get_scene_dist(p - h.yyx).dist,
    ));
}

struct Sdf {
    dist: f32,
    mat: u32,
}

fn sdf_min(a: Sdf, b: Sdf) -> Sdf {
    if a.dist <= b.dist {
        return a;
    }
    return b;
}

fn get_scene_dist(pos: vec3<f32>) -> Sdf {    
    var res: Sdf;
    res.dist = 1000000.;
    let instance_count = arrayLength(&instances);
    if instance_count == 0 {
        return res;
    }

    for (var i = 0u; i < instance_count; i++) {
        let instance = instances[i];
        let relative_pos = instance.matrix * (pos - instance.translation);
        // TODO: Use a shape that lets us test rotation
        var current: Sdf;
        current.dist = sd_sphere(relative_pos / instance.scale, 0.5) * instance.scale;
        current.mat = instance.material;

        res = sdf_min(res, current);
    }

    return res;
}

fn op_extrude(z: f32, dist2d: f32, h: f32) -> f32 {
    let w = vec2<f32>(dist2d, abs(z) - h);
    return min(max(w.x, w.y), 0.) + length(max(w, vec2<f32>(0.)));
}

fn sd_plane(pos: vec3<f32>) -> f32 {
    return dot(pos, vec3<f32>(0., 1., 0.));
}

fn sd_sphere(pos: vec3<f32>, r: f32) -> f32 {
    return length(pos) - r;
}

fn sd_moon(pos: vec2<f32>, d: f32, ra: f32, rb: f32) -> f32 {
    var p = vec2<f32>(pos.x, abs(pos.y));
    let a = (ra*ra - rb*rb + d*d) / (2.*d);
    let b = sqrt(max(ra*ra - a*a, 0.));
    if d * (p.x*b - p.y*a) > d*d * max(b-p.y, 0.) {
        return length(p - vec2<f32>(a, b));
    }
    return max(length(p) - ra, -(length(p - vec2<f32>(d, 0.)) - rb));
}
