
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
    res.dist = 1e9;
    let instance_count = arrayLength(&instances);
    if instance_count == 0 {
        return res;
    }

    for (var i = 0u; i < instance_count; i++) {
        let instance = instances[i];
        // TODO: Encode the `/ instance.scale` in the matrix
        let relative_pos = instance.matrix * (pos - instance.translation);
        var current: Sdf;
        current.dist = sdf(relative_pos / instance.scale, instance.shape_offset) * instance.scale;
        current.mat = instance.material;

        res = sdf_min(res, current);
    }

    return res;
}

fn sdf(pos: vec3<f32>, start: u32) -> f32 {
    if shape_data[start] != 0 || shape_data[start+1] != 1 {
        return 1e9;
    }

    let shape_kind = shape_data[start+2];
    let offset = start + 4;

    if shape_kind == 0 { // Sphere
        return sd_sphere(pos, bitcast<f32>(shape_data[offset]));
    } else if shape_kind == 1 { // Capsule
        return 1e9; // TODO
    } else if shape_kind == 2 { // Cylinder
        return sd_cylinder(pos, bitcast<f32>(shape_data[offset]), bitcast<f32>(shape_data[offset+1]));
    } else if shape_kind == 3 { // Cuboid
        let x = bitcast<f32>(shape_data[offset]);
        let y = bitcast<f32>(shape_data[offset + 1]);
        let z = bitcast<f32>(shape_data[offset + 2]);
        return sd_cuboid(pos, vec3<f32>(x, y, z));
    } else if shape_kind == 4 { // Extruded
        let half_height = bitcast<f32>(shape_data[offset]);
        let extruded_kind = shape_data[offset+1];
        return sd_extrude(pos, half_height, extruded_kind, offset+2);
    }
    return 1e9;
}

fn sd_sphere(pos: vec3<f32>, radius: f32) -> f32 {
    return length(pos) - radius;
}

fn sd_cylinder(pos: vec3<f32>, half_height: f32, radius: f32) -> f32 {
    let d = abs(vec2<f32>(length(pos.xz), pos.y)) - vec2<f32>(radius, half_height);
    return min(max(d.x, d.y), 0.0) + length(max(d, vec2<f32>(0.)));
}

fn sd_cuboid(pos: vec3<f32>, bounds: vec3<f32>) -> f32 {
    let q = abs(pos) - bounds;
    return length(max(q, vec3<f32>(0.))) + min(max(q.x, max(q.y, q.z)), 0.);
}

fn sd_extrude(pos: vec3<f32>, half_height: f32, shape_kind: u32, offset: u32) -> f32 {
    let d = sdf2d(pos.xz, shape_kind, offset);
    let w = vec2<f32>(d, abs(pos.y) - half_height);

    return min(max(w.x, w.y), 0.) + length(max(w, vec2<f32>(0.)));
}
fn sdf2d(pos: vec2<f32>, shape_kind: u32, offset: u32) -> f32 {
    if shape_kind == 3 {
        let radius = bitcast<f32>(shape_data[offset]);
        let thickness = bitcast<f32>(shape_data[offset+1]);
        let segment = bitcast<f32>(shape_data[offset+2]);
        return sd_arc(pos, radius, thickness, segment);
    }

    return 1e9;
}

fn sd_arc(pos: vec2<f32>, radius: f32, thickness: f32, segment: f32) -> f32 {
    let sc = vec2<f32>(sin(segment), cos(segment));
    let p = vec2<f32>(abs(pos.x), pos.y);
    if sc.y * p.x > sc.x * p.y {
        let w = p - radius * sc;
        let l = length(w);
        return l - thickness;
    } else {
        let l = length(pos);
        let w = l - radius;
        return abs(w) - thickness;
    }
}
