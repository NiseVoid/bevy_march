#import bevy_march::{get_initial_settings, settings, NearestSdf, get_ray_dir_invz, nodes, get_node_min, instances, get_aabb_hit, EPSILON_PER_DIST, EPSILON_MIN, EPSILON_MAX};
#import bevy_prototype_sdf::sdf;

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

    let cluster_start = vec2<f32>(position) * pixel_factor;
    let cluster_end = vec2<f32>(position + 1) * pixel_factor;
    let cluster_center = (cluster_start + cluster_end) * 0.5;
    var march = get_initial_settings(cluster_center, 0.);

    let tl = get_ray_dir_invz(cluster_start);
    let tr = get_ray_dir_invz(vec2<f32>(cluster_end.x, cluster_start.y));
    let bl = get_ray_dir_invz(vec2<f32>(cluster_start.x, cluster_end.y));
    let br = get_ray_dir_invz(cluster_end);
    let radius_per_unit = sqrt(max(max(max(
        len_sq(tl.xyz-march.direction),
        len_sq(tr.xyz-march.direction)),
        len_sq(bl.xyz-march.direction)),
        len_sq(br.xyz-march.direction),
    ));

    let frustum = get_cone_frustum(tl, tr, bl, br, march.direction);

    let epsilon_per_dist = march.scale * EPSILON_PER_DIST;
    let min_epsilon = march.scale * EPSILON_MIN;
    let max_epsilon = march.scale * EPSILON_MAX;

    let dir_recip = 1. / march.direction;
    let ray_positive = sign(march.direction) == vec3<f32>(1.);

    var cluster_size: f32;
    var traveled = 1e9;

    var stack: array<u32, 16>;
    stack[0] = 0u;
    var stack_location = 1u;

    while true {
        if stack_location == 0 {
            break;
        }
        stack_location -= 1u;
        let node = nodes[stack[stack_location]];

        if !check_aabb_frustum(frustum, node.min-max_epsilon, node.max+max_epsilon) {
            continue;
        }

        if node.count == 0 {
            let a = nodes[node.index];
            let hit_a = project_node(a.min-max_epsilon, a.max+max_epsilon, march.origin, march.direction, ray_positive);
            let b = nodes[node.index+1];
            let hit_b = project_node(b.min-max_epsilon, b.max+max_epsilon, march.origin, march.direction, ray_positive);

            if hit_a.x > hit_b.x {
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

        // If the node is a leaf it should have one node to traverse

        let instance_id = node.index;
        let instance = instances[instance_id];

        let min = node.min - max_epsilon;
        let max = node.max + max_epsilon;
        var hit = project_node(min, max, march.origin, march.direction, ray_positive);
        hit = vec2<f32>(max(hit.x, march.start), min(hit.y, traveled));
        if hit.x > hit.y {
            continue;
        }

        let start = max(hit.x, 0.);
        let end = (min(hit.y, march.limit) - start) / instance.scale;

        let start_pos = march.origin + march.direction * start;

        let relative_pos = instance.matrix * (start_pos - instance.translation);
        let relative_dir = instance.matrix * march.direction * instance.scale;

        var dist = 0.;
        var local_traveled = 0.;

        let start_epsilon = clamp(epsilon_per_dist * hit.x, min_epsilon, max_epsilon) / instance.scale;
        var epsilon = start_epsilon;
        let max_epsilon = max_epsilon / instance.scale;

        let start_radius = start * radius_per_unit / instance.scale;
        let radius_per_scaled_unit = radius_per_unit * instance.scale;

        for (var i = 0u; i < 512u; i++) {
            let pos = relative_pos + relative_dir * local_traveled;
            dist = sdf(pos, instance.order_start, instance.data_start);

            epsilon = min(start_epsilon + local_traveled * epsilon_per_dist, max_epsilon);
            cluster_size = start_radius + radius_per_scaled_unit * local_traveled;

            if local_traveled > end || dist < cluster_size + epsilon {
                break;
            }

            local_traveled += max(dist - cluster_size, epsilon);
        }

        let cur_travel = start + local_traveled * instance.scale;
        if dist < cluster_size + epsilon && cur_travel < traveled {
            traveled = cur_travel;
        }
    }

    textureStore(cone_texture, position, vec4<f32>(traveled, 0., 0., 0.));
}

fn project_node(aabb_min: vec3<f32>, aabb_max: vec3<f32>, origin: vec3<f32>, direction: vec3<f32>, ray_sign: vec3<bool>) -> vec2<f32> {
    let min = aabb_min - origin;
    let max = aabb_max - origin;
    let axis_min = select(max, min, ray_sign);
    let axis_max = select(min, max, ray_sign);
    let t_min = dot(direction, axis_min);
    let t_max = dot(direction, axis_max);

    return vec2<f32>(t_min, t_max);
}

struct Frustum {
    planes: array<vec4<f32>, 6>,
    points: array<vec3<f32>, 8>,
    axes: array<vec3<f32>, 6>,
    mins: array<vec3<f32>, 6>,
    maxs: array<vec3<f32>, 6>,
}

fn get_cone_frustum(tl: vec4<f32>, tr: vec4<f32>, bl: vec4<f32>, br: vec4<f32>, center: vec3<f32>) -> Frustum{
    // Calculate points on the near plane
    let near_tl = settings.origin + tl.xyz * settings.near * tl.w;
    let near_tr = settings.origin + tr.xyz * settings.near * tr.w;
    let near_bl = settings.origin + bl.xyz * settings.near * bl.w;
    let near_br = settings.origin + br.xyz * settings.near * br.w;

    // Calculate points on the far plane
    let far_tl = settings.origin + tl.xyz * settings.far * tl.w;
    let far_tr = settings.origin + tr.xyz * settings.far * tr.w;
    let far_bl = settings.origin + bl.xyz * settings.far * bl.w;
    let far_br = settings.origin + br.xyz * settings.far * br.w;

    var frustum: Frustum;
    frustum.planes[0] = vec4<f32>(center, -dot(center, near_tl));
    frustum.planes[1] = vec4<f32>(-center, -dot(-center, far_tl));
    frustum.planes[2] = get_plane(near_bl, far_bl, far_tl);
    frustum.planes[3] = get_plane(near_tl, far_tl, far_tr);
    frustum.planes[4] = get_plane(near_tr, far_tr, far_br);
    frustum.planes[5] = get_plane(near_br, far_br, far_bl);

    frustum.points[0] = near_tl;
    frustum.points[1] = near_tr;
    frustum.points[2] = near_br;
    frustum.points[3] = near_bl;
    frustum.points[4] = far_tl;
    frustum.points[5] = far_tr;
    frustum.points[6] = far_br;
    frustum.points[7] = far_bl;

    let frustum_axes = array(normalize(far_tr - far_tl), normalize(far_tl - far_bl), tl.xyz, tr.xyz, bl.xyz, br.xyz);
    for (var i = 0u; i < 6; i++) {
        let a = frustum_axes[i];
        frustum.axes[i] = a;
        // Calculate the cross products between the current frustum edge and each aligned axis.
        // These axes doesn't need to be normalized since the min/max in both
        //   sides of the SAT check are always scaled according to it
        let axes = array(
            vec3<f32>(0., -a.z, a.y),
            vec3<f32>(a.z, 0., -a.x),
            vec3<f32>(-a.y, a.x, 0.),
        );
        for (var j = 0u; j < 3; j++) {
            let axis = axes[j];
            let dntl = dot(axis, near_tl);
            let dntr = dot(axis, near_tr);
            let dnbl = dot(axis, near_bl);
            let dnbr = dot(axis, near_br);
            let dftl = dot(axis, far_tl);
            let dftr = dot(axis, far_tr);
            let dfbl = dot(axis, far_bl);
            let dfbr = dot(axis, far_br);
            frustum.mins[i][j] = min(min(min(min(min(min(min(
                dntl, dntr), dnbl), dnbr),
                dftl), dftr), dfbl), dfbr
            );
            frustum.maxs[i][j] = max(max(max(max(max(max(max(
                dntl, dntr), dnbl), dnbr),
                dftl), dftr), dfbl), dfbr
            );
        }
    }

    return frustum;
}

fn get_plane(p0: vec3<f32>, p1: vec3<f32>, p2: vec3<f32>) -> vec4<f32> {
    let normal = normalize(cross(p1-p0, p2-p1));
    let offset = dot(normal, p0);
    return vec4<f32>(normal, -offset);
}

fn len_sq(v: vec3<f32>) -> f32 {
    return dot(v, v);
}

fn check_aabb_frustum(frustum: Frustum, min: vec3<f32>, max: vec3<f32>) -> bool {
    // check box outside/inside of frustum
    for (var i = 0u; i < 6; i++) {
        if dot(frustum.planes[i].xyzw, vec4(max + min, 2.0f)) + dot(abs(frustum.planes[i].xyz), max - min) < 0. {
          return false;
        }
    }

    // check frustum outside/inside of box
    var outMin = vec3<u32>(1);
    var outMax = vec3<u32>(1);

    for( var i=0; i<8; i++ ) {
        outMin = min(outMin, vec3<u32>(-sign(frustum.points[i].xyz - min)));
        outMax = min(outMax, vec3<u32>(sign(frustum.points[i].xyz - max)));
    }

    let out = max(outMin, outMax);
    if out.x == 1 || out.y == 1 || out.z == 1 {
        return false;
    }

    // Separating Axis Theorem for the cross products between each unique edge direction
    for (var i = 0u; i < 6; i++) {
        let a = frustum.axes[i];
        let f_mins = frustum.mins[i];
        let f_maxs = frustum.maxs[i];

        let axes = array(
            vec3<f32>(0., -a.z, a.y),
            vec3<f32>(a.z, 0., -a.x),
            vec3<f32>(-a.y, a.x, 0.),
        );
        for (var j = 0u; j < 3; j++) {
            let cross_axis = axes[j];
            let sign = sign(cross_axis) == vec3<f32>(1.);
            let axis_min = select(max, min, sign);
            let axis_max = select(min, max, sign);

            let t_min = dot(cross_axis, axis_min);
            let t_max = dot(cross_axis, axis_max);

            if f_mins[j] > t_max || t_min > f_maxs[j] {
                return false;
            }
        }
    }

    return true;
}
