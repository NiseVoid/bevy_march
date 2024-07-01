#import "ray_marcher.wgsl"::{march_ray, MarchSettings, get_initial_settings, settings, calc_normal, get_occlusion};
#import "ray_marcher.wgsl"::{shape_data, instance_data};
#import bevy_pbr::view_transformations::view_z_to_depth_ndc;

@group(1) @binding(0) var depth_texture: texture_storage_2d<r32float, write>;
@group(1) @binding(1) var color_texture: texture_storage_2d<rgba16float, write>;
@group(1) @binding(3) var<storage, read> materials: array<Material>;

struct Material {
    base_color: vec3<f32>,
    emissive: vec3<f32>,
    reflective: f32,
}

// TODO:
// struct RayMarcherSettings {
//     // TODO: Light direction
//     // TODO: Light color
//     // TODO: Ambient color
// }
// @group(0) @binding(2) var<uniform> pass_settings: MainPassSettings;

@compute @workgroup_size(8, 8, 1)
fn march(
    @builtin(global_invocation_id) invocation_id: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>,
) {
    var size = textureDimensions(color_texture);
    let pixel_factor = 1. / vec2<f32>(size);
    let position = invocation_id.xy * 8;

    if position.x >= size.x || position.y >= size.y {
        return;
    }
    let n = min(size - position, vec2<u32>(8, 8));

    // TODO: Cone march as an early pass
    for (var x = 0u; x < n.x; x++) {
        for (var y = 0u; y < n.y; y++) {
            let coords = position + vec2<u32>(x, y);
            let out = march_single(vec2<f32>(coords) * pixel_factor);

            textureStore(depth_texture, coords, vec4<f32>(settings.near / out.w, 0., 0., 0.));
            textureStore(color_texture, coords, vec4<f32>(out.rgb, 1.));
        }
    }
}

fn march_single(uv: vec2<f32>) -> vec4<f32> {
    var march = get_initial_settings(uv);
    let res = march_ray(march);

    let c = cos(settings.t / 2. % 6.35);
    let s = sin(settings.t / 2. % 6.35);
    let light_dir = normalize(vec3<f32>(s, 0.7, c));

    if res.traveled > 50. {
        return vec4<f32>(skybox(march.direction), settings.far);
    }
    if res.distance < 0.05 {
        let hit = march.origin + march.direction * (res.traveled - 0.01);
        let normal = calc_normal(hit);
        var diffuse = max(dot(normal, light_dir), 0.);

        var material = materials[res.material];
        if res.material == 50000 {
            material.reflective = 0.97;
            material.base_color = vec3<f32>(0.5, 1., 0.5);
        }
        var albedo = material.base_color;
        var emission = material.emissive;
        if material.reflective > 0.01 {
            let base_strength = (1. - material.reflective);
            let base_color = base_strength * material.base_color;

            var reflected: MarchSettings;
            reflected.origin = march.origin + march.direction * res.traveled;
            reflected.direction = reflect(march.direction, normal);
            reflected.limit = settings.far - res.traveled;
            reflected.ignored = 50000u;
            let res = march_ray(reflected);
            let refl_mat = materials[res.material];
            if res.distance < 0.1 {
                emission += refl_mat.emissive * material.reflective;
                albedo = base_color + refl_mat.base_color * material.reflective;

                let reflected_hit = hit + reflected.direction * (res.traveled - 0.01);
                let reflected_normal = calc_normal(reflected_hit);
                diffuse = max(dot(reflected_normal, light_dir), 0.);
            } else {
                albedo = base_color + skybox(reflected.direction) * material.reflective;
            }
        }
        let ao = get_occlusion(march.origin + march.direction * res.traveled, normal);
        let light = max(diffuse, ao * 0.4);
        let color = max(emission, vec3<f32>(albedo * light));
        if res.traveled > 35. {
            let factor = min((res.traveled - 35.) / 15., 1.);
            return vec4<f32>(
                (1. - factor) * color,
                res.traveled + res.distance
            );
        }
        return vec4<f32>(color, res.traveled + res.distance);
    }

    return vec4<f32>(skybox(march.direction), settings.far);
}

fn skybox(direction: vec3<f32>) -> vec3<f32> {
    let sphere_uv = healpix(direction);
    let cell = floor(sphere_uv * 5.);
    let uv = (sphere_uv - cell * 0.2) * 5. - 0.5;

    var dist = 999.;
    for (var i = 0; i < 4; i++) {
        let relative = vec2<f32>(f32(i) % 2. * 2. - 1., 2. * floor(f32(i) * 0.5) - 1.);
        let pos = uv - relative * 0.5;
        let origin = hash2(cell * 2. + relative) - 0.5;
        let corner_dist = sd_star(pos - origin, 0.03, 4u, 3.);
        dist = min(dist, corner_dist);
    }

    let star = -sign(dist) * 2.;

    let noise = perlinNoise2(sphere_uv * 2.);

    let background = vec3<f32>(
        0.,
        0.,
        noise * 0.005,
    );

    return max(background, vec3<f32>(star));
}

// From https://www.shadertoy.com/view/4sjXW1
fn healpix(p: vec3<f32>) -> vec2<f32> {
    let a = atan(p.x / p.z) * 0.63662;
    let h = 3.*abs(p.y);
    var h2 = .75*p.y;
    var uv = vec2<f32>(a + h2, a - h2);
    h2 = sqrt(3. - h);
    let a2 = h2 * fract(a);
    uv = mix(uv, vec2(-h2 + a2, a2), step(2., h));

    return uv;
}

fn sd_star(pos: vec2<f32>, r: f32, n: u32, m: f32) -> f32 {
    var p = pos;
    // next 4 lines can be precomputed for a given shape
    let  an = 3.141593/f32(n);
    let  en = 3.141593/m;  // m is between 2 and n
    let acs = vec2<f32>(cos(an), sin(an));
    let ecs = vec2<f32>(cos(en), sin(en)); // ecs=vec2(0,1) for regular polygon

    let bn = modulo(atan(p.x / p.y), (2.0*an)) - an;
    p = length(p) * vec2<f32>(cos(bn), abs(sin(bn)));
    p -= r * acs;
    p += ecs * clamp(-dot(p, ecs), 0.0, r*acs.y/ecs.y);
    return length(p)*sign(p.x);
}

fn modulo(x: f32, y: f32) -> f32 {
    return x - y * floor(x/y);
}

fn hash2(in: vec2<f32>) -> vec2<f32> {
    // procedural white noise
    return fract(sin(vec2<f32>(
        dot(in, vec2(127.1,311.7)),
        dot(in, vec2(269.5,183.3))
    )) * 43758.5453);
}

// MIT License. © Stefan Gustavson, Munrocket
//
fn permute4(x: vec4<f32>) -> vec4<f32> { return ((x * 34. + 1.) * x) % vec4<f32>(289.); }
fn fade2(t: vec2<f32>) -> vec2<f32> { return t * t * t * (t * (t * 6. - 15.) + 10.); }

fn perlinNoise2(P: vec2<f32>) -> f32 {
  var Pi: vec4<f32> = floor(P.xyxy) + vec4<f32>(0., 0., 1., 1.);
  let Pf = fract(P.xyxy) - vec4<f32>(0., 0., 1., 1.);
  Pi = Pi % vec4<f32>(289.); // To avoid truncation effects in permutation
  let ix = Pi.xzxz;
  let iy = Pi.yyww;
  let fx = Pf.xzxz;
  let fy = Pf.yyww;
  let i = permute4(permute4(ix) + iy);
  var gx: vec4<f32> = 2. * fract(i * 0.0243902439) - 1.; // 1/41 = 0.024...
  let gy = abs(gx) - 0.5;
  let tx = floor(gx + 0.5);
  gx = gx - tx;
  var g00: vec2<f32> = vec2<f32>(gx.x, gy.x);
  var g10: vec2<f32> = vec2<f32>(gx.y, gy.y);
  var g01: vec2<f32> = vec2<f32>(gx.z, gy.z);
  var g11: vec2<f32> = vec2<f32>(gx.w, gy.w);
  let norm = 1.79284291400159 - 0.85373472095314 *
      vec4<f32>(dot(g00, g00), dot(g01, g01), dot(g10, g10), dot(g11, g11));
  g00 = g00 * norm.x;
  g01 = g01 * norm.y;
  g10 = g10 * norm.z;
  g11 = g11 * norm.w;
  let n00 = dot(g00, vec2<f32>(fx.x, fy.x));
  let n10 = dot(g10, vec2<f32>(fx.y, fy.y));
  let n01 = dot(g01, vec2<f32>(fx.z, fy.z));
  let n11 = dot(g11, vec2<f32>(fx.w, fy.w));
  let fade_xy = fade2(Pf.xy);
  let n_x = mix(vec2<f32>(n00, n01), vec2<f32>(n10, n11), vec2<f32>(fade_xy.x));
  let n_xy = mix(n_x.x, n_x.y, fade_xy.y);
  return 2.3 * n_xy;
}
