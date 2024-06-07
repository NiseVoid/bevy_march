#import "ray_marcher.wgsl"::{march_ray, MarchSettings, get_initial_settings, settings, calc_normal, get_occlusion};

@group(1) @binding(0) var depth_texture: texture_storage_2d<r32float, write>;
@group(1) @binding(1) var color_texture: texture_storage_2d<rgba16float, write>;

// TODO:
// struct RayMarcherSettings {
//     // TODO: Light direction
//     // TODO: Light color
//     // TODO: Ambient color
// }
// @group(0) @binding(2) var<uniform> pass_settings: MainPassSettings;

// TODO: Material buffer
//   - Base color, emissive, reflective, AO, etc

@compute @workgroup_size(8, 8, 1)
fn march(
    @builtin(global_invocation_id) invocation_id: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>,
) {
    let pixel_factor = 1. / vec2<f32>(num_workgroups.xy * 8 * 8);
    let uv = vec2<f32>(invocation_id.xy) / vec2<f32>(num_workgroups.xy * 8);

    // TODO: Cone march as an early pass
    for (var x = 0u; x < 8; x++) {
        for (var y = 0u; y < 8; y++) {
            // TODO: Early out fot all pixels that are outside the texture
            let coords = invocation_id.xy * 8 + vec2<u32>(x, y);
            let out = march_single(uv + vec2<f32>(f32(x), f32(y)) * pixel_factor);

            textureStore(depth_texture, coords, vec4<f32>(out.w / settings.far, 0., 0., 0.));
            textureStore(color_texture, coords, vec4<f32>(out.rgb, 1.));
        }
    }
}

fn march_single(uv: vec2<f32>) -> vec4<f32> {
    let march = get_initial_settings(uv);

    let res = march_ray(march);

    let c = cos(settings.t / 2. % 6.35);
    let s = sin(settings.t / 2. % 6.35);
    let light_dir = normalize(vec3<f32>(s, 0.7, c));

    var tint = vec3<f32>(1.);
    if res.distance < 0.1 {
        let hit = march.origin + march.direction * (res.traveled - 0.01);
        let normal = calc_normal(hit);
        var diffuse = max(dot(normal, light_dir), 0.);
        tint = get_albedo(res.material);
        var emission = get_emission(res.material);
        if res.material == 4u {
            var reflected: MarchSettings;
            reflected.origin = hit;
            reflected.direction = reflect(march.direction, normal);
            reflected.limit = settings.far - res.traveled;
            let res = march_ray(reflected);
            if res.distance < 0.1 {
                emission = tint * get_emission(res.material);
                tint *= get_albedo(res.material);
                let reflected_hit = hit + reflected.direction * (res.traveled - 0.01);
                let reflected_normal = calc_normal(reflected_hit);
                diffuse = max(dot(reflected_normal, light_dir), 0.);
            } else {
                tint *= skybox(reflected.direction);
            }
        }
        let ao = get_occlusion(march.origin + march.direction * res.traveled, normal);
        let light = max(diffuse, ao * 0.4);
        let color = max(emission, vec3<f32>(tint * light));
        if res.traveled > 40. {
            let factor = min((res.traveled - 40.) / 40., 1.);
            return vec4<f32>(
                factor * skybox(march.direction) + (1. - factor) * color,
                res.traveled + res.distance
            );
        }
        return vec4<f32>(color, res.traveled + res.distance);
    }

    return vec4<f32>(skybox(march.direction), settings.far);
}

fn skybox(direction: vec3<f32>) -> vec3<f32> {
    let angle = atan(direction.x / direction.z);
    let angle2 = atan(direction.y / direction.z);
    let angles = vec2<f32>(angle, angle2);
    let uv = fract(angles) % 0.1 * 10. - 0.5;
    let cell = vec2<f32>(floor(angle * 10.), floor(angle2 * 10.));

    var dist = 999.;
    for (var i = 0; i < 4; i++) {
        let relative = vec2<f32>(f32(i) % 2. * 2. - 1., 2. * floor(f32(i) * 0.5) - 1.);
        let pos = uv - relative * 0.5;
        let origin = hash2(cell * 2. + relative) - 0.5;
        let corner_dist = sd_star(pos - origin, 0.07, 4u, 3.);
        dist = min(dist, corner_dist);
    }

    let star = -sign(dist) * 2.4;

    let noise = perlinNoise2(angles * 4.);
    let noise2 = perlinNoise2(angles * 4. + 12);

    let background = vec3<f32>(
        noise * 0.1 + 0.9,
        0.3,
        noise2 * 0.2 + 0.9,
    );

    return max(background, vec3<f32>(star));
}

fn get_albedo(material: u32) -> vec3<f32> {
    switch material {
        case 1u: {
            return vec3<f32>(0.9, 1.5, 1.7);
        }
        case 2u: {
            return vec3<f32>(1.);
        }
        case 3u: {
            return vec3<f32>(0.6, 1., 0.9);
        }
        case 4u: {
            return vec3<f32>(0.75, 0.75, 1.);
        }
        default: {
            return vec3<f32>(0.);
        }
    }
}

fn get_emission(material: u32) -> vec3<f32> {
    switch material {
        case 2u: {
            return vec3<f32>(0., 1.8, 2.);
        }
        default: {
            return vec3<f32>(0.);
        }
    }
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
