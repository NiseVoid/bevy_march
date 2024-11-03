use bevy_march::*;
use bevy_prototype_sdf::Sdf3d;

use bevy::{
    core_pipeline::bloom::Bloom,
    diagnostic::{DiagnosticsStore, FrameTimeDiagnosticsPlugin},
    math::vec3,
    prelude::*,
    render::{render_resource::ShaderType, renderer::RenderDevice, view::RenderLayers},
    sprite::Anchor,
};

fn main() {
    let mut app = App::new();
    app.add_plugins((
        DefaultPlugins.set::<WindowPlugin>(WindowPlugin {
            primary_window: Some(Window {
                present_mode: bevy::window::PresentMode::AutoNoVsync,
                ..default()
            }),
            ..default()
        }),
        FrameTimeDiagnosticsPlugin,
    ));

    let main_pass_shader = app.world().resource::<AssetServer>().load("simple.wgsl");

    app.add_plugins(RayMarcherPlugin::<SdfMaterial>::new(main_pass_shader))
        .add_systems(Startup, setup)
        .add_systems(Update, (update_fps,))
        .run();
}

#[derive(Asset, ShaderType, TypePath, Clone, Debug, Default)]
struct SdfMaterial {
    base_color: Vec3,
    emissive: Vec3,
}

impl MarcherMaterial for SdfMaterial {}

#[derive(Component)]
struct FpsText;

fn setup(
    mut commands: Commands,
    mut images: ResMut<Assets<Image>>,
    mut sdfs: ResMut<Assets<Sdf3d>>,
    mut mats: ResMut<Assets<SdfMaterial>>,
    device: Res<RenderDevice>,
) {
    commands.spawn((
        Camera2d,
        Camera {
            order: 1,
            hdr: true,
            clear_color: ClearColorConfig::None,
            ..default()
        },
    ));
    commands.spawn((
        Text2d::default(),
        TextFont {
            font_size: 18.0,
            ..default()
        },
        Anchor::TopLeft,
        FpsText,
    ));

    // Camera
    commands.spawn((
        Camera3d::default(),
        Camera {
            hdr: true,
            ..default()
        },
        Projection::Perspective(PerspectiveProjection {
            far: 100.,
            ..default()
        }),
        Transform::from_translation(vec3(0.0, 0.0, 5.0)).looking_at(Vec3::X, Vec3::Y),
        RenderLayers::from_layers(&[0, 1]),
        MarcherSettings::default(),
        MarcherMainTextures::new(&mut images, (8, 8)),
        MarcherConeTexture::new(&mut images, &device, (8, 8)),
        MarcherScale(1),
        Bloom {
            intensity: 0.3,
            composite_mode: bevy::core_pipeline::bloom::BloomCompositeMode::Additive,
            prefilter: bevy::core_pipeline::bloom::BloomPrefilter {
                threshold: 1.,
                threshold_softness: 0.0,
            },
            ..default()
        },
    ));

    // Light
    commands.spawn((
        DirectionalLight {
            color: Color::srgb(1., 1., 0.9),
            illuminance: 5_000.,
            shadows_enabled: false,
            ..default()
        },
        Transform::from_xyz(1., 1.5, 1.).looking_at(Vec3::ZERO, Vec3::Y),
        MarcherShadowSettings::default(),
        MarcherShadowTextures::new(&mut images),
    ));

    let cube = sdfs.add(Sdf3d::from(Cuboid::default()));
    let cube_material = mats.add(SdfMaterial {
        base_color: LinearRgba::BLACK.to_vec3(),
        emissive: LinearRgba::rgb(0., 1.5, 1.75).to_vec3(),
    });

    for i in 0..100 {
        let x = (i as f32 * 59292.15) % 80. - 40.;
        let y = (i as f32 * 928.19) % 20. - 10.;
        let z = -5. - (i as f32 * 29578.92) % 80.;
        let scale = 0.2 + (i as f32 * 9829.72) % 1.8;

        commands.spawn((
            Transform::from_xyz(x, y, z).with_scale(Vec3::splat(scale)),
            RenderedSdf {
                sdf: cube.clone(),
                material: cube_material.clone(),
            },
        ));
    }

    let sphere = sdfs.add(Sdf3d::from(Sphere::default()));
    let sphere_material = mats.add(SdfMaterial {
        base_color: LinearRgba::gray(0.7).to_vec3(),
        emissive: LinearRgba::BLACK.to_vec3(),
    });

    for i in 0..400 {
        let x = (i as f32 * 247825.27) % 100. - 50.;
        let y = (i as f32 * 29752.25) % 20. - 10.;
        let z = -5. - (i as f32 * 85285.29) % 50.;
        let scale = 0.5 + (i as f32 * 927.19) % 1.5;

        commands.spawn((
            Transform::from_xyz(x, y, z).with_scale(Vec3::splat(scale)),
            RenderedSdf {
                sdf: sphere.clone(),
                material: sphere_material.clone(),
            },
        ));
    }
}

fn update_fps(
    window: Single<&Window>,
    mut text: Single<(&mut Transform, &mut Text2d), With<FpsText>>,
    diag_store: Res<DiagnosticsStore>,
) {
    let half_size = window.resolution.size() * 0.5;
    let (ref mut transform, ref mut text) = *text;
    let Some(fps) = diag_store.get(&FrameTimeDiagnosticsPlugin::FPS) else {
        return;
    };
    let Some(fps) = fps.smoothed() else {
        return;
    };
    transform.translation = Vec3::new(-half_size.x, half_size.y, 0.);
    text.clear();
    text.push_str(&format!("FPS: {:.1}", fps))
}
