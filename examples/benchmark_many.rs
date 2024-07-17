use bevy_march::*;
use bevy_prototype_sdf::Sdf3d;

use bevy::{
    core_pipeline::bloom::BloomSettings,
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

    app.insert_gizmo_config(
        DefaultGizmoConfigGroup,
        GizmoConfig {
            render_layers: RenderLayers::layer(1),
            ..default()
        },
    );

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
    fn text_style(size: f32, gray: f32) -> TextStyle {
        TextStyle {
            font: default(),
            font_size: size,
            color: Color::srgb(gray, gray, gray),
        }
    }
    commands.spawn((
        Text2dBundle {
            text: Text {
                sections: vec![
                    TextSection::new("FPS: ", text_style(20., 0.7)),
                    TextSection::new("?", text_style(20., 0.8)),
                ],
                ..default()
            },
            text_anchor: Anchor::TopLeft,
            ..default()
        },
        FpsText,
    ));
    commands.spawn(Camera2dBundle {
        camera: Camera {
            order: 1,
            hdr: true,
            clear_color: ClearColorConfig::None,
            ..default()
        },
        ..default()
    });

    // Camera
    commands.spawn((
        Camera3dBundle {
            transform: Transform::from_translation(vec3(0.0, 0.0, 5.0))
                .looking_at(Vec3::X, Vec3::Y),
            camera: Camera {
                hdr: true,
                ..default()
            },
            projection: Projection::Perspective(PerspectiveProjection {
                far: 100.,
                ..default()
            }),
            ..default()
        },
        RenderLayers::from_layers(&[0, 1]),
        MarcherSettings::default(),
        MarcherMainTextures::new(&mut images, (8, 8)),
        MarcherConeTexture::new(&mut images, &device, (8, 8)),
        MarcherScale(1),
        BloomSettings {
            intensity: 0.3,
            composite_mode: bevy::core_pipeline::bloom::BloomCompositeMode::Additive,
            prefilter_settings: bevy::core_pipeline::bloom::BloomPrefilterSettings {
                threshold: 1.,
                threshold_softness: 0.0,
            },
            ..default()
        },
    ));

    // Light
    commands.spawn((
        DirectionalLightBundle {
            directional_light: DirectionalLight {
                color: Color::srgb(1., 1., 0.9),
                illuminance: 5_000.,
                shadows_enabled: false,
                ..default()
            },
            transform: Transform::from_xyz(1., 1.5, 1.).looking_at(Vec3::ZERO, Vec3::Y),
            ..default()
        },
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
            TransformBundle::from_transform(
                Transform::from_xyz(x, y, z).with_scale(Vec3::splat(scale)),
            ),
            cube.clone(),
            cube_material.clone(),
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
            TransformBundle::from_transform(
                Transform::from_xyz(x, y, z).with_scale(Vec3::splat(scale)),
            ),
            sphere.clone(),
            sphere_material.clone(),
        ));
    }
}

fn update_fps(
    window: Query<&Window>,
    mut text: Query<(&mut Transform, &mut Text), With<FpsText>>,
    diag_store: Res<DiagnosticsStore>,
) {
    let half_size = window
        .get_single()
        .map(|w| w.resolution.size() * 0.5)
        .unwrap_or_default();
    let Ok((mut transform, mut text)) = text.get_single_mut() else {
        return;
    };
    let Some(fps) = diag_store.get(&FrameTimeDiagnosticsPlugin::FPS) else {
        return;
    };
    let Some(fps) = fps.smoothed() else {
        return;
    };
    transform.translation = Vec3::new(-half_size.x, half_size.y, 0.);
    text.sections[1].value = format!("{:.1}", fps);
}
