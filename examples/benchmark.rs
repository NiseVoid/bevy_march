use bevy_march::*;
use bevy_prototype_sdf::Sdf3d;

use bevy::{
    core_pipeline::bloom::BloomSettings,
    diagnostic::{DiagnosticsStore, FrameTimeDiagnosticsPlugin},
    input::mouse::MouseMotion,
    math::vec3,
    prelude::*,
    render::{render_resource::ShaderType, renderer::RenderDevice, view::RenderLayers},
    sprite::Anchor,
    window::CursorGrabMode,
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

    let main_pass_shader = app.world().resource::<AssetServer>().load("benchmark.wgsl");

    app.add_plugins(RayMarcherPlugin::<SdfMaterial>::new(main_pass_shader))
        .init_resource::<CursorState>()
        .add_systems(Startup, setup)
        .add_systems(
            Update,
            (
                (grab_cursor, rotate_and_move).chain(),
                update_offsets,
                rotate_light,
                update_fps,
            ),
        )
        .run();
}

#[derive(Asset, ShaderType, TypePath, Clone, Debug, Default)]
struct SdfMaterial {
    base_color: Vec3,
    emissive: Vec3,
    reflective: f32,
}

impl MarcherMaterial for SdfMaterial {}

#[derive(Component)]
struct FpsText;
#[derive(Component)]
struct HelpText;

fn setup(
    mut commands: Commands,
    mut images: ResMut<Assets<Image>>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut std_mats: ResMut<Assets<StandardMaterial>>,
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
    commands.spawn((
        Text2dBundle {
            text: Text {
                sections: vec![TextSection::new(
                    "Use WASD to move.\n\
                    Space and Ctrl to go up and down.\n\
                    Left click and Escape to lock and release the cursor",
                    text_style(40., 1.),
                )],
                justify: JustifyText::Center,
                ..default()
            },
            ..default()
        },
        HelpText,
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

    // Spawn some meshes to test interaction with raymarcher
    let mesh = meshes.add(Cuboid::new(2., 2., 0.01));
    let mat = std_mats.add(StandardMaterial::from(Color::srgb(1., 0.5, 0.5)));

    // Plane 1
    commands.spawn(PbrBundle {
        mesh: mesh.clone(),
        material: mat.clone(),
        transform: Transform::from_xyz(0.5, 0., -5.).with_scale(Vec3::splat(2.)),
        ..default()
    });

    // Plane 2
    commands.spawn(PbrBundle {
        mesh: mesh.clone(),
        material: mat.clone(),
        transform: Transform::from_xyz(4., 1.5, 0.),
        ..default()
    });

    // Plane 3
    commands.spawn(PbrBundle {
        mesh: mesh.clone(),
        material: mat.clone(),
        transform: Transform::from_xyz(1.5, 1., -20.).with_scale(Vec3::splat(1.5)),
        ..default()
    });

    let mesh = meshes.add(Sphere::default().mesh().ico(3).unwrap());

    // Sphere
    commands.spawn((
        PbrBundle {
            mesh,
            material: mat.clone(),
            transform: Transform::from_xyz(1.5, -1., 0.),
            ..default()
        },
        Offset {
            t: 0.,
            scale: 0.5,
            speed: 1.,
        },
    ));

    let mesh = meshes.add(Cuboid::default());

    // Cube
    commands.spawn(PbrBundle {
        mesh,
        material: mat.clone(),
        transform: Transform::from_xyz(5., -1., -5.),
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
            intensity: 0.5,
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

    // TODO: Spawn moons
    // let moon2d = sd_moon(pos.xz - vec2<f32>(-0.5, -10.), -1., 4., 3.3);
    // var moon: Sdf;
    // moon.dist = op_extrude(pos.y + 2.25, moon2d, 0.05) - 0.03;
    // moon.mat = 1u;

    // let moon2_2d = sd_moon(pos.xz - vec2<f32>(2., -10.), -0.5, 1., 0.75);
    // var moon2: Sdf;
    // moon2.dist = op_extrude(pos.y + 2.15, moon2_2d, 0.2) - 0.03;
    // moon2.mat = 1u;

    // let moon3_2d = sd_moon(pos.xz - vec2<f32>(2.9, -9.), -0.4, 0.75, 0.5);
    // var moon3: Sdf;
    // moon3.dist = op_extrude(pos.y + 2.1, moon3_2d, 0.3) - 0.03;
    // moon3.mat = 1u;

    let cube = sdfs.add(Sdf3d::from(Cuboid::default()));
    let cube_material = mats.add(SdfMaterial {
        base_color: LinearRgba::BLACK.to_vec3(),
        emissive: LinearRgba::rgb(0., 1.5, 1.75).to_vec3(),
        reflective: 0.,
    });

    commands.spawn((
        TransformBundle::from_transform(Transform::from_xyz(0., -0.5, -10.)),
        cube,
        cube_material,
    ));

    let sphere = sdfs.add(Sdf3d::from(Sphere::default()));
    let sphere_material = mats.add(SdfMaterial {
        base_color: LinearRgba::gray(0.7).to_vec3(),
        emissive: LinearRgba::BLACK.to_vec3(),
        reflective: 0.,
    });

    for (pos, scale, speed) in [
        (vec3(3., -1.6, -15.), 0.44, 0.8),
        (vec3(-5., -1.3, -12.), 0.35, -1.),
        (vec3(6., -1.4, -9.), 0.3, -1.2),
        (vec3(-5., -1.5, -7.), 0.5, 0.4),
    ] {
        commands.spawn((
            TransformBundle::from_transform(
                Transform::from_translation(pos).with_scale(Vec3::splat(0.6)),
            ),
            sphere.clone(),
            sphere_material.clone(),
            Offset {
                t: 0.,
                scale,
                speed,
            },
        ));
    }

    let water_plane = sdfs.add(Sdf3d::from(InfinitePlane3d::default()));
    let water_material = mats.add(SdfMaterial {
        base_color: LinearRgba::rgb(0.5, 1., 0.5).to_vec3(),
        emissive: LinearRgba::BLACK.to_vec3(),
        reflective: 0.97,
    });

    commands.spawn((
        TransformBundle::from_transform(Transform::from_xyz(0., -2.25, 0.)),
        water_plane,
        water_material,
    ));
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

#[derive(Resource, Default, PartialEq, Eq)]
enum CursorState {
    #[default]
    Free,
    Locked,
}

fn grab_cursor(
    mut windows: Query<&mut Window>,
    keyboard_input: Res<ButtonInput<KeyCode>>,
    mouse_input: Res<ButtonInput<MouseButton>>,
    mut cursor_state: ResMut<CursorState>,
    mut text: Query<&mut Visibility, With<HelpText>>,
) {
    let grabbed = if keyboard_input.just_pressed(KeyCode::Escape) {
        false
    } else if mouse_input.just_pressed(MouseButton::Left) {
        true
    } else {
        return;
    };

    let Ok(mut window) = windows.get_single_mut() else {
        return;
    };
    let Ok(mut help_vis) = text.get_single_mut() else {
        return;
    };

    (window.cursor.grab_mode, *cursor_state, *help_vis) = if grabbed {
        (
            CursorGrabMode::Confined,
            CursorState::Locked,
            Visibility::Hidden,
        )
    } else {
        (
            CursorGrabMode::None,
            CursorState::Free,
            Visibility::Inherited,
        )
    };
    window.cursor.visible = !grabbed;
}

fn rotate_and_move(
    time: Res<Time>,
    mut cameras: Query<&mut Transform, With<Camera3d>>,
    keyboard_input: Res<ButtonInput<KeyCode>>,
    mut mouse_motion: EventReader<MouseMotion>,
    cursor_state: Res<CursorState>,
) {
    let rotation_input = -mouse_motion
        .read()
        .fold(Vec2::ZERO, |acc, mot| acc + mot.delta);
    let movement_input = Vec3::new(
        keyboard_input.pressed(KeyCode::KeyD) as u32 as f32
            - keyboard_input.pressed(KeyCode::KeyA) as u32 as f32,
        keyboard_input.pressed(KeyCode::Space) as u32 as f32
            - keyboard_input.pressed(KeyCode::ControlLeft) as u32 as f32,
        keyboard_input.pressed(KeyCode::KeyS) as u32 as f32
            - keyboard_input.pressed(KeyCode::KeyW) as u32 as f32,
    );

    if rotation_input.length_squared() < 0.001 && movement_input.length_squared() < 0.001 {
        return;
    }

    for mut transform in cameras.iter_mut() {
        let translation = movement_input * time.delta_seconds() * 5.;
        let translation = transform.rotation * translation;
        transform.translation += translation;
        transform.translation.y = transform.translation.y.max(-2.);

        if *cursor_state == CursorState::Locked {
            let mut euler = transform.rotation.to_euler(EulerRot::YXZ);
            euler.0 += rotation_input.x * 0.003;
            euler.1 += rotation_input.y * 0.003;
            transform.rotation = Quat::from_euler(EulerRot::YXZ, euler.0, euler.1, 0.);
        }
    }
}

fn rotate_light(time: Res<Time>, mut lights: Query<&mut Transform, With<DirectionalLight>>) {
    for mut transform in lights.iter_mut() {
        let mut euler = transform.rotation.to_euler(EulerRot::YXZ);
        euler.0 += 0.2 * time.delta_seconds();
        transform.rotation = Quat::from_euler(EulerRot::YXZ, euler.0, euler.1, euler.2);
    }
}

#[derive(Component)]
struct Offset {
    t: f32,
    scale: f32,
    speed: f32,
}

fn update_offsets(time: Res<Time>, mut spheres: Query<(&mut Transform, &mut Offset)>) {
    for (mut transform, mut offset) in spheres.iter_mut() {
        // Remove old offset
        transform.translation.y -= offset.t.sin() * offset.speed;

        // Calculate and apply new offset
        offset.t += offset.scale * time.delta_seconds();
        transform.translation.y += offset.t.sin() * offset.speed;
    }
}
