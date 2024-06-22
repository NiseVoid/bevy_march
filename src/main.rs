use std::marker::PhantomData;

use bevy::{
    core_pipeline::bloom::BloomSettings,
    math::vec3,
    prelude::*,
    render::render_resource::{encase::internal::WriteInto, *},
};
use bevy_prototype_sdf::Sdf3d;

// TODO: We only need to re-render if any of the buffers change, the light changes, or the camera is moved

mod buffers;
use buffers::BufferPlugin;

mod main_pass;
use main_pass::{MainPassPlugin, MarcherMainTextures, MarcherSettings};

mod shadow_pass;
use shadow_pass::{MarcherShadowSettings, MarcherShadowTextures};

mod settings;
use settings::SettingsPlugin;

pub trait MarcherMaterial: Asset + ShaderType + WriteInto + std::fmt::Debug + Clone {}

/// It is generally encouraged to set up post processing effects as a plugin
#[derive(Default)]
struct RayMarcherPlugin<Material: MarcherMaterial> {
    // TODO: Store shader handle for main pass compute shader
    _phantom: PhantomData<Material>,
}

impl<Material: MarcherMaterial> Plugin for RayMarcherPlugin<Material> {
    fn build(&self, app: &mut App) {
        app.init_asset::<Sdf3d>()
            .init_asset::<Material>()
            .add_plugins((
                BufferPlugin::<Material>::default(),
                MainPassPlugin,
                SettingsPlugin,
            ));
    }
}

// TODO: Use resolution or a integer division of it instead of a hardcoded resolution

const WORKGROUP_SIZE: u32 = 8;
const CONE_SIZE: u32 = 8;
const BLOCK_SIZE: u32 = WORKGROUP_SIZE * CONE_SIZE;

// TODO: Turn the main code below into an example

fn main() {
    App::new()
        // TODO: Pass in shader handle for main pass
        .add_plugins((DefaultPlugins, RayMarcherPlugin::<SdfMaterial>::default()))
        .add_systems(Startup, setup)
        .run();
}

#[derive(Asset, ShaderType, TypePath, Clone, Debug, Default)]
struct SdfMaterial {
    base_color: Vec3,
    emissive: Vec3,
    reflective: f32,
}

impl MarcherMaterial for SdfMaterial {}

fn setup(
    mut commands: Commands,
    mut images: ResMut<Assets<Image>>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut std_mats: ResMut<Assets<StandardMaterial>>,
    mut sdfs: ResMut<Assets<Sdf3d>>,
    mut mats: ResMut<Assets<SdfMaterial>>,
) {
    let main_textures = MarcherMainTextures::new(&mut images);
    let shadow_textures = MarcherShadowTextures::new(&mut images);

    let mesh = meshes.add(Plane3d::new(Vec3::Z, Vec2::ONE));

    // Color plane
    commands.spawn(PbrBundle {
        mesh: mesh.clone(),
        material: std_mats.add(StandardMaterial {
            base_color_texture: Some(main_textures.color.clone()),
            ..default()
        }),
        ..default()
    });

    // Depth plane
    commands.spawn(PbrBundle {
        mesh: mesh.clone(),
        material: std_mats.add(StandardMaterial {
            base_color_texture: Some(main_textures.depth.clone()),
            ..default()
        }),
        transform: Transform::from_xyz(2., 0., 0.),
        ..default()
    });

    // Shadow plane
    commands.spawn(PbrBundle {
        mesh,
        material: std_mats.add(StandardMaterial {
            base_color_texture: Some(shadow_textures.shadow_map.clone()),
            ..default()
        }),
        transform: Transform::from_xyz(1., 2., 0.),
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
            ..default()
        },
        MarcherSettings::default(),
        main_textures,
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
        shadow_textures,
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
        emissive: LinearRgba::rgb(0., 1.8, 2.).to_vec3(),
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

    for pos in [
        vec3(3., -1.6, -15.),
        vec3(-5., -1.3, -12.),
        vec3(6., -1.4, -9.),
        vec3(-5., -1.5, -7.),
    ] {
        commands.spawn((
            TransformBundle::from_transform(
                Transform::from_translation(pos).with_scale(Vec3::splat(0.6)),
            ),
            sphere.clone(),
            sphere_material.clone(),
        ));
    }

    // TODO
    // let water_plane = sdfs.add(Sdf3d::from(InfinitePlane3d::default()));
    let water_material = mats.add(SdfMaterial {
        base_color: LinearRgba::rgb(0., 0.3, 1.).to_vec3(),
        emissive: LinearRgba::BLACK.to_vec3(),
        reflective: 0.7,
    });
    std::mem::forget(water_material);

    // commands.spawn((
    //     TransformBundle::from_transform(Transform::from_xyz(0., -2.25, 0.)),
    //     water_plane,
    //     water_material,
    // ));
}
