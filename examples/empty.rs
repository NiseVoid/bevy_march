use bevy_march::*;

use bevy::{
    prelude::*,
    render::{render_resource::ShaderType, renderer::RenderDevice},
};

fn main() {
    let mut app = App::new();
    app.add_plugins(DefaultPlugins);

    let main_pass_shader = app.world().resource::<AssetServer>().load("simple.wgsl");

    app.add_plugins(RayMarcherPlugin::<SdfMaterial>::new(main_pass_shader))
        .add_systems(Startup, setup)
        .run();
}

#[derive(Asset, ShaderType, TypePath, Clone, Debug, Default)]
struct SdfMaterial {
    base_color: Vec3,
    emissive: Vec3,
}

impl MarcherMaterial for SdfMaterial {}

fn setup(mut commands: Commands, mut images: ResMut<Assets<Image>>, device: Res<RenderDevice>) {
    commands.spawn((
        Camera3d::default(),
        Camera {
            // TODO: We should not need HDR for a minimal setup
            hdr: true,
            ..default()
        },
        MarcherSettings::default(),
        MarcherMainTextures::new(&mut images, (8, 8)),
        MarcherConeTexture::new(&mut images, &device, (8, 8)),
    ));

    commands.spawn((
        Transform::from_xyz(1., 1.5, 1.).looking_at(Vec3::ZERO, Vec3::Y),
        DirectionalLight::default(),
        MarcherShadowSettings::default(),
        MarcherShadowTextures::new(&mut images),
    ));
}
