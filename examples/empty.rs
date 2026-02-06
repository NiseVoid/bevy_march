use bevy_march::*;

use bevy::{
    diagnostic::{DiagnosticsStore, FrameTimeDiagnosticsPlugin},
    prelude::*,
    render::{render_resource::ShaderType, renderer::RenderDevice, view::Hdr},
    sprite::Anchor,
};

fn make_single_threaded(schedule: &mut Schedule) {
    schedule.set_executor_kind(bevy::ecs::schedule::ExecutorKind::SingleThreaded);
}

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
        FrameTimeDiagnosticsPlugin::default(),
    ))
    .edit_schedule(PreUpdate, make_single_threaded)
    .edit_schedule(Update, make_single_threaded)
    .edit_schedule(PostUpdate, make_single_threaded);

    let main_pass_shader = app.world().resource::<AssetServer>().load("simple.wgsl");

    app.add_plugins(RayMarcherPlugin::<SdfMaterial>::new(main_pass_shader))
        .add_systems(Startup, setup)
        .add_systems(Update, update_fps)
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

fn setup(mut commands: Commands, mut images: ResMut<Assets<Image>>, device: Res<RenderDevice>) {
    commands.spawn((
        Text2d::default(),
        TextFont {
            font_size: 18.0,
            ..default()
        },
        Anchor::TOP_LEFT,
        FpsText,
    ));
    commands.spawn((
        Camera2d,
        Hdr,
        Camera {
            order: 1,
            clear_color: ClearColorConfig::None,
            ..default()
        },
    ));

    commands.spawn((
        Camera3d::default(),
        // TODO: We should not need HDR for a minimal setup
        Hdr,
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
