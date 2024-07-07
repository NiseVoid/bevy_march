use crate::shadow_pass::{MarcherShadowArea, MarcherShadowSettings};

use bevy::{
    math::{bounding::Aabb3d, vec3a},
    prelude::*,
    render::{
        extract_component::{ExtractComponent, ExtractComponentPlugin, UniformComponentPlugin},
        render_resource::ShaderType,
        view::RenderLayers,
    },
};

pub struct SettingsPlugin;

impl Plugin for SettingsPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins((
            ExtractComponentPlugin::<MarcherSettings>::default(),
            UniformComponentPlugin::<MarcherSettings>::default(),
        ))
        .add_systems(
            PostUpdate,
            update_settings.after(TransformSystem::TransformPropagate),
        );
    }
}

// TODO: Split settings between ones relevant for the main pass and all other passes
#[derive(Component, ShaderType, ExtractComponent, Clone, Copy, Default, Debug)]
pub struct MarcherSettings {
    pub origin: Vec3,
    pub rotation: Mat3,
    pub t: f32,
    pub aspect_ratio: f32,
    pub perspective_factor: f32,
    pub near: f32,
    pub far: f32,
    pub light_dir: Vec3,
}

fn update_settings(
    mut cameras: Query<(
        &Projection,
        &GlobalTransform,
        &mut MarcherSettings,
        Option<&RenderLayers>,
    )>,
    mut lights: Query<
        (
            Option<&mut MarcherShadowSettings>,
            Option<&mut MarcherShadowArea>,
            &GlobalTransform,
            Option<&RenderLayers>,
        ),
        With<DirectionalLight>,
    >,
    time: Res<Time<Virtual>>,
) {
    for (projection, transform, mut settings, layers) in cameras.iter_mut() {
        let Projection::Perspective(projection) = projection else {
            return;
        };
        let rotation = transform.affine().matrix3;
        settings.origin = transform.translation();
        settings.rotation = rotation.into();
        settings.t = time.elapsed_seconds_wrapped();
        settings.aspect_ratio = projection.aspect_ratio;
        settings.perspective_factor = 1. / (projection.fov / 2.).tan();
        settings.near = projection.near;
        settings.far = projection.far;

        let x = 0.5 * settings.aspect_ratio;
        let rotation = Quat::from_mat3a(&rotation);

        let tl_dir = Dir3A::new(vec3a(-x, 0.5, -settings.perspective_factor)).unwrap();
        let tr_dir = rotation * Dir3A::new_unchecked(vec3a(-tl_dir.x, tl_dir.y, tl_dir.z));
        let bl_dir = rotation * Dir3A::new_unchecked(vec3a(tl_dir.x, -tl_dir.y, tl_dir.z));
        let br_dir = rotation * Dir3A::new_unchecked(vec3a(-tl_dir.x, -tl_dir.y, tl_dir.z));
        let tl_dir = rotation * tl_dir;

        let fwd = rotation * Dir3::NEG_Z;
        let mut frustum_aabb = Aabb3d::new(fwd * settings.near, Vec3::ZERO);
        for vertex in [tl_dir, tr_dir, bl_dir, br_dir]
            .iter()
            .map(|&d| [d * settings.near, d * settings.far])
            .flatten()
        {
            frustum_aabb.min = frustum_aabb.min.min(vertex);
            frustum_aabb.max = frustum_aabb.max.max(vertex);
        }

        for (_, mut area, transform, light_layers) in lights.iter_mut() {
            if !light_layers
                .cloned()
                .unwrap_or_default()
                .intersects(&layers.cloned().unwrap_or_default())
            {
                continue;
            }
            settings.light_dir = transform.forward().into();
            _ = (&mut area, transform);
            // TODO
        }
        // TODO: Transform bounding box to camera location
        // TODO: Add up bounding boxes for all cameras that can see a directional light
    }
}
