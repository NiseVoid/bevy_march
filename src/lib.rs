use std::marker::PhantomData;

use bevy::{
    asset::embedded_asset,
    prelude::*,
    render::{
        render_resource::{encase::internal::WriteInto, *},
        RenderApp,
    },
};
use bevy_prototype_sdf::Sdf3d;

// TODO: We only need to re-render if any of the buffers change, the light changes, or the camera is moved
// TODO: Document everything and warn(missing_docs)

mod buffers;
use buffers::BufferPlugin;

mod cone_pass;
use cone_pass::ConePassPlugin;
pub use cone_pass::MarcherConeTexture;

mod main_pass;
pub use main_pass::MarcherMainTextures;
use main_pass::{MainPassPlugin, MainPassShader};

mod shadow_pass;
use ploc_bvh::dim3::BoundingVolume;
pub use shadow_pass::{MarcherShadowSettings, MarcherShadowTextures};

mod settings;
pub use settings::MarcherSettings;
use settings::SettingsPlugin;

mod writeback;
use writeback::WritebackPlugin;

pub trait MarcherMaterial: Asset + ShaderType + WriteInto + std::fmt::Debug + Clone {}

#[derive(Component, Deref, DerefMut)]
pub struct MarcherScale(pub u8);

impl Default for MarcherScale {
    fn default() -> Self {
        Self(1)
    }
}

pub struct RayMarcherPlugin<Material: MarcherMaterial> {
    shader: Handle<Shader>,
    _phantom: PhantomData<Material>,
}

impl<Material: MarcherMaterial> RayMarcherPlugin<Material> {
    pub fn new(shader: Handle<Shader>) -> Self {
        Self {
            shader,
            _phantom: PhantomData,
        }
    }
}

impl<Material: MarcherMaterial> Plugin for RayMarcherPlugin<Material> {
    fn build(&self, app: &mut App) {
        embedded_asset!(app, "sdf_marcher.wgsl");
        std::mem::forget(
            app.world()
                .resource::<AssetServer>()
                .load::<Shader>("embedded://bevy_march/sdf_marcher.wgsl"),
        );

        app.add_systems(
            Last,
            bvh_gizmos.after(buffers::upload_new_buffers::<Material>),
        );

        app.init_asset::<Sdf3d>()
            .init_asset::<Material>()
            .add_plugins((
                BufferPlugin::<Material>::default(),
                ConePassPlugin,
                MainPassPlugin,
                SettingsPlugin,
                WritebackPlugin,
            ))
            .sub_app_mut(RenderApp)
            .insert_resource(MainPassShader(self.shader.clone()));
    }
}

const WORKGROUP_SIZE: u32 = 8;
const CONE_SIZE: u32 = 8;

fn bvh_gizmos(mut gizmos: Gizmos, bvh: Res<buffers::CurrentBvh>) {
    for node in bvh.nodes() {
        if node.count == 1 {
            continue;
        }
        gizmos.primitive_3d(
            &Cuboid {
                half_size: node.volume.half_size().into(),
            },
            node.volume.center().into(),
            default(),
            Color::srgb(0., 0., 1.),
        );
    }

    for item in bvh.items() {
        gizmos.primitive_3d(
            &Cuboid {
                half_size: item.volume.half_size().into(),
            },
            item.volume.center().into(),
            default(),
            Color::srgb(1., 0., 0.),
        );
    }
}
