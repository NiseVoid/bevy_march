use bevy::{
    math::bounding::Aabb2d,
    prelude::*,
    render::{
        extract_component::ExtractComponent,
        render_asset::RenderAssetUsages,
        render_resource::{Extent3d, ShaderType, TextureDimension, TextureFormat, TextureUsages},
    },
};

#[derive(Component, ShaderType, ExtractComponent, Clone, Copy, Default, Debug)]
pub struct MarcherShadowSettings {
    origin: Vec3,
    direction: Vec3,
    t: f32,
    aspect_ratio: f32,
    near: f32,
    far: f32,
}

// TODO
#[allow(dead_code)]
#[derive(Component, ExtractComponent, Clone)]
pub struct MarcherShadowArea {
    current: Aabb2d,
    desired: Aabb2d,
}

#[derive(Component, ExtractComponent, Clone)]
pub struct MarcherShadowTextures {
    pub shadow_map: Handle<Image>,
}

impl MarcherShadowTextures {
    pub fn new(images: &mut Assets<Image>) -> Self {
        let mut shadow_map = Image::new_fill(
            Extent3d {
                width: 2048,
                height: 2048,
                depth_or_array_layers: 1,
            },
            TextureDimension::D2,
            &1f32.to_ne_bytes(),
            TextureFormat::R32Float,
            RenderAssetUsages::RENDER_WORLD,
        );
        shadow_map.texture_descriptor.usage = TextureUsages::COPY_DST
            | TextureUsages::STORAGE_BINDING
            | TextureUsages::TEXTURE_BINDING;
        let shadow_map = images.add(shadow_map);

        Self { shadow_map }
    }
}
