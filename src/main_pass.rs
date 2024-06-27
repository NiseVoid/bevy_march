use crate::{
    buffers::{BufferSet, Instance, MaterialSize},
    MarcherScale, BLOCK_SIZE,
};

use std::borrow::Cow;

use bevy::{
    core_pipeline::core_3d::graph::{Core3d, Node3d},
    ecs::query::QueryItem,
    prelude::*,
    render::{
        camera::RenderTarget,
        extract_component::{
            ComponentUniforms, ExtractComponent, ExtractComponentPlugin, UniformComponentPlugin,
        },
        render_asset::{RenderAssetUsages, RenderAssets},
        render_graph::{
            NodeRunError, RenderGraphApp, RenderGraphContext, RenderLabel, ViewNode, ViewNodeRunner,
        },
        render_resource::{
            binding_types::{
                storage_buffer_read_only, storage_buffer_read_only_sized, texture_storage_2d,
                uniform_buffer,
            },
            BindGroup, BindGroupEntries, BindGroupLayout, BindGroupLayoutEntries,
            CachedComputePipelineId, ComputePassDescriptor, ComputePipelineDescriptor, Extent3d,
            PipelineCache, ShaderStages, ShaderType, StorageTextureAccess, TextureDimension,
            TextureFormat, TextureUsages,
        },
        renderer::{RenderContext, RenderDevice},
        texture::GpuImage,
        Render, RenderApp, RenderSet,
    },
    window::{PrimaryWindow, WindowRef, WindowResized},
};

pub struct MainPassPlugin;

impl Plugin for MainPassPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins((
            ExtractComponentPlugin::<MarcherSettings>::default(),
            UniformComponentPlugin::<MarcherSettings>::default(),
            ExtractComponentPlugin::<MarcherMainTextures>::default(),
        ))
        .add_systems(Last, resize_textures);

        app.sub_app_mut(RenderApp)
            .add_systems(
                Render,
                prepare_bind_group.in_set(RenderSet::PrepareBindGroups),
            )
            .add_render_graph_node::<ViewNodeRunner<MarcherMainPass>>(Core3d, MarcherMainPass)
            .add_render_graph_edges(Core3d, (MarcherMainPass, Node3d::StartMainPass));
    }

    fn finish(&self, app: &mut App) {
        app.sub_app_mut(RenderApp)
            .init_resource::<RayMarcherPipeline>();
    }
}

#[derive(Component, ShaderType, ExtractComponent, Clone, Copy, Default, Debug)]
pub struct MarcherSettings {
    pub origin: Vec3,
    pub rotation: Mat3,
    pub t: f32,
    pub aspect_ratio: f32,
    pub perspective_factor: f32,
    pub near: f32,
    pub far: f32,
}

// TODO: Use gpu textures instead of Image handles?
#[derive(Component, ExtractComponent, Clone)]
pub struct MarcherMainTextures {
    pub depth: Handle<Image>,
    pub color: Handle<Image>,
}

impl MarcherMainTextures {
    pub fn new(images: &mut Assets<Image>, size: (u32, u32)) -> Self {
        let mut depth = Image::new_fill(
            Extent3d {
                width: size.0,
                height: size.1,
                depth_or_array_layers: 1,
            },
            TextureDimension::D2,
            &[0u8; 4],
            TextureFormat::R32Float,
            RenderAssetUsages::RENDER_WORLD,
        );
        depth.texture_descriptor.usage = TextureUsages::COPY_DST
            | TextureUsages::STORAGE_BINDING
            | TextureUsages::TEXTURE_BINDING;
        let depth = images.add(depth);

        let mut color = Image::new_fill(
            Extent3d {
                width: size.0,
                height: size.1,
                depth_or_array_layers: 1,
            },
            TextureDimension::D2,
            &[0; 8],
            TextureFormat::Rgba16Float,
            RenderAssetUsages::RENDER_WORLD,
        );
        color.texture_descriptor.usage = TextureUsages::COPY_DST
            | TextureUsages::STORAGE_BINDING
            | TextureUsages::TEXTURE_BINDING;
        let color = images.add(color);

        Self { color, depth }
    }
}

fn resize_textures(
    mut resized: EventReader<WindowResized>,
    mut textures: Query<(&Camera, &mut MarcherMainTextures, Option<&MarcherScale>)>,
    windows: Query<&Window>,
    primary_window: Query<&Window, With<PrimaryWindow>>,
    mut images: ResMut<Assets<Image>>,
) {
    if resized.read().last().is_none() {
        return;
    }

    for (camera, mut textures, scale) in textures.iter_mut() {
        let RenderTarget::Window(window) = camera.target else {
            continue;
        };
        let Some(window) = (match window {
            WindowRef::Primary => primary_window.get_single().ok(),
            WindowRef::Entity(e) => windows.get(e).ok(),
        }) else {
            continue;
        };

        let scale = scale.map(|s| **s).unwrap_or(1) as u32;
        let new_width = window.physical_width() / scale;
        let new_height = window.physical_height() / scale;

        *textures = MarcherMainTextures::new(&mut images, (new_width, new_height));
    }
}

#[derive(RenderLabel, Clone, PartialEq, Eq, Hash, Default, Debug)]
pub struct MarcherMainPass;

impl ViewNode for MarcherMainPass {
    type ViewQuery = (&'static MarcherSettings, &'static MarcherStorageBindGroup);

    fn run(
        &self,
        _graph: &mut RenderGraphContext,
        render_context: &mut RenderContext,
        (_settings, texture_bind_group): QueryItem<Self::ViewQuery>,
        world: &World,
    ) -> Result<(), NodeRunError> {
        let ray_marcher_pipeline = world.resource::<RayMarcherPipeline>();
        let pipeline_cache = world.resource::<PipelineCache>();
        let Some(pipeline) = pipeline_cache.get_compute_pipeline(ray_marcher_pipeline.pipeline_id)
        else {
            return Ok(());
        };

        let settings_uniforms = world.resource::<ComponentUniforms<MarcherSettings>>();
        let Some(settings_binding) = settings_uniforms.uniforms().binding() else {
            return Ok(());
        };

        let settings_bind_group = render_context.render_device().create_bind_group(
            "marcher_settings_bind_group",
            &ray_marcher_pipeline.settings_layout,
            &BindGroupEntries::sequential((settings_binding.clone(),)),
        );

        let mut pass = render_context
            .command_encoder()
            .begin_compute_pass(&ComputePassDescriptor::default());

        pass.set_bind_group(0, &settings_bind_group, &[]);
        pass.set_bind_group(1, &texture_bind_group.bind_group, &[]);
        pass.set_pipeline(pipeline);
        pass.dispatch_workgroups(
            texture_bind_group.size.x.div_ceil(BLOCK_SIZE),
            texture_bind_group.size.y.div_ceil(BLOCK_SIZE),
            1,
        );

        Ok(())
    }
}

// TODO: Split data shared between passes and data that's unique to each pass
#[derive(Component)]
pub struct MarcherStorageBindGroup {
    bind_group: BindGroup,
    size: UVec2,
}

fn prepare_bind_group(
    mut commands: Commands,
    pipeline: Res<RayMarcherPipeline>,
    gpu_images: Res<RenderAssets<GpuImage>>,
    textures: Query<(Entity, &MarcherMainTextures)>,
    buffer_set: Res<BufferSet>,
    render_device: Res<RenderDevice>,
) {
    for (entity, textures) in textures.iter() {
        let Some(color) = gpu_images.get(textures.color.id()) else {
            continue;
        };
        let Some(depth) = gpu_images.get(textures.depth.id()) else {
            continue;
        };
        let bind_group = render_device.create_bind_group(
            None,
            &pipeline.storage_layout,
            &BindGroupEntries::sequential((
                &depth.texture_view,
                &color.texture_view,
                buffer_set.sdfs.as_entire_binding(),
                buffer_set.materials.as_entire_binding(),
                buffer_set.instances.as_entire_binding(),
            )),
        );
        commands.entity(entity).insert(MarcherStorageBindGroup {
            bind_group,
            size: color.size,
        });
    }
}

#[derive(Resource)]
struct RayMarcherPipeline {
    settings_layout: BindGroupLayout,
    storage_layout: BindGroupLayout,
    pipeline_id: CachedComputePipelineId,
}

impl FromWorld for RayMarcherPipeline {
    fn from_world(world: &mut World) -> Self {
        let mat_size = **world.resource::<MaterialSize>();
        let render_device = world.resource::<RenderDevice>();

        let settings_layout = render_device.create_bind_group_layout(
            "marcher_settings_bind_group_layout",
            &BindGroupLayoutEntries::sequential(
                ShaderStages::COMPUTE,
                (uniform_buffer::<MarcherSettings>(false),),
            ),
        );
        let storage_layout = render_device.create_bind_group_layout(
            "marcher_storage_bind_group_layout",
            &BindGroupLayoutEntries::sequential(
                ShaderStages::COMPUTE,
                (
                    // Depth texture
                    texture_storage_2d(TextureFormat::R32Float, StorageTextureAccess::WriteOnly),
                    // Color texture
                    texture_storage_2d(TextureFormat::Rgba16Float, StorageTextureAccess::WriteOnly),
                    // SDFs
                    storage_buffer_read_only::<u32>(false),
                    // Materials
                    storage_buffer_read_only_sized(false, Some(mat_size)),
                    // Instances
                    storage_buffer_read_only::<Instance>(false),
                ),
            ),
        );

        let shader = world.resource::<AssetServer>().load("main_pass.wgsl");

        let pipeline_id = world
            .resource_mut::<PipelineCache>()
            .queue_compute_pipeline(ComputePipelineDescriptor {
                label: Some("ray_marcher_pipeline".into()),
                layout: vec![settings_layout.clone(), storage_layout.clone()],
                push_constant_ranges: Vec::new(),
                shader: shader.clone(),
                shader_defs: vec![],
                entry_point: Cow::from("march"),
            });

        Self {
            settings_layout,
            storage_layout,
            pipeline_id,
        }
    }
}
