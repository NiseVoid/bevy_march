use std::{borrow::Cow, marker::PhantomData};

use bevy::{
    core_pipeline::{
        bloom::BloomSettings,
        core_3d::graph::{Core3d, Node3d},
    },
    ecs::query::QueryItem,
    math::{
        bounding::{Aabb2d, Aabb3d},
        Vec3A,
    },
    prelude::*,
    render::{
        extract_component::{
            ComponentUniforms, ExtractComponent, ExtractComponentPlugin, UniformComponentPlugin,
        },
        render_asset::{RenderAssetUsages, RenderAssets},
        render_graph::{
            NodeRunError, RenderGraphApp, RenderGraphContext, RenderLabel, ViewNode, ViewNodeRunner,
        },
        render_resource::{
            binding_types::{texture_storage_2d, uniform_buffer},
            *,
        },
        renderer::{RenderContext, RenderDevice},
        texture::GpuImage,
        view::RenderLayers,
        Render, RenderApp, RenderSet,
    },
};

fn main() {
    App::new()
        // TODO: Pass in shader handle for main pass
        .add_plugins((DefaultPlugins, RayMarcherPlugin::<SdfMaterial>::default()))
        .add_systems(Startup, setup)
        .add_systems(PostUpdate, update_settings)
        .run();
}

#[derive(Asset, TypePath, Default)]
struct SdfMaterial {
    base_color: LinearRgba,
    emissive: LinearRgba,
    reflective: f32,
}

/// It is generally encouraged to set up post processing effects as a plugin
#[derive(Default)]
struct RayMarcherPlugin<Material: Asset> {
    // TODO: Store shader handle for main pass compute shader
    _phantom: PhantomData<Material>,
}

impl<Material: Asset> Plugin for RayMarcherPlugin<Material> {
    fn build(&self, app: &mut App) {
        // TODO: More descriptive panic if we already have another RayMarcherPlugin
        app.add_plugins((
            ExtractComponentPlugin::<MarcherSettings>::default(),
            UniformComponentPlugin::<MarcherSettings>::default(),
            ExtractComponentPlugin::<MarcherMainTextures>::default(),
        ));

        app.sub_app_mut(RenderApp)
            .add_systems(
                Render,
                prepare_bind_group.in_set(RenderSet::PrepareBindGroups),
            )
            .add_render_graph_node::<ViewNodeRunner<RayMarcherNode>>(Core3d, RayMarcherLabel)
            .add_render_graph_edges(Core3d, (RayMarcherLabel, Node3d::EndPrepasses));
    }

    fn finish(&self, app: &mut App) {
        app.sub_app_mut(RenderApp)
            .init_resource::<RayMarcherPipeline>();
    }
}

// TODO: Use resolution or a integer division of it instead of a hardcoded 1280x720
const SIZE: (u32, u32) = (1280, 720);
const WORKGROUP_SIZE: u32 = 8;
const CONE_SIZE: u32 = 8;
const BLOCK_SIZE: u32 = WORKGROUP_SIZE * CONE_SIZE;

#[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
struct RayMarcherLabel;

// The post process node used for the render graph
#[derive(Default)]
struct RayMarcherNode;

impl ViewNode for RayMarcherNode {
    type ViewQuery = (&'static MarcherSettings, &'static MarcherTextureBindGroup);

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

        // Get the settings uniform binding
        let settings_uniforms = world.resource::<ComponentUniforms<MarcherSettings>>();
        let Some(settings_binding) = settings_uniforms.uniforms().binding() else {
            return Ok(());
        };

        let settings_bind_group = render_context.render_device().create_bind_group(
            "marcher_settings_bind_group",
            &ray_marcher_pipeline.settings_layout,
            &BindGroupEntries::sequential((
                // TODO: Pass in shapes
                settings_binding.clone(),
            )),
        );

        let mut pass = render_context
            .command_encoder()
            .begin_compute_pass(&ComputePassDescriptor::default());

        pass.set_bind_group(0, &settings_bind_group, &[]);
        pass.set_bind_group(1, &texture_bind_group.0, &[]);
        pass.set_pipeline(pipeline);
        pass.dispatch_workgroups(SIZE.0.div_ceil(BLOCK_SIZE), SIZE.1.div_ceil(BLOCK_SIZE), 1);

        Ok(())
    }
}

#[derive(Component)]
struct MarcherTextureBindGroup(BindGroup);

fn prepare_bind_group(
    mut commands: Commands,
    pipeline: Res<RayMarcherPipeline>,
    gpu_images: Res<RenderAssets<GpuImage>>,
    textures: Query<(Entity, &MarcherMainTextures)>,
    render_device: Res<RenderDevice>,
) {
    for (entity, textures) in textures.iter() {
        let depth = gpu_images.get(textures.depth.id()).unwrap();
        let color = gpu_images.get(textures.color.id()).unwrap();
        let bind_group = render_device.create_bind_group(
            None,
            &pipeline.textures_layout,
            &BindGroupEntries::sequential((&depth.texture_view, &color.texture_view)),
        );
        commands
            .entity(entity)
            .insert(MarcherTextureBindGroup(bind_group));
    }
}

#[derive(Resource)]
struct RayMarcherPipeline {
    settings_layout: BindGroupLayout,
    textures_layout: BindGroupLayout,
    pipeline_id: CachedComputePipelineId,
}

impl FromWorld for RayMarcherPipeline {
    fn from_world(world: &mut World) -> Self {
        let render_device = world.resource::<RenderDevice>();

        let settings_layout = render_device.create_bind_group_layout(
            "marcher_settings_bind_group_layout",
            &BindGroupLayoutEntries::sequential(
                ShaderStages::COMPUTE,
                (uniform_buffer::<MarcherSettings>(false),),
            ),
        );
        let textures_layout = render_device.create_bind_group_layout(
            "marcher_settings_bind_group_layout",
            &BindGroupLayoutEntries::sequential(
                ShaderStages::COMPUTE,
                (
                    texture_storage_2d(TextureFormat::R32Float, StorageTextureAccess::WriteOnly),
                    texture_storage_2d(TextureFormat::Rgba16Float, StorageTextureAccess::WriteOnly),
                ),
            ),
        );

        let shader = world.resource::<AssetServer>().load("main_pass.wgsl");

        let pipeline_id = world
            .resource_mut::<PipelineCache>()
            .queue_compute_pipeline(ComputePipelineDescriptor {
                label: Some("ray_marcher_pipeline".into()),
                layout: vec![settings_layout.clone(), textures_layout.clone()],
                push_constant_ranges: Vec::new(),
                shader: shader.clone(),
                shader_defs: vec![],
                entry_point: Cow::from("march"),
            });

        Self {
            settings_layout,
            textures_layout,
            pipeline_id,
        }
    }
}

#[derive(Component, ShaderType, ExtractComponent, Clone, Copy, Default, Debug)]
struct MarcherSettings {
    origin: Vec3,
    rotation: Mat3,
    t: f32,
    aspect_ratio: f32,
    perspective_factor: f32,
    near: f32,
    far: f32,
}

#[derive(Component, ExtractComponent, Clone)]
pub struct MarcherMainTextures {
    depth: Handle<Image>,
    color: Handle<Image>,
}

impl MarcherMainTextures {
    fn new(images: &mut Assets<Image>) -> Self {
        let mut depth = Image::new_fill(
            Extent3d {
                width: 1280,
                height: 720,
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
                width: 1280,
                height: 720,
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

#[derive(Component, ShaderType, ExtractComponent, Clone, Copy, Default, Debug)]
struct MarcherShadowSettings {
    origin: Vec3,
    direction: Vec3,
    t: f32,
    aspect_ratio: f32,
    near: f32,
    far: f32,
}

#[derive(Component, ExtractComponent, Clone)]
pub struct MarcherShadowTextures {
    shadow_map: Handle<Image>,
}

// TODO
#[allow(dead_code)]
#[derive(Component, ExtractComponent, Clone)]
pub struct MarcherShadowArea {
    current: Aabb2d,
    desired: Aabb2d,
}

impl MarcherShadowTextures {
    fn new(images: &mut Assets<Image>) -> Self {
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

fn setup(
    mut commands: Commands,
    mut images: ResMut<Assets<Image>>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut mats: ResMut<Assets<StandardMaterial>>,
) {
    let main_textures = MarcherMainTextures::new(&mut images);
    let shadow_textures = MarcherShadowTextures::new(&mut images);

    let mesh = meshes.add(Plane3d::new(Vec3::Z, Vec2::ONE));

    // Color plane
    commands.spawn(PbrBundle {
        mesh: mesh.clone(),
        material: mats.add(StandardMaterial {
            base_color_texture: Some(main_textures.color.clone()),
            ..default()
        }),
        ..default()
    });

    // Depth plane
    commands.spawn(PbrBundle {
        mesh: mesh.clone(),
        material: mats.add(StandardMaterial {
            base_color_texture: Some(main_textures.depth.clone()),
            ..default()
        }),
        transform: Transform::from_xyz(2., 0., 0.),
        ..default()
    });

    // Shadow plane
    commands.spawn(PbrBundle {
        mesh,
        material: mats.add(StandardMaterial {
            base_color_texture: Some(shadow_textures.shadow_map.clone()),
            ..default()
        }),
        transform: Transform::from_xyz(1., 2., 0.),
        ..default()
    });

    // Camera
    commands.spawn((
        Camera3dBundle {
            transform: Transform::from_translation(Vec3::new(0.0, 0.0, 5.0))
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
            &mut MarcherShadowSettings,
            &mut MarcherShadowArea,
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

        let tl_dir = Dir3A::new(Vec3A::new(-x, 0.5, -settings.perspective_factor)).unwrap();
        let tr_dir = rotation * Dir3A::new_unchecked(Vec3A::new(-tl_dir.x, tl_dir.y, tl_dir.z));
        let bl_dir = rotation * Dir3A::new_unchecked(Vec3A::new(tl_dir.x, -tl_dir.y, tl_dir.z));
        let br_dir = rotation * Dir3A::new_unchecked(Vec3A::new(-tl_dir.x, -tl_dir.y, tl_dir.z));
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
            _ = (&mut area, transform);
            // TODO
        }
        // TODO: Transform bounding box to camera location
        // TODO: Add up bounding boxes for all cameras that can see a directional light
    }
}

// TODO: Get AABB for camera's view frustom in the local space of the light
// Use the size of the AABB to determine the orthographic projection params
// Translate the start of the projection back to worldspace and use it as the origin
