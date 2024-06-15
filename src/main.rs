use std::{borrow::Cow, marker::PhantomData, num::NonZeroU64};

use bevy::{
    core_pipeline::{
        bloom::BloomSettings,
        core_3d::graph::{Core3d, Node3d},
    },
    ecs::query::QueryItem,
    math::{
        bounding::{Aabb2d, Aabb3d},
        vec3, vec3a,
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
            binding_types::{
                storage_buffer_read_only, storage_buffer_read_only_sized, texture_storage_2d,
                uniform_buffer,
            },
            encase::internal::WriteInto,
            *,
        },
        renderer::{RenderContext, RenderDevice, RenderQueue},
        texture::GpuImage,
        view::RenderLayers,
        Extract, Render, RenderApp, RenderSet,
    },
};
use bevy_prototype_sdf::Sdf3d;

// TODO: We only need to re-render if any of the buffers change, the light changes, or the camera is moved

fn main() {
    App::new()
        // TODO: Pass in shader handle for main pass
        .add_plugins((DefaultPlugins, RayMarcherPlugin::<SdfMaterial>::default()))
        .add_systems(Startup, setup)
        .add_systems(PostUpdate, update_settings)
        .run();
}

#[derive(Asset, ShaderType, TypePath, Clone, Debug, Default)]
struct SdfMaterial {
    base_color: Vec3,
    emissive: Vec3,
    reflective: f32,
}

impl MarcherMaterial for SdfMaterial {}

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
                // TODO: More descriptive panic if we already have another RayMarcherPlugin
                ExtractComponentPlugin::<MarcherSettings>::default(),
                UniformComponentPlugin::<MarcherSettings>::default(),
                ExtractComponentPlugin::<MarcherMainTextures>::default(),
            ))
            .add_systems(PostUpdate, upload_new_buffers::<Material>)
            .init_resource::<SdfIndices>()
            .init_resource::<MaterialIndices>();

        app.sub_app_mut(RenderApp)
            .insert_resource(MaterialSize(Material::min_size()))
            .add_systems(ExtractSchedule, extract_buffers)
            .add_systems(
                Render,
                prepare_bind_group.in_set(RenderSet::PrepareBindGroups),
            )
            .add_render_graph_node::<ViewNodeRunner<RayMarcherNode>>(Core3d, RayMarcherLabel)
            .add_render_graph_edges(Core3d, (RayMarcherLabel, Node3d::EndPrepasses));
    }

    fn finish(&self, app: &mut App) {
        app.init_resource::<Buffers>()
            .sub_app_mut(RenderApp)
            .init_resource::<BufferSet>()
            .init_resource::<RayMarcherPipeline>();
    }
}

#[derive(Resource)]
pub struct Buffers {
    current: BufferSet,
    new: Option<BufferSet>,
}

impl FromWorld for Buffers {
    fn from_world(world: &mut World) -> Self {
        Self {
            current: BufferSet::from_world(world),
            new: None,
        }
    }
}

#[derive(Resource, Clone, Debug)]
pub struct BufferSet {
    sdfs: Buffer,
    materials: Buffer,
    instances: Buffer,
}

impl FromWorld for BufferSet {
    fn from_world(world: &mut World) -> Self {
        let render_device = world.resource::<RenderDevice>();

        let empty_buffer = render_device.create_buffer_with_data(&BufferInitDescriptor {
            usage: BufferUsages::STORAGE,
            label: Some("Empty"),
            contents: &[0, 0, 0, 0],
        });

        Self {
            sdfs: empty_buffer.clone(),
            materials: empty_buffer.clone(),
            instances: empty_buffer,
        }
    }
}

#[derive(Resource, Deref, DerefMut, Default)]
pub struct SdfIndices(Vec<u32>);

#[derive(Resource, Deref, DerefMut, Default)]
pub struct MaterialIndices(Vec<u32>);

// TODO: This type becomes 80 bytes on the GPU, investigate performance of alternatives
#[derive(ShaderType)]
struct Instance {
    sdf_index: u32,
    mat_index: u32,
    scale: f32,
    translation: Vec3,
    rotation: Mat3,
}

// TODO: We can probably split this up into three systems each with different scheduling constraints
fn upload_new_buffers<Material: MarcherMaterial>(
    mut buffers: ResMut<Buffers>,
    mut sdf_events: EventReader<AssetEvent<Sdf3d>>,
    sdfs: Res<Assets<Sdf3d>>,
    mut sdf_indices: ResMut<SdfIndices>,
    mut mat_events: EventReader<AssetEvent<Material>>,
    mats: ResMut<Assets<Material>>,
    mut mat_indices: ResMut<MaterialIndices>,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    changed_query: Query<
        (),
        Or<(
            Changed<Handle<Sdf3d>>,
            Changed<Handle<Material>>,
            Changed<GlobalTransform>,
        )>,
    >,
    instances: Query<(&Handle<Sdf3d>, &Handle<Material>, &GlobalTransform)>,
) {
    if let Some(previous_new) = std::mem::take(&mut buffers.new) {
        buffers.current = previous_new;
    }

    let mut new_set = None;

    if sdfs.is_added() || sdf_events.read().last().is_some() {
        info!("Updating SDF buffer");
        sdf_indices.clear();
        let mut sdf_buffer: Vec<u8> = Vec::with_capacity(buffers.current.sdfs.size() as usize);
        for (id, sdf) in sdfs.iter() {
            let AssetId::Index { index, .. } = id else {
                warn_once!("Only dense asset storage is supported for sdfs");
                continue;
            };
            let index = (index.to_bits() & (u32::MAX as u64)) as usize;
            let start_offset = (sdf_buffer.len() / 4) as u32;

            if !sdf.operations.is_empty() {
                warn!("SDF operations are not supported");
                continue;
            }

            // TODO: Change SDF trees to be stored in a better format that can be
            //       sent to the GPU directly
            sdf.to_buffer(&mut sdf_buffer);

            while index >= sdf_indices.len() {
                sdf_indices.push(0);
            }
            eprintln!(
                "- Added shape {:?} with start offset: {:?}",
                index, start_offset
            );
            sdf_indices[index] = start_offset;
        }

        if sdf_buffer.len() < 4 {
            sdf_buffer.extend([0; 4]);
        }

        new_set = new_set
            .or_else(|| Some(buffers.current.clone()))
            .map(|mut s| {
                s.sdfs = render_device.create_buffer_with_data(&BufferInitDescriptor {
                    usage: BufferUsages::STORAGE,
                    label: Some("SDF buffer"),
                    contents: &sdf_buffer,
                });
                s
            });
    }

    if mats.is_added() || mat_events.read().last().is_some() {
        info!("Updating materials buffer");
        mat_indices.clear();
        let mut mats_buffer = BufferVec::<Material>::new(BufferUsages::STORAGE);
        for (id, mat) in mats.iter() {
            let AssetId::Index { index, .. } = id else {
                warn_once!("Only dense asset storage is supported for materials");
                continue;
            };
            let index = (index.to_bits() & (u32::MAX as u64)) as usize;
            let start_offset = mats_buffer.len() as u32;

            mats_buffer.push(mat.clone());

            while index >= mat_indices.len() {
                mat_indices.push(0);
            }
            mat_indices[index] = start_offset;
        }

        new_set = new_set
            .or_else(|| Some(buffers.current.clone()))
            .map(|mut s| {
                mats_buffer.write_buffer(&*render_device, &*render_queue);
                s.materials = mats_buffer.buffer().unwrap().clone();
                s
            });
    }

    if new_set.is_none() && changed_query.is_empty() {
        return;
    }

    info!("Buffers or transforms changed, rebuilding instance buffer",);

    let mut instance_buffer = BufferVec::<Instance>::new(BufferUsages::STORAGE);
    for (sdf, mat, transform) in instances.iter() {
        let AssetId::Index { index, .. } = sdf.id() else {
            continue;
        };
        let index = (index.to_bits() & (u32::MAX as u64)) as usize;
        let sdf_index = sdf_indices[index];

        let AssetId::Index { index, .. } = mat.id() else {
            continue;
        };
        let index = (index.to_bits() & (u32::MAX as u64)) as usize;
        let mat_index = mat_indices[index];

        let matrix = transform.affine().matrix3;
        let matrix_transpose = matrix.transpose();
        let difference = matrix * matrix_transpose;
        if difference.x_axis.yz().max_element() > 0.0001
            || difference.y_axis.xz().max_element() > 0.0001
            || difference.z_axis.xy().max_element() > 0.0001
        {
            warn!(
                "GlobalTransform can only contain translation, rotation and uniform scale.
                Expected uniformly scaled matrix after canceling rotation but got {:?}",
                difference
            );
            continue;
        }

        let squared_scale = vec3(
            difference.x_axis.x,
            difference.y_axis.y,
            difference.z_axis.z,
        );
        if (squared_scale.x - squared_scale.y).abs() > 0.0001
            || (squared_scale.x - squared_scale.z).abs() > 0.0001
        {
            warn!(
                "Non-uniform scaling is not supported, but found scale: {:?}",
                Vec3::from(squared_scale.to_array().map(|f| f.sqrt()))
            );
            continue;
        }

        let scale = squared_scale.x.sqrt();
        let inv_scale = scale.recip();

        let translation = transform.translation();
        let rotation = matrix * inv_scale;

        info!("Instance: {:?} {:?} {:?}", translation, rotation, scale);

        instance_buffer.push(Instance {
            sdf_index,
            mat_index,
            scale,
            translation,
            rotation: rotation.into(),
        });
    }

    buffers.new = new_set
        .or_else(|| Some(buffers.current.clone()))
        .map(|mut s| {
            instance_buffer.write_buffer(&*render_device, &*render_queue);
            s.instances = instance_buffer.buffer().unwrap().clone();
            s
        });
}

// TODO: Use resolution or a integer division of it instead of a hardcoded 1280x720
const SIZE: (u32, u32) = (720, 720);
const WORKGROUP_SIZE: u32 = 8;
const CONE_SIZE: u32 = 8;
const BLOCK_SIZE: u32 = WORKGROUP_SIZE * CONE_SIZE;

#[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
struct RayMarcherLabel;

// The post process node used for the render graph
#[derive(Default)]
struct RayMarcherNode;

impl ViewNode for RayMarcherNode {
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

fn extract_buffers(buffers: Extract<Res<Buffers>>, mut extracted: ResMut<BufferSet>) {
    let Some(new_buffers) = &buffers.new else {
        return;
    };

    info!("Replacing buffers: {:?}", *extracted);
    if new_buffers.sdfs.id() != extracted.sdfs.id() {
        info!("- Destroying sdf buffer");
        extracted.sdfs.destroy();
    }
    if new_buffers.materials.id() != extracted.materials.id() {
        info!("- Destroying material buffer");
        extracted.materials.destroy();
    }
    if new_buffers.instances.id() != extracted.instances.id() {
        info!("- Destroying instance buffer");
        extracted.instances.destroy();
    }

    *extracted = (*new_buffers).clone();
    info!("- New buffers: {:?}", *extracted);
}

#[derive(Component)]
struct MarcherStorageBindGroup(BindGroup);

fn prepare_bind_group(
    mut commands: Commands,
    pipeline: Res<RayMarcherPipeline>,
    gpu_images: Res<RenderAssets<GpuImage>>,
    textures: Query<(Entity, &MarcherMainTextures)>,
    buffer_set: Res<BufferSet>,
    render_device: Res<RenderDevice>,
) {
    // TODO: Swap buffers if new ones are available, and drop old ones if necessary
    for (entity, textures) in textures.iter() {
        let depth = gpu_images.get(textures.depth.id()).unwrap();
        let color = gpu_images.get(textures.color.id()).unwrap();
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
        commands
            .entity(entity)
            .insert(MarcherStorageBindGroup(bind_group));
    }
}

#[derive(Resource, Deref)]
struct MaterialSize(NonZeroU64);

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
                width: SIZE.0,
                height: SIZE.1,
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
                width: SIZE.0,
                height: SIZE.1,
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
