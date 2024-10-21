use crate::{MarcherMaterial, MarcherSettings};

use std::{marker::PhantomData, num::NonZeroU64};

use bevy::{
    asset::AssetEvents,
    math::{vec3, Vec3A},
    prelude::*,
    render::{
        render_resource::{
            binding_types::{storage_buffer_read_only, storage_buffer_read_only_sized},
            BindGroup, BindGroupEntries, BindGroupLayout, BindGroupLayoutEntries, Buffer,
            BufferInitDescriptor, BufferUsages, BufferVec, ShaderStages, ShaderType,
        },
        renderer::{RenderDevice, RenderQueue},
        Extract, Render, RenderApp, RenderSet,
    },
};
use bevy_prototype_sdf::{Sdf3d, SdfBounding};
use ploc_bvh::dim3::{Aabb3d, BoundingVolume, BvhAabb3d};

pub struct BufferPlugin<Material: MarcherMaterial> {
    _phantom: PhantomData<Material>,
}

impl<Material: MarcherMaterial> Default for BufferPlugin<Material> {
    fn default() -> Self {
        Self {
            _phantom: PhantomData,
        }
    }
}

impl<Material: MarcherMaterial> Plugin for BufferPlugin<Material> {
    fn build(&self, app: &mut App) {
        app.add_systems(Last, upload_new_buffers::<Material>.after(AssetEvents))
            .init_resource::<SdfIndices>()
            .init_resource::<MaterialIndices>();

        app.sub_app_mut(RenderApp)
            .insert_resource(MaterialSize(Material::min_size()))
            .add_systems(ExtractSchedule, extract_buffers)
            .add_systems(
                Render,
                prepare_bind_group.in_set(RenderSet::PrepareBindGroups),
            );
    }

    fn finish(&self, app: &mut App) {
        app.init_resource::<Buffers>();

        app.sub_app_mut(RenderApp)
            .init_resource::<BufferLayout>()
            .init_resource::<CurrentBufferSet>();
    }
}

#[derive(Resource, Deref)]
struct MaterialSize(NonZeroU64);

#[derive(Resource, Default)]
pub struct Buffers {
    current: Option<BufferSet>,
    new: Option<BufferSet>,
}

#[derive(Resource, Deref, DerefMut, Default, Debug)]
struct CurrentBufferSet(Option<BufferSet>);

#[derive(Clone, Debug)]
pub struct BufferSet {
    pub sdfs: Buffer,
    pub materials: Buffer,
    pub nodes: Buffer,
    pub instances: Buffer,
}

#[derive(Resource, Deref, DerefMut, Default)]
pub struct SdfIndices(Vec<u32>);

#[derive(Resource, Deref, DerefMut, Default)]
pub struct MaterialIndices(Vec<u32>);

// TODO: investigate performance of better ways to pack this data with less wasted alignment space
#[derive(ShaderType, Clone)]
pub struct Instance {
    sdf_index: u32,
    mat_index: u32,
    scale: f32,
    translation: Vec3,
    matrix: Mat3,
    min: Vec3,
    max: Vec3,
}

#[derive(ShaderType)]
pub struct BvhNode {
    min: Vec3,
    count: u32,
    max: Vec3,
    index: u32,
}

// TODO: We can probably split this up into three systems each with different scheduling constraints
fn upload_new_buffers<Material: MarcherMaterial>(
    mut buffers: ResMut<Buffers>,
    mut sdf_events: EventReader<AssetEvent<Sdf3d>>,
    sdfs: Res<Assets<Sdf3d>>,
    mut sdf_indices: ResMut<SdfIndices>,
    mut mat_events: EventReader<AssetEvent<Material>>,
    mats: Res<Assets<Material>>,
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
    if sdfs.len() == 0 || mats.len() == 0 || instances.iter().len() == 0 {
        buffers.current = None;
        buffers.new = None;
        return;
    }

    if let Some(previous_new) = std::mem::take(&mut buffers.new) {
        buffers.current = Some(previous_new);
    }

    let mut new_buffers = (None, None);

    if buffers.current.is_none() || sdf_events.read().last().is_some() {
        sdf_indices.clear();
        let mut sdf_buffer: Vec<u8> = Vec::with_capacity(
            buffers
                .current
                .as_ref()
                .map(|b| b.sdfs.size() as usize)
                .unwrap_or(0),
        );
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
            sdf_indices[index] = start_offset;
        }

        if sdf_buffer.len() < 4 {
            sdf_buffer.extend([0; 4]);
        }

        new_buffers.0 = Some(
            render_device.create_buffer_with_data(&BufferInitDescriptor {
                usage: BufferUsages::STORAGE,
                label: Some("SDF buffer"),
                contents: &sdf_buffer,
            }),
        );
    }

    if buffers.current.is_none() || mat_events.read().last().is_some() {
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

        mats_buffer.write_buffer(&*render_device, &*render_queue);
        new_buffers.1 = mats_buffer.buffer().cloned();
    }

    if new_buffers.0.is_none() && new_buffers.1.is_none() && changed_query.is_empty() {
        return;
    }

    let mut unordered_instances = Vec::<Instance>::with_capacity(instances.iter().len());
    let bvh = BvhAabb3d::new(
        instances.iter().len(),
        instances.iter().filter_map(|(sdf, mat, transform)| {
            let AssetId::Index { index, .. } = sdf.id() else {
                return None;
            };
            let index = (index.to_bits() & (u32::MAX as u64)) as usize;
            let sdf_index = sdf_indices[index];
            let sdf = sdfs.get(sdf.id()).unwrap();

            let AssetId::Index { index, .. } = mat.id() else {
                return None;
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
                return None;
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
                return None;
            }

            let scale = squared_scale.x.sqrt();
            let inv_scale = scale.recip();

            let translation = transform.translation_vec3a();
            let rot_matrix = matrix * inv_scale;
            let rotation = Quat::from_mat3a(&rot_matrix);
            let matrix = matrix_transpose * (inv_scale * inv_scale);

            let aabb = sdf.aabb(default(), rotation);
            let scaled_aabb = Aabb3d::new(
                aabb.center() * scale + translation,
                (aabb.half_size() * scale).min(Vec3A::splat(1e9)),
            );

            let instance_index = unordered_instances.len();
            unordered_instances.push(Instance {
                sdf_index,
                mat_index,
                scale,
                translation: translation.into(),
                matrix: matrix.into(),
                min: scaled_aabb.min.into(),
                max: scaled_aabb.max.into(),
            });

            Some((instance_index, scaled_aabb))
        }),
    );

    let mut nodes = BufferVec::<BvhNode>::new(BufferUsages::STORAGE);
    bvh.nodes().for_each(|n| {
        nodes.push(BvhNode {
            min: n.volume.min.into(),
            max: n.volume.max.into(),
            count: n.count,
            index: n.start_index,
        });
    });
    nodes.write_buffer(&*render_device, &*render_queue);

    let mut instance_buffer = BufferVec::<Instance>::new(BufferUsages::STORAGE);
    bvh.items().for_each(|i| {
        instance_buffer.push(unordered_instances[i.t].clone());
    });
    instance_buffer.write_buffer(&*render_device, &*render_queue);

    let cur_ref = buffers.current.as_ref();
    buffers.new = Some(BufferSet {
        sdfs: new_buffers
            .0
            .or_else(|| cur_ref.map(|c| c.sdfs.clone()))
            .unwrap(),
        materials: new_buffers
            .1
            .or_else(|| cur_ref.map(|c| c.materials.clone()))
            .unwrap(),
        nodes: nodes.buffer().unwrap().clone(),
        instances: instance_buffer.buffer().unwrap().clone(),
    });
}

fn extract_buffers(buffers: Extract<Res<Buffers>>, mut extracted: ResMut<CurrentBufferSet>) {
    if buffers.current.is_none() && buffers.new.is_none() {
        if let Some(previous) = &**extracted {
            previous.sdfs.destroy();
            previous.materials.destroy();
            previous.instances.destroy();
        }
        **extracted = None;
    }

    let Some(new_buffers) = &buffers.new else {
        return;
    };

    if let Some(previous) = &**extracted {
        if new_buffers.sdfs.id() != previous.sdfs.id() {
            previous.sdfs.destroy();
        }
        if new_buffers.materials.id() != previous.materials.id() {
            previous.materials.destroy();
        }
        if new_buffers.instances.id() != previous.instances.id() {
            previous.instances.destroy();
        }
    }

    **extracted = Some((*new_buffers).clone());
}

#[derive(Resource, Deref)]
pub struct BufferLayout(BindGroupLayout);

impl FromWorld for BufferLayout {
    fn from_world(world: &mut World) -> Self {
        let mat_size = **world.resource::<MaterialSize>();
        let render_device = world.resource::<RenderDevice>();

        let storage_layout = render_device.create_bind_group_layout(
            "marcher_storage_bind_group_layout",
            &BindGroupLayoutEntries::sequential(
                ShaderStages::COMPUTE,
                (
                    // SDFs
                    storage_buffer_read_only::<u32>(false),
                    // Materials
                    storage_buffer_read_only_sized(false, Some(mat_size)),
                    // Nodes
                    storage_buffer_read_only::<BvhNode>(false),
                    // Instances
                    storage_buffer_read_only::<Instance>(false),
                ),
            ),
        );
        Self(storage_layout)
    }
}

#[derive(Component, Deref)]
pub struct MarcherStorageBindGroup(BindGroup);

fn prepare_bind_group(
    mut commands: Commands,
    marchers: Query<Entity, With<MarcherSettings>>,
    buffer_set: Res<CurrentBufferSet>,
    render_device: Res<RenderDevice>,
    buffer_layout: Res<BufferLayout>,
) {
    for entity in marchers.iter() {
        let Some(buffer_set) = &**buffer_set else {
            continue;
        };
        let bind_group = render_device.create_bind_group(
            None,
            &buffer_layout,
            &BindGroupEntries::sequential((
                buffer_set.sdfs.as_entire_binding(),
                buffer_set.materials.as_entire_binding(),
                buffer_set.nodes.as_entire_binding(),
                buffer_set.instances.as_entire_binding(),
            )),
        );
        commands
            .entity(entity)
            .insert(MarcherStorageBindGroup(bind_group));
    }
}
