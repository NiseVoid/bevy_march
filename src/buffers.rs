use crate::MarcherMaterial;

use std::{marker::PhantomData, num::NonZeroU64};

use bevy::{
    asset::AssetEvents,
    math::vec3,
    prelude::*,
    render::{
        render_resource::{Buffer, BufferInitDescriptor, BufferUsages, BufferVec, ShaderType},
        renderer::{RenderDevice, RenderQueue},
        Extract, RenderApp,
    },
};
use bevy_prototype_sdf::Sdf3d;

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
            .add_systems(ExtractSchedule, extract_buffers);
    }

    fn finish(&self, app: &mut App) {
        app.init_resource::<Buffers>();

        app.sub_app_mut(RenderApp)
            .init_resource::<CurrentBufferSet>();
    }
}

#[derive(Resource, Deref)]
pub struct MaterialSize(NonZeroU64);

#[derive(Resource, Default)]
pub struct Buffers {
    current: Option<BufferSet>,
    new: Option<BufferSet>,
}

#[derive(Resource, Deref, DerefMut, Default, Debug)]
pub struct CurrentBufferSet(Option<BufferSet>);

#[derive(Clone, Debug)]
pub struct BufferSet {
    pub sdfs: Buffer,
    pub materials: Buffer,
    pub instances: Buffer,
}

#[derive(Resource, Deref, DerefMut, Default)]
pub struct SdfIndices(Vec<u32>);

#[derive(Resource, Deref, DerefMut, Default)]
pub struct MaterialIndices(Vec<u32>);

// TODO: This type becomes 80 bytes on the GPU, investigate performance of alternatives
#[derive(ShaderType)]
pub struct Instance {
    sdf_index: u32,
    mat_index: u32,
    scale: f32,
    translation: Vec3,
    matrix: Mat3,
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
        let matrix = matrix_transpose * (inv_scale * inv_scale);

        instance_buffer.push(Instance {
            sdf_index,
            mat_index,
            scale,
            translation,
            matrix: matrix.into(),
        });
    }

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
