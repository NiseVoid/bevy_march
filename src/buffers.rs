use crate::MarcherMaterial;

use std::{marker::PhantomData, num::NonZeroU64};

use bevy::{
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
        app.add_systems(PostUpdate, upload_new_buffers::<Material>)
            .init_resource::<SdfIndices>()
            .init_resource::<MaterialIndices>();

        app.sub_app_mut(RenderApp)
            .insert_resource(MaterialSize(Material::min_size()))
            .add_systems(ExtractSchedule, extract_buffers);
    }

    fn finish(&self, app: &mut App) {
        app.init_resource::<Buffers>();

        app.sub_app_mut(RenderApp).init_resource::<BufferSet>();
    }
}

#[derive(Resource, Deref)]
pub struct MaterialSize(NonZeroU64);

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
    pub sdfs: Buffer,
    pub materials: Buffer,
    pub instances: Buffer,
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
pub struct Instance {
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
