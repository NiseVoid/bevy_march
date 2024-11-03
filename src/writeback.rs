use bevy::{
    asset::embedded_asset,
    core_pipeline::{
        core_3d::graph::{Core3d, Node3d},
        fullscreen_vertex_shader::fullscreen_shader_vertex_state,
        prepass::ViewPrepassTextures,
    },
    ecs::query::QueryItem,
    prelude::*,
    render::{
        render_asset::RenderAssets,
        render_graph::{
            NodeRunError, RenderGraphApp, RenderGraphContext, RenderLabel, ViewNode, ViewNodeRunner,
        },
        render_resource::{
            binding_types::{sampler, texture_2d},
            BindGroup, BindGroupEntries, BindGroupLayout, BindGroupLayoutEntries,
            CachedRenderPipelineId, ColorTargetState, ColorWrites, CompareFunction, DepthBiasState,
            DepthStencilState, FilterMode, FragmentState, MultisampleState, PipelineCache,
            PrimitiveState, RenderPassDescriptor, RenderPipelineDescriptor, Sampler,
            SamplerBindingType, SamplerDescriptor, ShaderDefVal, ShaderStages, StencilFaceState,
            StencilState, StoreOp, TextureFormat, TextureSampleType,
        },
        renderer::{RenderContext, RenderDevice},
        texture::{DepthAttachment, GpuImage},
        view::{ViewDepthTexture, ViewTarget},
        Render, RenderApp, RenderSet,
    },
};

use crate::main_pass::{MarcherMainPass, MarcherMainTextures};

pub struct WritebackPlugin;

impl Plugin for WritebackPlugin {
    fn build(&self, app: &mut App) {
        embedded_asset!(app, "writeback.wgsl");
        app.sub_app_mut(RenderApp)
            .add_systems(
                Render,
                prepare_bind_group.in_set(RenderSet::PrepareBindGroups),
            )
            .add_render_graph_node::<ViewNodeRunner<MarcherWriteback>>(Core3d, MarcherWriteback)
            .add_render_graph_edges(Core3d, (MarcherMainPass, MarcherWriteback))
            .add_render_graph_edges(
                Core3d,
                (
                    Node3d::MainOpaquePass,
                    MarcherWriteback,
                    Node3d::MainTransmissivePass,
                ),
            );
    }

    fn finish(&self, app: &mut App) {
        app.sub_app_mut(RenderApp)
            .init_resource::<WritebackPipelines>();
    }
}

#[derive(RenderLabel, Clone, PartialEq, Eq, Hash, Default, Debug)]
pub struct MarcherWriteback;

impl ViewNode for MarcherWriteback {
    type ViewQuery = (
        &'static ViewTarget,
        &'static ViewDepthTexture,
        Option<&'static ViewPrepassTextures>,
        &'static WritebackBindGroup,
        Option<&'static Msaa>,
    );

    fn run(
        &self,
        _graph: &mut RenderGraphContext,
        render_context: &mut RenderContext,
        (view_target, depth, prepass, bind_group, msaa): QueryItem<Self::ViewQuery>,
        world: &World,
    ) -> Result<(), NodeRunError> {
        let pipelines = world.resource::<WritebackPipelines>();
        let writeback_pipeline = &pipelines.pipelines[match msaa {
            None | Some(Msaa::Off) => 0,
            Some(Msaa::Sample2) => 1,
            Some(Msaa::Sample4) => 2,
            Some(Msaa::Sample8) => 3,
        }];

        let pipeline_cache = world.resource::<PipelineCache>();

        if let Some(prepass) = prepass.and_then(|p| p.depth_view()) {
            let Some(pipeline) =
                pipeline_cache.get_render_pipeline(writeback_pipeline.depth_pipeline_id)
            else {
                return Ok(());
            };
            let depth = DepthAttachment::new(prepass.clone(), None);
            let mut render_pass = render_context.begin_tracked_render_pass(RenderPassDescriptor {
                label: Some("marcher_depth_writeback"),
                color_attachments: &[],
                depth_stencil_attachment: Some(depth.get_attachment(StoreOp::Store)),
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            render_pass.set_render_pipeline(pipeline);
            render_pass.set_bind_group(0, &bind_group.0, &[]);
            render_pass.draw(0..3, 0..1);
        }

        let Some(pipeline) =
            pipeline_cache.get_render_pipeline(writeback_pipeline.main_pipeline_id)
        else {
            return Ok(());
        };
        let depth_stencil_attachment = Some(depth.get_attachment(StoreOp::Store));

        let mut render_pass = render_context.begin_tracked_render_pass(RenderPassDescriptor {
            label: Some("marcher_writeback"),
            color_attachments: &[Some(view_target.get_color_attachment())],
            depth_stencil_attachment,
            timestamp_writes: None,
            occlusion_query_set: None,
        });

        render_pass.set_render_pipeline(pipeline);
        render_pass.set_bind_group(0, &bind_group.0, &[]);
        render_pass.draw(0..3, 0..1);

        Ok(())
    }
}

#[derive(Component)]
pub struct WritebackBindGroup(BindGroup);

fn prepare_bind_group(
    mut commands: Commands,
    pipeline: Res<WritebackPipelines>,
    gpu_images: Res<RenderAssets<GpuImage>>,
    textures: Query<(Entity, &MarcherMainTextures)>,
    render_device: Res<RenderDevice>,
) {
    for (entity, textures) in textures.iter() {
        // TODO: Render previous textures if the current ones can't be found
        let Some(color) = gpu_images.get(textures.color.id()) else {
            continue;
        };
        let Some(depth) = gpu_images.get(textures.depth.id()) else {
            continue;
        };
        let bind_group = render_device.create_bind_group(
            None,
            &pipeline.layout,
            &BindGroupEntries::sequential((
                &color.texture_view,
                &depth.texture_view,
                &pipeline.sampler,
            )),
        );
        commands
            .entity(entity)
            .insert(WritebackBindGroup(bind_group));
    }
}

#[derive(Resource)]
struct WritebackPipelines {
    layout: BindGroupLayout,
    sampler: Sampler,
    pipelines: [WritebackPipeline; 4],
}

struct WritebackPipeline {
    main_pipeline_id: CachedRenderPipelineId,
    depth_pipeline_id: CachedRenderPipelineId,
}

impl FromWorld for WritebackPipelines {
    fn from_world(world: &mut World) -> Self {
        let render_device = world.resource::<RenderDevice>();

        // We need to define the bind group layout used for our pipeline
        let layout = render_device.create_bind_group_layout(
            "post_process_bind_group_layout",
            &BindGroupLayoutEntries::sequential(
                // The layout entries will only be visible in the fragment stage
                ShaderStages::FRAGMENT,
                (
                    texture_2d(TextureSampleType::Float { filterable: true }),
                    texture_2d(TextureSampleType::Float { filterable: true }),
                    sampler(SamplerBindingType::Filtering),
                ),
            ),
        );

        let sampler = render_device.create_sampler(&SamplerDescriptor {
            mag_filter: FilterMode::Linear,
            ..default()
        });

        let shader = world.load_asset("embedded://bevy_march/writeback.wgsl");

        let pipelines = [
            get_pipeline(world, layout.clone(), shader.clone(), Msaa::Off),
            get_pipeline(world, layout.clone(), shader.clone(), Msaa::Sample2),
            get_pipeline(world, layout.clone(), shader.clone(), Msaa::Sample4),
            get_pipeline(world, layout.clone(), shader.clone(), Msaa::Sample8),
        ];

        Self {
            layout,
            sampler,
            pipelines,
        }
    }
}

fn get_pipeline(
    world: &mut World,
    layout: BindGroupLayout,
    shader: Handle<Shader>,
    msaa: Msaa,
) -> WritebackPipeline {
    let multisample = MultisampleState {
        count: msaa.samples(),
        mask: !0,
        alpha_to_coverage_enabled: false,
    };
    let stencil_state = DepthStencilState {
        format: TextureFormat::Depth32Float,
        depth_write_enabled: true,
        depth_compare: CompareFunction::Greater,
        stencil: StencilState {
            front: StencilFaceState::IGNORE,
            back: StencilFaceState::IGNORE,
            read_mask: 0,
            write_mask: 0,
        },
        bias: DepthBiasState {
            constant: 0,
            slope_scale: 0.,
            clamp: 0.,
        },
    };
    let main_pipeline_id =
        world
            .resource_mut::<PipelineCache>()
            .queue_render_pipeline(RenderPipelineDescriptor {
                label: Some("writeback_pipeline".into()),
                layout: vec![layout.clone()],
                vertex: fullscreen_shader_vertex_state(),
                fragment: Some(FragmentState {
                    shader: shader.clone(),
                    shader_defs: vec![ShaderDefVal::Bool("COLOR".into(), true)],
                    entry_point: "fragment".into(),
                    targets: vec![Some(ColorTargetState {
                        // TODO: Use whatever the view has
                        format: TextureFormat::Rgba16Float,
                        blend: None,
                        write_mask: ColorWrites::ALL,
                    })],
                }),
                primitive: PrimitiveState::default(),
                depth_stencil: Some(stencil_state.clone()),
                multisample,
                push_constant_ranges: vec![],
            });

    let depth_pipeline_id =
        world
            .resource_mut::<PipelineCache>()
            .queue_render_pipeline(RenderPipelineDescriptor {
                label: Some("writeback_depth_pipeline".into()),
                layout: vec![layout],
                vertex: fullscreen_shader_vertex_state(),
                fragment: Some(FragmentState {
                    shader,
                    shader_defs: vec![],
                    entry_point: "fragment".into(),
                    targets: vec![],
                }),
                primitive: PrimitiveState::default(),
                depth_stencil: Some(stencil_state),
                multisample,
                push_constant_ranges: vec![],
            });

    WritebackPipeline {
        main_pipeline_id,
        depth_pipeline_id,
    }
}
