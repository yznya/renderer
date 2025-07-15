use crate::math::power_2_floor;
use lava::{
    buffer_barrier,
    device::LDeviceRef,
    image_barrier,
    resources::{DescriptorInfo, LCommandBuffer},
    utils::{clear_color, clear_color_int, clear_depth},
    with,
};

use super::{GpuResources, SceneResources, WindowResources};
use anyhow::Result;
use ash::vk;
use glam::{IVec4, Vec4, vec2};
use scenery::scene::Scene;

pub(crate) trait Pass {
    unsafe fn record_cmd_buffer(
        &self,
        device: LDeviceRef,
        cmd: &LCommandBuffer,
        image_index: u32,
        frame_index: usize,
        win_res: &WindowResources,
        scene_resources: &SceneResources,
        gpu_resources: &GpuResources,
        scene: &Scene,
    ) -> Result<()>;

    fn name(&self) -> &'static str {
        std::any::type_name::<Self>()
    }
}

pub(crate) enum PassType {
    Early,
    Late,
    MaskAlpha,
}
pub(crate) struct CullPass {
    pub(super) late: bool,
    pub(super) pass_type: PassType,
    pub(super) name: &'static str,
}

impl Pass for CullPass {
    unsafe fn record_cmd_buffer(
        &self,
        _device: LDeviceRef,
        cmd: &LCommandBuffer,
        _image_index: u32,
        frame_index: usize,
        win_res: &WindowResources,
        scene_resources: &SceneResources,
        gpu_resources: &GpuResources,
        scene: &Scene,
    ) -> Result<()> {
        let draw_commands_count_barrier = buffer_barrier(
            &gpu_resources.draw_commands_count_buffer.inner,
            with!(vk::AccessFlags2 => { SHADER_WRITE | SHADER_READ }),
            vk::AccessFlags2::TRANSFER_WRITE,
            vk::PipelineStageFlags2::COMPUTE_SHADER,
            vk::PipelineStageFlags2::TRANSFER,
        );

        unsafe { cmd.buffer_barrier(&[draw_commands_count_barrier]) };

        unsafe { cmd.fill_buffer(&gpu_resources.draw_commands_count_buffer, 0, size_of::<u32>() as u64, 0) }

        unsafe {
            cmd.fill_buffer(
                &gpu_resources.draw_commands_count_buffer,
                size_of::<u32>() as u64,
                2 * size_of::<u32>() as u64,
                1,
            )
        };

        let draw_commands_barrier = buffer_barrier(
            &scene_resources.draw_commands_buffer.inner,
            vk::AccessFlags2::INDIRECT_COMMAND_READ,
            vk::AccessFlags2::SHADER_WRITE,
            with!(vk::PipelineStageFlags2=> { DRAW_INDIRECT | COMPUTE_SHADER }),
            vk::PipelineStageFlags2::COMPUTE_SHADER,
        );

        let draw_commands_count_barrier = buffer_barrier(
            &gpu_resources.draw_commands_count_buffer.inner,
            vk::AccessFlags2::TRANSFER_WRITE,
            with!(vk::AccessFlags2 => { SHADER_WRITE | SHADER_READ }),
            vk::PipelineStageFlags2::TRANSFER,
            vk::PipelineStageFlags2::COMPUTE_SHADER,
        );

        let draw_visibility_barrier = buffer_barrier(
            &scene_resources.draw_visiablity_buffer.inner,
            if self.late {
                vk::AccessFlags2::SHADER_READ
            } else {
                vk::AccessFlags2::TRANSFER_WRITE
            },
            with!(vk::AccessFlags2 => { SHADER_WRITE | SHADER_READ }),
            with!(vk::PipelineStageFlags2 => {COMPUTE_SHADER | TRANSFER}),
            vk::PipelineStageFlags2::COMPUTE_SHADER,
        );

        unsafe {
            cmd.buffer_barrier(&[
                draw_commands_barrier,
                draw_commands_count_barrier,
                draw_visibility_barrier,
            ])
        };

        let descriptor_writes = [
            DescriptorInfo::buffer(win_res.uniform_buffers[frame_index].buffer.inner),
            DescriptorInfo::buffer(scene_resources.meshs_buffer.inner),
            DescriptorInfo::buffer(scene_resources.draws_buffer.inner),
            DescriptorInfo::buffer(scene_resources.draw_commands_buffer.inner),
            DescriptorInfo::buffer(gpu_resources.draw_commands_count_buffer.inner),
            DescriptorInfo::buffer(scene_resources.draw_visiablity_buffer.inner),
            DescriptorInfo::buffer(scene_resources.models_buffer.inner),
            DescriptorInfo::image_sampler(
                &win_res.depth_sampler,
                &win_res.depth_pyramid_image.image_view,
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            ),
        ];

        let program = match self.pass_type {
            PassType::Early => &win_res.early_cull_program,
            PassType::Late => &win_res.late_cull_program,
            PassType::MaskAlpha => &win_res.mask_alpha_cull_program,
        }
        .as_ref()
        .unwrap();

        unsafe { cmd.bind_compute_pipeline(program) };
        unsafe { cmd.push_descriptors(program, 0, &descriptor_writes) };
        unsafe { cmd.dispatch((scene.mesh_draws.len() as u32).div_ceil(program.local_size_x), 1, 1) };

        let draw_commands_barrier = buffer_barrier(
            &scene_resources.draw_commands_buffer.inner,
            vk::AccessFlags2::SHADER_WRITE,
            vk::AccessFlags2::INDIRECT_COMMAND_READ,
            vk::PipelineStageFlags2::COMPUTE_SHADER,
            vk::PipelineStageFlags2::DRAW_INDIRECT,
        );

        let draw_commands_count_barrier = buffer_barrier(
            &gpu_resources.draw_commands_count_buffer.inner,
            vk::AccessFlags2::SHADER_WRITE,
            vk::AccessFlags2::INDIRECT_COMMAND_READ,
            vk::PipelineStageFlags2::COMPUTE_SHADER,
            vk::PipelineStageFlags2::DRAW_INDIRECT,
        );

        unsafe { cmd.buffer_barrier(&[draw_commands_barrier, draw_commands_count_barrier]) };

        Ok(())
    }

    fn name(&self) -> &'static str {
        self.name
    }
}

pub(crate) struct DrawOpaquePass {
    pub(super) late: bool,
    pub(super) pass_type: PassType,
    pub(super) name: &'static str,
}

impl Pass for DrawOpaquePass {
    unsafe fn record_cmd_buffer(
        &self,
        _device: LDeviceRef,
        cmd: &LCommandBuffer,
        image_index: u32,
        frame_index: usize,
        win_res: &WindowResources,
        scene_resources: &SceneResources,
        gpu_resources: &GpuResources,
        _scene: &Scene,
    ) -> Result<()> {
        if self.late {
            let depth_image_barrier = image_barrier!(
                image: win_res.depth_image.image.inner,
                access: DEPTH_STENCIL_ATTACHMENT_WRITE => DEPTH_STENCIL_ATTACHMENT_READ,
                layout: ATTACHMENT_OPTIMAL =>  ATTACHMENT_OPTIMAL,
                stage: LATE_FRAGMENT_TESTS => EARLY_FRAGMENT_TESTS,
                aspect: DEPTH
            );

            unsafe { cmd.image_barrier(&[depth_image_barrier]) };
        }

        let program = match self.pass_type {
            PassType::Early => &win_res.early_draw_program,
            PassType::Late => &win_res.late_draw_program,
            PassType::MaskAlpha => &win_res.mask_alpha_draw_program,
        }
        .as_ref()
        .unwrap();

        unsafe {
            cmd.bind_descriptor_sets(
                vk::PipelineBindPoint::GRAPHICS,
                program,
                1,
                &[scene_resources.texture_set],
                &[],
            )
        };

        let descriptor_writes = [
            DescriptorInfo::buffer(win_res.uniform_buffers[frame_index].buffer.inner),
            DescriptorInfo::buffer(scene_resources.vertex_buffer.inner),
            DescriptorInfo::buffer(scene_resources.meshlets_buffer.inner),
            DescriptorInfo::buffer(scene_resources.meshlet_vertices_buffer.inner),
            DescriptorInfo::buffer(scene_resources.meshlet_triangles_buffer.inner),
            DescriptorInfo::buffer(scene_resources.draws_buffer.inner),
            DescriptorInfo::buffer(scene_resources.draw_commands_buffer.inner),
            DescriptorInfo::buffer(scene_resources.meshlet_visibility_buffer.inner),
            DescriptorInfo::image_sampler(
                &win_res.depth_sampler,
                &win_res.depth_pyramid_image.image_view,
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            ),
            DescriptorInfo::acc_struct(scene_resources.tlas.inner),
            DescriptorInfo::buffer(scene_resources.materials_buffer.inner),
            DescriptorInfo::buffer(scene_resources.meshs_buffer.inner),
            DescriptorInfo::buffer(scene_resources.models_buffer.inner),
        ];

        unsafe { cmd.push_descriptors(program, 0, &descriptor_writes) };

        let load_op = if self.late {
            vk::AttachmentLoadOp::LOAD
        } else {
            vk::AttachmentLoadOp::CLEAR
        };

        let swapchain = win_res.swapchain.as_ref().unwrap();
        unsafe {
            cmd.begin_rendering(
                &win_res.color_images[image_index as usize].image_view,
                load_op,
                vk::AttachmentStoreOp::STORE,
                clear_color_int(IVec4::new(!0, !0, !0, !0)),
                &win_res.depth_image.image_view,
                load_op,
                vk::AttachmentStoreOp::STORE,
                clear_depth(0.0, 0),
                vk::Rect2D::default().extent(swapchain.extent()),
            )
        };

        unsafe { cmd.bind_graphics_pipeline(program) };
        unsafe { cmd.draw_mesh_tasks_indirect(gpu_resources.draw_commands_count_buffer.inner, 0, 1, 0) };
        unsafe { cmd.end_rendering() };

        Ok(())
    }

    fn name(&self) -> &'static str {
        self.name
    }
}

pub(crate) struct DepthPyramidPass {}

impl Pass for DepthPyramidPass {
    unsafe fn record_cmd_buffer(
        &self,
        _device: LDeviceRef,
        cmd: &LCommandBuffer,
        _image_index: u32,
        _frame_index: usize,
        win_res: &WindowResources,
        _scene_resources: &SceneResources,
        _gpu_resources: &GpuResources,
        _scene: &Scene,
    ) -> Result<()> {
        let depth_image_barrier = image_barrier!(
            image: win_res.depth_image.image.inner,
            access: DEPTH_STENCIL_ATTACHMENT_WRITE => SHADER_READ,
            layout: UNDEFINED =>  SHADER_READ_ONLY_OPTIMAL,
            stage: LATE_FRAGMENT_TESTS => COMPUTE_SHADER,
            aspect: DEPTH
        );

        let depth_pyramid_image_barrier = image_barrier!(
            image: win_res.depth_pyramid_image.image.inner,
            access: empty => SHADER_WRITE,
            layout: UNDEFINED => GENERAL,
            stage: LATE_FRAGMENT_TESTS => COMPUTE_SHADER,
            aspect: COLOR
        );

        unsafe { cmd.image_barrier(&[depth_image_barrier, depth_pyramid_image_barrier]) };
        unsafe { cmd.bind_compute_pipeline(win_res.depth_reduce_program.as_ref().unwrap()) };

        let swapchain = win_res.swapchain.as_ref().unwrap();
        let pyramid_width = power_2_floor(swapchain.extent().width);
        let pyramid_height = power_2_floor(swapchain.extent().height);

        for i in 0..win_res.depth_pyramid_mips.len() {
            let source = if i == 0 {
                DescriptorInfo::image_sampler(
                    &win_res.depth_sampler,
                    &win_res.depth_image.image_view,
                    vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                )
            } else {
                DescriptorInfo::image_sampler(
                    &win_res.depth_sampler,
                    &win_res.depth_pyramid_mips[i - 1],
                    vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                )
            };

            let descriptor_writes = [
                source,
                DescriptorInfo::image_sampler(
                    // TODO: verify this is okay to not be null
                    &win_res.depth_sampler,
                    &win_res.depth_pyramid_mips[i],
                    vk::ImageLayout::GENERAL,
                ),
            ];

            unsafe { cmd.push_descriptors(win_res.depth_reduce_program.as_ref().unwrap(), 0, &descriptor_writes) };

            let width = u32::max(pyramid_width >> i, 1);
            let height = u32::max(pyramid_height >> i, 1);

            let image_size = vec2(width as f32, height as f32);
            unsafe {
                cmd.push_constants(
                    win_res.depth_reduce_program.as_ref().unwrap().pipeline_layout,
                    vk::ShaderStageFlags::COMPUTE,
                    0,
                    bytemuck::cast_slice(image_size.as_ref().as_slice()),
                )
            };

            unsafe {
                cmd.dispatch(
                    width.div_ceil(win_res.depth_reduce_program.as_ref().unwrap().local_size_x),
                    height.div_ceil(win_res.depth_reduce_program.as_ref().unwrap().local_size_y),
                    1,
                )
            };

            let depth_pyramid_image_barrier = vk::ImageMemoryBarrier2::default()
                .image(win_res.depth_pyramid_image.image.inner)
                .src_access_mask(vk::AccessFlags2::empty())
                .dst_access_mask(vk::AccessFlags2::SHADER_READ)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .old_layout(vk::ImageLayout::GENERAL)
                .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .src_stage_mask(vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS)
                .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                .subresource_range(
                    vk::ImageSubresourceRange::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .base_mip_level(i as u32)
                        .level_count(1)
                        .layer_count(vk::REMAINING_ARRAY_LAYERS),
                );

            unsafe { cmd.image_barrier(&[depth_pyramid_image_barrier]) };
        }

        let depth_image_barrier = image_barrier!(
            image: win_res.depth_image.image.inner,
            access: SHADER_READ => DEPTH_STENCIL_ATTACHMENT_WRITE | DEPTH_STENCIL_ATTACHMENT_READ,
            layout: SHADER_READ_ONLY_OPTIMAL => DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
            stage: COMPUTE_SHADER => LATE_FRAGMENT_TESTS,
            aspect: DEPTH
        );

        unsafe { cmd.image_barrier(&[depth_image_barrier]) };

        Ok(())
    }

    fn name(&self) -> &'static str {
        "DepthPyramidPass"
    }
}

pub(crate) struct SkyBoxPass {}

impl Pass for SkyBoxPass {
    unsafe fn record_cmd_buffer(
        &self,
        _device: LDeviceRef,
        cmd: &LCommandBuffer,
        image_index: u32,
        frame_index: usize,
        win_res: &WindowResources,
        scene_resources: &SceneResources,
        gpu_resources: &GpuResources,
        _scene: &Scene,
    ) -> Result<()> {
        let descriptor_writes = [
            DescriptorInfo::buffer(win_res.uniform_buffers[frame_index].buffer.inner),
            DescriptorInfo::image_sampler(
                &gpu_resources.texture_sampler,
                &scene_resources.cubemap_image.image_view,
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            ),
        ];

        unsafe {
            cmd.push_descriptors(win_res.skybox_program.as_ref().unwrap(), 0, &descriptor_writes);

            let swapchain = win_res.swapchain.as_ref().unwrap();
            cmd.begin_rendering(
                &win_res.swapchain_image_views[image_index as usize],
                vk::AttachmentLoadOp::LOAD,
                vk::AttachmentStoreOp::STORE,
                clear_color(Vec4::ZERO),
                &win_res.depth_image.image_view,
                vk::AttachmentLoadOp::LOAD,
                vk::AttachmentStoreOp::DONT_CARE,
                clear_depth(0.0, 0),
                vk::Rect2D::default().extent(swapchain.extent()),
            );

            cmd.bind_graphics_pipeline(win_res.skybox_program.as_ref().unwrap());
            cmd.draw(36, 1, 0, 0);
            cmd.end_rendering();
        };

        Ok(())
    }

    fn name(&self) -> &'static str {
        "SkyBoxPass"
    }
}

pub(crate) struct GridPass {}

impl Pass for GridPass {
    unsafe fn record_cmd_buffer(
        &self,
        _device: LDeviceRef,
        cmd: &LCommandBuffer,
        image_index: u32,
        frame_index: usize,
        win_res: &WindowResources,
        _scene_resources: &SceneResources,
        _gpu_resources: &GpuResources,
        _scene: &Scene,
    ) -> Result<()> {
        let descriptor_writes = [DescriptorInfo::buffer(
            win_res.uniform_buffers[frame_index].buffer.inner,
        )];

        unsafe {
            cmd.push_descriptors(win_res.grid_program.as_ref().unwrap(), 0, &descriptor_writes);
            let swapchain = win_res.swapchain.as_ref().unwrap();
            cmd.begin_rendering(
                &win_res.swapchain_image_views[image_index as usize],
                vk::AttachmentLoadOp::LOAD,
                vk::AttachmentStoreOp::STORE,
                clear_color(Vec4::ZERO),
                &win_res.depth_image.image_view,
                vk::AttachmentLoadOp::LOAD,
                vk::AttachmentStoreOp::DONT_CARE,
                clear_depth(0.0, 0),
                vk::Rect2D::default().extent(swapchain.extent()),
            );

            cmd.bind_graphics_pipeline(win_res.grid_program.as_ref().unwrap());
            cmd.draw(6, 1, 0, 0);
            cmd.end_rendering();
        };

        Ok(())
    }

    fn name(&self) -> &'static str {
        "GridPass"
    }
}

pub(crate) struct ShadingPass {}

impl Pass for ShadingPass {
    unsafe fn record_cmd_buffer(
        &self,
        _device: LDeviceRef,
        cmd: &LCommandBuffer,
        image_index: u32,
        frame_index: usize,
        win_res: &WindowResources,
        scene_resources: &SceneResources,
        gpu_resources: &GpuResources,
        _scene: &Scene,
    ) -> Result<()> {
        let color_image_barrier = image_barrier!(
            image: win_res.color_images[image_index as usize].image.inner,
            access: COLOR_ATTACHMENT_WRITE => SHADER_READ,
            layout: COLOR_ATTACHMENT_OPTIMAL => GENERAL,
            stage: COLOR_ATTACHMENT_OUTPUT => FRAGMENT_SHADER,
            aspect: COLOR
        );

        let present_image_barrier = image_barrier!(
            image: win_res.swapchain.as_ref().unwrap().images()[image_index as usize],
            access: MEMORY_READ => COLOR_ATTACHMENT_WRITE,
            layout: UNDEFINED =>  COLOR_ATTACHMENT_OPTIMAL,
            stage: TRANSFER => COLOR_ATTACHMENT_OUTPUT,
            aspect: COLOR
        );

        unsafe { cmd.image_barrier(&[color_image_barrier, present_image_barrier]) };

        let descriptor_writes = [
            DescriptorInfo::buffer(win_res.uniform_buffers[frame_index].buffer.inner),
            DescriptorInfo::image_sampler(
                &gpu_resources.texture_sampler,
                &win_res.color_images[image_index as usize].image_view,
                vk::ImageLayout::GENERAL,
            ),
            DescriptorInfo::buffer(scene_resources.vertex_buffer.inner),
            DescriptorInfo::buffer(scene_resources.meshlets_buffer.inner),
            DescriptorInfo::buffer(scene_resources.meshlet_vertices_buffer.inner),
            DescriptorInfo::buffer(scene_resources.meshlet_triangles_buffer.inner),
            DescriptorInfo::buffer(scene_resources.draws_buffer.inner),
            DescriptorInfo::buffer(scene_resources.meshs_buffer.inner),
            DescriptorInfo::buffer(scene_resources.models_buffer.inner),
            DescriptorInfo::image_sampler(
                &gpu_resources.texture_sampler,
                &scene_resources.brdf_lut_image.image_view,
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            ),
            DescriptorInfo::image_sampler(
                &gpu_resources.texture_sampler,
                &scene_resources.lambertian_image.image_view,
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            ),
            DescriptorInfo::image_sampler(
                &gpu_resources.texture_sampler,
                &scene_resources.ggx_image.image_view,
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            ),
            DescriptorInfo::image_sampler(
                &gpu_resources.texture_sampler,
                &scene_resources.charlie_image.image_view,
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            ),
            DescriptorInfo::acc_struct(scene_resources.tlas.inner),
            DescriptorInfo::buffer(scene_resources.materials_buffer.inner),
        ];

        unsafe {
            cmd.bind_descriptor_sets(
                vk::PipelineBindPoint::GRAPHICS,
                win_res.shading_program.as_ref().unwrap(),
                1,
                &[scene_resources.texture_set],
                &[],
            );
            cmd.push_descriptors(win_res.shading_program.as_ref().unwrap(), 0, &descriptor_writes);
            cmd.begin_rendering(
                &win_res.swapchain_image_views[image_index as usize],
                vk::AttachmentLoadOp::CLEAR,
                vk::AttachmentStoreOp::STORE,
                clear_color(Vec4::ZERO),
                &win_res.depth_image.image_view,
                vk::AttachmentLoadOp::DONT_CARE,
                vk::AttachmentStoreOp::DONT_CARE,
                clear_depth(0.0, 0),
                vk::Rect2D::default().extent(win_res.swapchain.as_ref().unwrap().extent()),
            );

            cmd.bind_graphics_pipeline(win_res.shading_program.as_ref().unwrap());
            cmd.draw(3, 1, 0, 0);
            cmd.end_rendering();
        };

        Ok(())
    }

    fn name(&self) -> &'static str {
        "ShadingPass"
    }
}
