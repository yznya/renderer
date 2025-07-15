use core::ffi;
use std::{marker::PhantomData, mem::ManuallyDrop, ops::Deref};

use ash::vk;
use gpu_allocator::vulkan::Allocation;

use crate::{device::LDeviceRef, program::Program};

#[repr(C)]
pub union DescriptorInfo {
    #[allow(dead_code)]
    pub buffer: vk::DescriptorBufferInfo,
    pub image: vk::DescriptorImageInfo,
    pub acceleration_structure: vk::AccelerationStructureKHR,
}

impl DescriptorInfo {
    pub fn buffer(buffer: vk::Buffer) -> Self {
        Self {
            buffer: vk::DescriptorBufferInfo {
                buffer,
                offset: 0,
                range: vk::WHOLE_SIZE,
            },
        }
    }

    pub fn image_sampler(sampler: &LSampler, image_view: &LImageView, image_layout: vk::ImageLayout) -> Self {
        Self {
            image: vk::DescriptorImageInfo {
                sampler: sampler.inner,
                image_view: image_view.inner,
                image_layout,
            },
        }
    }

    pub fn acc_struct(acc: vk::AccelerationStructureKHR) -> Self {
        Self {
            acceleration_structure: acc,
        }
    }
}

pub struct LBuffer<'d> {
    pub inner: vk::Buffer,
    pub allocation: ManuallyDrop<Allocation>,
    pub(crate) device_ref: LDeviceRef<'d>,
}

impl LBuffer<'_> {
    pub(crate) fn new(inner: vk::Buffer, allocation: Allocation) -> Self {
        Self {
            inner,
            allocation: ManuallyDrop::new(allocation),
            device_ref: LDeviceRef { _marker: PhantomData },
        }
    }

    pub fn get_device_address(&self) -> vk::DeviceAddress {
        unsafe {
            self.device_ref
                .get_buffer_device_address(&vk::BufferDeviceAddressInfo::default().buffer(self.inner))
        }
    }
}

impl Drop for LBuffer<'_> {
    fn drop(&mut self) {
        unsafe {
            self.device_ref
                .device_mut()
                .allocator
                .free(ManuallyDrop::take(&mut self.allocation))
                .expect("Failed to free allocation");

            self.device_ref.destroy_buffer(self.inner, None);
        }
    }
}

pub struct LMappedBuffer<'d> {
    pub buffer: LBuffer<'d>,
    pub memory_map: *mut ffi::c_void,
}

impl<'d> LMappedBuffer<'d> {
    pub(crate) fn new(buffer: LBuffer<'d>, memory_map: *mut ffi::c_void) -> Self {
        Self { buffer, memory_map }
    }
}

impl<'d> Deref for LMappedBuffer<'d> {
    type Target = LBuffer<'d>;

    fn deref(&self) -> &Self::Target {
        &self.buffer
    }
}

pub struct LImage<'d> {
    pub inner: vk::Image,
    pub allocation: ManuallyDrop<Allocation>,
    pub(crate) device_ref: LDeviceRef<'d>,
}

impl Drop for LImage<'_> {
    fn drop(&mut self) {
        unsafe {
            self.device_ref.destroy_image(self.inner, None);

            self.device_ref
                .device_mut()
                .allocator
                .free(ManuallyDrop::take(&mut self.allocation))
                .expect("Failed to free allocation");
        }
    }
}

pub struct LImageView<'d> {
    pub inner: vk::ImageView,
    pub(crate) device_ref: LDeviceRef<'d>,
}

impl LImageView<'_> {
    pub(crate) fn new(image_view: vk::ImageView) -> Self {
        Self {
            inner: image_view,
            device_ref: LDeviceRef { _marker: PhantomData },
        }
    }
}

impl Drop for LImageView<'_> {
    fn drop(&mut self) {
        unsafe {
            self.device_ref.destroy_image_view(self.inner, None);
        }
    }
}

pub struct LImageWithView<'d> {
    pub image: LImage<'d>,
    pub image_view: LImageView<'d>,
}

pub struct LCommandBuffer<'d> {
    pub inner: vk::CommandBuffer,
    pub(crate) device_ref: LDeviceRef<'d>,
}

impl LCommandBuffer<'_> {
    pub(crate) fn new(command_buffer: vk::CommandBuffer) -> Self {
        Self {
            inner: command_buffer,
            device_ref: LDeviceRef { _marker: PhantomData },
        }
    }

    pub unsafe fn fill_buffer(&self, buffer: &LBuffer, offset: vk::DeviceSize, size: vk::DeviceSize, data: u32) {
        unsafe {
            self.device_ref
                .cmd_fill_buffer(self.inner, buffer.inner, offset, size, data)
        };
    }

    pub unsafe fn buffer_barrier(&self, barriers: &[vk::BufferMemoryBarrier2]) {
        let dependency_info = vk::DependencyInfo::default().buffer_memory_barriers(barriers);
        unsafe { self.device_ref.cmd_pipeline_barrier2(self.inner, &dependency_info) };
    }

    pub unsafe fn bind_graphics_pipeline(&self, program: &Program) {
        unsafe {
            self.device_ref
                .cmd_bind_pipeline(self.inner, vk::PipelineBindPoint::GRAPHICS, program.pipeline)
        };
    }

    pub unsafe fn bind_compute_pipeline(&self, program: &Program) {
        unsafe {
            self.device_ref
                .cmd_bind_pipeline(self.inner, vk::PipelineBindPoint::COMPUTE, program.pipeline)
        };
    }

    pub unsafe fn push_descriptors(&self, program: &Program, set: u32, info: &[DescriptorInfo]) {
        unsafe {
            self.device_ref
                .device()
                .push_descriptors_loader
                .cmd_push_descriptor_set_with_template(
                    self.inner,
                    program.descriptor_update_template,
                    program.pipeline_layout,
                    set,
                    info.as_ptr().cast(),
                )
        };
    }

    pub unsafe fn dispatch(&self, group_count_x: u32, group_count_y: u32, group_count_z: u32) {
        unsafe {
            self.device_ref
                .cmd_dispatch(self.inner, group_count_x, group_count_y, group_count_z)
        };
    }

    pub unsafe fn bind_descriptor_sets(
        &self,
        pipeline_bind_point: vk::PipelineBindPoint,
        layout: &Program,
        first_set: u32,
        descriptor_sets: &[vk::DescriptorSet],
        dynamic_offsets: &[u32],
    ) {
        unsafe {
            self.device_ref.cmd_bind_descriptor_sets(
                self.inner,
                pipeline_bind_point,
                layout.pipeline_layout,
                first_set,
                descriptor_sets,
                dynamic_offsets,
            )
        }
    }

    pub unsafe fn image_barrier(&self, barriers: &[vk::ImageMemoryBarrier2]) {
        let dependency_info = vk::DependencyInfo::default().image_memory_barriers(barriers);
        unsafe { self.device_ref.cmd_pipeline_barrier2(self.inner, &dependency_info) };
    }

    pub unsafe fn copy_buffer_to_image(
        &self,
        src_buffer: vk::Buffer,
        dst_image: vk::Image,
        dst_image_layout: vk::ImageLayout,
        regions: &[vk::BufferImageCopy],
    ) {
        unsafe {
            self.device_ref
                .cmd_copy_buffer_to_image(self.inner, src_buffer, dst_image, dst_image_layout, regions)
        };
    }

    pub unsafe fn begin_query(&self, query_pool: vk::QueryPool, query: u32, flags: vk::QueryControlFlags) {
        unsafe { self.device_ref.cmd_begin_query(self.inner, query_pool, query, flags) };
    }

    pub unsafe fn end_query(&self, query_pool: vk::QueryPool, query: u32) {
        unsafe { self.device_ref.cmd_end_query(self.inner, query_pool, query) };
    }

    pub unsafe fn reset_query_pool(&self, pool: vk::QueryPool, first_query: u32, query_count: u32) {
        unsafe {
            self.device_ref
                .cmd_reset_query_pool(self.inner, pool, first_query, query_count)
        };
    }

    pub unsafe fn write_timestamp(
        &self,
        pipeline_stage: vk::PipelineStageFlags,
        query_pool: vk::QueryPool,
        query: u32,
    ) {
        unsafe {
            self.device_ref
                .cmd_write_timestamp(self.inner, pipeline_stage, query_pool, query)
        };
    }

    pub unsafe fn begin_rendering(
        &self,
        color_image_view: &LImageView,
        color_load_op: vk::AttachmentLoadOp,
        color_store_op: vk::AttachmentStoreOp,
        clear_color: vk::ClearValue,
        depth_image_view: &LImageView,
        depth_load_op: vk::AttachmentLoadOp,
        depth_store_op: vk::AttachmentStoreOp,
        clear_depth: vk::ClearValue,
        render_area: vk::Rect2D,
    ) {
        let color_attachments = [vk::RenderingAttachmentInfo::default()
            .clear_value(clear_color)
            .image_view(color_image_view.inner)
            .image_layout(vk::ImageLayout::ATTACHMENT_OPTIMAL)
            .load_op(color_load_op)
            .store_op(color_store_op)];

        let depth_attachment = vk::RenderingAttachmentInfo::default()
            .clear_value(clear_depth)
            .image_view(depth_image_view.inner)
            .image_layout(vk::ImageLayout::ATTACHMENT_OPTIMAL)
            .load_op(depth_load_op)
            .store_op(depth_store_op);

        let rendering_info = vk::RenderingInfo::default()
            .render_area(render_area)
            .layer_count(1)
            .color_attachments(&color_attachments)
            .depth_attachment(&depth_attachment);

        unsafe {
            self.device_ref
                .vk_device()
                .cmd_begin_rendering(self.inner, &rendering_info)
        };
    }

    pub unsafe fn begin_rendering_color(
        &self,
        color_image_view: &LImageView,
        color_load_op: vk::AttachmentLoadOp,
        color_store_op: vk::AttachmentStoreOp,
        clear_color: vk::ClearValue,
        render_area: vk::Rect2D,
    ) {
        let color_attachments = [vk::RenderingAttachmentInfo::default()
            .clear_value(clear_color)
            .image_view(color_image_view.inner)
            .image_layout(vk::ImageLayout::ATTACHMENT_OPTIMAL)
            .load_op(color_load_op)
            .store_op(color_store_op)];

        let rendering_info = vk::RenderingInfo::default()
            .render_area(render_area)
            .layer_count(1)
            .color_attachments(&color_attachments);

        unsafe {
            self.device_ref
                .vk_device()
                .cmd_begin_rendering(self.inner, &rendering_info)
        };
    }

    pub unsafe fn end_rendering(&self) {
        unsafe {
            self.device_ref.vk_device().cmd_end_rendering(self.inner);
        }
    }

    pub unsafe fn draw(&self, vertex_count: u32, instance_count: u32, first_vertex: u32, first_instance: u32) {
        unsafe {
            self.device_ref.vk_device().cmd_draw(
                self.inner,
                vertex_count,
                instance_count,
                first_vertex,
                first_instance,
            );
        }
    }

    pub unsafe fn draw_mesh_tasks_indirect(
        &self,
        buffer: vk::Buffer,
        offset: vk::DeviceSize,
        draw_count: u32,
        stride: u32,
    ) {
        unsafe {
            self.device_ref
                .device()
                .mesh_ext_loader
                .cmd_draw_mesh_tasks_indirect(self.inner, buffer, offset, draw_count, stride)
        };
    }

    pub unsafe fn push_constants(
        &self,
        layout: vk::PipelineLayout,
        stage_flags: vk::ShaderStageFlags,
        offset: u32,
        constants: &[u8],
    ) {
        unsafe {
            self.device_ref
                .vk_device()
                .cmd_push_constants(self.inner, layout, stage_flags, offset, constants);
        }
    }

    pub unsafe fn build_acceleration_structures(
        &self,
        infos: &[vk::AccelerationStructureBuildGeometryInfoKHR<'_>],
        build_range_infos: &[&[vk::AccelerationStructureBuildRangeInfoKHR]],
    ) {
        unsafe {
            self.device_ref
                .device()
                .acc_struct_loader
                .cmd_build_acceleration_structures(self.inner, infos, build_range_infos);
        }
    }

    pub unsafe fn write_acceleration_structures_properties(
        &self,
        structures: &[vk::AccelerationStructureKHR],
        query_type: vk::QueryType,
        query_pool: vk::QueryPool,
        first_query: u32,
    ) {
        unsafe {
            self.device_ref
                .device()
                .acc_struct_loader
                .cmd_write_acceleration_structures_properties(
                    self.inner,
                    structures,
                    query_type,
                    query_pool,
                    first_query,
                );
        }
    }

    pub unsafe fn copy_acceleration_structure(&self, info: &vk::CopyAccelerationStructureInfoKHR<'_>) {
        unsafe {
            self.device_ref
                .device()
                .acc_struct_loader
                .cmd_copy_acceleration_structure(self.inner, info);
        }
    }
}

impl Drop for LCommandBuffer<'_> {
    fn drop(&mut self) {
        unsafe {
            self.device_ref
                .free_command_buffers(self.device_ref.command_pool(), &[self.inner]);
        }
    }
}

pub struct LSampler<'d> {
    pub inner: vk::Sampler,
    pub(crate) device_ref: LDeviceRef<'d>,
}

impl LSampler<'_> {
    pub(crate) fn new(sampler: vk::Sampler) -> Self {
        Self {
            inner: sampler,
            device_ref: LDeviceRef { _marker: PhantomData },
        }
    }
}

impl Drop for LSampler<'_> {
    fn drop(&mut self) {
        unsafe {
            self.device_ref.destroy_sampler(self.inner, None);
        }
    }
}

pub struct LSemaphore<'d> {
    pub inner: vk::Semaphore,
    pub(crate) device_ref: LDeviceRef<'d>,
}

impl LSemaphore<'_> {
    pub(crate) fn new(semaphore: vk::Semaphore) -> Self {
        Self {
            inner: semaphore,
            device_ref: LDeviceRef { _marker: PhantomData },
        }
    }
}

impl Drop for LSemaphore<'_> {
    fn drop(&mut self) {
        unsafe {
            self.device_ref.destroy_semaphore(self.inner, None);
        }
    }
}

pub struct LFence<'d> {
    pub inner: vk::Fence,
    pub(crate) device_ref: LDeviceRef<'d>,
}

impl LFence<'_> {
    pub(crate) fn new(fence: vk::Fence) -> Self {
        Self {
            inner: fence,
            device_ref: LDeviceRef { _marker: PhantomData },
        }
    }
}

impl Drop for LFence<'_> {
    fn drop(&mut self) {
        unsafe {
            self.device_ref.destroy_fence(self.inner, None);
        }
    }
}

pub struct LDescriptorPool<'d> {
    pub inner: vk::DescriptorPool,
    pub(crate) device_ref: LDeviceRef<'d>,
}

impl LDescriptorPool<'_> {
    pub(crate) fn new(pool: vk::DescriptorPool) -> Self {
        Self {
            inner: pool,
            device_ref: LDeviceRef { _marker: PhantomData },
        }
    }
}

impl Drop for LDescriptorPool<'_> {
    fn drop(&mut self) {
        unsafe {
            self.device_ref.destroy_descriptor_pool(self.inner, None);
        }
    }
}

#[repr(transparent)]
pub struct LAccelerationStructureKHR<'d> {
    pub inner: vk::AccelerationStructureKHR,
    pub(crate) device_ref: LDeviceRef<'d>,
}

impl LAccelerationStructureKHR<'_> {
    pub(crate) fn new(inner: vk::AccelerationStructureKHR) -> Self {
        Self {
            inner,
            device_ref: LDeviceRef { _marker: PhantomData },
        }
    }
}

impl Drop for LAccelerationStructureKHR<'_> {
    fn drop(&mut self) {
        unsafe {
            self.device_ref
                .device()
                .acc_struct_loader
                .destroy_acceleration_structure(self.inner, None);
        }
    }
}
