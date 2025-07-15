#![allow(clippy::too_many_arguments)]
#![allow(clippy::missing_safety_doc)]
use ash::vk;

pub mod device;
pub mod errors;
pub mod gpu_stats;
pub mod logger;
pub mod macros;
pub mod pipelines;
pub mod program;
pub mod resources;
pub mod shaders;
pub mod swapchain;
pub mod utils;

pub const MAX_FRAMES_IN_FLIGHT: usize = 3;

pub fn buffer_barrier(
    buffer: &vk::Buffer,
    src_access: vk::AccessFlags2,
    dst_access: vk::AccessFlags2,
    src_stage_mask: vk::PipelineStageFlags2,
    dst_stage_mask: vk::PipelineStageFlags2,
) -> vk::BufferMemoryBarrier2<'_> {
    vk::BufferMemoryBarrier2::default()
        .buffer(*buffer)
        .src_access_mask(src_access)
        .dst_access_mask(dst_access)
        .src_stage_mask(src_stage_mask)
        .dst_stage_mask(dst_stage_mask)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .offset(0)
        .size(vk::WHOLE_SIZE)
}
