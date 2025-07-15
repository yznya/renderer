use crate::image_utils::{write_cubemap_to_ktx_file, write_image_to_ktx_file};
use anyhow::Result;
use ash::vk;
use glam::vec4;
use lava::device::LDeviceRef;
use lava::pipelines::{NoSpecializationConstants, compile_shaders, create_compute_pipeline};
use lava::pipelines::{PipelineInfo, create_pipeline};
use lava::resources::{DescriptorInfo, LImageView};
use lava::utils::clear_color;
use lava::{image_barrier, with};

pub(crate) fn compute_brdf_lut(device: LDeviceRef, command_pool: vk::CommandPool, output_file: &str) -> Result<()> {
    let width: u32 = 256;
    let height: u32 = 256;
    let buffer_size = (width * height * 4) as usize * size_of::<f32>();
    let buffer = device.create_buffer(
        "BRDF",
        (buffer_size) as u64,
        vk::BufferUsageFlags::STORAGE_BUFFER,
        gpu_allocator::MemoryLocation::CpuToGpu,
    )?;
    compile_shaders()?;

    let program = create_compute_pipeline(device, "brdf.comp.glsl", NoSpecializationConstants)?;

    let info = vk::CommandBufferAllocateInfo::default()
        .command_pool(command_pool)
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_buffer_count(1);

    let cmd = unsafe { device.allocate_command_buffers(&info) }?.remove(0);

    let info = vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

    unsafe {
        device.reset_command_pool(command_pool, vk::CommandPoolResetFlags::empty())?;
        device.begin_command_buffer(cmd.inner, &info)?;
        let descriptor_writes = [DescriptorInfo::buffer(buffer.inner)];

        cmd.bind_compute_pipeline(&program);
        cmd.push_descriptors(&program, 0, &descriptor_writes);

        device.cmd_dispatch(
            cmd.inner,
            width / program.local_size_x,
            height / program.local_size_y,
            1,
        );

        device.end_command_buffer(cmd.inner)?;
        let commands = [cmd.inner];
        let info = vk::SubmitInfo::default().command_buffers(&commands);
        device.submit_compute(&info, vk::Fence::null())?;

        device.queue_wait_idle(device.compute_queue())?;
    };

    unsafe {
        write_image_to_ktx_file(
            width,
            height,
            vk::Format::R32G32B32A32_SFLOAT,
            output_file,
            false,
            1,
            move |get_data_dst| {
                std::ptr::copy_nonoverlapping::<u8>(
                    buffer.allocation.mapped_ptr().unwrap().cast().as_ptr(),
                    get_data_dst(0).cast(),
                    buffer_size,
                );
            },
        )?;
    };

    std::mem::forget(cmd);
    Ok(())
}

#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
struct EnvMapPC {
    face: u32,
    roughness: f32,
    sample_count: u32,
    width: u32,
    height: u32,
    distribution: u32,
}

#[derive(Debug, Clone, Copy)]
#[repr(u32)]
pub(crate) enum Distributions {
    Lambertian,
    Ggx,
    Charlie,
}

pub(crate) fn compute_env_map(
    device: LDeviceRef,
    distribution: Distributions,
    input: &str,
    output: &str,
) -> Result<()> {
    let staging_buffer = device.create_mapped_buffer(
        "staging",
        (2000 << 20) as u64,
        vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST,
    )?;

    let (skybox, _) = unsafe { device.load_cubemap(&staging_buffer, input) }?;

    let width: u32 = 256;
    let height: u32 = 256;
    let mipmaps = width.ilog2();
    let format = vk::Format::R32G32B32A32_SFLOAT;

    assert_eq!(width, height);
    let program = create_pipeline(
        device,
        PipelineInfo {
            shaders: &["fullscreen_triangle.vert.glsl", "filter_envmap.frag.glsl"],
            viewport_extent: vk::Extent2D { width, height },
            color_format: format,
            specialization_constants: Some(NoSpecializationConstants),
            cull_mode: vk::CullModeFlags::NONE,
            depth_test: false,
            has_depth_attachment: false,
            dynamic_states: vec![vk::DynamicState::VIEWPORT],
            ..Default::default()
        },
    )?;
    let info = vk::CommandPoolCreateInfo::default()
        .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
        .queue_family_index(device.graphics_queue_index());
    let command_pool = unsafe { device.create_command_pool(&info, None) }?;
    let info = vk::CommandBufferAllocateInfo::default()
        .command_pool(command_pool)
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_buffer_count(1);
    let cmd = unsafe { device.allocate_command_buffers(&info) }?.remove(0);

    let env_map_image = unsafe {
        device.create_cubemap_image_with_view(
            width,
            height,
            format,
            mipmaps,
            with!(vk::ImageUsageFlags => {SAMPLED | TRANSFER_SRC | COLOR_ATTACHMENT}),
        )?
    };

    let sampler = unsafe {
        device.create_sampler(
            &vk::SamplerCreateInfo::default()
                .min_lod(0.0)
                .max_lod(16.0)
                .mag_filter(vk::Filter::LINEAR)
                .min_filter(vk::Filter::LINEAR)
                .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
                .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE),
        )
    }?;

    let image_views: Vec<Vec<LImageView>> = (0..6)
        .map(|i| {
            let mut image_view_mips: Vec<LImageView> = Vec::new();
            for level in 0..mipmaps {
                image_view_mips.push(unsafe {
                    device
                        .create_image_view_for_layer(
                            env_map_image.image.inner,
                            format,
                            vk::ImageAspectFlags::COLOR,
                            level,
                            1,
                            i,
                            1,
                        )
                        .unwrap()
                });
            }

            image_view_mips
        })
        .collect();

    for (face, face_image_views) in image_views.iter().enumerate() {
        for mip_level in 0..mipmaps {
            unsafe {
                device.reset_command_buffer(cmd.inner, vk::CommandBufferResetFlags::empty())?;
                let info = vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
                device.begin_command_buffer(cmd.inner, &info)?;

                let env_barrier = image_barrier!(
                    image: env_map_image.image.inner,
                    access: empty => COLOR_ATTACHMENT_WRITE,
                    layout: UNDEFINED => COLOR_ATTACHMENT_OPTIMAL,
                    stage: empty => COLOR_ATTACHMENT_OUTPUT,
                    aspect: COLOR
                );
                cmd.image_barrier(&[env_barrier]);

                cmd.begin_rendering_color(
                    &face_image_views[mip_level as usize],
                    vk::AttachmentLoadOp::CLEAR,
                    vk::AttachmentStoreOp::STORE,
                    clear_color(vec4(0.0, 0.0, 0.0, 1.0)),
                    vk::Rect2D::default().extent(vk::Extent2D {
                        width: width >> mip_level,
                        height: height >> mip_level,
                    }),
                );
                device.cmd_set_viewport(
                    cmd.inner,
                    0,
                    &[vk::Viewport::default()
                        .x(0.0)
                        .y((height >> mip_level) as f32)
                        .width((width >> mip_level) as f32)
                        .height(-((height >> mip_level) as f32))
                        .min_depth(0.0)
                        .max_depth(1.0)],
                );

                let descriptor_writes = [DescriptorInfo::image_sampler(
                    &sampler,
                    &skybox.image_view,
                    vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                )];

                cmd.push_descriptors(&program, 0, &descriptor_writes);
                let constants = EnvMapPC {
                    face: face as u32,
                    roughness: mip_level as f32 / (mipmaps - 1) as f32,
                    sample_count: 1024,
                    width,
                    height,
                    distribution: distribution as u32,
                };

                cmd.push_constants(
                    program.pipeline_layout,
                    vk::ShaderStageFlags::ALL_GRAPHICS,
                    0,
                    bytemuck::bytes_of(&constants),
                );

                cmd.bind_graphics_pipeline(&program);
                cmd.draw(3, 1, 0, 0);
                cmd.end_rendering();

                device.end_command_buffer(cmd.inner)?;
                let commands = [cmd.inner];
                let info = vk::SubmitInfo::default().command_buffers(&commands);
                device.submit_graphics(&info, vk::Fence::null())?;

                device.queue_wait_idle(device.graphics_queue())?;
            };
        }
    }

    unsafe {
        device.reset_command_buffer(cmd.inner, vk::CommandBufferResetFlags::empty())?;
        let info = vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        device.begin_command_buffer(cmd.inner, &info)?;

        let env_barrier = image_barrier!(
            image: env_map_image.image.inner,
            access: COLOR_ATTACHMENT_WRITE => TRANSFER_READ,
            layout: COLOR_ATTACHMENT_OPTIMAL => TRANSFER_SRC_OPTIMAL,
            stage: COLOR_ATTACHMENT_OUTPUT => TRANSFER,
            aspect: COLOR
        );
        cmd.image_barrier(&[env_barrier]);

        device.end_command_buffer(cmd.inner)?;
        let commands = [cmd.inner];
        let info = vk::SubmitInfo::default().command_buffers(&commands);
        device.submit_graphics(&info, vk::Fence::null())?;

        device.queue_wait_idle(device.graphics_queue())?;

        write_cubemap_to_ktx_file(width, format, output, false, mipmaps, |face, get_dst| {
            for level in 0..mipmaps {
                device
                    .reset_command_buffer(cmd.inner, vk::CommandBufferResetFlags::empty())
                    .unwrap();
                let info = vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
                device.begin_command_buffer(cmd.inner, &info).unwrap();

                let region = vk::BufferImageCopy::default()
                    .image_extent(vk::Extent3D {
                        width: width >> level,
                        height: height >> level,
                        depth: 1,
                    })
                    .image_subresource(
                        vk::ImageSubresourceLayers::default()
                            .aspect_mask(vk::ImageAspectFlags::COLOR)
                            .mip_level(level)
                            .base_array_layer(face as u32)
                            .layer_count(1),
                    );

                device.cmd_copy_image_to_buffer(
                    cmd.inner,
                    env_map_image.image.inner,
                    vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    staging_buffer.buffer.inner,
                    &[region],
                );

                device.end_command_buffer(cmd.inner).unwrap();
                let commands = [cmd.inner];
                let info = vk::SubmitInfo::default().command_buffers(&commands);
                device.submit_graphics(&info, vk::Fence::null()).unwrap();
                device.queue_wait_idle(device.graphics_queue()).unwrap();

                std::ptr::copy_nonoverlapping::<u8>(
                    staging_buffer.memory_map.cast(),
                    get_dst(level).cast(),
                    ((width >> level) * (height >> level) * 4 * 4) as usize,
                );
            }
        })?;
    };

    // TODO: very hacky workaround because cmd buffers don't know which pool they were allocated from
    std::mem::forget(cmd);
    // TODO: move command pools to LDevice
    unsafe {
        device.destroy_command_pool(command_pool, None);
    }
    Ok(())
}
