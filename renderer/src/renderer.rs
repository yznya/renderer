use acceleration_sturct_utils::{build_blas, build_tlas};
use core::f32;
use egui::Ui;
use lava::MAX_FRAMES_IN_FLIGHT;
use rayon::prelude::*;
use render_graph::RenderGraph;
use scenery::RendererConfig;
use std::collections::HashMap;
use std::{fmt::Debug, path::Path, sync::mpsc::Receiver, time::Instant};

use anyhow::{Context, Result, anyhow};
use ash::vk;
use glam::{Mat4, Vec3, Vec4};
use gpu_allocator::MemoryLocation;
use image_loader::{load_image, load_pixel_data};
use lava::pipelines::{
    MAX_DESCRIPTOR_COUNT, NoSpecializationConstants, PipelineInfo, compile_shaders, create_compute_pipeline,
    create_pipeline,
};
use lava::resources::{
    LAccelerationStructureKHR, LCommandBuffer, LDescriptorPool, LFence, LImageView, LSampler, LSemaphore,
};
use lava::{
    buffer_barrier,
    device::{LDeviceRef, PresentResult},
    gpu_stats::GpuStats,
    resources::{LBuffer, LImageWithView, LMappedBuffer},
    swapchain::LSwapchain,
};
use lava::{image_barrier, program::Program, with};
use notify::{RecursiveMode, Watcher};
use pass::{CullPass, DepthPyramidPass, DrawOpaquePass, GridPass, ShadingPass, SkyBoxPass};
use scenery::mesh::AlphaMode;
use scenery::scene::Scene;
use winit::window::Window;

use crate::math::{get_mip_levels, normalize_plane, power_2_floor};

pub(crate) mod acceleration_sturct_utils;
pub(crate) mod egui_integration;
pub mod image_loader;
pub(crate) mod pass;
pub(crate) mod render_graph;

struct MeshShaderSpecialization {
    late: bool,
    pass: AlphaMode,
}

impl From<MeshShaderSpecialization> for [u32; 2] {
    fn from(val: MeshShaderSpecialization) -> Self {
        [val.late as u32, val.pass as u32]
    }
}

#[repr(C)]
#[derive(Debug)]
struct UniformBufferObject {
    view_proj: Mat4,
    view: Mat4,
    proj: Mat4,
    camera_position: Vec3,
    _padding: u32,
    near: f32,
    far: f32,
    depth_pyramid_width: f32,
    depth_pyramid_height: f32,
    frustum: Vec4,
    draw_count: u32,
    lod_enabled: u32,
    cull_enabled: u32,
    lod_target: f32,
    screen_width: f32,
    screen_height: f32,
    sun_direction: Vec4,
}

pub(crate) struct WindowResources<'d> {
    swapchain: Option<LSwapchain<'d>>,
    egui_integration: egui_integration::Integration<'d>,

    early_draw_program: Option<Program<'d>>,
    late_draw_program: Option<Program<'d>>,
    mask_alpha_draw_program: Option<Program<'d>>,
    early_cull_program: Option<Program<'d>>,
    late_cull_program: Option<Program<'d>>,
    mask_alpha_cull_program: Option<Program<'d>>,
    depth_reduce_program: Option<Program<'d>>,
    skybox_program: Option<Program<'d>>,
    grid_program: Option<Program<'d>>,
    shading_program: Option<Program<'d>>,

    command_buffers: Vec<LCommandBuffer<'d>>,
    image_available_semaphore: Vec<LSemaphore<'d>>,
    render_finished_semaphore: Vec<LSemaphore<'d>>,
    in_flight_fence: Vec<LFence<'d>>,

    uniform_buffers: Vec<LMappedBuffer<'d>>,

    depth_image: LImageWithView<'d>,

    color_images: Vec<LImageWithView<'d>>,
    swapchain_image_views: Vec<LImageView<'d>>,

    depth_pyramid_image: LImageWithView<'d>,
    depth_pyramid_mips: Vec<LImageView<'d>>,
    depth_sampler: LSampler<'d>,
}

const COLOR_FORMAT: vk::Format = vk::Format::R32G32_UINT;

impl<'d> WindowResources<'d> {
    unsafe fn new(device: LDeviceRef<'d>, swapchain: LSwapchain<'d>, window: &Window) -> Result<Self> {
        let _span = tracy_client::span!("Create Window Resources");
        let color_images = (0..swapchain.images().len())
            .map(|_| unsafe {
                device.create_image_with_view_and_mips(
                    swapchain.extent().width,
                    swapchain.extent().height,
                    COLOR_FORMAT,
                    with!(vk::ImageUsageFlags => { COLOR_ATTACHMENT | STORAGE  }),
                    1,
                )
            })
            .collect::<Result<Vec<_>, _>>()?;

        let swapchain_image_views = swapchain
            .images()
            .iter()
            .map(|image| unsafe {
                device.create_image_view(*image, swapchain.format(), vk::ImageAspectFlags::COLOR, 0, 1)
            })
            .collect::<Result<Vec<_>, _>>()?;

        let uniform_buffers = unsafe { create_uniform_buffers(device, MAX_FRAMES_IN_FLIGHT) }?;

        let depth_image = unsafe {
            device.create_depth_image_with_view(
                swapchain.extent().width,
                swapchain.extent().height,
                vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT | vk::ImageUsageFlags::SAMPLED,
            )
        }?;

        let pyramid_width = power_2_floor(swapchain.extent().width);
        let pyramid_height = power_2_floor(swapchain.extent().height);
        let mip_levels = get_mip_levels(pyramid_width, pyramid_height);
        let depth_pyramid_image = unsafe {
            device.create_image_with_view_and_mips(
                pyramid_width,
                pyramid_height,
                vk::Format::R32_SFLOAT,
                with!(vk::ImageUsageFlags => {SAMPLED | STORAGE | TRANSFER_SRC}),
                mip_levels,
            )?
        };

        let depth_pyramid_mips = (0..mip_levels)
            .map(|l| unsafe {
                device.create_image_view(
                    depth_pyramid_image.image.inner,
                    vk::Format::R32_SFLOAT,
                    vk::ImageAspectFlags::COLOR,
                    l,
                    1,
                )
            })
            .collect::<Result<Vec<_>, _>>()?;

        let mut reduction_info =
            vk::SamplerReductionModeCreateInfo::default().reduction_mode(vk::SamplerReductionMode::MIN);

        let info = vk::SamplerCreateInfo::default()
            .min_lod(0.0)
            .max_lod(16.0)
            .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .mipmap_mode(vk::SamplerMipmapMode::NEAREST)
            .mag_filter(vk::Filter::LINEAR)
            .min_filter(vk::Filter::LINEAR)
            .push_next(&mut reduction_info);

        let depth_sampler = unsafe { device.create_sampler(&info) }?;

        let command_buffers = unsafe { create_command_buffer(device, MAX_FRAMES_IN_FLIGHT) }?;

        let image_available_semaphore = (0..MAX_FRAMES_IN_FLIGHT)
            .map(|_| unsafe { device.create_semaphore(&vk::SemaphoreCreateInfo::default()) })
            .collect::<Result<Vec<_>, _>>()?;

        let render_finished_semaphore = (0..MAX_FRAMES_IN_FLIGHT)
            .map(|_| unsafe { device.create_semaphore(&vk::SemaphoreCreateInfo::default()) })
            .collect::<Result<Vec<_>, _>>()?;

        let fence_info = vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED);
        let in_flight_fence = (0..MAX_FRAMES_IN_FLIGHT)
            .map(|_| unsafe { device.create_fence(&fence_info) })
            .collect::<Result<Vec<_>, _>>()?;

        let cmd = unsafe { device.begin_single_time_command()? };

        // rendering loop expects these images to be in a specific format
        let mut barriers = vec![image_barrier!(
            image: depth_pyramid_image.image.inner,
            access: empty => SHADER_WRITE,
            layout: UNDEFINED => SHADER_READ_ONLY_OPTIMAL,
            stage: NONE => COMPUTE_SHADER,
            aspect: COLOR
        )];

        barriers.extend(color_images.iter().map(|image| {
            image_barrier!(
                image: image.image.inner,
                access: empty => COLOR_ATTACHMENT_WRITE ,
                layout: UNDEFINED => COLOR_ATTACHMENT_OPTIMAL,
                stage: NONE => COLOR_ATTACHMENT_OUTPUT,
                aspect: COLOR
            )
        }));

        unsafe {
            cmd.image_barrier(&barriers);
        };

        unsafe { device.end_single_time_command(cmd)? };

        let egui_integration = egui_integration::Integration::new(
            device,
            &swapchain,
            window,
            window.scale_factor(),
            egui::FontDefinitions::default(),
            egui::Style::default(),
        );
        egui_integration.context().set_theme(egui::Theme::Dark);

        let mut res = Self {
            swapchain: Some(swapchain),
            egui_integration,
            early_draw_program: Default::default(),
            late_draw_program: Default::default(),
            mask_alpha_draw_program: Default::default(),
            early_cull_program: Default::default(),
            late_cull_program: Default::default(),
            mask_alpha_cull_program: Default::default(),
            depth_reduce_program: Default::default(),
            skybox_program: Default::default(),
            grid_program: Default::default(),
            shading_program: Default::default(),
            command_buffers,
            image_available_semaphore,
            render_finished_semaphore,
            in_flight_fence,
            uniform_buffers,
            depth_image,
            color_images,
            swapchain_image_views,
            depth_pyramid_image,
            depth_pyramid_mips,
            depth_sampler,
        };
        unsafe { res.init_pipelines(device) }?;

        Ok(res)
    }

    unsafe fn init_pipelines(&mut self, device: LDeviceRef<'d>) -> Result<()> {
        compile_shaders()?;
        unsafe { self.destroy_pipelines(device) };

        let swapchain = self.swapchain.as_ref().context("Swapchain not initialized")?;

        self.early_draw_program = Some(create_pipeline(
            device,
            PipelineInfo {
                shaders: &["shader.task.glsl", "shader.mesh.glsl", "shader.frag.glsl"],
                specialization_constants: Some(MeshShaderSpecialization {
                    late: false,
                    pass: AlphaMode::Opaque,
                }),
                viewport_extent: swapchain.extent(),

                color_format: COLOR_FORMAT,
                ..Default::default()
            },
        )?);

        self.late_draw_program = Some(create_pipeline(
            device,
            PipelineInfo {
                shaders: &["shader.task.glsl", "shader.mesh.glsl", "shader.frag.glsl"],
                specialization_constants: Some(MeshShaderSpecialization {
                    late: true,
                    pass: AlphaMode::Opaque,
                }),
                viewport_extent: swapchain.extent(),
                color_format: COLOR_FORMAT,
                ..Default::default()
            },
        )?);

        self.mask_alpha_draw_program = Some(create_pipeline(
            device,
            PipelineInfo {
                shaders: &["shader.task.glsl", "shader.mesh.glsl", "shader.frag.glsl"],
                specialization_constants: Some(MeshShaderSpecialization {
                    late: true,
                    pass: AlphaMode::Mask,
                }),
                viewport_extent: swapchain.extent(),
                color_format: COLOR_FORMAT,
                cull_mode: vk::CullModeFlags::NONE,
                ..Default::default()
            },
        )?);

        self.early_cull_program = Some(create_compute_pipeline(
            device,
            "cull.comp.glsl",
            MeshShaderSpecialization {
                late: false,
                pass: AlphaMode::Opaque,
            },
        )?);

        self.late_cull_program = Some(create_compute_pipeline(
            device,
            "cull.comp.glsl",
            MeshShaderSpecialization {
                late: true,
                pass: AlphaMode::Opaque,
            },
        )?);

        self.mask_alpha_cull_program = Some(create_compute_pipeline(
            device,
            "cull.comp.glsl",
            MeshShaderSpecialization {
                late: true,
                pass: AlphaMode::Mask,
            },
        )?);

        self.depth_reduce_program = Some(create_compute_pipeline(
            device,
            "depth_reduce.comp.glsl",
            NoSpecializationConstants,
        )?);

        self.skybox_program = Some(create_pipeline(
            device,
            PipelineInfo::<0, NoSpecializationConstants> {
                shaders: &["cubemap.vert.glsl", "cubemap.frag.glsl"],
                viewport_extent: swapchain.extent(),
                color_format: swapchain.format(),
                cull_mode: vk::CullModeFlags::NONE,
                ..Default::default()
            },
        )?);

        self.grid_program = Some(create_pipeline(
            device,
            PipelineInfo::<0, NoSpecializationConstants> {
                shaders: &["grid.vert.glsl", "grid.frag.glsl"],
                viewport_extent: swapchain.extent(),
                color_format: swapchain.format(),
                cull_mode: vk::CullModeFlags::NONE,
                depth_write: false,
                blend_enable: true,
                ..Default::default()
            },
        )?);

        self.shading_program = Some(create_pipeline(
            device,
            PipelineInfo {
                shaders: &["fullscreen_triangle.vert.glsl", "shading.frag.glsl"],
                viewport_extent: swapchain.extent(),
                color_format: swapchain.format(),
                specialization_constants: Some(NoSpecializationConstants),
                cull_mode: vk::CullModeFlags::NONE,
                depth_test: false,
                ..Default::default()
            },
        )?);

        Ok(())
    }

    pub(crate) fn take_swapchain(&mut self) -> Option<LSwapchain<'d>> {
        self.swapchain.take()
    }

    unsafe fn destroy_pipelines(&mut self, device: LDeviceRef) {
        unsafe { device.device_wait_idle().unwrap() };
    }
}

pub(crate) struct SceneResources<'d> {
    vertex_buffer: LBuffer<'d>,
    _blas_buffer: LBuffer<'d>,
    _tlas_buffer: LBuffer<'d>,
    _blases: Vec<LAccelerationStructureKHR<'d>>,
    tlas: LAccelerationStructureKHR<'d>,
    meshlet_vertices_buffer: LBuffer<'d>,
    meshlet_triangles_buffer: LBuffer<'d>,
    meshlets_buffer: LBuffer<'d>,
    meshs_buffer: LBuffer<'d>,
    materials_buffer: LBuffer<'d>,
    models_buffer: LBuffer<'d>,
    draws_buffer: LBuffer<'d>,
    draw_commands_buffer: LBuffer<'d>,
    draw_visiablity_buffer: LBuffer<'d>,
    meshlet_visibility_buffer: LBuffer<'d>,
    _images: HashMap<usize, LImageWithView<'d>>,
    cubemap_image: LImageWithView<'d>,
    brdf_lut_image: LImageWithView<'d>,
    lambertian_image: LImageWithView<'d>,
    ggx_image: LImageWithView<'d>,
    charlie_image: LImageWithView<'d>,
    texture_set: vk::DescriptorSet,
}

impl<'d> SceneResources<'d> {
    fn load_scene(device: LDeviceRef<'d>, gpu_res: &mut GpuResources, scene: &Scene) -> Result<Self> {
        let _span = tracy_client::span!("Loading scene resources");
        let loading_geometry_span = tracy_client::span!("Loading geometry");
        let staging_buffer = device.create_mapped_buffer(
            "staging",
            (500 << 20) as u64,
            vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::STORAGE_BUFFER,
        )?;
        // TODO: check how much memeory we are using, can we use a smaller staging buffer?
        // TODO: we can avoid using the staging buffer by using DEVICE_LOCAL | HOST_VISIBLE memory
        let vertex_buffer = unsafe {
            device.create_buffer_with_data(
                "vertex_buffer",
                &staging_buffer,
                &scene.geometry.vertices,
                with!(vk::BufferUsageFlags => {STORAGE_BUFFER | ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR}),
            )?
        };

        let index_buffer = unsafe {
            device.create_buffer_with_data(
                "index_buffer",
                &staging_buffer,
                &scene.geometry.indicies,
                with!(vk::BufferUsageFlags => {STORAGE_BUFFER | ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR}),
            )?
        };

        let meshlet_vertices_buffer = unsafe {
            device.create_buffer_with_data(
                "meshlet_vertices_buffer",
                &staging_buffer,
                &scene.geometry.meshlet_vertices,
                vk::BufferUsageFlags::STORAGE_BUFFER,
            )?
        };

        let meshlet_triangles_buffer = unsafe {
            device.create_buffer_with_data(
                "meshlet_triangles_buffer",
                &staging_buffer,
                &scene.geometry.meshlet_triangles,
                vk::BufferUsageFlags::STORAGE_BUFFER,
            )?
        };

        let meshlets_buffer = unsafe {
            device.create_buffer_with_data(
                "meshlets_buffer",
                &staging_buffer,
                &scene.geometry.meshlets,
                vk::BufferUsageFlags::STORAGE_BUFFER,
            )?
        };

        let meshs_buffer = unsafe {
            device.create_buffer_with_data(
                "meshs_buffer",
                &staging_buffer,
                &scene.geometry.meshes,
                vk::BufferUsageFlags::STORAGE_BUFFER,
            )?
        };

        let draw_visiablity_buffer = device.create_buffer(
            "draw_visiablity_buffer",
            size_of::<u32>() as u64 * scene.mesh_draws.len() as u64,
            with!(vk::BufferUsageFlags => { STORAGE_BUFFER | TRANSFER_DST }),
            MemoryLocation::GpuOnly,
        )?;

        let meshlet_visibility_count = scene.meshlet_visibility_count.div_ceil(32);
        let meshlet_visibility_buffer = device.create_buffer(
            "meshlet_visibility_buffer",
            size_of::<u32>() as u64 * meshlet_visibility_count as u64,
            with!(vk::BufferUsageFlags => { STORAGE_BUFFER | TRANSFER_DST }),
            MemoryLocation::GpuOnly,
        )?;

        let draws_buffer = unsafe {
            device.create_buffer_with_data(
                "draws_buffer",
                &staging_buffer,
                &scene.mesh_draws,
                vk::BufferUsageFlags::STORAGE_BUFFER,
            )
        }?;

        let materials_buffer = unsafe {
            device.create_buffer_with_data(
                "materials_buffer",
                &staging_buffer,
                &scene.materials,
                vk::BufferUsageFlags::STORAGE_BUFFER,
            )
        }?;

        let models_buffer = unsafe {
            device.create_buffer_with_data(
                "models_buffer",
                &staging_buffer,
                &scene.global_transforms,
                vk::BufferUsageFlags::STORAGE_BUFFER,
            )
        }?;

        let meshlet_count = scene.geometry.meshlets.len();
        let draw_commands_buffer = device.create_buffer(
            "draw_commands_buffer",
            scene.mesh_draws.len() as u64 * meshlet_count.div_ceil(64) as u64 * size_of::<u32>() as u64 * 8,
            with!(vk::BufferUsageFlags => { STORAGE_BUFFER | INDIRECT_BUFFER }),
            MemoryLocation::GpuOnly,
        )?;

        let cmd = unsafe { device.begin_single_time_command()? };

        let draw_buffer_barrier = buffer_barrier(
            &draw_visiablity_buffer.inner,
            vk::AccessFlags2::NONE,
            vk::AccessFlags2::TRANSFER_WRITE,
            vk::PipelineStageFlags2::NONE,
            vk::PipelineStageFlags2::TRANSFER,
        );

        let meshlet_visibility_buffer_barrier = buffer_barrier(
            &meshlet_visibility_buffer.inner,
            vk::AccessFlags2::NONE,
            vk::AccessFlags2::TRANSFER_WRITE,
            vk::PipelineStageFlags2::NONE,
            vk::PipelineStageFlags2::TRANSFER,
        );

        unsafe {
            cmd.buffer_barrier(&[draw_buffer_barrier, meshlet_visibility_buffer_barrier]);
        };

        unsafe { cmd.fill_buffer(&draw_visiablity_buffer, 0, vk::WHOLE_SIZE, 0) };
        unsafe { cmd.fill_buffer(&meshlet_visibility_buffer, 0, vk::WHOLE_SIZE, 0) };

        // rendering loop expects these buffers to be in a specific format
        let draw_buffer_barrier = buffer_barrier(
            &draw_visiablity_buffer.inner,
            vk::AccessFlags2::TRANSFER_WRITE,
            vk::AccessFlags2::SHADER_WRITE,
            vk::PipelineStageFlags2::TRANSFER,
            vk::PipelineStageFlags2::COMPUTE_SHADER,
        );

        let meshlet_visibility_buffer_barrier = buffer_barrier(
            &meshlet_visibility_buffer.inner,
            vk::AccessFlags2::TRANSFER_WRITE,
            vk::AccessFlags2::SHADER_WRITE,
            vk::PipelineStageFlags2::TRANSFER,
            vk::PipelineStageFlags2::COMPUTE_SHADER,
        );

        unsafe {
            cmd.buffer_barrier(&[draw_buffer_barrier, meshlet_visibility_buffer_barrier]);
        };

        unsafe { device.end_single_time_command(cmd)? };
        drop(loading_geometry_span);

        let t = Instant::now();
        let (blases, blas_buffer) =
            unsafe { build_blas(device, scene, &vertex_buffer, &index_buffer, &staging_buffer) }?;

        let (tlas, tlas_buffer) = unsafe { build_tlas(device, scene, &blases, &staging_buffer) }?;

        log::info!("BLAS build time: {:?}", t.elapsed());

        let images: Vec<_> = scene
            .images
            .par_iter()
            .filter_map(|(i, path)| {
                if let Ok(image) = load_pixel_data(path) {
                    Some((*i, image))
                } else {
                    None
                }
            })
            .collect();

        let images: HashMap<_, _> = images
            .iter()
            .map(|(i, image)| (*i, load_image(device, &staging_buffer, image).unwrap()))
            .collect();

        let (cubemap_image, _) = unsafe { device.load_cubemap(&staging_buffer, &scene.env_map.skybox) }?;
        let brdf_lut_image = load_image(device, &staging_buffer, &load_pixel_data(&scene.env_map.brdf_lut)?)?;
        let (lambertian_image, _) = unsafe { device.load_cubemap(&staging_buffer, &scene.env_map.lambertian) }?;
        let (ggx_image, _) = unsafe { device.load_cubemap(&staging_buffer, &scene.env_map.ggx) }?;
        let (charlie_image, _) = unsafe { device.load_cubemap(&staging_buffer, &scene.env_map.charlie) }?;

        drop(staging_buffer);

        let bindings = [vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(MAX_DESCRIPTOR_COUNT)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT)];
        let binding_flags =
            [vk::DescriptorBindingFlags::PARTIALLY_BOUND | vk::DescriptorBindingFlags::VARIABLE_DESCRIPTOR_COUNT];
        let mut binding_flags_info =
            vk::DescriptorSetLayoutBindingFlagsCreateInfo::default().binding_flags(&binding_flags);

        let info = vk::DescriptorSetLayoutCreateInfo::default()
            .bindings(&bindings)
            .push_next(&mut binding_flags_info);

        let descriptor_set_layout = unsafe { device.create_descriptor_set_layout(&info)? };

        let max_dst = [(scene.images.iter().map(|i| i.0).max().unwrap_or_default() + 1) as u32];
        let mut set_allocate_info_count =
            vk::DescriptorSetVariableDescriptorCountAllocateInfo::default().descriptor_counts(&max_dst);

        let set_layouts = [descriptor_set_layout];
        let info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(gpu_res.descriptor_pool.inner)
            .set_layouts(&set_layouts)
            .push_next(&mut set_allocate_info_count);

        let texture_set = unsafe { device.allocate_descriptor_sets(&info)?[0] };

        for image in scene.images.iter() {
            let Some(l_image) = images.get(&image.0) else {
                log::error!("Cannot load image {}", image.1);
                continue;
            };
            let image_info = [vk::DescriptorImageInfo::default()
                .sampler(gpu_res.texture_sampler.inner)
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .image_view(l_image.image_view.inner)];

            let write = [vk::WriteDescriptorSet::default()
                .dst_set(texture_set)
                .dst_binding(0)
                .dst_array_element(image.0 as u32)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(&image_info)
                .descriptor_count(1)];
            unsafe { device.update_descriptor_sets(&write, &[]) };
        }

        unsafe {
            device.destroy_descriptor_set_layout(descriptor_set_layout, None);
        }

        Ok(Self {
            vertex_buffer,
            _blas_buffer: blas_buffer,
            _tlas_buffer: tlas_buffer,
            tlas,
            _blases: blases,
            meshlet_vertices_buffer,
            meshlet_triangles_buffer,
            meshlets_buffer,
            meshs_buffer,
            draws_buffer,
            materials_buffer,
            models_buffer,
            draw_commands_buffer,
            draw_visiablity_buffer,
            meshlet_visibility_buffer,
            _images: images,
            cubemap_image,
            lambertian_image,
            brdf_lut_image,
            ggx_image,
            charlie_image,
            texture_set,
        })
    }

    fn upload_transforms(&self, device: LDeviceRef<'d>, scene: &Scene) -> Result<()> {
        let staging_buffer = device.create_mapped_buffer(
            "staging",
            (50 << 20) as u64,
            vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::STORAGE_BUFFER,
        )?;

        unsafe {
            std::ptr::copy_nonoverlapping(
                scene.global_transforms.as_ptr(),
                staging_buffer.memory_map.cast(),
                scene.global_transforms.len(),
            )
        };

        let size = std::mem::size_of_val(&scene.global_transforms[..]) as u64;

        unsafe { device.copy_buffer(staging_buffer.buffer.inner, self.models_buffer.inner, size) }?;
        unsafe { device.device_wait_idle() }?;

        log::info!("Updated Models buffer, {size}");
        Ok(())
    }
}

pub(crate) struct GpuResources<'d> {
    draw_commands_count_buffer: LBuffer<'d>,
    descriptor_pool: LDescriptorPool<'d>,
    texture_sampler: LSampler<'d>,
    window_resources: Option<WindowResources<'d>>,
    scene_resources: Option<SceneResources<'d>>,
}

impl<'d> GpuResources<'d> {
    fn new(device: LDeviceRef<'d>, window: &Window, config: &RendererConfig) -> Result<Self> {
        let _span = tracy_client::span!("Creating GPU resources");
        let draw_commands_count_buffer = device.create_buffer(
            "draw_commands_count_buffer",
            3 * size_of::<u32>() as u64,
            with!(vk::BufferUsageFlags => { STORAGE_BUFFER | INDIRECT_BUFFER | TRANSFER_DST }),
            MemoryLocation::GpuOnly,
        )?;

        let descriptor_count = 65536;
        let pool_sizes = [vk::DescriptorPoolSize {
            ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
            descriptor_count,
        }];

        let info = vk::DescriptorPoolCreateInfo::default()
            .max_sets(1)
            .pool_sizes(&pool_sizes);

        let descriptor_pool = unsafe { device.create_descriptor_pool(&info) }?;

        let info = vk::SamplerCreateInfo::default()
            .min_lod(0.0)
            .max_lod(16.0)
            .address_mode_u(vk::SamplerAddressMode::REPEAT)
            .address_mode_v(vk::SamplerAddressMode::REPEAT)
            .address_mode_w(vk::SamplerAddressMode::REPEAT)
            .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
            .mag_filter(vk::Filter::LINEAR)
            .min_filter(vk::Filter::LINEAR);

        let swapchain = LSwapchain::create(device, window, config.enable_vsync)?;

        let texture_sampler = unsafe { device.create_sampler(&info) }?;
        let window_resources = unsafe { WindowResources::new(device, swapchain, window)? };

        Ok(Self {
            draw_commands_count_buffer,
            descriptor_pool,
            texture_sampler,
            scene_resources: None,
            window_resources: Some(window_resources),
        })
    }
}

pub struct Renderer<'d> {
    pub(crate) device: LDeviceRef<'d>,
    gpu_resources: GpuResources<'d>,
    frame: usize,
    frame_counter: usize,
    pub config: RendererConfig,
    pub gpu_stats: GpuStats<'d>,
    pub(crate) render_graph: RenderGraph,
    _shader_files_watcher: notify::RecommendedWatcher,
    file_events_receiver: Receiver<notify::Result<notify::Event>>,
}

impl<'d> Renderer<'d> {
    pub fn new(device: LDeviceRef<'d>, window: &Window, config: RendererConfig) -> Result<Self> {
        let _span = tracy_client::span!("Renderer initailization");
        let (tx, rx) = std::sync::mpsc::channel::<notify::Result<notify::Event>>();
        let mut watcher = notify::recommended_watcher(tx)?;
        watcher.watch(Path::new("assets/shaders/"), RecursiveMode::NonRecursive)?;

        let gpu_resources = GpuResources::new(device, window, &config)?;

        let mut render_graph = RenderGraph::new();

        render_graph.add_pass(CullPass {
            late: false,
            pass_type: pass::PassType::Early,
            name: "Early Cull",
        });

        render_graph.add_pass(DrawOpaquePass {
            late: false,
            pass_type: pass::PassType::Early,
            name: "Early Draw",
        });

        render_graph.add_pass(DepthPyramidPass {});

        render_graph.add_pass(CullPass {
            late: true,
            pass_type: pass::PassType::Late,
            name: "Late Cull",
        });

        render_graph.add_pass(DrawOpaquePass {
            late: true,
            pass_type: pass::PassType::Late,
            name: "Late Draw",
        });

        render_graph.add_pass(CullPass {
            late: true,
            pass_type: pass::PassType::MaskAlpha,
            name: "Mask Cull",
        });

        render_graph.add_pass(DrawOpaquePass {
            late: true,
            pass_type: pass::PassType::MaskAlpha,
            name: "Mask Draw",
        });

        render_graph.add_pass(ShadingPass {});
        render_graph.add_pass(SkyBoxPass {});
        render_graph.add_pass(GridPass {});

        let gpu_stats = GpuStats::new(device, render_graph.passes.len())?;

        unsafe { device.device_wait_idle() }?;

        Ok(Self {
            device,
            gpu_resources,
            frame: 0,
            frame_counter: 0,
            config,
            gpu_stats,
            _shader_files_watcher: watcher,
            file_events_receiver: rx,
            render_graph,
        })
    }

    pub fn render(
        &mut self,
        window: &Window,
        time: Instant,
        scene: &mut Scene,
        changed_this_frame: &mut [Vec<u32>],
    ) -> Result<()> {
        let _span = tracy_client::span!();

        // TODO: move to window resources
        if self.file_events_receiver.try_recv().is_ok() {
            // drain all change envents
            let mut file_changed = false;
            while let Some(event) = self.file_events_receiver.try_iter().next() {
                if let notify::EventKind::Modify(modify_kind) = event.unwrap().kind
                    && let notify::event::ModifyKind::Data(_) = modify_kind
                {
                    file_changed = true;
                }
            }

            if file_changed {
                unsafe { self.device.device_wait_idle() }?;
                unsafe {
                    if let Err(e) = self
                        .gpu_resources
                        .window_resources
                        .as_mut()
                        .unwrap()
                        .init_pipelines(self.device)
                    {
                        log::error!("{e}");
                    }
                };

                return Ok(());
            }
        }

        if !changed_this_frame.iter().all(|change_list| change_list.is_empty()) {
            self.gpu_resources
                .scene_resources
                .as_ref()
                .unwrap()
                .upload_transforms(self.device, scene)?;

            for change_list in changed_this_frame.iter_mut() {
                change_list.clear();
            }
        }

        let win_res = self
            .gpu_resources
            .window_resources
            .as_ref()
            .expect("Window Resources should never be None when render is called");

        unsafe {
            self.device
                .wait_for_fences(&[win_res.in_flight_fence[self.frame].inner], true, u64::MAX)?;
            self.device.reset_fences(&[win_res.in_flight_fence[self.frame].inner])?;

            let wait_semaphore = win_res.image_available_semaphore[self.frame].inner;
            let signal_semaphore = win_res.render_finished_semaphore[self.frame].inner;

            let result = self.device.swapchain_loader().acquire_next_image(
                **win_res.swapchain.as_ref().context("Swapchain not initialized")?,
                u64::MAX,
                wait_semaphore,
                vk::Fence::null(),
            );

            let image_index = match result {
                Ok((image_index, _)) => image_index as usize,
                Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                    self.reinitialized_swapchain(window)?;
                    return Ok(());
                }
                Err(_) => return Err(anyhow!("Failed to acquire the next image")),
            };

            self.update_uniform_object(time, scene)?;
            self.record_command_buffer(image_index as u32, scene, window)?;

            let win_res = self
                .gpu_resources
                .window_resources
                .as_ref()
                .expect("Window Resources should never be None when render is called");

            let cmd = &win_res.command_buffers[self.frame];

            let gpu_query_span = tracy_client::span!("time_gpu_query_span");
            self.gpu_stats.update_gpu_time(cmd, self.frame)?;
            drop(gpu_query_span);

            self.device.end_command_buffer(cmd.inner)?;
            let signal_semaphores = [signal_semaphore];

            self.device.submit_graphics(
                &vk::SubmitInfo::default()
                    .wait_semaphores(&[wait_semaphore])
                    .wait_dst_stage_mask(&[vk::PipelineStageFlags::TRANSFER])
                    .command_buffers(&[cmd.inner])
                    .signal_semaphores(&signal_semaphores),
                self.gpu_resources.window_resources.as_ref().unwrap().in_flight_fence[self.frame].inner,
            )?;

            let swapchains = [**self
                .gpu_resources
                .window_resources
                .as_ref()
                .unwrap()
                .swapchain
                .as_ref()
                .unwrap()];
            let image_indicies = [image_index as u32];
            let present_info = vk::PresentInfoKHR::default()
                .wait_semaphores(&signal_semaphores)
                .swapchains(&swapchains)
                .image_indices(&image_indicies);

            if let PresentResult::Changed = self.device.present(&present_info)? {
                self.reinitialized_swapchain(window)?;
                return Ok(());
            }

            let gpu_query_span = tracy_client::span!("stats_gpu_query_span");
            self.gpu_stats.update_pipeline_stats()?;
            drop(gpu_query_span);

            self.gpu_stats.update();

            self.frame_counter += 1;
            self.frame = (self.frame + 1) % MAX_FRAMES_IN_FLIGHT;

            tracy_client::frame_mark();

            Ok(())
        }
    }

    pub fn set_scene(&mut self, scene: &Scene) -> Result<()> {
        self.gpu_resources.scene_resources =
            Some(SceneResources::load_scene(self.device, &mut self.gpu_resources, scene)?);

        Ok(())
    }

    pub fn egui(&mut self) -> egui::Context {
        self.gpu_resources
            .window_resources
            .as_mut()
            .unwrap()
            .egui_integration
            .context()
    }

    pub fn handle_events(&mut self, window: &Window, winit_event: &egui_winit::winit::event::WindowEvent) {
        let _ = self
            .gpu_resources
            .window_resources
            .as_mut()
            .unwrap()
            .egui_integration
            .handle_event(window, winit_event);
    }

    pub fn begin_frame(&mut self, window: &Window) {
        self.gpu_resources
            .window_resources
            .as_mut()
            .unwrap()
            .egui_integration
            .begin_frame(window);
    }

    pub fn register_texture(&mut self, image_view: vk::ImageView, sampler: vk::Sampler) {
        self.gpu_resources
            .window_resources
            .as_mut()
            .unwrap()
            .egui_integration
            .register_user_texture(image_view, sampler);
    }

    pub fn unregister_texture(&mut self, texture_id: egui::TextureId) {
        self.gpu_resources
            .window_resources
            .as_mut()
            .unwrap()
            .egui_integration
            .unregister_user_texture(texture_id);
    }

    pub fn display_ui(&mut self, ui: &mut Ui) {
        self.render_graph.display_ui(ui, &self.gpu_stats);
    }

    fn reinitialized_swapchain(&mut self, window: &Window) -> Result<(), anyhow::Error> {
        log::info!("Reinitializing swapchain");
        unsafe { self.device.device_wait_idle() }?;
        let mut swapchain = self
            .gpu_resources
            .window_resources
            .as_mut()
            .unwrap()
            .take_swapchain()
            .unwrap();

        swapchain.resize(self.device, window, self.config.enable_vsync)?;
        self.gpu_resources.window_resources.take();
        self.gpu_resources.window_resources = Some(unsafe { WindowResources::new(self.device, swapchain, window) }?);
        Ok(())
    }

    unsafe fn record_command_buffer(&mut self, image_index: u32, scene: &Scene, window: &Window) -> Result<()> {
        let cmd = &self.gpu_resources.window_resources.as_ref().unwrap().command_buffers[self.frame];

        unsafe {
            self.device
                .reset_command_buffer(cmd.inner, vk::CommandBufferResetFlags::empty())?;

            let begin_info = vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            self.device.begin_command_buffer(cmd.inner, &begin_info)?;

            self.gpu_stats.begin_pipeline_stats_query(cmd);
        };

        self.render_graph.render(
            self.device,
            &self.gpu_resources,
            &mut self.gpu_stats,
            cmd,
            image_index,
            scene,
            self.frame,
        )?;

        let cmd_inner = self.gpu_resources.window_resources.as_ref().unwrap().command_buffers[self.frame].inner;

        // TODO: this reborrowing is ugly, maybe partial borrows can help
        let win_res = self.gpu_resources.window_resources.as_mut().unwrap();
        let color_image = &win_res.color_images[image_index as usize].image;
        let swapchain = win_res.swapchain.as_ref().unwrap();

        let output = win_res.egui_integration.end_frame(window);
        let clipped_meshes = win_res
            .egui_integration
            .context()
            .tessellate(output.shapes, win_res.egui_integration.scale_factor as f32);

        win_res.egui_integration.paint(
            self.device,
            cmd_inner,
            swapchain,
            image_index as usize,
            clipped_meshes,
            output.textures_delta,
        );

        let present_image_barrier = image_barrier!(
            image: swapchain.images()[image_index as usize],
            access: COLOR_ATTACHMENT_WRITE => empty,
            layout: COLOR_ATTACHMENT_OPTIMAL =>  PRESENT_SRC_KHR,
            stage: COLOR_ATTACHMENT_OUTPUT => TRANSFER,
            aspect: COLOR
        );

        let color_image_barrier = image_barrier!(
            image: color_image.inner,
            access: SHADER_READ => COLOR_ATTACHMENT_WRITE | COLOR_ATTACHMENT_READ,
            layout: GENERAL => COLOR_ATTACHMENT_OPTIMAL,
            stage: FRAGMENT_SHADER => COLOR_ATTACHMENT_OUTPUT,
            aspect: COLOR
        );

        // TODO: this reborrowing is ugly, maybe we can use partial borrows
        let cmd = &self.gpu_resources.window_resources.as_ref().unwrap().command_buffers[self.frame];
        unsafe { cmd.image_barrier(&[present_image_barrier, color_image_barrier]) };

        unsafe { self.gpu_stats.end_pipeline_stats_query(cmd) };

        Ok(())
    }

    unsafe fn update_uniform_object(&self, _time: Instant, scene: &Scene) -> Result<()> {
        let win_res = self
            .gpu_resources
            .window_resources
            .as_ref()
            .expect("Window Resources should never be None when update_uniform_object is called");
        let swapchain = win_res.swapchain.as_ref().unwrap();

        // TODO: seems like we can use bytemuck to avoid unsafe code
        let ubo = unsafe { &mut *(win_res.uniform_buffers[self.frame].memory_map as *mut UniformBufferObject) };

        let camera_position = scene.camera.position;
        let near = 0.01;
        ubo.view = scene.camera.view_mat();
        ubo.proj = Mat4::perspective_infinite_reverse_lh(
            f32::to_radians(45.0),
            swapchain.extent().width as f32 / swapchain.extent().height as f32,
            near,
        );
        ubo.proj.col_mut(0)[0] *= -1.0;

        ubo.view_proj = ubo.proj * ubo.view;
        ubo.camera_position = camera_position;
        let frustum_x = normalize_plane(ubo.proj.row(3) + ubo.proj.row(0));
        let frustum_y = normalize_plane(ubo.proj.row(3) + ubo.proj.row(1));

        ubo.frustum.x = frustum_x.x;
        ubo.frustum.y = frustum_x.z;
        ubo.frustum.z = frustum_y.y;
        ubo.frustum.w = frustum_y.z;

        ubo.near = near;
        ubo.far = 500.0;

        ubo.lod_enabled = if self.config.lod_enabled { 1 } else { 0 };
        ubo.cull_enabled = if self.config.cull_enabled { 1 } else { 0 };
        ubo.draw_count = scene.mesh_draws.len() as u32;

        ubo.depth_pyramid_width = power_2_floor(swapchain.extent().width) as f32;
        ubo.depth_pyramid_height = power_2_floor(swapchain.extent().height) as f32;

        ubo.lod_target = (1.0 / ubo.proj.col(1)[1]) * (1.0 / swapchain.extent().height as f32);

        ubo.screen_width = swapchain.extent().width as f32;
        ubo.screen_height = swapchain.extent().height as f32;

        ubo.sun_direction = Vec4::new(scene.sun_direction.x, scene.sun_direction.y, scene.sun_direction.z, 0.0);

        Ok(())
    }
}

impl Drop for Renderer<'_> {
    fn drop(&mut self) {
        unsafe {
            if let Err(e) = self.device.device_wait_idle() {
                log::error!("Failed waiting for device: {e}");
            }
        }
    }
}

// TODO: inline these
unsafe fn create_uniform_buffers(device: LDeviceRef, count: usize) -> Result<Vec<LMappedBuffer>> {
    let mut uniform_buffers = Vec::new();
    for _ in 0..count {
        let buffer = device.create_mapped_buffer(
            "uniform_buffer",
            size_of::<UniformBufferObject>() as u64,
            vk::BufferUsageFlags::UNIFORM_BUFFER,
        )?;

        uniform_buffers.push(buffer);
    }
    Ok(uniform_buffers)
}

unsafe fn create_command_buffer(device: LDeviceRef, count: usize) -> Result<Vec<LCommandBuffer>> {
    let info = vk::CommandBufferAllocateInfo::default()
        .command_pool(device.command_pool())
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_buffer_count(count as u32);

    let command_buffers = unsafe { device.allocate_command_buffers(&info) }?;

    Ok(command_buffers)
}
