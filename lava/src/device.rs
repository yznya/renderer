#![allow(static_mut_refs)]
#![allow(clippy::mut_from_ref)]

use crate::{
    errors::SuitablityError,
    resources::{
        LAccelerationStructureKHR, LBuffer, LCommandBuffer, LDescriptorPool, LFence, LImage, LImageView,
        LImageWithView, LMappedBuffer, LSampler, LSemaphore,
    },
    with,
};
use anyhow::{Context, Result, anyhow};
use ash::{
    Device, Entry, Instance,
    ext::{debug_utils, mesh_shader},
    khr::{acceleration_structure, push_descriptor, surface, swapchain},
    vk::{self, Handle, QueueFlags},
};
use bytemuck::Zeroable;
use core::ffi;
use gpu_allocator::{
    MemoryLocation,
    vulkan::{Allocation, AllocationCreateDesc, AllocationScheme, Allocator, AllocatorCreateDesc},
};
use std::{
    cell::{Cell, UnsafeCell},
    collections::HashSet,
    ffi::CStr,
    marker::PhantomData,
    mem::{ManuallyDrop, MaybeUninit},
    ops::Deref,
    sync::{
        MutexGuard,
        atomic::{AtomicBool, Ordering},
    },
};
use winit::raw_window_handle::{HasDisplayHandle, HasWindowHandle, RawDisplayHandle, RawWindowHandle};

pub const DEVICE_EXTENSTIONS: &[&CStr] = &[
    vk::KHR_SWAPCHAIN_NAME,
    vk::KHR_PUSH_DESCRIPTOR_NAME,
    vk::EXT_MESH_SHADER_NAME,
    vk::KHR_RAY_QUERY_NAME,
    vk::KHR_ACCELERATION_STRUCTURE_NAME,
    vk::KHR_DEFERRED_HOST_OPERATIONS_NAME,
    vk::KHR_FORMAT_FEATURE_FLAGS2_NAME,
];

pub struct ImageInfo {
    pub width: u32,
    pub height: u32,
    pub mipmaps: u32,
}

pub enum PresentResult {
    Changed,
    None,
}

pub(crate) struct LDeviceImpl {
    entry: Entry,
    instance: Instance,
    debug_utils_loader: debug_utils::Instance,
    debug_messenger: vk::DebugUtilsMessengerEXT,
    surface_loader: Option<surface::Instance>,
    physical_device: vk::PhysicalDevice,
    surface: vk::SurfaceKHR,
    device: Device,
    graphics_queue: vk::Queue,
    graphics_queue_index: u32,
    compute_queue: vk::Queue,
    compute_queue_index: u32,
    swapchain_loader: swapchain::Device,
    command_pool: vk::CommandPool,
    props: vk::PhysicalDeviceProperties,
    pub(crate) push_descriptors_loader: push_descriptor::Device,
    pub(crate) mesh_ext_loader: mesh_shader::Device,
    pub(crate) acc_struct_loader: acceleration_structure::Device,
    pub(crate) allocator: ManuallyDrop<Allocator>,
}

#[repr(transparent)]
struct SyncUnsafeCellDevice(UnsafeCell<MaybeUninit<LDeviceImpl>>);

unsafe impl Sync for SyncUnsafeCellDevice {}

static DEVICE: SyncUnsafeCellDevice = SyncUnsafeCellDevice(UnsafeCell::new(MaybeUninit::uninit()));
static DEVICE_INITIATED: AtomicBool = AtomicBool::new(false);

pub type PhantomUnsync = PhantomData<Cell<()>>;
pub type PhantomUnsend = PhantomData<MutexGuard<'static, ()>>;

pub struct LDevice {
    _marker_unsync: PhantomUnsync,
    _marker_unsend: PhantomUnsend,
}

impl LDevice {
    pub fn create_without_surface() -> Result<Self> {
        assert!(!DEVICE_INITIATED.load(Ordering::SeqCst));

        let entry = unsafe { Entry::load()? };
        let instance = unsafe { create_instance(&entry)? };

        let debug_utils_loader = ash::ext::debug_utils::Instance::new(&entry, &instance);
        let mut debug_messenger = vk::DebugUtilsMessengerEXT::null();
        if cfg!(debug_assertions) {
            let debug_info = vk::DebugUtilsMessengerCreateInfoEXT::default()
                .pfn_user_callback(Some(debug_callback))
                .message_severity(with!(vk::DebugUtilsMessageSeverityFlagsEXT => {ERROR | WARNING | INFO | VERBOSE}))
                .message_type(with!(vk::DebugUtilsMessageTypeFlagsEXT => {GENERAL | PERFORMANCE | VALIDATION}));

            debug_messenger = unsafe { debug_utils_loader.create_debug_utils_messenger(&debug_info, None)? };
        }

        let physical_device = pick_physical_device_without_surface(&instance)?;

        let queue_priorities = &[1.0];
        let indices = unsafe { QueueFamilyIndices::get_without_surface(&instance, physical_device) }?;

        let queue_infos = [
            vk::DeviceQueueCreateInfo::default()
                .queue_family_index(indices.graphics)
                .queue_priorities(queue_priorities),
            vk::DeviceQueueCreateInfo::default()
                .queue_family_index(indices.compute)
                .queue_priorities(queue_priorities),
        ];

        let mut mesh_feature = vk::PhysicalDeviceMeshShaderFeaturesEXT::default()
            .mesh_shader(true)
            .task_shader(true);

        let mut features_11 = vk::PhysicalDeviceVulkan11Features::default()
            .storage_buffer16_bit_access(true)
            .shader_draw_parameters(true);

        let mut features_12 = vk::PhysicalDeviceVulkan12Features::default()
            .shader_int8(true)
            .descriptor_indexing(true)
            .shader_sampled_image_array_non_uniform_indexing(true)
            .descriptor_binding_sampled_image_update_after_bind(true)
            .descriptor_binding_update_unused_while_pending(true)
            .descriptor_binding_partially_bound(true)
            .descriptor_binding_variable_descriptor_count(true)
            .runtime_descriptor_array(true)
            .timeline_semaphore(true)
            .buffer_device_address(true)
            .uniform_and_storage_buffer8_bit_access(true)
            .scalar_block_layout(true)
            .sampler_filter_minmax(true)
            .storage_buffer8_bit_access(true)
            .draw_indirect_count(true);

        let mut features_13 = vk::PhysicalDeviceVulkan13Features::default()
            .robust_image_access(true)
            .maintenance4(true)
            .synchronization2(true)
            .dynamic_rendering(true);

        let mut feature_ray_query = vk::PhysicalDeviceRayQueryFeaturesKHR::default().ray_query(true);
        let mut feature_acceleration_struct =
            vk::PhysicalDeviceAccelerationStructureFeaturesKHR::default().acceleration_structure(true);

        let features = vk::PhysicalDeviceFeatures::default()
            .shader_storage_image_extended_formats(true)
            .robust_buffer_access(true)
            .full_draw_index_uint32(true)
            .image_cube_array(true)
            .fragment_stores_and_atomics(true)
            .vertex_pipeline_stores_and_atomics(true)
            .shader_int64(true)
            .shader_int16(true)
            .pipeline_statistics_query(true)
            .sampler_anisotropy(true)
            .multi_draw_indirect(true);

        let extenstions: Vec<_> = DEVICE_EXTENSTIONS.iter().map(|e| e.as_ptr()).collect();

        let info = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queue_infos)
            .enabled_extension_names(&extenstions)
            .enabled_features(&features)
            .push_next(&mut features_11)
            .push_next(&mut features_12)
            .push_next(&mut features_13)
            .push_next(&mut mesh_feature)
            .push_next(&mut feature_acceleration_struct)
            .push_next(&mut feature_ray_query);

        let device = unsafe { instance.create_device(physical_device, &info, None)? };
        let graphics_queue = unsafe { device.get_device_queue(indices.graphics, 0) };
        let compute_queue = unsafe { device.get_device_queue(indices.compute, 0) };

        let swapchain_loader = swapchain::Device::new(&instance, &device);
        let push_descriptors_loader = push_descriptor::Device::new(&instance, &device);

        let info = vk::CommandPoolCreateInfo::default()
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
            .queue_family_index(indices.graphics);

        let command_pool = unsafe { device.create_command_pool(&info, None)? };
        let mesh_ext_loader = mesh_shader::Device::new(&instance, &device);
        let acc_struct_loader = acceleration_structure::Device::new(&instance, &device);

        let props = unsafe { instance.get_physical_device_properties(physical_device) };

        let allocator = Allocator::new(&AllocatorCreateDesc {
            instance: instance.clone(),
            device: device.clone(),
            physical_device,
            debug_settings: Default::default(),
            buffer_device_address: true,
            allocation_sizes: Default::default(),
        })?;

        let device_impl = LDeviceImpl {
            entry,
            instance,
            debug_utils_loader,
            debug_messenger,
            surface_loader: None,
            device,
            graphics_queue,
            graphics_queue_index: indices.graphics,
            compute_queue,
            compute_queue_index: indices.compute,
            swapchain_loader,
            physical_device,
            surface: vk::SurfaceKHR::null(),
            command_pool,
            props,
            mesh_ext_loader,
            push_descriptors_loader,
            acc_struct_loader,
            allocator: ManuallyDrop::new(allocator),
        };
        unsafe {
            DEVICE_INITIATED.store(true, Ordering::SeqCst);
            (*DEVICE.0.get()).write(device_impl);
        }

        Ok(Self {
            _marker_unsend: PhantomData,
            _marker_unsync: PhantomData,
        })
    }

    pub fn device_ref(&self) -> LDeviceRef<'_> {
        LDeviceRef { _marker: PhantomData }
    }
}

#[derive(Copy, Clone, Zeroable)]
pub struct LDeviceRef<'d> {
    pub(crate) _marker: PhantomData<&'d LDevice>,
}

impl<'d> LDeviceRef<'d> {
    pub fn vk_device(&self) -> &Device {
        unsafe { &(*DEVICE.0.get()).assume_init_ref().device }
    }

    pub(crate) unsafe fn device(&self) -> &LDeviceImpl {
        unsafe { (*DEVICE.0.get()).assume_init_ref() }
    }

    pub(crate) unsafe fn device_mut(&self) -> &mut LDeviceImpl {
        unsafe { (*DEVICE.0.get()).assume_init_mut() }
    }

    pub fn report_memory_leaks(&self) {
        unsafe { self.device().allocator.report_memory_leaks(log::Level::Warn) };
    }

    pub fn set_surface(&self, display: &dyn HasDisplayHandle, window: &dyn HasWindowHandle) -> Result<()> {
        unsafe {
            let device_impl = self.device_mut();

            let surface = create_surface(&device_impl.entry, &device_impl.instance, display, window)?;

            let surface_loader = surface::Instance::new(&device_impl.entry, &device_impl.instance);
            device_impl.surface_loader = Some(surface_loader);
            device_impl.surface = surface;
        }
        Ok(())
    }

    pub unsafe fn get_acceleration_structure_build_sizes(
        &self,
        build_type: vk::AccelerationStructureBuildTypeKHR,
        build_info: &vk::AccelerationStructureBuildGeometryInfoKHR<'_>,
        max_primitive_counts: &[u32],
        size_info: &mut vk::AccelerationStructureBuildSizesInfoKHR<'_>,
    ) {
        unsafe {
            self.device().acc_struct_loader.get_acceleration_structure_build_sizes(
                build_type,
                build_info,
                max_primitive_counts,
                size_info,
            )
        };
    }

    pub unsafe fn create_acceleration_structure(
        &self,
        create_info: &vk::AccelerationStructureCreateInfoKHR,
    ) -> Result<LAccelerationStructureKHR<'d>> {
        unsafe {
            Ok(LAccelerationStructureKHR::new(
                self.device()
                    .acc_struct_loader
                    .create_acceleration_structure(create_info, None)?,
            ))
        }
    }

    pub unsafe fn get_acceleration_structure_device_address(
        &self,
        info: &vk::AccelerationStructureDeviceAddressInfoKHR<'_>,
    ) -> vk::DeviceAddress {
        unsafe {
            self.device()
                .acc_struct_loader
                .get_acceleration_structure_device_address(info)
        }
    }

    pub unsafe fn create_descriptor_pool(&self, info: &vk::DescriptorPoolCreateInfo) -> Result<LDescriptorPool<'d>> {
        let descriptor_pool = unsafe { self.vk_device().create_descriptor_pool(info, None) }?;
        Ok(LDescriptorPool::new(descriptor_pool))
    }

    pub unsafe fn create_semaphore(&self, info: &vk::SemaphoreCreateInfo) -> Result<LSemaphore<'d>> {
        let semaphore = unsafe { self.vk_device().create_semaphore(info, None)? };
        Ok(LSemaphore::new(semaphore))
    }

    pub unsafe fn create_fence(&self, info: &vk::FenceCreateInfo) -> Result<LFence<'d>> {
        let fence = unsafe { self.vk_device().create_fence(info, None)? };
        Ok(LFence::new(fence))
    }

    pub unsafe fn create_sampler(&self, info: &vk::SamplerCreateInfo) -> Result<LSampler<'d>> {
        let sampler = unsafe { self.vk_device().create_sampler(info, None)? };
        Ok(LSampler::new(sampler))
    }

    pub unsafe fn allocate_command_buffers(
        &self,
        info: &vk::CommandBufferAllocateInfo,
    ) -> Result<Vec<LCommandBuffer<'d>>> {
        let command_buffers = unsafe { self.vk_device().allocate_command_buffers(info)? };
        Ok(command_buffers.iter().map(|&c| LCommandBuffer::new(c)).collect())
    }

    pub unsafe fn create_image_view(
        &self,
        image: vk::Image,
        format: vk::Format,
        aspect: vk::ImageAspectFlags,
        mip_level: u32,
        mip_level_count: u32,
    ) -> Result<LImageView<'d>> {
        let components = vk::ComponentMapping::default();

        let subresource_range = vk::ImageSubresourceRange::default()
            .aspect_mask(aspect)
            .base_mip_level(mip_level)
            .level_count(mip_level_count)
            .layer_count(vk::REMAINING_ARRAY_LAYERS);

        let info = vk::ImageViewCreateInfo::default()
            .image(image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(format)
            .components(components)
            .subresource_range(subresource_range);

        let image_view = unsafe { self.vk_device().create_image_view(&info, None)? };
        Ok(LImageView {
            inner: image_view,
            device_ref: LDeviceRef { _marker: PhantomData },
        })
    }

    pub unsafe fn create_image_view_for_layer(
        &self,
        image: vk::Image,
        format: vk::Format,
        aspect: vk::ImageAspectFlags,
        mip_level: u32,
        mip_level_count: u32,
        layer: u32,
        layer_count: u32,
    ) -> Result<LImageView<'d>> {
        let components = vk::ComponentMapping::default();

        let subresource_range = vk::ImageSubresourceRange::default()
            .aspect_mask(aspect)
            .base_mip_level(mip_level)
            .level_count(mip_level_count)
            .base_array_layer(layer)
            .layer_count(layer_count);

        let info = vk::ImageViewCreateInfo::default()
            .image(image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(format)
            .components(components)
            .subresource_range(subresource_range);

        let image_view = unsafe { self.vk_device().create_image_view(&info, None)? };
        Ok(LImageView {
            inner: image_view,
            device_ref: LDeviceRef { _marker: PhantomData },
        })
    }

    pub unsafe fn create_descriptor_set_layout(
        &self,
        info: &vk::DescriptorSetLayoutCreateInfo,
    ) -> Result<vk::DescriptorSetLayout> {
        let descriptor_set_layout = unsafe { self.vk_device().create_descriptor_set_layout(info, None) }?;

        Ok(descriptor_set_layout)
    }

    pub fn get_depth_format(&self) -> Result<vk::Format> {
        let formats = [
            vk::Format::D32_SFLOAT,
            vk::Format::D32_SFLOAT_S8_UINT,
            vk::Format::D24_UNORM_S8_UINT,
        ];

        self.get_supported_formats(
            &formats,
            vk::ImageTiling::OPTIMAL,
            vk::FormatFeatureFlags::DEPTH_STENCIL_ATTACHMENT,
        )
    }

    pub unsafe fn push_descriptors_with_template(
        &self,
        cmd: vk::CommandBuffer,
        descriptor_update_template: vk::DescriptorUpdateTemplate,
        layout: vk::PipelineLayout,
        set: u32,
        p_data: *const ffi::c_void,
    ) {
        unsafe {
            self.device()
                .push_descriptors_loader
                .cmd_push_descriptor_set_with_template(cmd, descriptor_update_template, layout, set, p_data)
        };
    }

    pub unsafe fn submit_graphics(&self, submit_info: &vk::SubmitInfo, fence: vk::Fence) -> Result<()> {
        unsafe {
            self.vk_device()
                .queue_submit(self.device().graphics_queue, &[*submit_info], fence)
        }?;
        Ok(())
    }

    pub unsafe fn submit_compute(&self, submit_info: &vk::SubmitInfo, fence: vk::Fence) -> Result<()> {
        unsafe {
            self.vk_device()
                .queue_submit(self.device().compute_queue, &[*submit_info], fence)
        }?;
        Ok(())
    }

    pub unsafe fn present(&self, present_info: &vk::PresentInfoKHR) -> Result<PresentResult> {
        let result = unsafe {
            self.device()
                .swapchain_loader
                .queue_present(self.device().graphics_queue, present_info)
        };
        let changed = result == Ok(true) || result == Err(vk::Result::ERROR_OUT_OF_DATE_KHR);

        if changed {
            Ok(PresentResult::Changed)
        } else {
            Ok(PresentResult::None)
        }
    }

    pub fn create_mapped_buffer(
        &self,
        name: &str,
        size: vk::DeviceSize,
        usage: vk::BufferUsageFlags,
    ) -> Result<LMappedBuffer<'d>> {
        let (buffer, allocation) = self.create_buffer_impl(name, size, usage, MemoryLocation::CpuToGpu)?;

        let memory_map = allocation.mapped_ptr().context("Failed to map memory")?.as_ptr();

        Ok(LMappedBuffer::new(LBuffer::new(buffer, allocation), memory_map))
    }

    pub unsafe fn create_buffer_with_data<T>(
        &self,
        name: &str,
        staging_buffer: &LMappedBuffer,
        data: &[T],
        usage: vk::BufferUsageFlags,
    ) -> Result<LBuffer<'d>> {
        unsafe { std::ptr::copy_nonoverlapping(data.as_ptr(), staging_buffer.memory_map.cast(), data.len()) };

        let size = std::mem::size_of_val(data) as u64;

        let buffer = self.create_buffer(
            name,
            size,
            vk::BufferUsageFlags::TRANSFER_DST | usage,
            MemoryLocation::GpuOnly,
        )?;

        unsafe { self.copy_buffer(staging_buffer.buffer.inner, buffer.inner, size) }?;

        Ok(buffer)
    }

    pub fn create_buffer(
        &self,
        name: &str,
        size: vk::DeviceSize,
        usage: vk::BufferUsageFlags,
        location: MemoryLocation,
    ) -> Result<LBuffer<'d>> {
        let (buffer, allocation) = self.create_buffer_impl(name, size, usage, location)?;

        Ok(LBuffer {
            inner: buffer,
            allocation: ManuallyDrop::new(allocation),
            device_ref: LDeviceRef { _marker: PhantomData },
        })
    }

    fn create_buffer_impl(
        &self,
        name: &str,
        size: vk::DeviceSize,
        usage: vk::BufferUsageFlags,
        location: MemoryLocation,
    ) -> Result<(vk::Buffer, Allocation)> {
        let buffer_info = vk::BufferCreateInfo::default()
            .usage(usage | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS)
            .size(size)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let buffer = unsafe { self.vk_device().create_buffer(&buffer_info, None)? };

        let requirements = unsafe { self.vk_device().get_buffer_memory_requirements(buffer) };
        let allocation = unsafe {
            self.device_mut().allocator.allocate(&AllocationCreateDesc {
                name,
                requirements,
                location,
                linear: true,
                allocation_scheme: AllocationScheme::GpuAllocatorManaged,
            })
        }?;

        unsafe {
            self.vk_device()
                .bind_buffer_memory(buffer, allocation.memory(), allocation.offset())
        }?;

        Ok((buffer, allocation))
    }

    pub unsafe fn copy_buffer(&self, src: vk::Buffer, dst: vk::Buffer, size: vk::DeviceSize) -> Result<()> {
        let cmd_buffer = unsafe { self.begin_single_time_command() }?;

        let region = vk::BufferCopy::default().size(size);
        unsafe { self.vk_device().cmd_copy_buffer(cmd_buffer.inner, src, dst, &[region]) };

        unsafe { self.end_single_time_command(cmd_buffer) }?;
        Ok(())
    }

    pub unsafe fn create_image(
        &self,
        width: u32,
        height: u32,
        format: vk::Format,
        usage: vk::ImageUsageFlags,
    ) -> Result<LImage<'d>> {
        unsafe { self.create_image_with_mips(width, height, format, usage, 1) }
    }

    pub unsafe fn create_image_with_mips_impl(
        &self,
        width: u32,
        height: u32,
        format: vk::Format,
        usage: vk::ImageUsageFlags,
        mip_level: u32,
    ) -> Result<(vk::Image, Allocation)> {
        let info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .extent(vk::Extent3D {
                width,
                height,
                depth: 1,
            })
            .mip_levels(mip_level)
            .array_layers(1)
            .format(format)
            .tiling(vk::ImageTiling::OPTIMAL)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .samples(vk::SampleCountFlags::TYPE_1);

        let image = unsafe { self.vk_device().create_image(&info, None) }?;
        let requirements = unsafe { self.vk_device().get_image_memory_requirements(image) };
        let allocation = unsafe {
            self.device_mut().allocator.allocate(&AllocationCreateDesc {
                name: "image",
                requirements,
                location: MemoryLocation::GpuOnly,
                linear: false,
                allocation_scheme: AllocationScheme::GpuAllocatorManaged,
            })
        }?;
        unsafe {
            self.vk_device()
                .bind_image_memory(image, allocation.memory(), allocation.offset())
        }?;

        Ok((image, allocation))
    }

    pub unsafe fn create_image_with_mips(
        &self,
        width: u32,
        height: u32,
        format: vk::Format,
        usage: vk::ImageUsageFlags,
        mip_level: u32,
    ) -> Result<LImage<'d>> {
        let (image, allocation) = unsafe { self.create_image_with_mips_impl(width, height, format, usage, mip_level) }?;

        Ok(LImage {
            inner: image,
            allocation: ManuallyDrop::new(allocation),
            device_ref: LDeviceRef { _marker: PhantomData },
        })
    }

    pub unsafe fn create_image_with_view_and_mips(
        &self,
        width: u32,
        height: u32,
        format: vk::Format,
        usage: vk::ImageUsageFlags,
        mip_level: u32,
    ) -> Result<LImageWithView<'d>> {
        let (image, allocation) = unsafe { self.create_image_with_mips_impl(width, height, format, usage, mip_level) }?;

        let image_view = unsafe { self.create_image_view(image, format, vk::ImageAspectFlags::COLOR, 0, mip_level)? };

        Ok(LImageWithView {
            image: LImage {
                inner: image,
                allocation: ManuallyDrop::new(allocation),
                device_ref: LDeviceRef { _marker: PhantomData },
            },
            image_view,
        })
    }

    pub unsafe fn create_depth_image_with_view(
        &self,
        width: u32,
        height: u32,
        usage: vk::ImageUsageFlags,
    ) -> Result<LImageWithView<'d>> {
        let (image, allocation) =
            unsafe { self.create_image_with_mips_impl(width, height, self.get_depth_format()?, usage, 1) }?;

        let image_view =
            unsafe { self.create_image_view(image, self.get_depth_format()?, vk::ImageAspectFlags::DEPTH, 0, 1)? };

        Ok(LImageWithView {
            image: LImage {
                inner: image,
                allocation: ManuallyDrop::new(allocation),
                device_ref: LDeviceRef { _marker: PhantomData },
            },
            image_view,
        })
    }

    pub unsafe fn create_cubemap_image_with_view(
        &self,
        width: u32,
        height: u32,
        format: vk::Format,
        mip_maps: u32,
        usage: vk::ImageUsageFlags,
    ) -> Result<LImageWithView<'d>> {
        let info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .extent(vk::Extent3D {
                width,
                height,
                depth: 1,
            })
            .mip_levels(mip_maps)
            .array_layers(6)
            .format(format)
            .tiling(vk::ImageTiling::OPTIMAL)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .samples(vk::SampleCountFlags::TYPE_1)
            .flags(vk::ImageCreateFlags::CUBE_COMPATIBLE);

        let image = unsafe { self.vk_device().create_image(&info, None)? };

        let requirements = unsafe { self.vk_device().get_image_memory_requirements(image) };
        let allocation = unsafe {
            self.device_mut().allocator.allocate(&AllocationCreateDesc {
                name: "image",
                requirements,
                location: MemoryLocation::GpuOnly,
                linear: false,
                allocation_scheme: AllocationScheme::GpuAllocatorManaged,
            })
        }?;

        unsafe {
            self.vk_device()
                .bind_image_memory(image, allocation.memory(), allocation.offset())
        }?;

        let image_view = unsafe {
            self.vk_device().create_image_view(
                &vk::ImageViewCreateInfo::default()
                    .image(image)
                    .view_type(vk::ImageViewType::CUBE)
                    .format(format)
                    .components(vk::ComponentMapping::default())
                    .subresource_range(
                        vk::ImageSubresourceRange::default()
                            .aspect_mask(vk::ImageAspectFlags::COLOR)
                            .level_count(vk::REMAINING_MIP_LEVELS)
                            .layer_count(vk::REMAINING_ARRAY_LAYERS),
                    ),
                None,
            )?
        };

        Ok(LImageWithView {
            image: LImage {
                inner: image,
                allocation: ManuallyDrop::new(allocation),
                device_ref: LDeviceRef { _marker: PhantomData },
            },
            image_view: LImageView::new(image_view),
        })
    }

    pub unsafe fn load_cubemap(
        &self,
        staging_buffer: &LMappedBuffer,
        cubemap: &str,
    ) -> Result<(LImageWithView<'d>, ImageInfo)> {
        let _span = tracy_client::span!("Loading cubemap");
        log::info!("Loading cubemap: {cubemap}");

        let mut new_tex = std::ptr::null_mut();
        let cubemap_tex_data = std::fs::read(cubemap)?;
        let result = unsafe {
            ktxvulkan_sys::ktxTexture2_CreateFromMemory(
                cubemap_tex_data.as_ptr(),
                cubemap_tex_data.len(),
                ktxvulkan_sys::ktxTextureCreateFlagBits_KTX_TEXTURE_CREATE_LOAD_IMAGE_DATA_BIT,
                &mut new_tex,
            )
        };
        assert_eq!(result, ktxvulkan_sys::ktx_error_code_e_KTX_SUCCESS);

        let texture_data = (unsafe { *new_tex }).pData;
        let texture_size = (unsafe { *new_tex }).dataSize;
        let format = vk::Format::from_raw((unsafe { *new_tex }).vkFormat as i32);
        let width = (unsafe { *new_tex }).baseWidth;
        let height = (unsafe { *new_tex }).baseHeight;
        let mipmaps = (unsafe { *new_tex }).numLevels;

        unsafe { std::ptr::copy_nonoverlapping(texture_data, staging_buffer.memory_map.cast(), texture_size) };

        let image = unsafe {
            self.create_cubemap_image_with_view(
                width,
                height,
                format,
                mipmaps,
                vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST,
            )?
        };

        let cmd_buffer = unsafe { self.begin_single_time_command() }?;

        unsafe {
            cmd_buffer.image_barrier(&[vk::ImageMemoryBarrier2::default()
                .image(image.image.inner)
                .src_access_mask(vk::AccessFlags2::empty())
                .dst_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .src_stage_mask(vk::PipelineStageFlags2::NONE)
                .dst_stage_mask(vk::PipelineStageFlags2::TRANSFER)
                .subresource_range(
                    vk::ImageSubresourceRange::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .level_count(vk::REMAINING_ARRAY_LAYERS)
                        .layer_count(vk::REMAINING_ARRAY_LAYERS),
                )])
        };
        for i in 0..(unsafe { *new_tex }).numFaces {
            for mip_level in 0..mipmaps {
                let mut buffer_offset = 0;
                let result =
                    unsafe { ktxvulkan_sys::ktxTexture2_GetImageOffset(new_tex, mip_level, 0, i, &mut buffer_offset) };

                assert_eq!(result, ktxvulkan_sys::ktx_error_code_e_KTX_SUCCESS);

                let region = vk::BufferImageCopy::default()
                    .buffer_offset(buffer_offset as u64)
                    .image_subresource(vk::ImageSubresourceLayers {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        mip_level,
                        base_array_layer: i,
                        layer_count: 1,
                    })
                    .image_offset(vk::Offset3D { x: 0, y: 0, z: 0 })
                    .image_extent(vk::Extent3D {
                        width: width >> mip_level,
                        height: height >> mip_level,
                        depth: 1,
                    });

                unsafe {
                    cmd_buffer.copy_buffer_to_image(
                        staging_buffer.buffer.inner,
                        image.image.inner,
                        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                        &[region],
                    )
                };

                unsafe {
                    cmd_buffer.image_barrier(&[vk::ImageMemoryBarrier2::default()
                        .image(image.image.inner)
                        .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                        .dst_access_mask(vk::AccessFlags2::SHADER_SAMPLED_READ)
                        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                        .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                        .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                        .src_stage_mask(vk::PipelineStageFlags2::TRANSFER)
                        .dst_stage_mask(vk::PipelineStageFlags2::ALL_GRAPHICS)
                        .subresource_range(
                            vk::ImageSubresourceRange::default()
                                .aspect_mask(vk::ImageAspectFlags::COLOR)
                                .base_array_layer(i)
                                .base_mip_level(mip_level)
                                .level_count(1)
                                .layer_count(1),
                        )])
                };
            }
        }
        unsafe { self.end_single_time_command(cmd_buffer) }?;

        let image_info = ImageInfo { width, height, mipmaps };

        unsafe { ktxvulkan_sys::ktxTexture2_Destroy(new_tex) };
        Ok((image, image_info))
    }

    pub unsafe fn begin_single_time_command(&self) -> Result<LCommandBuffer<'d>> {
        let info = vk::CommandBufferAllocateInfo::default()
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_pool(unsafe { self.device().command_pool })
            .command_buffer_count(1);
        let command_buffer = unsafe { self.allocate_command_buffers(&info) }?.remove(0);
        let info = vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        unsafe { self.vk_device().begin_command_buffer(command_buffer.inner, &info) }?;

        Ok(command_buffer)
    }

    pub unsafe fn end_single_time_command(&self, command_buffer: LCommandBuffer<'d>) -> Result<()> {
        unsafe {
            self.vk_device().end_command_buffer(command_buffer.inner)?;
            let command_buffers = [command_buffer.inner];
            let info = vk::SubmitInfo::default().command_buffers(&command_buffers);
            self.vk_device()
                .queue_submit(self.device().graphics_queue, &[info], vk::Fence::null())?;
            self.vk_device().queue_wait_idle(self.device().graphics_queue)?;
        };
        Ok(())
    }

    pub fn surface_loader(&self) -> &surface::Instance {
        unsafe {
            self.device()
                .surface_loader
                .as_ref()
                .expect("Surface loader not initialized")
        }
    }

    pub fn swapchain_loader(&self) -> &swapchain::Device {
        unsafe { &self.device().swapchain_loader }
    }

    pub fn graphics_queue(&self) -> vk::Queue {
        unsafe { self.device().graphics_queue }
    }

    pub fn compute_queue(&self) -> vk::Queue {
        unsafe { self.device().compute_queue }
    }

    pub fn graphics_queue_index(&self) -> u32 {
        unsafe { self.device().graphics_queue_index }
    }

    pub fn compute_queue_index(&self) -> u32 {
        unsafe { self.device().compute_queue_index }
    }

    pub fn surface(&self) -> vk::SurfaceKHR {
        unsafe { self.device().surface }
    }

    pub fn physical_device(&self) -> vk::PhysicalDevice {
        unsafe { self.device().physical_device }
    }

    pub fn command_pool(&self) -> vk::CommandPool {
        unsafe { self.device().command_pool }
    }

    pub fn props(&self) -> &vk::PhysicalDeviceProperties {
        unsafe { &self.device().props }
    }

    fn get_supported_formats(
        &self,
        candidates: &[vk::Format],
        tiling: vk::ImageTiling,
        features: vk::FormatFeatureFlags,
    ) -> Result<vk::Format> {
        candidates
            .iter()
            .find(|&f| {
                let properties = unsafe {
                    self.device()
                        .instance
                        .get_physical_device_format_properties(self.device().physical_device, *f)
                };
                match tiling {
                    vk::ImageTiling::LINEAR => properties.linear_tiling_features.contains(features),
                    vk::ImageTiling::OPTIMAL => properties.optimal_tiling_features.contains(features),
                    _ => false,
                }
            })
            .copied()
            .ok_or_else(|| anyhow!("Failed to find a supported format"))
    }
}

impl Drop for LDevice {
    fn drop(&mut self) {
        drop(unsafe { ManuallyDrop::take(&mut self.device_ref().device_mut().allocator) });
        unsafe {
            self.device_ref()
                .vk_device()
                .destroy_command_pool(self.device_ref().device().command_pool, None)
        };
        unsafe { self.device_ref().vk_device().destroy_device(None) };
        if !unsafe { self.device_ref().device().surface.is_null() } {
            unsafe {
                self.device_ref()
                    .device()
                    .surface_loader
                    .as_ref()
                    // We cannot create a surface without a surface loader
                    .unwrap()
                    .destroy_surface(self.device_ref().device().surface, None);
            }
        }

        if cfg!(debug_assertions) {
            unsafe {
                self.device_ref()
                    .device()
                    .debug_utils_loader
                    .destroy_debug_utils_messenger(self.device_ref().device().debug_messenger, None)
            };
        }

        unsafe { self.device_ref().device().instance.destroy_instance(None) };
    }
}

impl Deref for LDeviceRef<'_> {
    type Target = Device;

    fn deref(&self) -> &Self::Target {
        self.vk_device()
    }
}

pub struct QueueFamilyIndices {
    pub graphics: u32,
    pub compute: u32,
}

impl QueueFamilyIndices {
    pub unsafe fn get(
        instance: &Instance,
        surface_loader: &surface::Instance,
        physical_device: vk::PhysicalDevice,
        surface: vk::SurfaceKHR,
    ) -> Result<Self> {
        let props = unsafe { instance.get_physical_device_queue_family_properties(physical_device) };

        let mut graphics = None;
        for (index, prop) in props.iter().enumerate() {
            let supports_present =
                unsafe { surface_loader.get_physical_device_surface_support(physical_device, index as u32, surface) }?;
            if supports_present && prop.queue_flags.contains(QueueFlags::GRAPHICS) {
                graphics = Some(index as u32);
                break;
            }
        }

        let mut compute = None;
        for (index, prop) in props.iter().enumerate() {
            if prop.queue_flags.contains(QueueFlags::COMPUTE) && !prop.queue_flags.contains(QueueFlags::GRAPHICS) {
                compute = Some(index as u32);
                break;
            }
        }

        if let (Some(graphics), Some(compute)) = (graphics, compute) {
            log::trace!("Found graphics and compute: {graphics}, {compute}");
            Ok(Self { graphics, compute })
        } else {
            Err(anyhow!(SuitablityError("Didn't find a suitable queue")))
        }
    }

    pub unsafe fn get_without_surface(instance: &Instance, physical_device: vk::PhysicalDevice) -> Result<Self> {
        let props = unsafe { instance.get_physical_device_queue_family_properties(physical_device) };

        let mut graphics = None;
        for (index, prop) in props.iter().enumerate() {
            if prop.queue_flags.contains(QueueFlags::GRAPHICS) {
                graphics = Some(index as u32);
                break;
            }
        }

        let mut compute = None;
        for (index, prop) in props.iter().enumerate() {
            if prop.queue_flags.contains(QueueFlags::COMPUTE) && !prop.queue_flags.contains(QueueFlags::GRAPHICS) {
                compute = Some(index as u32);
                break;
            }
        }

        if let (Some(graphics), Some(compute)) = (graphics, compute) {
            log::trace!("Found graphics and compute: {graphics}, {compute}");
            Ok(Self { graphics, compute })
        } else {
            Err(anyhow!(SuitablityError("Didn't find a suitable queue")))
        }
    }
}

unsafe fn create_instance(entry: &Entry) -> Result<Instance> {
    let application_info = vk::ApplicationInfo::default()
        .application_name(c"Kilua")
        .application_version(vk::make_api_version(0, 1, 1, 0))
        .engine_name(c"Kilua")
        .engine_version(vk::make_api_version(0, 1, 0, 0))
        .api_version(vk::API_VERSION_1_3);

    let mut required_extensions = vec![
        vk::KHR_SURFACE_NAME.as_ptr(),
        #[cfg(target_os = "windows")]
        vk::KHR_WIN32_SURFACE_NAME.as_ptr(),
        #[cfg(target_os = "linux")]
        vk::KHR_XLIB_SURFACE_NAME.as_ptr(),
        #[cfg(target_os = "linux")]
        vk::KHR_WAYLAND_SURFACE_NAME.as_ptr(),
    ];
    let mut layers: Vec<*const i8> = Vec::new();

    if cfg!(debug_assertions) {
        layers.push(c"VK_LAYER_KHRONOS_validation".as_ptr());
        required_extensions.push(ash::ext::debug_utils::NAME.as_ptr());
    }

    let instance_info = vk::InstanceCreateInfo::default()
        .application_info(&application_info)
        .enabled_layer_names(&layers)
        .enabled_extension_names(&required_extensions);

    let enabled_validation_features = [vk::ValidationFeatureEnableEXT::SYNCHRONIZATION_VALIDATION];
    let mut validation_features =
        vk::ValidationFeaturesEXT::default().enabled_validation_features(&enabled_validation_features);

    let instance_info = if cfg!(debug_assertions) {
        instance_info.push_next(&mut validation_features)
    } else {
        instance_info
    };

    let instance = unsafe { entry.create_instance(&instance_info, None) }?;

    Ok(instance)
}

fn pick_physical_device_without_surface(instance: &Instance) -> Result<vk::PhysicalDevice> {
    for physical_device in unsafe { instance.enumerate_physical_devices() }? {
        let props = unsafe { instance.get_physical_device_properties(physical_device) };
        if props.device_type == vk::PhysicalDeviceType::DISCRETE_GPU
            && unsafe { check_physical_device(instance, physical_device).is_ok() }
        {
            return Ok(physical_device);
        }
    }

    Err(anyhow!(SuitablityError("Didn't find a suitable physical device")))
}

unsafe fn check_physical_device(instance: &Instance, physical_device: vk::PhysicalDevice) -> Result<()> {
    let properties = unsafe { instance.get_physical_device_properties(physical_device) };
    if properties.device_type != vk::PhysicalDeviceType::DISCRETE_GPU {
        return Err(anyhow!(SuitablityError("Device isn't a discrete GPU")));
    }

    let features = unsafe { instance.get_physical_device_features(physical_device) };
    if features.geometry_shader == vk::FALSE {
        return Err(anyhow!(SuitablityError("Device doesn't support geometry shaders")));
    }
    if features.sampler_anisotropy == vk::FALSE {
        return Err(anyhow!(SuitablityError("Device doesn't support anisotropy samplers")));
    }

    unsafe { QueueFamilyIndices::get_without_surface(instance, physical_device) }?;

    unsafe { check_physical_device_extensions(instance, physical_device) }?;

    Ok(())
}

unsafe fn check_physical_device_extensions(instance: &Instance, physical_device: vk::PhysicalDevice) -> Result<()> {
    let extentions: HashSet<_> = unsafe { instance.enumerate_device_extension_properties(physical_device) }?
        .into_iter()
        .map(|e| e.extension_name_as_c_str().unwrap_or(c"").to_owned())
        .collect();

    if DEVICE_EXTENSTIONS.iter().all(|e| extentions.contains(*e)) {
        Ok(())
    } else {
        Err(anyhow!(SuitablityError("Missing required device extenstions")))
    }
}

fn create_surface(
    entry: &Entry,
    instance: &Instance,
    display: &dyn HasDisplayHandle,
    window: &dyn HasWindowHandle,
) -> Result<vk::SurfaceKHR> {
    match (
        display.display_handle().map(|handle| handle.as_raw()),
        window.window_handle().map(|handle| handle.as_raw()),
    ) {
        #[cfg(target_os = "windows")]
        (Ok(RawDisplayHandle::Windows(_)), Ok(RawWindowHandle::Win32(window))) => {
            let win32_surface_loader = ash::khr::win32_surface::Instance::new(entry, instance);
            let hinstance_ptr = window
                .hinstance
                .map(|hinstance| hinstance.get() as vk::HINSTANCE)
                .unwrap();
            let hwnd_ptr = window.hwnd.get() as vk::HWND;

            let info = vk::Win32SurfaceCreateInfoKHR::default()
                .hinstance(hinstance_ptr)
                .hwnd(hwnd_ptr);
            let surface = unsafe { win32_surface_loader.create_win32_surface(&info, None) }?;

            Ok(surface)
        }
        #[cfg(target_os = "linux")]
        (Ok(RawDisplayHandle::Xlib(display)), Ok(RawWindowHandle::Xlib(window))) => {
            log::info!("Running on x11");
            let xlib_surface_loader = ash::khr::xlib_surface::Instance::new(entry, instance);
            let info = vk::XlibSurfaceCreateInfoKHR::default()
                .window(window.window)
                .dpy(display.display.context("Cannot find display pointer")?.as_ptr());

            let surface = unsafe { xlib_surface_loader.create_xlib_surface(&info, None) }?;

            Ok(surface)
        }
        #[cfg(target_os = "linux")]
        (Ok(RawDisplayHandle::Wayland(display)), Ok(RawWindowHandle::Wayland(window))) => {
            log::info!("Running on wayland");
            let wayland_surface_loader = ash::khr::wayland_surface::Instance::new(entry, instance);
            let info = vk::WaylandSurfaceCreateInfoKHR::default()
                .display(display.display.as_ptr())
                .surface(window.surface.as_ptr());

            let surface = unsafe { wayland_surface_loader.create_wayland_surface(&info, None) }?;

            Ok(surface)
        }
        // Unsupported (currently)
        _ => unimplemented!(),
    }
}

extern "system" fn debug_callback(
    severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    type_: vk::DebugUtilsMessageTypeFlagsEXT,
    data: *const vk::DebugUtilsMessengerCallbackDataEXT,
    _: *mut std::ffi::c_void,
) -> vk::Bool32 {
    let data = unsafe { *data };
    let message = unsafe { data.message_as_c_str().unwrap() }.to_string_lossy();

    if severity >= vk::DebugUtilsMessageSeverityFlagsEXT::ERROR {
        log::error!("({type_:?}) {message}\n");
    } else if severity >= vk::DebugUtilsMessageSeverityFlagsEXT::WARNING {
        log::warn!("({type_:?}) {message}\n");
    } else if severity >= vk::DebugUtilsMessageSeverityFlagsEXT::INFO {
        log::debug!("({type_:?}) {message}\n");
    } else {
        log::trace!("({type_:?}) {message}\n");
    }

    vk::FALSE
}
