use std::ops::Deref;

use crate::with;

use super::device::LDeviceRef;
use anyhow::Result;
use ash::{khr::surface, vk};
use winit::window::Window;

pub struct LSwapchain<'d> {
    swapchain: vk::SwapchainKHR,
    images: Vec<vk::Image>,
    format: vk::Format,
    extent: vk::Extent2D,
    _device: LDeviceRef<'d>,
}

impl<'d> LSwapchain<'d> {
    pub fn create(device: LDeviceRef<'d>, window: &Window, enable_vsync: bool) -> Result<Self> {
        let support =
            unsafe { SwapchainSupport::get(device.surface_loader(), device.surface(), device.physical_device()) }?;
        let format = support.get_swapchain_surface_formats();
        let extent = support.get_swapchain_extent(window);
        let swapchain = create_swapchain(device, window, vk::SwapchainKHR::null(), enable_vsync)?;
        let swapchain_images = unsafe { device.swapchain_loader().get_swapchain_images(swapchain) }?;
        let swapchain_format = format.format;
        let swapchain_extent = extent;
        Ok(Self {
            swapchain,
            images: swapchain_images,
            format: swapchain_format,
            extent: swapchain_extent,
            _device: device,
        })
    }

    pub fn resize(&mut self, device: LDeviceRef, window: &Window, enable_vsync: bool) -> Result<()> {
        let support =
            unsafe { SwapchainSupport::get(device.surface_loader(), device.surface(), device.physical_device()) }?;
        let format = support.get_swapchain_surface_formats();
        let extent = support.get_swapchain_extent(window);
        let old_swapchain = self.swapchain;
        let swapchain = create_swapchain(device, window, old_swapchain, enable_vsync)?;
        unsafe { device.swapchain_loader().destroy_swapchain(old_swapchain, None) };
        let swapchain_images = unsafe { device.swapchain_loader().get_swapchain_images(swapchain) }?;
        let swapchain_format = format.format;
        let swapchain_extent = extent;
        self.swapchain = swapchain;
        self.images = swapchain_images;
        self.format = swapchain_format;
        self.extent = swapchain_extent;
        Ok(())
    }

    pub fn images(&self) -> &[vk::Image] {
        &self.images
    }

    pub fn format(&self) -> vk::Format {
        self.format
    }

    pub fn extent(&self) -> vk::Extent2D {
        self.extent
    }
}

impl Drop for LSwapchain<'_> {
    fn drop(&mut self) {
        // destroying the swapchain on linux causes a segfault
        #[cfg(target_os = "windows")]
        unsafe {
            self._device.swapchain_loader().destroy_swapchain(self.swapchain, None)
        };
    }
}

#[derive(Debug, Clone)]
pub struct SwapchainSupport {
    pub capabilities: vk::SurfaceCapabilitiesKHR,
    pub formats: Vec<vk::SurfaceFormatKHR>,
    pub present_modes: Vec<vk::PresentModeKHR>,
}

impl SwapchainSupport {
    pub unsafe fn get(
        surface_loader: &surface::Instance,
        surface: vk::SurfaceKHR,
        physical_device: vk::PhysicalDevice,
    ) -> Result<Self> {
        Ok(Self {
            capabilities: unsafe { surface_loader.get_physical_device_surface_capabilities(physical_device, surface) }?,
            formats: unsafe { surface_loader.get_physical_device_surface_formats(physical_device, surface) }?,
            present_modes: unsafe {
                surface_loader.get_physical_device_surface_present_modes(physical_device, surface)
            }?,
        })
    }

    pub fn get_swapchain_surface_formats(&self) -> vk::SurfaceFormatKHR {
        if self.formats.len() == 1 && self.formats[0].format == vk::Format::UNDEFINED {
            return vk::SurfaceFormatKHR {
                format: vk::Format::R8G8B8_SRGB,
                color_space: vk::ColorSpaceKHR::SRGB_NONLINEAR,
            };
        }
        *self
            .formats
            .iter()
            .find(|f| {
                (f.format == vk::Format::R8G8B8A8_SRGB || f.format == vk::Format::B8G8R8A8_SRGB)
                    && f.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
            })
            .unwrap_or_else(|| &self.formats[0])
    }

    pub fn get_swapchain_present_mode(&self, enable_vsync: bool) -> vk::PresentModeKHR {
        if enable_vsync {
            vk::PresentModeKHR::FIFO
        } else {
            *self
                .present_modes
                .iter()
                .find(|p| **p == vk::PresentModeKHR::IMMEDIATE)
                .unwrap_or(&vk::PresentModeKHR::IMMEDIATE)
        }
    }

    pub fn get_swapchain_extent(&self, window: &Window) -> vk::Extent2D {
        if self.capabilities.current_extent.width != u32::MAX {
            self.capabilities.current_extent
        } else {
            vk::Extent2D::default()
                .width(window.inner_size().width.clamp(
                    self.capabilities.min_image_extent.width,
                    self.capabilities.max_image_extent.width,
                ))
                .height(window.inner_size().height.clamp(
                    self.capabilities.min_image_extent.height,
                    self.capabilities.max_image_extent.height,
                ))
        }
    }
}

impl Deref for LSwapchain<'_> {
    type Target = vk::SwapchainKHR;

    fn deref(&self) -> &Self::Target {
        &self.swapchain
    }
}

pub fn create_swapchain(
    device: LDeviceRef,
    window: &Window,
    old_swapchain: vk::SwapchainKHR,
    enable_vsync: bool,
) -> Result<vk::SwapchainKHR> {
    let support =
        unsafe { SwapchainSupport::get(device.surface_loader(), device.surface(), device.physical_device()) }?;

    let format = support.get_swapchain_surface_formats();
    let present_mode = support.get_swapchain_present_mode(enable_vsync);
    log::info!("Present mode: {present_mode:?}");
    let extent = support.get_swapchain_extent(window);

    let mut image_count = support.capabilities.min_image_count + 1;

    if support.capabilities.max_image_count != 0 && image_count > support.capabilities.max_image_count {
        image_count = support.capabilities.max_image_count;
    }

    let queues = [device.graphics_queue_index()];
    let info = vk::SwapchainCreateInfoKHR::default()
        .surface(device.surface())
        .min_image_count(image_count)
        .image_format(format.format)
        .image_color_space(format.color_space)
        .image_extent(extent)
        .image_array_layers(1)
        .image_usage(with!(vk::ImageUsageFlags => { TRANSFER_DST | COLOR_ATTACHMENT }))
        .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
        .queue_family_indices(&queues)
        .pre_transform(support.capabilities.current_transform)
        .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
        .present_mode(present_mode)
        .clipped(true)
        .old_swapchain(old_swapchain);

    Ok(unsafe { device.swapchain_loader().create_swapchain(&info, None)? })
}
