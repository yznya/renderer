use std::mem::ManuallyDrop;

use ash::vk::{self, CommandBufferResetFlags};
use bytemuck::bytes_of;
use egui::{
    Context, TextureId, TexturesDelta, ViewportId,
    epaint::{ImageDelta, ahash::AHashMap},
};
use egui_winit::{EventResponse, winit::window::Window};
use gpu_allocator::MemoryLocation;
use winit::raw_window_handle::HasDisplayHandle;

use lava::{
    device::LDeviceRef,
    image_barrier,
    resources::{LBuffer, LDescriptorPool, LImageView, LImageWithView, LSampler},
    shaders::Shader,
    swapchain::LSwapchain,
    with,
};

pub(crate) struct Integration<'d> {
    physical_width: u32,
    physical_height: u32,
    pub(crate) scale_factor: f64,
    egui_winit: ManuallyDrop<egui_winit::State>,
    descriptor_pool: LDescriptorPool<'d>,
    descriptor_set_layouts: Vec<vk::DescriptorSetLayout>,
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    sampler: LSampler<'d>,
    color_image_views: Vec<LImageView<'d>>,
    vertex_buffers: Vec<LBuffer<'d>>,
    index_buffers: Vec<LBuffer<'d>>,
    texture_desc_sets: AHashMap<TextureId, vk::DescriptorSet>,
    texture_images: AHashMap<TextureId, LImageWithView<'d>>,
    texture_image_infos: AHashMap<TextureId, vk::Extent3D>,

    user_texture_layout: vk::DescriptorSetLayout,
    user_textures: Vec<Option<vk::DescriptorSet>>,
    device: LDeviceRef<'d>,
}

impl<'d> Integration<'d> {
    pub(crate) fn new<H: HasDisplayHandle>(
        device: LDeviceRef<'d>,
        swapchain: &LSwapchain,
        display_target: &H,
        scale_factor: f64,
        font_definitions: egui::FontDefinitions,
        style: egui::Style,
    ) -> Self {
        // Create context
        let context = Context::default();
        context.set_fonts(font_definitions);
        context.set_style(style);

        // TODO: a bug when egui is opened on the first frame
        let egui_winit = egui_winit::State::new(
            context,
            ViewportId::default(),
            display_target,
            Some(scale_factor as f32),
            None,
            None,
        );

        let swap_images = swapchain.images();

        let descriptor_pool = unsafe {
            device
                .create_descriptor_pool(
                    &vk::DescriptorPoolCreateInfo::default()
                        .flags(vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET)
                        .max_sets(1024)
                        .pool_sizes(&[vk::DescriptorPoolSize::default()
                            .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                            .descriptor_count(1024)]),
                )
                .expect("Failed to create descriptor pool.")
        };

        let descriptor_set_layouts = {
            let mut sets = vec![];
            for _ in 0..swap_images.len() {
                sets.push(
                    unsafe {
                        device.create_descriptor_set_layout(
                            &vk::DescriptorSetLayoutCreateInfo::default().bindings(&[
                                vk::DescriptorSetLayoutBinding::default()
                                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                                    .descriptor_count(1)
                                    .binding(0)
                                    .stage_flags(vk::ShaderStageFlags::FRAGMENT),
                            ]),
                        )
                    }
                    .expect("Failed to create descriptor set layout."),
                );
            }
            sets
        };

        let pipeline_layout = unsafe {
            device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default()
                    .set_layouts(&descriptor_set_layouts)
                    .push_constant_ranges(&[
                        vk::PushConstantRange::default()
                            .stage_flags(vk::ShaderStageFlags::VERTEX)
                            .offset(0)
                            .size(std::mem::size_of::<f32>() as u32 * 2), // screen size
                    ]),
                None,
            )
        }
        .expect("Failed to create pipeline layout.");

        let pipeline = {
            // TODO: can we read these from shaders?
            let bindings = [vk::VertexInputBindingDescription::default()
                .binding(0)
                .input_rate(vk::VertexInputRate::VERTEX)
                .stride(4 * std::mem::size_of::<f32>() as u32 + 4 * std::mem::size_of::<u8>() as u32)];

            let attributes = [
                // position
                vk::VertexInputAttributeDescription::default()
                    .binding(0)
                    .offset(0)
                    .location(0)
                    .format(vk::Format::R32G32_SFLOAT),
                // uv
                vk::VertexInputAttributeDescription::default()
                    .binding(0)
                    .offset(8)
                    .location(1)
                    .format(vk::Format::R32G32_SFLOAT),
                // color
                vk::VertexInputAttributeDescription::default()
                    .binding(0)
                    .offset(16)
                    .location(2)
                    .format(vk::Format::R8G8B8A8_UNORM),
            ];

            let shader_files = [
                "assets/out/shaders/egui.vert.glsl.spv",
                "assets/out/shaders/egui.frag.glsl.spv",
            ];

            let mut shaders: Vec<Shader> = shader_files
                .iter()
                .map(|f| Shader::from_file(device, f))
                .collect::<Result<Vec<_>, _>>()
                .expect("Failed to create egui shaders");

            let stages = Shader::extract_stages(&shaders);

            let color_formats = [swapchain.format()];
            let mut rendering_info =
                vk::PipelineRenderingCreateInfo::default().color_attachment_formats(&color_formats);

            let input_assembly_info =
                vk::PipelineInputAssemblyStateCreateInfo::default().topology(vk::PrimitiveTopology::TRIANGLE_LIST);
            let viewport_info = vk::PipelineViewportStateCreateInfo::default()
                .viewport_count(1)
                .scissor_count(1);
            let rasterization_info = vk::PipelineRasterizationStateCreateInfo::default().line_width(1.0);
            let depth_stencil_info = vk::PipelineDepthStencilStateCreateInfo::default();
            let color_blend_attachments = [vk::PipelineColorBlendAttachmentState::default()
                .color_write_mask(vk::ColorComponentFlags::RGBA)
                .blend_enable(true)
                .src_color_blend_factor(vk::BlendFactor::ONE)
                .dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)];
            let color_blend_info =
                vk::PipelineColorBlendStateCreateInfo::default().attachments(&color_blend_attachments);
            let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
            let dynamic_state_info = vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);
            let vertex_input_state = vk::PipelineVertexInputStateCreateInfo::default()
                .vertex_attribute_descriptions(&attributes)
                .vertex_binding_descriptions(&bindings);
            let multisample_info =
                vk::PipelineMultisampleStateCreateInfo::default().rasterization_samples(vk::SampleCountFlags::TYPE_1);

            let pipeline_create_info = [vk::GraphicsPipelineCreateInfo::default()
                .stages(&stages)
                .vertex_input_state(&vertex_input_state)
                .input_assembly_state(&input_assembly_info)
                .viewport_state(&viewport_info)
                .rasterization_state(&rasterization_info)
                .multisample_state(&multisample_info)
                .depth_stencil_state(&depth_stencil_info)
                .color_blend_state(&color_blend_info)
                .dynamic_state(&dynamic_state_info)
                .layout(pipeline_layout)
                .push_next(&mut rendering_info)];

            let pipeline =
                unsafe { device.create_graphics_pipelines(vk::PipelineCache::null(), &pipeline_create_info, None) }
                    .expect("Failed to create graphics pipeline.")[0];
            shaders.drain(..).for_each(|s| s.destroy(device));
            pipeline
        };

        let sampler = unsafe {
            device.create_sampler(
                &vk::SamplerCreateInfo::default()
                    .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                    .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                    .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                    .anisotropy_enable(false)
                    .min_filter(vk::Filter::LINEAR)
                    .mag_filter(vk::Filter::LINEAR)
                    .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
                    .min_lod(0.0)
                    .max_lod(vk::LOD_CLAMP_NONE),
            )
        }
        .expect("Failed to create sampler.");

        // Create Framebuffers
        let framebuffer_color_image_views = swap_images
            .iter()
            .map(|swapchain_image| unsafe {
                device
                    .create_image_view(*swapchain_image, swapchain.format(), vk::ImageAspectFlags::COLOR, 0, 1)
                    .expect("Failed to create image view.")
            })
            .collect::<Vec<_>>();

        // Create vertex buffer and index buffer
        let mut vertex_buffers = vec![];
        let mut index_buffers = vec![];
        for _ in 0..swapchain.images().len() {
            let vertex_buffer = device
                .create_buffer(
                    "egui_vertex_buffer",
                    Self::vertex_buffer_size(),
                    vk::BufferUsageFlags::VERTEX_BUFFER,
                    MemoryLocation::CpuToGpu,
                )
                .expect("Failed to create vertex buffer.");

            let index_buffer = device
                .create_buffer(
                    "egui_index_buffer",
                    Self::index_buffer_size(),
                    vk::BufferUsageFlags::INDEX_BUFFER,
                    MemoryLocation::CpuToGpu,
                )
                .expect("Failed to create index buffer.");

            vertex_buffers.push(vertex_buffer);
            index_buffers.push(index_buffer);
        }

        // User Textures
        let user_texture_layout = unsafe {
            device.create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default().bindings(&[vk::DescriptorSetLayoutBinding::default()
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .descriptor_count(1)
                    .binding(0)
                    .stage_flags(vk::ShaderStageFlags::FRAGMENT)]),
            )
        }
        .expect("Failed to create descriptor set layout.");
        let user_textures = vec![];

        let physical_width = swapchain.extent().width;
        let physical_height = swapchain.extent().height;

        Self {
            physical_width,
            physical_height,
            scale_factor,
            egui_winit: ManuallyDrop::new(egui_winit),
            descriptor_pool,
            descriptor_set_layouts,
            pipeline_layout,
            pipeline,
            sampler,
            color_image_views: framebuffer_color_image_views,
            vertex_buffers,
            index_buffers,
            texture_desc_sets: AHashMap::new(),
            texture_images: AHashMap::new(),
            texture_image_infos: AHashMap::new(),

            user_texture_layout,
            user_textures,
            device,
        }
    }

    fn vertex_buffer_size() -> u64 {
        1024 * 1024 * 4
    }

    fn index_buffer_size() -> u64 {
        1024 * 1024 * 2
    }

    pub(crate) fn handle_event(
        &mut self,
        window: &Window,
        winit_event: &egui_winit::winit::event::WindowEvent,
    ) -> EventResponse {
        self.egui_winit.on_window_event(window, winit_event)
    }

    pub(crate) fn begin_frame(&mut self, window: &Window) {
        let raw_input = self.egui_winit.take_egui_input(window);
        self.context().begin_pass(raw_input);
    }

    pub(crate) fn end_frame(&mut self, window: &Window) -> egui::FullOutput {
        let output = self.context().end_pass();

        self.egui_winit
            .handle_platform_output(window, output.platform_output.clone());

        output
    }

    pub(crate) fn context(&self) -> Context {
        self.egui_winit.egui_ctx().clone()
    }

    // TODO: refactor this file
    pub(crate) fn paint(
        &mut self,
        device: LDeviceRef<'d>,
        command_buffer: vk::CommandBuffer,
        swapchain: &LSwapchain,
        swapchain_image_index: usize,
        clipped_meshes: Vec<egui::ClippedPrimitive>,
        textures_delta: TexturesDelta,
    ) {
        let index = swapchain_image_index;

        for (id, image_delta) in textures_delta.set {
            self.update_texture(device, id, image_delta);
        }

        let mut vertex_buffer_ptr = self.vertex_buffers[index].allocation.mapped_ptr().unwrap().as_ptr() as *mut u8;
        let vertex_buffer_ptr_end = unsafe { vertex_buffer_ptr.add(Self::vertex_buffer_size() as usize) };
        let mut index_buffer_ptr = self.index_buffers[index].allocation.mapped_ptr().unwrap().as_ptr() as *mut u8;
        let index_buffer_ptr_end = unsafe { index_buffer_ptr.add(Self::index_buffer_size() as usize) };

        let color_attachments = [vk::RenderingAttachmentInfo::default()
            .image_view(self.color_image_views[index].inner)
            .image_layout(vk::ImageLayout::ATTACHMENT_OPTIMAL)
            .load_op(vk::AttachmentLoadOp::LOAD)
            .store_op(vk::AttachmentStoreOp::STORE)];

        let rendering_info = vk::RenderingInfo::default()
            .render_area(vk::Rect2D::default().extent(swapchain.extent()))
            .layer_count(1)
            .color_attachments(&color_attachments);

        unsafe { device.cmd_begin_rendering(command_buffer, &rendering_info) };

        // bind resources
        unsafe {
            device.cmd_bind_pipeline(command_buffer, vk::PipelineBindPoint::GRAPHICS, self.pipeline);
            device.cmd_bind_vertex_buffers(command_buffer, 0, &[self.vertex_buffers[index].inner], &[0]);
            device.cmd_bind_index_buffer(
                command_buffer,
                self.index_buffers[index].inner,
                0,
                vk::IndexType::UINT32,
            );
            device.cmd_set_viewport(
                command_buffer,
                0,
                &[vk::Viewport::default()
                    .x(0.0)
                    .y(0.0)
                    .width(self.physical_width as f32)
                    .height(self.physical_height as f32)
                    .min_depth(0.0)
                    .max_depth(1.0)],
            );
            let width_points = self.physical_width as f32 / self.scale_factor as f32;
            let height_points = self.physical_height as f32 / self.scale_factor as f32;
            device.cmd_push_constants(
                command_buffer,
                self.pipeline_layout,
                vk::ShaderStageFlags::VERTEX,
                0,
                bytes_of(&width_points),
            );
            device.cmd_push_constants(
                command_buffer,
                self.pipeline_layout,
                vk::ShaderStageFlags::VERTEX,
                std::mem::size_of_val(&width_points) as u32,
                bytes_of(&height_points),
            );
        }

        // render meshes
        let mut vertex_base = 0;
        let mut index_base = 0;
        for egui::ClippedPrimitive { clip_rect, primitive } in clipped_meshes {
            let mesh = match primitive {
                egui::epaint::Primitive::Mesh(mesh) => mesh,
                egui::epaint::Primitive::Callback(_) => todo!(),
            };
            if mesh.vertices.is_empty() || mesh.indices.is_empty() {
                continue;
            }

            unsafe {
                if let egui::TextureId::User(id) = mesh.texture_id {
                    if let Some(descriptor_set) = self.user_textures[id as usize] {
                        device.cmd_bind_descriptor_sets(
                            command_buffer,
                            vk::PipelineBindPoint::GRAPHICS,
                            self.pipeline_layout,
                            0,
                            &[descriptor_set],
                            &[],
                        );
                    } else {
                        eprintln!("This UserTexture has already been unregistered: {:?}", mesh.texture_id);
                        continue;
                    }
                } else {
                    device.cmd_bind_descriptor_sets(
                        command_buffer,
                        vk::PipelineBindPoint::GRAPHICS,
                        self.pipeline_layout,
                        0,
                        &[*self.texture_desc_sets.get(&mesh.texture_id).unwrap()],
                        &[],
                    );
                }
            }
            let v_slice = &mesh.vertices;
            let v_size = std::mem::size_of_val(&v_slice[0]);
            let v_copy_size = v_slice.len() * v_size;

            let i_slice = &mesh.indices;
            let i_size = std::mem::size_of_val(&i_slice[0]);
            let i_copy_size = i_slice.len() * i_size;

            let vertex_buffer_ptr_next = unsafe { vertex_buffer_ptr.add(v_copy_size) };
            let index_buffer_ptr_next = unsafe { index_buffer_ptr.add(i_copy_size) };

            if vertex_buffer_ptr_next >= vertex_buffer_ptr_end || index_buffer_ptr_next >= index_buffer_ptr_end {
                panic!("egui paint out of memory");
            }

            // map memory
            unsafe { vertex_buffer_ptr.copy_from(v_slice.as_ptr() as *const u8, v_copy_size) };
            unsafe { index_buffer_ptr.copy_from(i_slice.as_ptr() as *const u8, i_copy_size) };

            vertex_buffer_ptr = vertex_buffer_ptr_next;
            index_buffer_ptr = index_buffer_ptr_next;

            // record draw commands
            unsafe {
                let min = clip_rect.min;
                let min = egui::Pos2 {
                    x: min.x * self.scale_factor as f32,
                    y: min.y * self.scale_factor as f32,
                };
                let min = egui::Pos2 {
                    x: f32::clamp(min.x, 0.0, self.physical_width as f32),
                    y: f32::clamp(min.y, 0.0, self.physical_height as f32),
                };
                let max = clip_rect.max;
                let max = egui::Pos2 {
                    x: max.x * self.scale_factor as f32,
                    y: max.y * self.scale_factor as f32,
                };
                let max = egui::Pos2 {
                    x: f32::clamp(max.x, min.x, self.physical_width as f32),
                    y: f32::clamp(max.y, min.y, self.physical_height as f32),
                };
                device.cmd_set_scissor(
                    command_buffer,
                    0,
                    &[vk::Rect2D::default()
                        .offset(vk::Offset2D::default().x(min.x.round() as i32).y(min.y.round() as i32))
                        .extent(
                            vk::Extent2D::default()
                                .width((max.x.round() - min.x) as u32)
                                .height((max.y.round() - min.y) as u32),
                        )],
                );
                device.cmd_draw_indexed(command_buffer, mesh.indices.len() as u32, 1, index_base, vertex_base, 0);
            }

            vertex_base += mesh.vertices.len() as i32;
            index_base += mesh.indices.len() as u32;
        }

        unsafe {
            device.cmd_end_rendering(command_buffer);
        }

        for &id in &textures_delta.free {
            self.texture_desc_sets.remove_entry(&id); // dsc_set is destroyed with dsc_pool
            self.texture_image_infos.remove_entry(&id);
        }
    }

    fn update_texture(&mut self, device: LDeviceRef<'d>, texture_id: TextureId, delta: ImageDelta) {
        // Extract pixel data from egui
        let data: Vec<u8> = match &delta.image {
            egui::ImageData::Color(image) => {
                assert_eq!(
                    image.width() * image.height(),
                    image.pixels.len(),
                    "Mismatch between texture size and texel count"
                );
                image.pixels.iter().flat_map(|color| color.to_array()).collect()
            }
            egui::ImageData::Font(image) => image.srgba_pixels(None).flat_map(|color| color.to_array()).collect(),
        };
        let cmd_buff = {
            let cmd_buff_alloc_info = vk::CommandBufferAllocateInfo::default()
                .command_buffer_count(1u32)
                .command_pool(device.command_pool())
                .level(vk::CommandBufferLevel::PRIMARY);
            unsafe { device.allocate_command_buffers(&cmd_buff_alloc_info).unwrap().remove(0) }
        };
        let fence_info = vk::FenceCreateInfo::default();
        let cmd_buff_fence = unsafe { device.create_fence(&fence_info).unwrap() };

        let staging_buffer = {
            let buffer_size = data.len() as vk::DeviceSize;
            device
                .create_buffer(
                    "egui_texture_buffer",
                    buffer_size,
                    vk::BufferUsageFlags::TRANSFER_SRC,
                    MemoryLocation::CpuToGpu,
                )
                .unwrap()
        };
        let ptr = staging_buffer.allocation.mapped_ptr().unwrap().as_ptr() as *mut u8;
        unsafe {
            ptr.copy_from_nonoverlapping(data.as_ptr(), data.len());
        }
        let texture_image = {
            unsafe {
                device.create_image_with_view_and_mips(
                    delta.image.width() as u32,
                    delta.image.height() as u32,
                    vk::Format::R8G8B8A8_UNORM,
                    with!(vk::ImageUsageFlags => {SAMPLED | TRANSFER_DST | TRANSFER_SRC}),
                    1,
                )
            }
            .unwrap()
        };
        self.texture_image_infos.insert(
            texture_id,
            vk::Extent3D {
                width: delta.image.width() as u32,
                height: delta.image.height() as u32,
                depth: 0,
            },
        );

        unsafe {
            let cmd_buff_begin_info =
                vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            device
                .begin_command_buffer(cmd_buff.inner, &cmd_buff_begin_info)
                .unwrap();
        }
        let barrier = image_barrier!(
            image: texture_image.image.inner,
            access: NONE => TRANSFER_WRITE,
            layout: UNDEFINED => TRANSFER_DST_OPTIMAL,
            stage: HOST => TRANSFER,
            aspect: COLOR
        );
        unsafe { cmd_buff.image_barrier(&[barrier]) };

        let region = vk::BufferImageCopy::default()
            .buffer_offset(0)
            .buffer_row_length(delta.image.width() as u32)
            .buffer_image_height(delta.image.height() as u32)
            .image_subresource(
                vk::ImageSubresourceLayers::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .base_array_layer(0)
                    .layer_count(1)
                    .mip_level(0),
            )
            .image_offset(vk::Offset3D { x: 0, y: 0, z: 0 })
            .image_extent(vk::Extent3D {
                width: delta.image.width() as u32,
                height: delta.image.height() as u32,
                depth: 1,
            });
        unsafe {
            device.cmd_copy_buffer_to_image(
                cmd_buff.inner,
                staging_buffer.inner,
                texture_image.image.inner,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[region],
            );
        }

        let barrier = image_barrier!(
            image: texture_image.image.inner,
            access: TRANSFER_WRITE => SHADER_READ,
            layout: TRANSFER_DST_OPTIMAL => SHADER_READ_ONLY_OPTIMAL,
            stage: TRANSFER => VERTEX_SHADER,
            aspect: COLOR
        );
        unsafe { cmd_buff.image_barrier(&[barrier]) };

        unsafe {
            device.end_command_buffer(cmd_buff.inner).unwrap();
        }
        let cmd_buffs = [cmd_buff.inner];
        let submit_infos = [vk::SubmitInfo::default().command_buffers(&cmd_buffs)];
        unsafe {
            device
                .queue_submit(device.graphics_queue(), &submit_infos, cmd_buff_fence.inner)
                .unwrap();
            device.wait_for_fences(&[cmd_buff_fence.inner], true, u64::MAX).unwrap();
        }

        // texture is now in GPU memory, now we need to decide whether we should register it as new or update existing

        if let Some(pos) = delta.pos {
            // Blit texture data to existing texture if delta pos exists (e.g. font changed)
            let existing_texture = self.texture_images.get(&texture_id);
            if let Some(existing_texture) = existing_texture {
                let info = self.texture_image_infos.get(&texture_id).unwrap();
                unsafe {
                    device
                        .reset_command_buffer(cmd_buff.inner, CommandBufferResetFlags::empty())
                        .unwrap();
                    device.reset_fences(&[cmd_buff_fence.inner]).unwrap();
                    let cmd_buff_begin_info =
                        vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
                    device
                        .begin_command_buffer(cmd_buff.inner, &cmd_buff_begin_info)
                        .unwrap();

                    let barrier_dst = image_barrier!(
                        image: existing_texture.image.inner,
                        access: SHADER_READ => TRANSFER_WRITE,
                        layout: UNDEFINED => TRANSFER_DST_OPTIMAL,
                        stage: FRAGMENT_SHADER => TRANSFER,
                        aspect: COLOR
                    );

                    let barrier_src = image_barrier!(
                        image: texture_image.image.inner,
                        access: SHADER_READ => TRANSFER_READ,
                        layout: SHADER_READ_ONLY_OPTIMAL => TRANSFER_SRC_OPTIMAL,
                        stage: FRAGMENT_SHADER => TRANSFER,
                        aspect: COLOR
                    );
                    cmd_buff.image_barrier(&[barrier_src, barrier_dst]);

                    let top_left = vk::Offset3D {
                        x: pos[0] as i32,
                        y: pos[1] as i32,
                        z: 0,
                    };
                    let bottom_right = vk::Offset3D {
                        x: pos[0] as i32 + delta.image.width() as i32,
                        y: pos[1] as i32 + delta.image.height() as i32,
                        z: 1,
                    };

                    let region = vk::ImageBlit {
                        src_subresource: vk::ImageSubresourceLayers {
                            aspect_mask: vk::ImageAspectFlags::COLOR,
                            mip_level: 0,
                            base_array_layer: 0,
                            layer_count: 1,
                        },
                        src_offsets: [
                            vk::Offset3D { x: 0, y: 0, z: 0 },
                            vk::Offset3D {
                                x: info.width as i32,
                                y: info.height as i32,
                                z: info.depth as i32,
                            },
                        ],
                        dst_subresource: vk::ImageSubresourceLayers {
                            aspect_mask: vk::ImageAspectFlags::COLOR,
                            mip_level: 0,
                            base_array_layer: 0,
                            layer_count: 1,
                        },
                        dst_offsets: [top_left, bottom_right],
                    };
                    device.cmd_blit_image(
                        cmd_buff.inner,
                        texture_image.image.inner,
                        vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                        existing_texture.image.inner,
                        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                        &[region],
                        vk::Filter::NEAREST,
                    );

                    let barrier_src = image_barrier!(
                        image: existing_texture.image.inner,
                        access: SHADER_WRITE => SHADER_READ,
                        layout: TRANSFER_DST_OPTIMAL => SHADER_READ_ONLY_OPTIMAL,
                        stage: TRANSFER => FRAGMENT_SHADER,
                        aspect: COLOR
                    );
                    cmd_buff.image_barrier(&[barrier_src, barrier_dst]);

                    device.end_command_buffer(cmd_buff.inner).unwrap();
                    let cmd_buffs = [cmd_buff.inner];
                    let submit_infos = [vk::SubmitInfo::default().command_buffers(&cmd_buffs)];
                    device
                        .queue_submit(device.graphics_queue(), &submit_infos, cmd_buff_fence.inner)
                        .unwrap();
                    device.wait_for_fences(&[cmd_buff_fence.inner], true, u64::MAX).unwrap();

                    // texture_image.destroy(device);
                }
            }
        } else {
            // Otherwise save the newly created texture

            // update dsc set
            let descritptor_set_layouts = [self.descriptor_set_layouts[0]];
            let dsc_set = {
                let dsc_alloc_info = vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(self.descriptor_pool.inner)
                    .set_layouts(&descritptor_set_layouts);
                unsafe { device.allocate_descriptor_sets(&dsc_alloc_info).unwrap()[0] }
            };
            let image_infos = [vk::DescriptorImageInfo::default()
                .image_view(texture_image.image_view.inner)
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .sampler(self.sampler.inner)];

            let dsc_writes = [vk::WriteDescriptorSet::default()
                .dst_set(dsc_set)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .dst_array_element(0_u32)
                .dst_binding(0_u32)
                .image_info(&image_infos)];
            unsafe {
                device.update_descriptor_sets(&dsc_writes, &[]);
            }
            // register new texture
            self.texture_images.insert(texture_id, texture_image);
            self.texture_desc_sets.insert(texture_id, dsc_set);
        }
    }

    /// Registering user texture.
    ///
    /// Pass the Vulkan ImageView and Sampler.
    /// `image_view`'s image layout must be `SHADER_READ_ONLY_OPTIMAL`.
    ///
    /// UserTexture needs to be unregistered when it is no longer needed.
    pub(crate) fn register_user_texture(&mut self, image_view: vk::ImageView, sampler: vk::Sampler) -> egui::TextureId {
        // get texture id
        let mut id = None;
        for (i, user_texture) in self.user_textures.iter().enumerate() {
            if user_texture.is_none() {
                id = Some(i as u64);
                break;
            }
        }
        let id = if let Some(i) = id {
            i
        } else {
            self.user_textures.len() as u64
        };

        let layouts = [self.user_texture_layout];
        let descriptor_set = unsafe {
            self.device.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(self.descriptor_pool.inner)
                    .set_layouts(&layouts),
            )
        }
        .expect("Failed to create descriptor sets.")[0];
        unsafe {
            self.device.update_descriptor_sets(
                &[vk::WriteDescriptorSet::default()
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .dst_set(descriptor_set)
                    .image_info(&[vk::DescriptorImageInfo::default()
                        .image_view(image_view)
                        .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                        .sampler(sampler)])
                    .dst_binding(0)],
                &[],
            );
        }

        if id == self.user_textures.len() as u64 {
            self.user_textures.push(Some(descriptor_set));
        } else {
            self.user_textures[id as usize] = Some(descriptor_set);
        }

        egui::TextureId::User(id)
    }

    /// Unregister user texture.
    pub(crate) fn unregister_user_texture(&mut self, texture_id: egui::TextureId) {
        if let egui::TextureId::User(id) = texture_id {
            if let Some(descriptor_set) = self.user_textures[id as usize] {
                unsafe {
                    self.device
                        .free_descriptor_sets(self.descriptor_pool.inner, &[descriptor_set])
                        .expect("Failed to free descriptor sets.");
                }
                self.user_textures[id as usize] = None;
            }
        } else {
            eprintln!("The internal texture cannot be unregistered; please pass the texture ID of UserTexture.");
        }
    }
}

impl Drop for Integration<'_> {
    fn drop(&mut self) {
        unsafe {
            // destroying the swapchain on linux causes a segfault
            #[cfg(target_os = "windows")]
            ManuallyDrop::drop(&mut self.egui_winit);

            self.device
                .destroy_descriptor_set_layout(self.user_texture_layout, None);
            self.device.destroy_pipeline(self.pipeline, None);
            self.device.destroy_pipeline_layout(self.pipeline_layout, None);
            for &descriptor_set_layout in self.descriptor_set_layouts.iter() {
                self.device.destroy_descriptor_set_layout(descriptor_set_layout, None);
            }
        }
    }
}
