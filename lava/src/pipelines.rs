use crate::{
    device::LDeviceRef,
    program::Program,
    resources::DescriptorInfo,
    shaders::{Shader, compile_shader},
};
use anyhow::{Result, anyhow};
use ash::vk;
use std::fmt::{Debug, Display};
use std::path::{Path, PathBuf};

pub const MAX_DESCRIPTOR_COUNT: u32 = 1 << 15;
pub const SHADERS_PATH: &str = "assets/shaders/";
pub const SHADERS_OUT_PATH: &str = "assets/out/shaders/";

pub struct NoSpecializationConstants;

impl From<NoSpecializationConstants> for [u32; 0] {
    fn from(_: NoSpecializationConstants) -> Self {
        []
    }
}

pub struct PipelineInfo<const N: usize = 0, SC: Into<[u32; N]> = NoSpecializationConstants> {
    pub shaders: &'static [&'static str],
    pub specialization_constants: Option<SC>,
    pub viewport_extent: vk::Extent2D,
    pub color_format: vk::Format,
    pub cull_mode: vk::CullModeFlags,
    pub depth_write: bool,
    pub depth_test: bool,
    pub has_depth_attachment: bool,
    pub blend_enable: bool,
    pub dynamic_states: Vec<vk::DynamicState>,
}

impl<const N: usize, SC: Into<[u32; N]>> Default for PipelineInfo<N, SC> {
    fn default() -> Self {
        Self {
            shaders: &[],
            specialization_constants: None,
            viewport_extent: vk::Extent2D::default(),
            color_format: vk::Format::UNDEFINED,
            cull_mode: vk::CullModeFlags::FRONT,
            depth_write: true,
            depth_test: true,
            blend_enable: false,
            has_depth_attachment: true,
            dynamic_states: Vec::new(),
        }
    }
}

pub fn create_pipeline<const N: usize, SC: Into<[u32; N]>>(
    device: LDeviceRef,
    pipeline_info: PipelineInfo<N, SC>,
) -> Result<Program> {
    let mut shaders: Vec<Shader> = pipeline_info
        .shaders
        .iter()
        .map(|s| format!("{SHADERS_OUT_PATH}/{s}.spv"))
        .map(|f| Shader::from_file(device, f))
        .collect::<Result<Vec<_>, _>>()?;

    let mut stages = Shader::extract_stages(&shaders);

    let constants: [u32; N];
    let specialization_map: Vec<_>;
    let specialization_info: vk::SpecializationInfo;
    if let Some(specialization_constants) = pipeline_info.specialization_constants {
        constants = specialization_constants.into();
        specialization_map = constants
            .iter()
            .enumerate()
            .map(|(i, _)| vk::SpecializationMapEntry {
                constant_id: i as u32,
                offset: (size_of::<u32>() * i) as u32,
                size: size_of::<u32>(),
            })
            .collect();

        specialization_info = vk::SpecializationInfo::default()
            .map_entries(&specialization_map)
            .data(bytemuck::cast_slice(&constants));

        for stage in &mut stages {
            stage.p_specialization_info = &specialization_info;
        }
    }

    let bindings = Shader::extract_bindings(&shaders, 0);
    let info = vk::DescriptorSetLayoutCreateInfo::default()
        .bindings(&bindings)
        .flags(vk::DescriptorSetLayoutCreateFlags::PUSH_DESCRIPTOR_KHR);
    let descriptor_set_layout = unsafe { device.create_descriptor_set_layout(&info)? };

    let texture_binding = [vk::DescriptorSetLayoutBinding::default()
        .binding(0)
        .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
        .stage_flags(vk::ShaderStageFlags::FRAGMENT)
        .descriptor_count(MAX_DESCRIPTOR_COUNT)];

    let binding_flags =
        [vk::DescriptorBindingFlags::PARTIALLY_BOUND | vk::DescriptorBindingFlags::VARIABLE_DESCRIPTOR_COUNT];
    let mut binding_flags_info = vk::DescriptorSetLayoutBindingFlagsCreateInfo::default().binding_flags(&binding_flags);

    let info = vk::DescriptorSetLayoutCreateInfo::default()
        .bindings(&texture_binding)
        .push_next(&mut binding_flags_info);

    let texture_set_layout = unsafe { device.create_descriptor_set_layout(&info)? };

    let layout_sets = [descriptor_set_layout, texture_set_layout];
    let pipeline_layout_info = vk::PipelineLayoutCreateInfo::default().set_layouts(&layout_sets);
    let push_constants_size = shaders.iter().map(|s| s.push_constants_size).max();
    let push_constant_ranges = [vk::PushConstantRange::default()
        .offset(0)
        .size(push_constants_size.unwrap_or_default())
        .stage_flags(vk::ShaderStageFlags::ALL_GRAPHICS)];

    let pipeline_layout_info = if push_constant_ranges[0].size > 0 {
        pipeline_layout_info.push_constant_ranges(&push_constant_ranges)
    } else {
        pipeline_layout_info
    };
    let pipeline_layout = unsafe { device.create_pipeline_layout(&pipeline_layout_info, None)? };

    let mut descriptor_update_entries = Vec::new();

    for binding in &bindings {
        descriptor_update_entries.push(
            vk::DescriptorUpdateTemplateEntry::default()
                .dst_binding(binding.binding)
                .descriptor_count(1)
                .descriptor_type(binding.descriptor_type)
                .offset((binding.binding as usize) * size_of::<DescriptorInfo>())
                .stride(size_of::<DescriptorInfo>()),
        );
    }
    let info = vk::DescriptorUpdateTemplateCreateInfo::default()
        .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
        .template_type(vk::DescriptorUpdateTemplateType::PUSH_DESCRIPTORS_KHR)
        .descriptor_set_layout(descriptor_set_layout)
        .pipeline_layout(pipeline_layout)
        .descriptor_update_entries(&descriptor_update_entries);

    let descriptor_update_template = unsafe { device.create_descriptor_update_template(&info, None)? };

    let viewport = vk::Viewport::default()
        .x(0.0)
        .y(pipeline_info.viewport_extent.height as f32)
        .width(pipeline_info.viewport_extent.width as f32)
        .height(-(pipeline_info.viewport_extent.height as f32))
        .min_depth(0.0)
        .max_depth(1.0);

    let scissor = vk::Rect2D::default()
        .offset(vk::Offset2D { x: 0, y: 0 })
        .extent(pipeline_info.viewport_extent);

    let viewports = [viewport];
    let scissors = [scissor];

    let viewport_state = vk::PipelineViewportStateCreateInfo::default()
        .viewports(&viewports)
        .scissors(&scissors);

    let rasterization_state = vk::PipelineRasterizationStateCreateInfo::default()
        .line_width(1.0)
        .cull_mode(pipeline_info.cull_mode)
        .front_face(vk::FrontFace::CLOCKWISE);

    let multisample_state =
        vk::PipelineMultisampleStateCreateInfo::default().rasterization_samples(vk::SampleCountFlags::TYPE_1);

    let attachment = vk::PipelineColorBlendAttachmentState::default()
        .color_write_mask(vk::ColorComponentFlags::RGBA)
        .src_color_blend_factor(vk::BlendFactor::SRC_ALPHA)
        .dst_color_blend_factor(vk::BlendFactor::DST_ALPHA)
        .color_blend_op(vk::BlendOp::ADD)
        .blend_enable(pipeline_info.blend_enable);

    let attachments = &[attachment];
    let color_blend_state = vk::PipelineColorBlendStateCreateInfo::default()
        .attachments(attachments)
        .blend_constants([0.0, 0.0, 0.0, 0.0]);

    let depth_stencil_state = vk::PipelineDepthStencilStateCreateInfo::default()
        .depth_test_enable(pipeline_info.depth_test)
        .depth_write_enable(pipeline_info.depth_write)
        .depth_compare_op(vk::CompareOp::GREATER);

    let color_formats = [pipeline_info.color_format];
    let mut rendering_info = vk::PipelineRenderingCreateInfo::default().color_attachment_formats(&color_formats);
    if pipeline_info.has_depth_attachment {
        rendering_info.depth_attachment_format = device.get_depth_format()?;
    }

    let dynamic_states = vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&pipeline_info.dynamic_states);

    let mut info = vk::GraphicsPipelineCreateInfo::default()
        .dynamic_state(&dynamic_states)
        .stages(&stages)
        .viewport_state(&viewport_state)
        .rasterization_state(&rasterization_state)
        .multisample_state(&multisample_state)
        .color_blend_state(&color_blend_state)
        .depth_stencil_state(&depth_stencil_state)
        .layout(pipeline_layout)
        .subpass(0)
        .push_next(&mut rendering_info);

    let has_vertex_shader = stages.iter().any(|s| s.stage == vk::ShaderStageFlags::VERTEX);
    let vertex_input_state = vk::PipelineVertexInputStateCreateInfo::default()
        .vertex_attribute_descriptions(&[])
        .vertex_binding_descriptions(&[]);
    let input_assembly_state =
        vk::PipelineInputAssemblyStateCreateInfo::default().topology(vk::PrimitiveTopology::TRIANGLE_LIST);

    if has_vertex_shader {
        info.p_vertex_input_state = &vertex_input_state;
        info.p_input_assembly_state = &input_assembly_state;
    }

    let pipeline = match unsafe { device.create_graphics_pipelines(vk::PipelineCache::null(), &[info], None) } {
        Ok(pipelines) => pipelines[0],
        Err(_) => return Err(anyhow!("Failed to create pipeline")),
    };

    unsafe { device.destroy_descriptor_set_layout(texture_set_layout, None) };
    shaders.drain(..).for_each(|s| s.destroy(device));

    Ok(Program {
        descriptor_set_layout,
        descriptor_update_template,
        pipeline_layout,
        pipeline,
        local_size_x: 1,
        local_size_y: 1,
        local_size_z: 1,
        device,
    })
}

pub fn create_compute_pipeline<P, SC, const N: usize>(
    device: LDeviceRef,
    shader_file: P,
    constants: SC,
) -> Result<Program>
where
    P: AsRef<Path> + Debug + Display,
    SC: Into<[u32; N]>,
{
    let shader = [Shader::from_file(
        device,
        format!("{SHADERS_OUT_PATH}/{shader_file}.spv"),
    )?];

    let stage = Shader::extract_stages(&shader)[0];
    let bindings = Shader::extract_bindings(&shader, 0);

    let constants: [u32; N] = constants.into();
    let specialization_map: Vec<_> = constants
        .iter()
        .enumerate()
        .map(|(i, _)| vk::SpecializationMapEntry {
            constant_id: i as u32,
            offset: (size_of::<u32>() * i) as u32,
            size: size_of::<u32>(),
        })
        .collect();

    let specialization_info = vk::SpecializationInfo::default()
        .map_entries(&specialization_map)
        .data(bytemuck::cast_slice(&constants));

    let stage = stage.specialization_info(&specialization_info);

    let info = vk::DescriptorSetLayoutCreateInfo::default()
        .bindings(&bindings)
        .flags(vk::DescriptorSetLayoutCreateFlags::PUSH_DESCRIPTOR_KHR);

    let descriptor_set_layout = unsafe { device.create_descriptor_set_layout(&info)? };

    let layout_sets = [descriptor_set_layout];
    let pipeline_layout_info = vk::PipelineLayoutCreateInfo::default().set_layouts(&layout_sets);

    let push_constants_size = shader.iter().map(|s| s.push_constants_size).max();

    let push_constant_ranges = [vk::PushConstantRange::default()
        .offset(0)
        .size(push_constants_size.unwrap_or_default())
        .stage_flags(vk::ShaderStageFlags::COMPUTE)];

    let pipeline_layout_info = if push_constant_ranges[0].size > 0 {
        pipeline_layout_info.push_constant_ranges(&push_constant_ranges)
    } else {
        pipeline_layout_info
    };

    let pipeline_layout = unsafe { device.create_pipeline_layout(&pipeline_layout_info, None) }?;

    let mut descriptor_update_entries = Vec::new();

    for binding in &bindings {
        descriptor_update_entries.push(
            vk::DescriptorUpdateTemplateEntry::default()
                .dst_binding(binding.binding)
                .descriptor_count(1)
                .descriptor_type(binding.descriptor_type)
                .offset((binding.binding as usize) * size_of::<DescriptorInfo>())
                .stride(size_of::<DescriptorInfo>()),
        );
    }
    let info = vk::DescriptorUpdateTemplateCreateInfo::default()
        .pipeline_bind_point(vk::PipelineBindPoint::COMPUTE)
        .template_type(vk::DescriptorUpdateTemplateType::PUSH_DESCRIPTORS_KHR)
        .pipeline_layout(pipeline_layout)
        .descriptor_set_layout(descriptor_set_layout)
        .descriptor_update_entries(&descriptor_update_entries);

    let descriptor_update_template = unsafe { device.create_descriptor_update_template(&info, None)? };

    let create_info = vk::ComputePipelineCreateInfo::default()
        .stage(stage)
        .layout(pipeline_layout);

    let pipeline = match unsafe { device.create_compute_pipelines(vk::PipelineCache::null(), &[create_info], None) } {
        Ok(pipelines) => pipelines[0],
        Err(_) => return Err(anyhow!("Failed to create pipeline")),
    };

    let local_size_x = shader[0].local_size_x;
    let local_size_y = shader[0].local_size_y;
    let local_size_z = shader[0].local_size_z;

    shader[0].destroy(device);

    Ok(Program {
        descriptor_set_layout,
        descriptor_update_template,
        pipeline_layout,
        pipeline,
        local_size_x,
        local_size_y,
        local_size_z,
        device,
    })
}

pub fn compile_shaders() -> Result<()> {
    for glsl_file in (Path::new(SHADERS_PATH).read_dir()?).flatten() {
        if glsl_file.path().extension().is_some_and(|ext| ext == "glsl") {
            let output_file = PathBuf::from(SHADERS_OUT_PATH).join(format!(
                "{}.spv",
                glsl_file.path().file_name().unwrap().to_str().unwrap()
            ));
            compile_shader(glsl_file.path(), output_file)?;
        }
    }

    Ok(())
}
