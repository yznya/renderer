use anyhow::{Context, Result};
use ash::vk;
use spirv_cross2::{
    Compiler, Module,
    reflect::{ExecutionModeArguments, ResourceType, ShaderResources},
    spirv::{Decoration, ExecutionMode, ExecutionModel},
    targets,
};
use std::{fmt::Debug, fs::File, mem::transmute, path::Path};

use super::device::LDeviceRef;

#[derive(Debug)]
pub struct ShaderResource {
    pub binding: u32,
    pub set: u32,
    pub descriptor_type: vk::DescriptorType,
}

pub struct Shader {
    pub resource_bindings: Vec<ShaderResource>,
    pub stage: vk::ShaderStageFlags,
    pub shader_module: vk::ShaderModule,
    pub local_size_x: u32,
    pub local_size_y: u32,
    pub local_size_z: u32,
    pub push_constants_size: u32,
}

impl Shader {
    pub fn from_file<P>(device: LDeviceRef, path: P) -> Result<Self>
    where
        P: AsRef<Path> + Debug,
    {
        let shader_bytecode = std::fs::read(path)?;

        let mut x = Vec::<u32>::new();
        for i in (0..shader_bytecode.len()).step_by(4) {
            x.push(u32::from_be_bytes([
                shader_bytecode[i],
                shader_bytecode[i + 1],
                shader_bytecode[i + 2],
                shader_bytecode[i + 3],
            ]))
        }

        let module = Module::from_words(&x);

        let compiler = Compiler::<targets::None>::new(module)?;

        let resources = compiler.shader_resources()?;

        let resource_bindings = get_resource_bindings(&resources, &compiler)?;

        let mut push_constants_size: u32 = 0;
        for resource in resources.resources_for_type(ResourceType::PushConstant)? {
            let buffer_range = compiler.active_buffer_ranges(resource.id)?;
            for r in buffer_range {
                push_constants_size += r.range as u32;
            }
        }

        let stage = match compiler.execution_model()? {
            ExecutionModel::Vertex => vk::ShaderStageFlags::VERTEX,
            ExecutionModel::TaskEXT => vk::ShaderStageFlags::TASK_EXT,
            ExecutionModel::MeshEXT => vk::ShaderStageFlags::MESH_EXT,
            ExecutionModel::Fragment => vk::ShaderStageFlags::FRAGMENT,
            ExecutionModel::GLCompute => vk::ShaderStageFlags::COMPUTE,
            _ => panic!("Exectuion model not supported"),
        };

        let (local_size_x, local_size_y, local_size_z) =
            match compiler.execution_mode_arguments(ExecutionMode::LocalSize)? {
                Some(ExecutionModeArguments::LocalSize { x, y, z }) => (x, y, z),
                Some(_) => (1, 1, 1),
                None => (1, 1, 1),
            };

        let info = vk::ShaderModuleCreateInfo {
            p_code: unsafe { transmute::<*const u8, *const u32>(shader_bytecode.as_ptr()) },
            code_size: shader_bytecode.len(),
            ..Default::default()
        };

        let shader_module = unsafe { device.create_shader_module(&info, None) }?;

        Ok(Self {
            resource_bindings,
            stage,
            shader_module,
            local_size_x,
            local_size_y,
            local_size_z,
            push_constants_size,
        })
    }

    pub fn extract_bindings(shaders: &[Shader], set: u32) -> Vec<vk::DescriptorSetLayoutBinding<'_>> {
        let mut binding_slots: [Option<vk::DescriptorSetLayoutBinding>; 64] = [None; 64];

        for shader in shaders {
            for bind_point in shader.resource_bindings.iter().filter(|b| b.set == set) {
                assert!(bind_point.binding < 64);

                if let Some(ref mut set_layout_binding) = binding_slots[bind_point.binding as usize] {
                    if set_layout_binding.descriptor_type != bind_point.descriptor_type {
                        panic!("Conflicting descriptor types found");
                    }
                    set_layout_binding.stage_flags |= shader.stage;
                } else {
                    binding_slots[bind_point.binding as usize] = Some(
                        vk::DescriptorSetLayoutBinding::default()
                            .binding(bind_point.binding)
                            .descriptor_count(1)
                            .descriptor_type(bind_point.descriptor_type)
                            .stage_flags(shader.stage),
                    );
                }
            }
        }
        binding_slots
            .iter()
            .flatten()
            .cloned()
            .filter(|b| b.descriptor_count > 0)
            .collect::<Vec<_>>()
    }

    pub fn extract_stages(shaders: &[Shader]) -> Vec<vk::PipelineShaderStageCreateInfo<'_>> {
        let stages: Vec<vk::PipelineShaderStageCreateInfo> = shaders
            .iter()
            .map(|shader| {
                vk::PipelineShaderStageCreateInfo::default()
                    .module(shader.shader_module)
                    .stage(shader.stage)
                    .name(c"main")
            })
            .collect();
        stages
    }

    pub fn destroy(&self, device: LDeviceRef) {
        unsafe { device.destroy_shader_module(self.shader_module, None) };
    }
}

fn get_resource_bindings<T>(resources: &ShaderResources, compiler: &Compiler<T>) -> Result<Vec<ShaderResource>> {
    let resource_types = [
        ResourceType::StorageBuffer,
        ResourceType::UniformBuffer,
        ResourceType::StorageImage,
        ResourceType::SampledImage,
        ResourceType::SeparateImage,
        ResourceType::SeparateSamplers,
        ResourceType::AccelerationStructure,
    ];

    let mut shader_resource = Vec::new();
    for &resource_type in &resource_types {
        for resource in resources.resources_for_type(resource_type)? {
            let set = compiler
                .decoration(resource.id, Decoration::DescriptorSet)?
                .unwrap()
                .as_literal()
                .unwrap();

            let binding = compiler
                .decoration(resource.id, Decoration::Binding)
                .expect("Storage buffer doesn't belong to any binding")
                .unwrap()
                .as_literal()
                .unwrap();

            let descriptor_type = match resource_type {
                ResourceType::StorageBuffer => vk::DescriptorType::STORAGE_BUFFER,
                ResourceType::UniformBuffer => vk::DescriptorType::UNIFORM_BUFFER,
                ResourceType::StorageImage => vk::DescriptorType::STORAGE_IMAGE,
                ResourceType::SampledImage => vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                ResourceType::AccelerationStructure => vk::DescriptorType::ACCELERATION_STRUCTURE_KHR,
                _ => panic!("Descriptor type is not supported: {resource_type:?}"),
            };

            shader_resource.push(ShaderResource {
                binding,
                descriptor_type,
                set,
            });
        }
    }
    Ok(shader_resource)
}

pub fn compile_shader(input: impl AsRef<Path> + Debug, output: impl AsRef<Path> + Debug) -> Result<()> {
    let output_file = File::open(&output);
    let input_file = File::open(&input);
    if let (Ok(input_file), Ok(output_file)) = (input_file, output_file) {
        let input_metadata = input_file.metadata()?;
        let output_metadata = output_file.metadata()?;
        if input_metadata.modified()? < output_metadata.modified()? {
            log::trace!("Skipping compilation of {input:?}, output is up to date");
            return Ok(());
        }
    }

    let parent = input.as_ref().parent().context("File has no parent")?;
    let compiler = shaderc::Compiler::new().context("Initializing shader compiler failed")?;
    let mut options = shaderc::CompileOptions::new().context("Initializing compile options failed")?;
    options.set_target_spirv(shaderc::SpirvVersion::V1_6);
    options.set_include_callback(|filename, _type, _source, _include_depth| {
        let path = parent.join(filename);
        let source = std::fs::read_to_string(&path).unwrap();
        Ok(shaderc::ResolvedInclude {
            resolved_name: path.to_str().unwrap().to_string(),
            content: source,
        })
    });

    let s = input.as_ref().file_name().unwrap().to_str().unwrap();
    let kind = if s.ends_with(".frag.glsl") {
        shaderc::ShaderKind::Fragment
    } else if s.ends_with(".vert.glsl") {
        shaderc::ShaderKind::Vertex
    } else if s.ends_with(".mesh.glsl") {
        shaderc::ShaderKind::Mesh
    } else if s.ends_with(".task.glsl") {
        shaderc::ShaderKind::Task
    } else if s.ends_with(".comp.glsl") {
        shaderc::ShaderKind::Compute
    } else if s.ends_with(".inc.glsl") {
        return Ok(());
    } else {
        panic!("Unknown shader type {input:?}")
    };

    log::info!("Compiling {input:?}, output {output:?}");

    let source = std::fs::read_to_string(&input)?;

    let binary_result = compiler.compile_into_spirv(
        &source,
        kind,
        input.as_ref().as_os_str().to_str().unwrap(),
        "main",
        Some(&options),
    )?;

    std::fs::write(output, binary_result.as_binary_u8())?;

    Ok(())
}
