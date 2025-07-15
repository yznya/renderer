use std::mem::transmute;

use anyhow::Result;
use ash::vk::{self, Packed24_8};
use gpu_allocator::MemoryLocation;
use lava::{
    device::LDeviceRef,
    resources::{LAccelerationStructureKHR, LBuffer, LMappedBuffer},
    with,
};

use scenery::scene::Scene;
use scenery::vertex::Vertex;

pub(crate) unsafe fn build_blas<'d>(
    device: LDeviceRef<'d>,
    scene: &Scene,
    vertex_buffer: &LBuffer<'d>,
    index_buffer: &LBuffer<'d>,
    staging_buffer: &LMappedBuffer<'d>,
) -> Result<(Vec<LAccelerationStructureKHR<'d>>, LBuffer<'d>)> {
    let _span = tracy_client::span!("Build BLAS");
    let acc_struct_geo = scene
        .geometry
        .meshes
        .iter()
        .map(|m| {
            let mut acc_struct_data = vk::AccelerationStructureGeometryDataKHR::default();
            let vertex_address = vk::DeviceOrHostAddressConstKHR {
                device_address: vertex_buffer.get_device_address()
                    + m.vertex_offset as u64 * size_of::<Vertex>() as u64,
            };
            let index_address = vk::DeviceOrHostAddressConstKHR {
                device_address: index_buffer.get_device_address() + m.index_offset as u64 * size_of::<u32>() as u64,
            };
            acc_struct_data.triangles = vk::AccelerationStructureGeometryTrianglesDataKHR::default()
                .vertex_format(vk::Format::R32G32B32_SFLOAT)
                .vertex_data(vertex_address)
                .vertex_stride(size_of::<Vertex>() as u64)
                .max_vertex(m.vertex_count - 1)
                .index_type(vk::IndexType::UINT32)
                .index_data(index_address);

            vk::AccelerationStructureGeometryKHR::default()
                .flags(vk::GeometryFlagsKHR::OPAQUE)
                .geometry_type(vk::GeometryTypeKHR::TRIANGLES)
                .geometry(acc_struct_data)
        })
        .collect::<Vec<_>>();

    let mut max_primitive_counts = Vec::new();
    for mi in 0..(scene.geometry.meshes.len() - 1) {
        let size = scene.geometry.meshes[mi + 1].index_offset - scene.geometry.meshes[mi].index_offset;
        max_primitive_counts.push(size / 3);
    }

    max_primitive_counts.push(
        (scene.geometry.indicies.len() as u32 - scene.geometry.meshes[scene.geometry.meshes.len() - 1].index_offset)
            / 3,
    );

    let alignment = 256;
    let mut scratch_size = 0;
    let mut acc_struct_size = 0;
    let mut size_infos = Vec::new();
    let mut acc_offsets = Vec::new();
    let mut scratch_offsets = Vec::new();
    let build_flags = vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE
        | vk::BuildAccelerationStructureFlagsKHR::ALLOW_COMPACTION;

    for (i, acc_geo) in acc_struct_geo.iter().enumerate() {
        let geo = [*acc_geo];
        let build_info = vk::AccelerationStructureBuildGeometryInfoKHR::default()
            .geometries(&geo)
            .ty(vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL)
            .flags(build_flags)
            .mode(vk::BuildAccelerationStructureModeKHR::BUILD);

        let mut size_info = vk::AccelerationStructureBuildSizesInfoKHR::default();

        unsafe {
            device.get_acceleration_structure_build_sizes(
                vk::AccelerationStructureBuildTypeKHR::DEVICE,
                &build_info,
                &[max_primitive_counts[i]],
                &mut size_info,
            )
        };

        acc_offsets.push(acc_struct_size);
        scratch_offsets.push(scratch_size);
        scratch_size += (size_info.build_scratch_size + alignment - 1) & !(alignment - 1);
        acc_struct_size += (size_info.acceleration_structure_size + alignment - 1) & !(alignment - 1);
        size_infos.push(size_info);
    }

    log::info!(
        "BLAS acceleration structure Size : {:?}",
        (acc_struct_size as f64) / 1e6
    );
    log::info!("BLAS scratch Size : {:?}", (scratch_size as f64) / 1e6);

    let blas_buffer = device.create_buffer(
        "blas_buffer",
        acc_struct_size,
        vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR,
        MemoryLocation::GpuOnly,
    )?;

    let mut blases = Vec::new();
    let mut build_infos = Vec::new();
    let mut build_range_info = Vec::new();

    for (i, acc_geo) in acc_struct_geo.iter().enumerate() {
        let info = vk::AccelerationStructureCreateInfoKHR::default()
            .buffer(blas_buffer.inner)
            .offset(acc_offsets[i])
            .size(size_infos[i].acceleration_structure_size)
            .ty(vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL);

        blases.push(unsafe { device.create_acceleration_structure(&info) }?);

        let scratch_address = vk::DeviceOrHostAddressKHR {
            device_address: staging_buffer.get_device_address() + scratch_offsets[i],
        };
        let mut build_info = vk::AccelerationStructureBuildGeometryInfoKHR::default()
            .ty(vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL)
            .flags(build_flags)
            .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
            .scratch_data(scratch_address)
            .dst_acceleration_structure(blases[i].inner);

        build_info.geometry_count = 1;
        build_info.p_geometries = acc_geo;

        build_infos.push(build_info);

        build_range_info
            .push(vk::AccelerationStructureBuildRangeInfoKHR::default().primitive_count(max_primitive_counts[i]))
    }

    let mut build_range_info_ptrs = Vec::new();
    for i in 0..build_range_info.len() {
        build_range_info_ptrs.push(&build_range_info[i..(i + 1)]);
    }

    let query_pool = unsafe {
        device.create_query_pool(
            &vk::QueryPoolCreateInfo::default()
                .query_type(vk::QueryType::ACCELERATION_STRUCTURE_COMPACTED_SIZE_KHR)
                .query_count(blases.len() as u32),
            None,
        )
    }?;

    unsafe {
        let cmd = device.begin_single_time_command()?;
        cmd.build_acceleration_structures(&build_infos, build_range_info_ptrs.as_slice());
        cmd.reset_query_pool(query_pool, 0, blases.len() as u32);
        cmd.write_acceleration_structures_properties(
            transmute::<&[LAccelerationStructureKHR<'_>], &[vk::AccelerationStructureKHR]>(blases.as_slice()),
            vk::QueryType::ACCELERATION_STRUCTURE_COMPACTED_SIZE_KHR,
            query_pool,
            0,
        );

        device.end_single_time_command(cmd)?;
    }

    unsafe {
        let mut compacted_sizes = vec![0u64; blases.len()];
        device.get_query_pool_results(
            query_pool,
            0,
            &mut compacted_sizes,
            vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT,
        )?;

        device.destroy_query_pool(query_pool, None);

        let mut compated_offsets = vec![0u64; blases.len()];
        let mut total_compacted_size: u64 = 0;

        for i in 0..blases.len() {
            compated_offsets[i] = total_compacted_size;
            total_compacted_size = (total_compacted_size + compacted_sizes[i] + alignment - 1) & !(alignment - 1);
        }

        log::info!(
            "Total compated size: {} MB",
            (total_compacted_size as f64) / (1024.0 * 1024.0)
        );

        let compated_blas_buffer = device.create_buffer(
            "compated_blas_buffer",
            total_compacted_size,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR,
            MemoryLocation::GpuOnly,
        )?;

        let mut compacted_blases = Vec::new();
        for i in 0..blases.len() {
            let info = vk::AccelerationStructureCreateInfoKHR::default()
                .buffer(compated_blas_buffer.inner)
                .offset(compated_offsets[i])
                .size(compacted_sizes[i])
                .ty(vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL);

            compacted_blases.push(device.create_acceleration_structure(&info)?);
        }

        let cmd = device.begin_single_time_command()?;

        for i in 0..blases.len() {
            cmd.copy_acceleration_structure(
                &vk::CopyAccelerationStructureInfoKHR::default()
                    .src(blases[i].inner)
                    .dst(compacted_blases[i].inner)
                    .mode(vk::CopyAccelerationStructureModeKHR::COMPACT),
            );
        }

        device.end_single_time_command(cmd)?;

        Ok((compacted_blases, compated_blas_buffer))
    }
}

pub(crate) unsafe fn build_tlas<'d>(
    device: LDeviceRef<'d>,
    scene: &Scene,
    blases: &[LAccelerationStructureKHR<'d>],
    staging_buffer: &LMappedBuffer<'d>,
) -> Result<(LAccelerationStructureKHR<'d>, LBuffer<'d>)> {
    let _span = tracy_client::span!("Build TLAS");
    let blas_addresses = scene
        .geometry
        .meshes
        .iter()
        .enumerate()
        .map(|(i, _)| {
            let info = vk::AccelerationStructureDeviceAddressInfoKHR::default().acceleration_structure(blases[i].inner);
            unsafe { device.get_acceleration_structure_device_address(&info) }
        })
        .collect::<Vec<_>>();

    let instances = device.create_mapped_buffer(
        "Instances",
        (size_of::<vk::AccelerationStructureInstanceKHR>() * scene.mesh_draws.len()) as u64,
        with!(vk::BufferUsageFlags => {STORAGE_BUFFER | ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR}),
    )?;

    for (i, mesh_draw) in scene.mesh_draws.iter().enumerate() {
        let model = &scene.global_transforms[mesh_draw.model_index as usize];
        let transform = vk::TransformMatrixKHR {
            matrix: [
                model.x_axis[0],
                model.y_axis[0],
                model.z_axis[0],
                model.w_axis[0],
                model.x_axis[1],
                model.y_axis[1],
                model.z_axis[1],
                model.w_axis[1],
                model.x_axis[2],
                model.y_axis[2],
                model.z_axis[2],
                model.w_axis[2],
            ],
        };
        let acceleration_structure_reference = vk::AccelerationStructureReferenceKHR {
            device_handle: blas_addresses[mesh_draw.mesh_index as usize],
        };
        let instance = vk::AccelerationStructureInstanceKHR {
            transform,
            instance_custom_index_and_mask: Packed24_8::new(i as u32, 0xFF),
            instance_shader_binding_table_record_offset_and_flags: Packed24_8::new(0, 0),
            acceleration_structure_reference,
        };

        unsafe {
            std::ptr::copy_nonoverlapping(
                &instance,
                instances
                    .memory_map
                    .cast::<vk::AccelerationStructureInstanceKHR>()
                    .add(i),
                1,
            )
        };
    }

    let mut acc_struct_data = vk::AccelerationStructureGeometryDataKHR::default();
    let address = vk::DeviceOrHostAddressConstKHR {
        device_address: instances.get_device_address(),
    };
    acc_struct_data.instances = vk::AccelerationStructureGeometryInstancesDataKHR::default().data(address);

    let acc_struct_geo = vk::AccelerationStructureGeometryKHR::default()
        .flags(vk::GeometryFlagsKHR::OPAQUE)
        .geometry_type(vk::GeometryTypeKHR::INSTANCES)
        .geometry(acc_struct_data);

    let geometries = [acc_struct_geo];
    let build_info = vk::AccelerationStructureBuildGeometryInfoKHR::default()
        .geometries(&geometries)
        .ty(vk::AccelerationStructureTypeKHR::TOP_LEVEL)
        .flags(
            vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE
                | vk::BuildAccelerationStructureFlagsKHR::ALLOW_COMPACTION,
        )
        .mode(vk::BuildAccelerationStructureModeKHR::BUILD);

    let max_primitive_counts = [scene.mesh_draws.len() as u32];
    let mut size_info = vk::AccelerationStructureBuildSizesInfoKHR::default();

    unsafe {
        device.get_acceleration_structure_build_sizes(
            vk::AccelerationStructureBuildTypeKHR::DEVICE,
            &build_info,
            &max_primitive_counts,
            &mut size_info,
        )
    };

    log::info!(
        "TLAS acceleration structure size: {}",
        (size_info.acceleration_structure_size as f64) / 1e6
    );
    log::info!("TLAS scratch size: {}", (size_info.build_scratch_size as f64) / 1e6);

    let tlas_buffer = device.create_buffer(
        "TLAS buffer",
        size_info.acceleration_structure_size,
        vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR,
        MemoryLocation::GpuOnly,
    )?;

    let create_info = vk::AccelerationStructureCreateInfoKHR::default()
        .buffer(tlas_buffer.inner)
        .offset(0)
        .size(size_info.acceleration_structure_size)
        .ty(vk::AccelerationStructureTypeKHR::TOP_LEVEL);

    let tlas = unsafe { device.create_acceleration_structure(&create_info)? };
    let scratch_address = vk::DeviceOrHostAddressKHR {
        device_address: staging_buffer.get_device_address(),
    };

    let build_info = build_info
        .dst_acceleration_structure(tlas.inner)
        .scratch_data(scratch_address);

    let build_range =
        vk::AccelerationStructureBuildRangeInfoKHR::default().primitive_count(scene.mesh_draws.len() as u32);

    let query_pool = unsafe {
        device.create_query_pool(
            &vk::QueryPoolCreateInfo::default()
                .query_type(vk::QueryType::ACCELERATION_STRUCTURE_COMPACTED_SIZE_KHR)
                .query_count(1),
            None,
        )
    }?;

    unsafe {
        let cmd = device.begin_single_time_command()?;
        cmd.build_acceleration_structures(&[build_info], &[&[build_range]]);
        cmd.reset_query_pool(query_pool, 0, 1);
        cmd.write_acceleration_structures_properties(
            &[tlas.inner],
            vk::QueryType::ACCELERATION_STRUCTURE_COMPACTED_SIZE_KHR,
            query_pool,
            0,
        );
        device.end_single_time_command(cmd)?;
    }

    unsafe {
        let mut compacted_size = vec![0u64; 1];
        device.get_query_pool_results(
            query_pool,
            0,
            &mut compacted_size,
            vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT,
        )?;

        device.destroy_query_pool(query_pool, None);

        log::info!(
            "Total compated size: {} MB",
            (compacted_size[0] as f64) / (1024.0 * 1024.0)
        );

        let compated_tlas_buffer = device.create_buffer(
            "compated_tlas_buffer",
            compacted_size[0],
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR,
            MemoryLocation::GpuOnly,
        )?;

        let info = vk::AccelerationStructureCreateInfoKHR::default()
            .buffer(compated_tlas_buffer.inner)
            .offset(0)
            .size(compacted_size[0])
            .ty(vk::AccelerationStructureTypeKHR::TOP_LEVEL);

        let compacted_tlas = device.create_acceleration_structure(&info)?;

        let cmd = device.begin_single_time_command()?;

        cmd.copy_acceleration_structure(
            &vk::CopyAccelerationStructureInfoKHR::default()
                .src(tlas.inner)
                .dst(compacted_tlas.inner)
                .mode(vk::CopyAccelerationStructureModeKHR::COMPACT),
        );

        device.end_single_time_command(cmd)?;

        Ok((compacted_tlas, compated_tlas_buffer))
    }
}
