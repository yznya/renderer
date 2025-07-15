use std::{fs::File, io::BufReader, mem::offset_of};

use anyhow::Result;
use bitflags::bitflags;
use glam::{Vec3, Vec4, Vec4Swizzles, vec3};
use meshopt::VertexDataAdapter;
use obj::Obj;

use super::vertex::Vertex;

#[repr(u32)]
pub enum AlphaMode {
    Opaque = 0,
    Mask = 1,
}

#[repr(C)]
#[derive(Default, Debug, Clone, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
pub struct Meshlet {
    center: Vec3,
    radius: f32,
    cone_axis: [i8; 3],
    cone_cutoff: i8,
    vertex_offset: u32,
    triangle_offset: u32,
    triangle_count: u8,
    vertex_count: u8,
}

#[repr(C)]
#[derive(Default, Debug, Clone, Copy, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
pub struct MeshLod {
    pub meshlet_offset: u32,
    pub meshlet_count: u32,
    pub error: f32,
    _padding: u32,
}

#[repr(C)]
#[derive(Default, Debug, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
pub struct Mesh {
    pub center: Vec3,
    pub radius: f32,
    pub vertex_offset: u32,
    pub vertex_count: u32,
    pub lod_count: u32,
    pub index_offset: u32,
    pub lods: [MeshLod; 8],
}

#[derive(Default, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
pub struct Geometry {
    pub vertices: Vec<Vertex>,
    pub indicies: Vec<u32>,
    pub meshlet_vertices: Vec<u32>,
    pub meshlet_triangles: Vec<u8>,
    pub meshlets: Vec<Meshlet>,
    pub meshes: Vec<Mesh>,
}

#[repr(C)]
#[derive(Default, Debug, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
pub struct MeshDraw {
    pub alpha_mode: u32,
    // can use u16 for meshlet_visibility_offset and model_index
    pub meshlet_visibility_offset: u32,
    pub model_index: u32,
    pub mesh_index: u16,
    pub material_index: u16,
}

bitflags! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
    pub struct MaterialType: u32 {
        const METAL_ROUGHNESS = 1;
        const SPEC_GLOSS = (1 << 1);
        const CLEARCOAT = (1 << 2);
        const SHEEN = (1 << 3);
        const SPEC = (1 << 4);
    }
}

#[repr(C)]
#[derive(Debug, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
// TODO: most of these fields can be packed
pub struct Material {
    pub base_color_factor: Vec4,
    pub metallic_roughness_normal_occlusion: Vec4,
    pub emissive_factor_alpha_cutoff: Vec4,
    pub specular_glossiness_factor: Vec4,
    pub clearcoat_factor: Vec4,
    pub sheen_color_roughness_factor: Vec4,
    pub albedo_texture_transform: Vec4,
    pub normal_texture_transform: Vec4,
    pub metallic_roughness_texture_transform: Vec4,
    pub emissive_texture_transform: Vec4,
    pub occlusion_texture_transform: Vec4,
    pub specular_factor: Vec4,

    pub albedo_texture: u32,
    pub albedo_texture_uv: u32,
    pub normal_texture: u32,
    pub normal_texture_uv: u32,

    pub metallic_roughness_texture: u32,
    pub metallic_roughness_texture_uv: u32,
    pub emissive_texture: u32,
    pub emissive_texture_uv: u32,

    pub occlusion_texture: u32,
    pub occlusion_texture_uv: u32,
    pub clearcoat_texture: u32,
    pub clearcoat_texture_uv: u32,

    pub specular_color_texture: u32,
    pub specular_color_texture_uv: u32,
    pub specular_texture: u32,
    pub specular_texture_uv: u32,

    pub ior: f32,
    pub alpha_mode: u32,
    pub material_type: u32,
}

impl Default for Material {
    fn default() -> Self {
        Self {
            base_color_factor: Vec4::ONE,
            metallic_roughness_normal_occlusion: Vec4::ONE,
            emissive_factor_alpha_cutoff: Vec4::new(0.0, 0.0, 0.0, 1.0),
            specular_glossiness_factor: Vec4::ONE,
            clearcoat_factor: Vec4::ONE,
            sheen_color_roughness_factor: Vec4::ONE,
            albedo_texture_transform: Vec4::new(1.0, 1.0, 0.0, 0.0),
            normal_texture_transform: Vec4::new(1.0, 1.0, 0.0, 0.0),
            metallic_roughness_texture_transform: Vec4::new(1.0, 1.0, 0.0, 0.0),
            emissive_texture_transform: Vec4::new(1.0, 1.0, 0.0, 0.0),
            occlusion_texture_transform: Vec4::new(1.0, 1.0, 0.0, 0.0),
            specular_factor: Vec4::ONE,
            albedo_texture: !0,
            albedo_texture_uv: 0,
            normal_texture: !0,
            normal_texture_uv: 0,
            metallic_roughness_texture: !0,
            metallic_roughness_texture_uv: 0,
            emissive_texture: !0,
            emissive_texture_uv: 0,
            occlusion_texture: !0,
            occlusion_texture_uv: 0,
            clearcoat_texture: !0,
            clearcoat_texture_uv: 0,
            specular_color_texture: !0,
            specular_color_texture_uv: 0,
            specular_texture: !0,
            specular_texture_uv: 0,
            ior: 1.5,
            alpha_mode: 0,
            material_type: MaterialType::METAL_ROUGHNESS.bits(),
        }
    }
}

impl Geometry {
    pub fn load_obj_file(&mut self, filename: &str, fast_mode: bool) -> Result<()> {
        let reader = BufReader::new(File::open(filename)?);

        let model: Obj<obj::Vertex, u32> = obj::load_obj(reader)?;

        let mut vertices = Vec::new();
        let mut normals_f32 = Vec::new();

        for v in &model.vertices {
            let position = vec3(v.position[0], v.position[1], v.position[2]);
            let normals = Vec4::new(v.normal[0], v.normal[1], v.normal[2], 0.0);

            normals_f32.push(v.normal[0]);
            normals_f32.push(v.normal[1]);
            normals_f32.push(v.normal[2]);

            let vertex = Vertex {
                position: Vec4::new(position.x, position.y, position.z, 0.0),
                normals,
                tex_coords: Vec4::new(0.0, 0.0, 0.0, 0.0),
                tangents: Vec4::new(1.0, 0.0, 0.0, 1.0),
            };

            vertices.push(vertex);
        }

        self.append_mesh(&vertices, &model.indices, fast_mode)?;

        log::info!("Loaded mesh: {filename}");

        Ok(())
    }

    pub fn append_mesh(&mut self, vertices: &[Vertex], indices: &[u32], fast_mode: bool) -> Result<()> {
        let _span = tracy_client::span!("Append mesh");

        let (vertex_count, remap_data) = meshopt::generate_vertex_remap(vertices, Some(indices));
        let mut vertices = meshopt::remap_vertex_buffer(vertices, vertex_count, &remap_data);
        let mut indices = meshopt::remap_index_buffer(Some(indices), vertex_count, &remap_data);

        indices = meshopt::optimize_vertex_cache(&indices, vertex_count);

        if fast_mode {
            indices = meshopt::optimize_vertex_cache_fifo(&indices, vertices.len(), 16);
        } else {
            indices = meshopt::optimize_vertex_cache(&indices, vertex_count);
        }
        vertices = meshopt::optimize_vertex_fetch(&mut indices, &vertices);

        let max_vertices = 64;
        let max_triangles = 124;
        let cone_weight: f32 = 0.25;

        let meshlets = meshopt::build_meshlets(
            &indices,
            &VertexDataAdapter::new(
                to_byte_slice(&vertices),
                size_of::<Vertex>(),
                offset_of!(Vertex, position),
            )?,
            max_vertices,
            max_triangles,
            cone_weight,
        );

        let meshlet_offset = self.meshlets.len() as u32;
        let vertex_offset = self.vertices.len() as u32;
        let meshlet_count = meshlets.len() as u32;
        let vertex_count = vertices.len() as u32;

        let mut center = Vec3::ZERO;
        for v in &vertices {
            center += v.position.xyz();
        }
        center /= vertices.len() as f32;

        let mut radius: f32 = 0.0;
        for v in &vertices {
            radius = f32::max(radius, Vec3::distance(v.position.xyz(), center));
        }

        let mut mesh = Mesh {
            vertex_offset,
            vertex_count,
            center,
            radius,
            lod_count: 1,
            index_offset: self.indicies.len() as u32,
            ..Default::default()
        };
        mesh.lods[0] = MeshLod {
            meshlet_offset,
            meshlet_count,
            error: 0.0,
            ..Default::default()
        };

        self.append_meshlets(&vertices, &meshlets);

        self.meshlet_vertices.extend(meshlets.vertices);
        self.meshlet_triangles.extend(meshlets.triangles);

        let normals_f32: Vec<_> = vertices
            .iter()
            .flat_map(|v| [v.normals.x, v.normals.y, v.normals.z])
            .collect();

        let mut target_count = ((indices.len() as f64) * 0.65) as usize;
        while (mesh.lod_count as usize) < mesh.lods.len() {
            let mut error: f32 = 0.0;

            let normals_weight = [1.0, 1.0, 1.0];
            let mut lod_indices = {
                let mut result: Vec<u32> = vec![0; indices.len()];
                let index_count = unsafe {
                    meshopt::ffi::meshopt_simplifyWithAttributes(
                        result.as_mut_ptr().cast(),
                        indices.as_ptr().cast(),
                        indices.len(),
                        vertices.as_ptr().cast::<f32>(),
                        vertices.len(),
                        size_of::<Vertex>(),
                        normals_f32.as_ptr(),
                        size_of::<f32>() * 3,
                        normals_weight.as_ptr(),
                        normals_weight.len(),
                        std::ptr::null(),
                        target_count,
                        1e-1,
                        0,
                        &mut error as *mut _,
                    )
                };
                result.resize(index_count, 0u32);
                result
            };

            if lod_indices.len() >= (indices.len() as f64 * 0.95) as usize || lod_indices.is_empty() {
                break;
            }

            if fast_mode {
                lod_indices = meshopt::optimize_vertex_cache_fifo(&lod_indices, vertices.len(), 16);
            } else {
                lod_indices = meshopt::optimize_vertex_cache(&lod_indices, vertices.len());
            }

            let lod_meshlets = meshopt::build_meshlets(
                &lod_indices,
                &VertexDataAdapter::new(
                    to_byte_slice(&vertices),
                    size_of::<Vertex>(),
                    offset_of!(Vertex, position),
                )?,
                max_vertices,
                max_triangles,
                cone_weight,
            );

            let lod_scale = meshopt::simplify_scale(&VertexDataAdapter::new(
                to_byte_slice(&vertices),
                size_of::<Vertex>(),
                offset_of!(Vertex, position),
            )?);

            mesh.lods[mesh.lod_count as usize] = MeshLod {
                meshlet_offset: self.meshlets.len() as u32,
                meshlet_count: lod_meshlets.len() as u32,
                error: f32::max(error, mesh.lods[mesh.lod_count as usize - 1].error) * lod_scale,
                ..Default::default()
            };

            self.append_meshlets(&vertices, &lod_meshlets);
            self.meshlet_vertices.extend(lod_meshlets.vertices);
            self.meshlet_triangles.extend(lod_meshlets.triangles);

            mesh.lod_count += 1;
            target_count = ((target_count as f64) * 0.65) as usize;
        }

        self.vertices.extend(vertices);
        self.indicies.extend(indices);

        self.meshes.push(mesh);
        Ok(())
    }

    fn append_meshlets(&mut self, vertices: &[Vertex], meshlets: &meshopt::Meshlets) {
        self.meshlets.extend(meshlets.meshlets.iter().enumerate().map(|(i, m)| {
            let bounds = meshopt::compute_meshlet_bounds(
                meshlets.get(i),
                &VertexDataAdapter::new(
                    to_byte_slice(vertices),
                    size_of::<Vertex>(),
                    offset_of!(Vertex, position),
                )
                .unwrap(),
            );

            Meshlet {
                center: Vec3::from_array(bounds.center),
                radius: bounds.radius,
                cone_axis: bounds.cone_axis_s8,
                cone_cutoff: bounds.cone_cutoff_s8,
                vertex_offset: m.vertex_offset + self.meshlet_vertices.len() as u32,
                triangle_offset: m.triangle_offset + self.meshlet_triangles.len() as u32,
                triangle_count: m.triangle_count as u8,
                vertex_count: m.vertex_count as u8,
            }
        }));
    }
}

fn to_byte_slice<T>(data: &[T]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(data.as_ptr().cast(), std::mem::size_of_val(data)) }
}
