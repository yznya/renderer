use anyhow::{Context, Result, anyhow};
use glam::{Mat4, Quat, Vec2, Vec3, Vec4, vec2, vec3, vec4};
use gltf::{camera::Projection, khr_lights_punctual, mesh::Mode};
use serde_json::Value;
use std::{collections::HashMap, fs::File, path::Path, time::Instant};

#[derive(Default)]
pub struct Input {
    pub strafe: f32,
    pub forward: f32,
    pub fly: f32,
    pub mouse: Vec2,
}

use crate::{
    mesh::{Material, MaterialType},
    vertex::generate_tangents,
};

use super::{
    camera::Camera,
    mesh::{AlphaMode, Geometry, MeshDraw},
    vertex::Vertex,
};

#[derive(rkyv::Archive, rkyv::Serialize, rkyv::Deserialize, serde::Serialize, serde::Deserialize)]
pub struct EnvMap {
    pub skybox: String,
    pub brdf_lut: String,
    pub lambertian: String,
    pub ggx: String,
    pub charlie: String,
}

#[derive(rkyv::Archive, rkyv::Serialize, rkyv::Deserialize, Debug)]
pub struct Hierarchy {
    pub parent: i32,
    pub first_child: i32,
    pub next_sibling: i32,
    pub last_sibling: i32,
    pub level: i32,
}

#[derive(rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
pub struct Scene {
    pub geometry: Geometry,
    pub mesh_draws: Vec<MeshDraw>,
    pub materials: Vec<Material>,
    pub global_transforms: Vec<Mat4>,
    pub scale: Vec<Vec3>,
    pub rotation: Vec<Quat>,
    pub translation: Vec<Vec3>,
    pub heirarchy: Vec<Hierarchy>,
    pub node_names: Vec<String>,
    pub meshlet_visibility_count: u32,
    pub images: Vec<(usize, String)>,
    pub camera: Camera,
    pub sun_direction: Vec3,
    pub env_map: EnvMap,
}

impl Scene {
    pub fn load_gltf_file(model_file: &str, env_map: EnvMap) -> Result<Self> {
        let _span = tracy_client::span!("load gtlf file");

        let mut geometry = Geometry::default();

        let path = Path::new(model_file);
        let document =
            gltf::Gltf::from_reader_without_validation(File::open(path)?).context("Failed to import GLTF file")?;
        let parent = path.parent();
        let buffers = gltf::import_buffers(&document, parent, None).context("Failed to load GLTF buffers")?;

        let mut images = Vec::new();
        for texture in document.textures() {
            if let Some(image) = texture.source() {
                if let gltf::image::Source::Uri { uri, .. } = image.source() {
                    let relative_path = Path::new(uri);
                    let path: String = parent.unwrap().join(relative_path).to_str().unwrap().to_string();
                    images.push((texture.index(), path));
                } else {
                    log::warn!(
                        "Unsupported image source type: {}, {:?}",
                        texture.index(),
                        texture.source().unwrap().name()
                    );
                }
            } else {
                log::warn!(
                    "Unsupported texture: {}, {:?}",
                    texture.index(),
                    texture.source().unwrap().name()
                );
            }
        }

        let get_buffer_data = |buffer: gltf::Buffer| buffers.get(buffer.index()).map(|b| &*b.0);

        let gltf_scene = document
            .default_scene()
            .or_else(|| document.scenes().next())
            .context("GLTF file doesn't have a default scene")?;

        let mut nodes = Vec::new();
        for node in gltf_scene.nodes() {
            nodes.push(node);
        }
        let mut current_node = 0;

        while current_node < nodes.len() {
            nodes.extend(nodes[current_node].children());
            current_node += 1;
        }

        let mut sun_direction = Vec3::new(-0.23623444, 0.9302103, 0.2808959);
        let mut camera = None;
        let mut mesh_index: usize = 0;
        // TODO: try doing this without a hashmap
        let mut primitive_mesh_index_map = HashMap::new();
        for node in &nodes {
            if let Some(c) = node.camera() {
                if let Projection::Perspective(_) = c.projection() {
                    let (translation, rotation, _) = node.transform().decomposed();
                    let translation = Vec3::from_array(translation);
                    let rotation = Quat::from_array(rotation);
                    camera = Some(Camera {
                        position: translation,
                        direction: rotation * Vec3::Z,
                    });
                } else {
                    return Err(anyhow!("only Perspective is supported"));
                }
            }

            if let Some(light) = node.light()
                && let khr_lights_punctual::Kind::Directional = light.kind()
            {
                let m = node.transform().matrix();
                sun_direction = Vec3::new(m[2][0], m[2][1], m[2][2]);
            }

            if let Some(mesh) = node.mesh() {
                for primitive in mesh.primitives() {
                    let mut positions = Vec::new();
                    let mut normals = Vec::new();
                    let mut tangents = Vec::new();
                    let mut tex_coords = Vec::new();
                    let mut tex_coords2 = Vec::new();
                    let mut indices = Vec::new();

                    if primitive.mode() != Mode::Triangles {
                        return Err(anyhow!("only Triangles are supported"));
                    }

                    let reader = primitive.reader(get_buffer_data);

                    if let Some(pos_reader) = reader.read_positions() {
                        for pos in pos_reader {
                            positions.push(vec3(pos[0], pos[1], pos[2]));
                        }
                    }

                    if let Some(normal_reader) = reader.read_normals() {
                        for norm in normal_reader {
                            normals.push(vec3(norm[0], norm[1], norm[2]));
                        }
                    }

                    if let Some(tex_coords_reader) = reader.read_tex_coords(0) {
                        for uv in tex_coords_reader.into_f32() {
                            tex_coords.push(vec2(uv[0], uv[1]));
                        }
                    }

                    if let Some(tex_coords_reader) = reader.read_tex_coords(1) {
                        for uv in tex_coords_reader.into_f32() {
                            tex_coords2.push(vec2(uv[0], uv[1]));
                        }
                    }

                    if let Some(tangents_reader) = reader.read_tangents() {
                        for tang in tangents_reader {
                            tangents.push(vec4(tang[0], tang[1], tang[2], tang[3]));
                        }
                    }

                    let mut vertices = Vec::new();
                    for (i, position) in positions.iter().enumerate() {
                        let normals = *normals.get(i).unwrap_or(&Vec3::ZERO);
                        let tex_coord = tex_coords.get(i).unwrap_or(&Vec2::ZERO);
                        let tex_coord2 = tex_coords2.get(i).unwrap_or(&Vec2::ZERO);
                        let tangents = *tangents.get(i).unwrap_or(&Vec4::new(1.0, 0.0, 0.0, 1.0));

                        vertices.push(Vertex {
                            position: Vec4::new(position.x, position.y, position.z, 0.0),
                            normals: Vec4::new(normals.x, normals.y, normals.z, 0.0),
                            tex_coords: Vec4::new(tex_coord.x, tex_coord.y, tex_coord2.x, tex_coord2.y),
                            tangents,
                        });
                    }

                    if let Some(indices_reader) = reader.read_indices() {
                        indices.extend(indices_reader.into_u32());
                    }

                    if reader.read_tangents().is_none() {
                        generate_tangents(&mut vertices, &indices);
                    }

                    geometry.append_mesh(&vertices, &indices, false)?;
                    primitive_mesh_index_map.insert(mesh.index() * 16 + primitive.index(), mesh_index);
                    mesh_index += 1;
                }
            }
        }

        let materials: Vec<_> = document.materials().map(|m| build_material(&m)).collect();

        let camera = camera.unwrap_or(Camera {
            position: Vec3::ZERO,
            direction: Vec3::Z,
        });

        let mut scene = Self {
            geometry,
            mesh_draws: Vec::new(),
            materials,
            global_transforms: Vec::new(),
            heirarchy: Vec::new(),
            node_names: Vec::new(),
            camera,
            meshlet_visibility_count: 0,
            images,
            sun_direction,
            scale: Vec::new(),
            rotation: Vec::new(),
            translation: Vec::new(),
            env_map,
        };

        for node in gltf_scene.nodes() {
            build_scene_mesh_draws(&primitive_mesh_index_map, &mut scene, node, Mat4::IDENTITY, -1, 0);
        }
        log::info!("Loaded GLTF file: {model_file:?}");

        Ok(scene)
    }

    pub fn load_ptto_file(path: &str) -> Result<Self> {
        let _span = tracy_client::span!("load ptto file");
        let t = Instant::now();
        let data = std::fs::read(path)?;
        let total_time = Instant::now() - t;
        log::info!("Reading file: {}", total_time.as_secs_f64());

        let scene = unsafe { rkyv::from_bytes_unchecked::<Scene, rkyv::rancor::Error>(&data)? };
        let total_time = Instant::now() - t;
        log::info!("Loading ptto file: {}", total_time.as_secs_f64());
        Ok(scene)
    }

    pub fn load_obj_file(model_files: &str, env_map: EnvMap) -> Result<Self> {
        let mut geometry = Geometry::default();

        let model_files = model_files.split_terminator(',');
        for file_name in model_files {
            geometry.load_obj_file(file_name, false)?;
        }

        let mut mesh_draws: Vec<MeshDraw> = Vec::new();
        let mut global_transforms: Vec<Mat4> = Vec::new();
        let mut meshlet_visibility_count = 0;

        let model = Mat4::IDENTITY;
        let mesh_index = 0;
        let model_index = global_transforms.len() as u32;
        global_transforms.push(model);
        mesh_draws.push(MeshDraw {
            alpha_mode: 0,
            mesh_index: mesh_index as u16,
            meshlet_visibility_offset: meshlet_visibility_count,
            material_index: 0,
            model_index,
        });

        meshlet_visibility_count += geometry.meshes[mesh_index].lods[0].meshlet_count;

        let camera = Camera {
            position: Vec3::new(0.0, 0.0, 2.0),
            direction: Vec3::Z,
        };

        Ok(Self {
            geometry,
            mesh_draws,
            materials: vec![Material::default()],
            global_transforms,
            // TODO: fill these arrays
            heirarchy: Vec::new(),
            node_names: Vec::new(),
            camera,
            meshlet_visibility_count,
            images: Vec::new(),
            sun_direction: Vec3::new(-0.23623444, 0.9302103, 0.2808959),
            scale: Vec::new(),
            rotation: Vec::new(),
            translation: Vec::new(),
            env_map,
        })
    }

    pub fn update(&mut self, input: &Input, delta: f32) {
        self.camera.update(input, delta);
    }
}

fn extract_texture_transform(extensions: Option<&serde_json::Map<String, Value>>) -> Vec4 {
    if let Some(ext) = extensions
        && let Some(tx_ext) = ext.get("KHR_texture_transform")
    {
        let obj = tx_ext.as_object().unwrap();
        let offset = obj
            .get("offset")
            .map(|json_arr| {
                let arr = json_arr.as_array().unwrap();

                [arr[0].as_f64().unwrap(), arr[1].as_f64().unwrap()]
            })
            .unwrap_or([0.0, 0.0]);

        let scale = obj
            .get("scale")
            .map(|json_arr| {
                let arr = json_arr.as_array().unwrap();

                [arr[0].as_f64().unwrap(), arr[1].as_f64().unwrap()]
            })
            .unwrap_or([1.0, 1.0]);

        return Vec4::new(scale[0] as f32, scale[1] as f32, offset[0] as f32, offset[1] as f32);
    }

    Vec4::new(1.0, 1.0, 0.0, 0.0)
}

fn build_material(gltf_mat: &gltf::Material) -> Material {
    let mut base_color_factor = Vec4::from_array(gltf_mat.pbr_metallic_roughness().base_color_factor());
    let metallic_factor = gltf_mat.pbr_metallic_roughness().metallic_factor();
    let roughness_factor = gltf_mat.pbr_metallic_roughness().roughness_factor();
    let normal_scale = gltf_mat.normal_texture().map(|n| n.scale()).unwrap_or(1.0);
    let occlusion_strength = gltf_mat.occlusion_texture().map(|info| info.strength()).unwrap_or(1.0);
    let emissive_factor = Vec3::from_array(gltf_mat.emissive_factor());
    let alpha_cutoff = gltf_mat.alpha_cutoff().unwrap_or(0.5);
    let ior = gltf_mat.ior().unwrap_or(1.5);

    let default_uv_transform = Vec4::new(1.0, 1.0, 0.0, 0.0);

    let (albedo_texture_index, albedo_texture_uv, albedo_texture_transform) = {
        if let Some(m) = gltf_mat.pbr_metallic_roughness().base_color_texture() {
            let uv_tx = extract_texture_transform(m.extensions());
            (m.texture().index(), m.tex_coord(), uv_tx)
        } else if let Some(spec) = gltf_mat.pbr_specular_glossiness() {
            spec.diffuse_texture()
                .map(|m| {
                    let uv_tx = extract_texture_transform(m.extensions());
                    (m.texture().index(), m.tex_coord(), uv_tx)
                })
                .unwrap_or((!0, 0, default_uv_transform))
        } else {
            (!0, 0, default_uv_transform)
        }
    };

    let (normal_texture_index, normal_texture_uv, normal_texture_transform) = gltf_mat
        .normal_texture()
        .map(|m| {
            let uv_tx = extract_texture_transform(m.extensions());
            (m.texture().index(), m.tex_coord(), uv_tx)
        })
        .unwrap_or((!0, 0, default_uv_transform));

    let (emissive_texture_index, emissive_texture_uv, emissive_texture_transform) = gltf_mat
        .emissive_texture()
        .map(|info| {
            let uv_tx = extract_texture_transform(info.extensions());
            (info.texture().index(), info.tex_coord(), uv_tx)
        })
        .unwrap_or((!0, 0, default_uv_transform));

    let (occlusion_texture_index, occlusion_texture_uv, occlusion_texture_transform) = gltf_mat
        .occlusion_texture()
        .map(|info| {
            let uv_tx = extract_texture_transform(info.extensions());
            (info.texture().index(), info.tex_coord(), uv_tx)
        })
        .unwrap_or((!0, 0, default_uv_transform));

    let (
        mut metallic_roughness_texture_index,
        mut metallic_roughness_texture_uv,
        mut metallic_roughness_texture_transform,
    ) = gltf_mat
        .pbr_metallic_roughness()
        .metallic_roughness_texture()
        .map(|info| {
            let uv_tx = extract_texture_transform(info.extensions());
            (info.texture().index(), info.tex_coord(), uv_tx)
        })
        .unwrap_or((!0, 0, default_uv_transform));

    let mut material_type = MaterialType::METAL_ROUGHNESS;
    let mut spec_gloss_factor = Vec4::ZERO;
    if metallic_roughness_texture_index == !0 {
        (
            metallic_roughness_texture_index,
            metallic_roughness_texture_uv,
            metallic_roughness_texture_transform,
        ) = gltf_mat
            .pbr_specular_glossiness()
            .map(|spec| {
                material_type = MaterialType::SPEC_GLOSS;
                let sf = spec.specular_factor();
                let gf = spec.glossiness_factor();
                spec_gloss_factor = vec4(sf[0], sf[1], sf[2], gf);
                base_color_factor = Vec4::from_array(spec.diffuse_factor());
                spec.specular_glossiness_texture()
                    .map(|info| {
                        let uv_tx = extract_texture_transform(info.extensions());
                        (info.texture().index(), info.tex_coord(), uv_tx)
                    })
                    .unwrap_or((!0, 0, default_uv_transform))
            })
            .unwrap_or((!0, 0, default_uv_transform));
    }

    let mut specular_color_texture: u32 = !0;
    let mut specular_color_texture_uv: u32 = 0;
    let mut specular_texture: u32 = !0;
    let mut specular_texture_uv: u32 = 0;
    let mut specular_factor = Vec4::ONE;

    if let Some(spec) = gltf_mat.specular() {
        specular_factor.x = spec.specular_color_factor()[0];
        specular_factor.y = spec.specular_color_factor()[1];
        specular_factor.z = spec.specular_color_factor()[2];
        specular_factor.w = spec.specular_factor();
        specular_color_texture = spec
            .specular_color_texture()
            .map(|i| i.texture().index() as u32)
            .unwrap_or(!0);
        specular_color_texture_uv = spec.specular_color_texture().map(|i| i.tex_coord()).unwrap_or(0);
        specular_texture = spec
            .specular_texture()
            .map(|i| i.texture().index() as u32)
            .unwrap_or(!0);

        specular_texture_uv = spec.specular_texture().map(|i| i.tex_coord()).unwrap_or(!0);

        material_type |= MaterialType::SPEC;
    }

    let mut clearcoat_factor = 1.0;
    let mut clearcoat_roughness_factor = 0.0;
    let mut clearcoat_texture = !0;
    // TODO: extract from gltf file
    let clearcoat_texture_uv = 0;

    if let Some(clearcoat) = gltf_mat.extension_value("KHR_materials_clearcoat") {
        let clearcoat_obj = clearcoat
            .as_object()
            .context("expected an object for KHR_materials_clearcoat")
            .unwrap();

        clearcoat_factor = clearcoat_obj
            .get("clearcoatFactor")
            .context("Clearcout factor not found")
            .unwrap()
            .as_f64()
            .unwrap() as f32;

        clearcoat_roughness_factor = clearcoat_obj
            .get("clearcoatRoughnessFactor")
            .context("Clearcout factor not found")
            .unwrap()
            .as_f64()
            .unwrap() as f32;

        clearcoat_texture = clearcoat_obj
            .get("clearcoatNormalTexture")
            .context("Clearcout factor not found")
            .unwrap()
            .as_object()
            .unwrap()
            .get("index")
            .unwrap()
            .as_number()
            .unwrap()
            .as_u64()
            .unwrap() as u32;

        material_type |= MaterialType::CLEARCOAT;
    }

    let mut sheen_color_factor = Vec3::ONE;
    let mut sheen_roughness_factor = 1.0;

    if let Some(sheen) = gltf_mat.extension_value("KHR_materials_sheen") {
        let sheen_obj = sheen
            .as_object()
            .context("expected an object for KHR_materials_seen")
            .unwrap();

        let sheen_factor_arr = sheen_obj
            .get("sheenColorFactor")
            .context("sheen factor not found")
            .unwrap()
            .as_array()
            .unwrap();

        sheen_color_factor = vec3(
            sheen_factor_arr[0].as_f64().unwrap() as f32,
            sheen_factor_arr[1].as_f64().unwrap() as f32,
            sheen_factor_arr[2].as_f64().unwrap() as f32,
        );

        sheen_roughness_factor = sheen_obj
            .get("sheenRoughnessFactor")
            .context("sheen roughness factor not found")
            .unwrap()
            .as_f64()
            .unwrap() as f32;

        material_type |= MaterialType::SHEEN;
    }

    Material {
        albedo_texture_transform,
        albedo_texture: albedo_texture_index as u32,
        albedo_texture_uv,
        normal_texture_transform,
        normal_texture: normal_texture_index as u32,
        normal_texture_uv,
        metallic_roughness_texture_transform,
        metallic_roughness_texture: metallic_roughness_texture_index as u32,
        metallic_roughness_texture_uv,
        emissive_texture_transform,
        emissive_texture: emissive_texture_index as u32,
        emissive_texture_uv,
        occlusion_texture_transform,
        occlusion_texture: occlusion_texture_index as u32,
        occlusion_texture_uv,
        clearcoat_texture,
        clearcoat_texture_uv,
        sheen_color_roughness_factor: sheen_color_factor.extend(sheen_roughness_factor),
        base_color_factor,
        metallic_roughness_normal_occlusion: Vec4::new(
            metallic_factor,
            roughness_factor,
            normal_scale,
            occlusion_strength,
        ),
        emissive_factor_alpha_cutoff: Vec4::new(emissive_factor.x, emissive_factor.y, emissive_factor.z, alpha_cutoff),
        alpha_mode: match gltf_mat.alpha_mode() {
            gltf::material::AlphaMode::Opaque => AlphaMode::Opaque as u32,
            _ => AlphaMode::Mask as u32,
        },
        material_type: material_type.bits(),
        specular_glossiness_factor: spec_gloss_factor,
        clearcoat_factor: Vec4::new(clearcoat_factor, clearcoat_roughness_factor, 0.0, 0.0),
        ior,
        specular_factor,
        specular_texture,
        specular_texture_uv,
        specular_color_texture,
        specular_color_texture_uv,
    }
}

fn build_scene_mesh_draws(
    primitive_mesh_index_map: &HashMap<usize, usize>,
    scene: &mut Scene,
    node: gltf::Node<'_>,
    mut transform: Mat4,
    parent: i32,
    level: i32,
) {
    let t = node.transform();
    let (translation, rotation, scale) = t.decomposed();
    let scale = Vec3::from_array(scale);
    let rotation = Quat::from_array(rotation);
    let translation = Vec3::from_array(translation);

    let local_transform = Mat4::from_scale_rotation_translation(scale, rotation, translation);

    scene.scale.push(scale);
    scene.rotation.push(rotation);
    scene.translation.push(translation);

    transform *= local_transform;

    let model_index = scene.global_transforms.len() as u32;
    scene.global_transforms.push(transform);
    scene.heirarchy.push(Hierarchy {
        parent,
        first_child: -1,
        next_sibling: -1,
        last_sibling: -1,
        level,
    });
    let current_node_index = scene.heirarchy.len() as i32 - 1;

    let mut node_name = node.name().unwrap_or("").to_owned();
    if node_name.is_empty() {
        node_name = format!("Node {current_node_index}");
    }
    scene.node_names.push(node_name);
    if parent > -1 {
        let first_child_index = scene.heirarchy[parent as usize].first_child;
        if first_child_index == -1 {
            scene.heirarchy[parent as usize].first_child = current_node_index;
            scene.heirarchy[current_node_index as usize].last_sibling = current_node_index;
        } else {
            let mut sibling = scene.heirarchy[first_child_index as usize].last_sibling;
            if sibling == -1 {
                sibling = first_child_index;
                while scene.heirarchy[sibling as usize].next_sibling != -1 {
                    log::info!("Why do we need this loop??");
                    sibling = scene.heirarchy[sibling as usize].next_sibling;
                }
            }
            scene.heirarchy[sibling as usize].next_sibling = current_node_index;
            scene.heirarchy[first_child_index as usize].last_sibling = current_node_index;
        }
    }

    if let Some(mesh) = node.mesh() {
        for prim in mesh.primitives() {
            let index = mesh.index() * 16 + prim.index();
            let mesh_index = primitive_mesh_index_map.get(&index).unwrap();

            scene.mesh_draws.push(MeshDraw {
                model_index,
                alpha_mode: match prim.material().alpha_mode() {
                    gltf::material::AlphaMode::Opaque => AlphaMode::Opaque as u32,
                    _ => AlphaMode::Mask as u32,
                },
                mesh_index: *mesh_index as u16,
                meshlet_visibility_offset: scene.meshlet_visibility_count,
                material_index: prim.material().index().unwrap_or(!0) as u16,
            });
            scene.meshlet_visibility_count += scene.geometry.meshes[*mesh_index].lods[0].meshlet_count;
        }
    }
    for node in node.children() {
        build_scene_mesh_draws(
            primitive_mesh_index_map,
            scene,
            node,
            transform,
            current_node_index,
            level + 1,
        )
    }
}
