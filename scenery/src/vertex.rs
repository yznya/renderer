use glam::Vec4;
use std::hash::Hash;

#[repr(C)]
#[derive(Clone, Default, Debug, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
pub struct Vertex {
    pub position: Vec4,
    pub normals: Vec4,
    pub tex_coords: Vec4,
    pub tangents: Vec4,
}

impl PartialEq for Vertex {
    fn eq(&self, other: &Self) -> bool {
        self.position == other.position && self.normals == other.normals && self.tex_coords == other.tex_coords
    }
}

impl Eq for Vertex {}

impl Hash for Vertex {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.position.x.to_bits().hash(state);
        self.position.y.to_bits().hash(state);
        self.position.z.to_bits().hash(state);
        self.position.w.to_bits().hash(state);
        self.normals.x.to_bits().hash(state);
        self.normals.y.to_bits().hash(state);
        self.normals.z.to_bits().hash(state);
        self.normals.w.to_bits().hash(state);
        self.tangents.x.to_bits().hash(state);
        self.tangents.y.to_bits().hash(state);
        self.tangents.z.to_bits().hash(state);
        self.tangents.w.to_bits().hash(state);
        self.tex_coords.x.to_bits().hash(state);
        self.tex_coords.y.to_bits().hash(state);
        self.tex_coords.z.to_bits().hash(state);
        self.tex_coords.w.to_bits().hash(state);
    }
}

struct MikktspaceGeometry<'a> {
    vertices: &'a mut [Vertex],
    indices: &'a [u32],
}

impl mikktspace::Geometry for MikktspaceGeometry<'_> {
    fn num_faces(&self) -> usize {
        self.indices.len() / 3
    }

    fn num_vertices_of_face(&self, _face: usize) -> usize {
        3
    }

    fn position(&self, face: usize, vert: usize) -> [f32; 3] {
        let index = face * 3 + vert;

        let p = self.vertices[self.indices[index] as usize].position;

        [p.x, p.y, p.z]
    }

    fn normal(&self, face: usize, vert: usize) -> [f32; 3] {
        let index = face * 3 + vert;

        let normal = self.vertices[self.indices[index] as usize].normals;
        [normal.x, normal.y, normal.z]
    }

    fn tex_coord(&self, face: usize, vert: usize) -> [f32; 2] {
        let index = face * 3 + vert;
        let uv = self.vertices[self.indices[index] as usize].tex_coords;

        [uv.x, uv.y]
    }

    fn set_tangent_encoded(&mut self, tangent: [f32; 4], face: usize, vert: usize) {
        let index = face * 3 + vert;

        self.vertices[self.indices[index] as usize].tangents =
            Vec4::new(tangent[0], tangent[1], tangent[2], tangent[3]);
    }
}

pub fn generate_tangents(vertices: &mut [Vertex], indices: &[u32]) {
    let mut m = MikktspaceGeometry { vertices, indices };

    mikktspace::generate_tangents(&mut m);
}
