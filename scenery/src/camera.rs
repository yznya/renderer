use glam::{Mat4, Vec3};

use crate::scene::Input;

#[derive(rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
pub struct Camera {
    pub position: Vec3,
    pub direction: Vec3,
}

impl Camera {
    pub fn view_mat(&self) -> Mat4 {
        Mat4::look_to_lh(self.position, -self.direction, Vec3::Y)
    }

    pub fn forward(&mut self, delta: f32) {
        self.position += self.direction * delta;
    }

    pub fn strafe(&mut self, delta: f32) {
        let right = self.direction.cross(Vec3::Y);
        self.position += right * delta;
    }

    pub fn fly(&mut self, delta: f32) {
        self.position.y += delta;
    }

    pub fn rotate(&mut self, x: f32, y: f32) {
        let right = self.direction.cross(Vec3::Y);
        let rotation = Mat4::from_axis_angle(Vec3::Y, x) * Mat4::from_axis_angle(right, y);
        self.direction = rotation.transform_vector3(self.direction);
    }

    pub fn update(&mut self, input: &Input, delta: f32) {
        self.strafe(input.strafe * delta);
        self.forward(input.forward * delta);
        self.fly(input.fly * delta);
        self.rotate(input.mouse.x, input.mouse.y);
    }
}
