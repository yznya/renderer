use glam::{Vec4, Vec4Swizzles};

pub fn normalize_plane(plane: Vec4) -> Vec4 {
    plane / plane.xyz().length()
}

pub fn get_mip_levels(mut width: u32, mut height: u32) -> u32 {
    let mut r: u32 = 1;
    while width > 1 || height > 1 {
        r += 1;
        width /= 2;
        height /= 2;
    }
    r
}

pub fn power_2_floor(x: u32) -> u32 {
    let mut r = 1;

    while r * 2 < x {
        r *= 2;
    }

    r
}
