use anyhow::Result;
use ash::vk;
use core::f32::{self, consts::PI};
use exr::prelude::*;
use glam::{U8Vec4, Vec3, Vec4, uvec2, vec3, vec4};

use crate::image_utils::write_cubemap_to_ktx_file;

pub(crate) fn face_coords_to_xyz(i: u32, j: u32, face_id: u32, face_size: u32) -> Vec3 {
    let a = 2.0 * i as f32 / face_size as f32;
    let b = 2.0 * j as f32 / face_size as f32;

    match face_id {
        0 => vec3(-1.0, a - 1.0, b - 1.0),
        1 => vec3(a - 1.0, -1.0, 1.0 - b),
        2 => vec3(1.0, a - 1.0, 1.0 - b),
        3 => vec3(1.0 - a, 1.0, 1.0 - b),
        4 => vec3(b - 1.0, a - 1.0, 1.0),
        5 => vec3(1.0 - b, a - 1.0, -1.0),
        _ => panic!("Invalid face ID"),
    }
}

pub(crate) fn convert_equirectangular_map_to_vertical_cross(data: &[Vec4], width: u32, height: u32) -> Vec<Vec4> {
    let face_size = width / 4;

    let w = face_size * 3;
    let h = face_size * 4;

    let mut res: Vec<Vec4> = vec![Vec4::ZERO; (w * h) as usize];

    let face_offsets = [
        uvec2(face_size, face_size * 3),
        uvec2(0, face_size),
        uvec2(face_size, face_size),
        uvec2(face_size * 2, face_size),
        uvec2(face_size, 0),
        uvec2(face_size, face_size * 2),
    ];

    let clamp_w = width - 1;
    let clamp_h = height - 1;

    for face in 0..6 {
        for i in 0..face_size {
            for j in 0..face_size {
                let p = face_coords_to_xyz(i, j, face, face_size);
                let r = f32::hypot(p.x, p.y);
                let theta = f32::atan2(p.y, p.x);
                let phi = f32::atan2(p.z, r);
                //	float point source coordinates
                let uf = 2.0 * face_size as f32 * (theta + PI) / PI;
                let vf = 2.0 * face_size as f32 * (PI / 2.0 - phi) / PI;
                // 4-samples for bilinear interpolation
                let u1 = u32::clamp(f32::floor(uf) as u32, 0, clamp_w);
                let v1 = u32::clamp(f32::floor(vf) as u32, 0, clamp_h);
                let u2 = u32::clamp(u1 + 1, 0, clamp_w);
                let v2 = u32::clamp(v1 + 1, 0, clamp_h);
                // fetch 4-samples
                let a = data[(u1 + v1 * width) as usize];
                let b = data[(u2 + v1 * width) as usize];
                let c = data[(u1 + v2 * width) as usize];
                let d = data[(u2 + v2 * width) as usize];
                // bilinear interpolation
                let s = uf - u1 as f32;
                let t = vf - v1 as f32;
                let color = a * (1.0 - s) * (1.0 - t) + b * s * (1.0 - t) + c * (1.0 - s) * t + d * s * t;

                let base = (i + face_offsets[face as usize].x + (j + face_offsets[face as usize].y) * w) as usize;

                res[base] = color;
            }
        }
    }

    res
}

pub(crate) fn convert_vertical_cross_to_cubemap_faces(data: &[Vec4], width: u32, height: u32) -> Vec<Vec4> {
    let face_width = width / 3;
    let face_height = height / 4;

    let mut res = vec![Vec4::ZERO; face_width as usize * face_height as usize * 6];

    /*
        ------
        | +Y |
     ----------------
     | -X | -Z | +X |
     ----------------
        | -Y |
        ------
        | +Z |
        ------
    */
    let mut dst = 0;
    for face in 0..6 {
        for j in 0..face_height {
            for i in 0..face_width {
                let (x, y) = match face {
                    0 => (2 * face_width + i, face_height + j),
                    1 => (i, face_height + j),
                    2 => (face_width + i, j),
                    3 => (face_width + i, 2 * face_height + j),
                    4 => (face_width + i, face_height + j),
                    _ => (2 * face_width - (i + 1), height - (j + 1)),
                };

                let base = (y * width + x) as usize;
                res[dst] = data[base];

                dst += 1;
            }
        }
    }

    res
}

pub(crate) fn convert_hdr_cubemap_to_ktx(input_file: &str, output_file: &str) -> Result<()> {
    let image = read_first_rgba_layer_from_file(
        input_file,
        |resolution, _| (vec![Vec4::ZERO; resolution.area()], resolution.width()),
        |(pixel_vector, width), position, (r, g, b, a): (f32, f32, f32, f32)| {
            pixel_vector[position.y() * *width + position.x()] = vec4(r, g, b, a);
        },
    )?;
    let face_size = image.layer_data.size.width() as u32 / 4;

    let cubemap_cross = convert_equirectangular_map_to_vertical_cross(
        &image.layer_data.channel_data.pixels.0,
        image.layer_data.size.width() as u32,
        image.layer_data.size.height() as u32,
    );

    drop(image);

    let w = face_size * 3;
    let h = face_size * 4;
    let mut faces = convert_vertical_cross_to_cubemap_faces(&cubemap_cross, w, h);
    drop(cubemap_cross);

    let mut faces: Vec<_> = faces
        .drain(..)
        .map(|p| {
            U8Vec4::new(
                (p.x * 255.0) as u8,
                (p.y * 255.0) as u8,
                (p.z * 255.0) as u8,
                (p.w * 255.0) as u8,
            )
        })
        .collect();

    unsafe {
        let face_size_sq = (face_size * face_size) as usize;
        let num_levels = u32::ilog2(face_size);
        write_cubemap_to_ktx_file(
            face_size,
            vk::Format::R8G8B8A8_UNORM,
            output_file,
            true,
            num_levels,
            |_face, get_data_dst| {
                let data: Vec<_> = faces.drain(..face_size_sq).flat_map(|p| [p.x, p.y, p.z, p.w]).collect();

                let mut image = image::RgbaImage::from_raw(face_size, face_size, data).unwrap();

                let mut width = face_size;
                for level in 0..num_levels {
                    std::ptr::copy_nonoverlapping::<U8Vec4>(
                        image.as_ptr().cast(),
                        get_data_dst(level).cast(),
                        (width * width) as usize,
                    );

                    width = u32::max(width >> 1, 1);
                    image = image::imageops::resize(&image, width, width, image::imageops::FilterType::Lanczos3);
                }
            },
        )?;
    };

    Ok(())
}
