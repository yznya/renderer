use anyhow::{Context, Result};
use ash::vk;
use glam::U8Vec4;
use renderer::renderer::image_loader::load_pixel_data;
use scenery::scene::{EnvMap, Scene};

use crate::image_utils::write_image_to_ktx_file;

pub(crate) fn convert_image_to_ktx(input_file: &str, output_file: &str) -> Result<()> {
    let image_data = load_pixel_data(input_file)?;
    let new_size = u32::max(image_data.width, image_data.height);
    let num_levels = u32::ilog2(new_size);
    let mut image = image::RgbaImage::from_raw(image_data.width, image_data.height, image_data.data)
        .context("Failed to read image")?;
    image = image::imageops::resize(&image, new_size, new_size, image::imageops::FilterType::Lanczos3);

    unsafe {
        write_image_to_ktx_file(
            image_data.width,
            image_data.height,
            vk::Format::R8G8B8A8_UNORM,
            output_file,
            true,
            num_levels,
            move |get_data_dst| {
                let mut size = new_size;
                for level in 0..num_levels {
                    std::ptr::copy_nonoverlapping::<U8Vec4>(
                        image.as_ptr().cast(),
                        get_data_dst(level).cast(),
                        (size * size) as usize,
                    );

                    size = u32::max(size >> 1, 1);
                    image = image::imageops::resize(&image, size, size, image::imageops::FilterType::Lanczos3);
                }
            },
        )?;
    };

    Ok(())
}

pub(crate) fn convert_to_ptto(model_file: &str, env_map: EnvMap, output_file: &str) -> Result<()> {
    let mut scene = Scene::load_gltf_file(model_file, env_map)?;

    for image in scene.images.iter_mut() {
        let (name, extension) = image.1.rsplit_once(".").context("Image doesn't have an extension")?;
        if let "jpg" | "jpeg" | "png" = extension {
            let new_name = format!("{}.{}", name, "ktx2");
            if convert_image_to_ktx(&image.1, &new_name).is_ok() {
                image.1 = new_name;
            } else {
                log::warn!("Failed to convert {}", image.1);
            }
        } else {
            continue;
        }
    }

    let data = rkyv::to_bytes::<rkyv::rancor::Error>(&scene)?;
    std::fs::write(output_file, data)?;

    Ok(())
}
