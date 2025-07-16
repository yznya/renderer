use std::{fmt::Debug, fs::File, path::Path};

use anyhow::{Context, Result, anyhow};
use ash::vk;
use ddsfile::{Dds, DxgiFormat};
use lava::device::LDeviceRef;
use lava::image_barrier;
use lava::resources::{LImageWithView, LMappedBuffer};
use png::Transformations;
use zune_jpeg::zune_core::options::DecoderOptions;

pub struct PixelData {
    pub data: Vec<u8>,
    pub width: u32,
    pub height: u32,
    pub mip_levels: u32,
    pub format: vk::Format,
    pub regions: Vec<vk::BufferImageCopy>,
}

impl PixelData {
    fn from_jpeg<P: AsRef<Path> + Debug>(path: P) -> Result<Self> {
        let data = std::fs::read(&path)?;
        let mut decoder = zune_jpeg::JpegDecoder::new(&data);
        decoder.set_options(
            DecoderOptions::default().jpeg_set_out_colorspace(zune_jpeg::zune_core::colorspace::ColorSpace::RGBA),
        );
        let pixels = decoder.decode()?;
        let (width, height) = decoder.dimensions().unwrap();

        let regions = vec![
            vk::BufferImageCopy::default()
                .image_subresource(vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level: 0,
                    base_array_layer: 0,
                    layer_count: 1,
                })
                .image_offset(vk::Offset3D { x: 0, y: 0, z: 0 })
                .image_extent(vk::Extent3D {
                    width: width as u32,
                    height: height as u32,
                    depth: 1,
                }),
        ];

        let format = if path.as_ref().to_str().unwrap().contains("basecolor") {
            vk::Format::R8G8B8A8_SRGB
        } else {
            vk::Format::R8G8B8A8_UNORM
        };

        Ok(Self {
            data: pixels,
            width: width as u32,
            height: height as u32,
            mip_levels: 1,
            format,
            regions,
        })
    }

    fn from_dds<P: AsRef<Path> + Debug>(path: P) -> Result<Self> {
        let file = File::open(&path)?;
        let dds_file = Dds::read(file)?;
        let dxgi_format = match dds_file.get_dxgi_format() {
            Some(f) => f,
            None => return Err(anyhow!("DDS file doesn't have a format")),
        };

        let format = match dxgi_format {
            DxgiFormat::BC1_UNorm | DxgiFormat::BC1_UNorm_sRGB => vk::Format::BC1_RGBA_UNORM_BLOCK,
            DxgiFormat::BC2_UNorm | DxgiFormat::BC2_UNorm_sRGB => vk::Format::BC2_UNORM_BLOCK,
            DxgiFormat::BC3_UNorm | DxgiFormat::BC3_UNorm_sRGB => vk::Format::BC3_UNORM_BLOCK,
            DxgiFormat::BC4_UNorm => vk::Format::BC4_UNORM_BLOCK,
            DxgiFormat::BC4_SNorm => vk::Format::BC4_SNORM_BLOCK,
            DxgiFormat::BC5_UNorm => vk::Format::BC4_UNORM_BLOCK,
            DxgiFormat::BC5_SNorm => vk::Format::BC4_SNORM_BLOCK,
            DxgiFormat::BC6H_UF16 => vk::Format::BC6H_UFLOAT_BLOCK,
            DxgiFormat::BC6H_SF16 => vk::Format::BC6H_SFLOAT_BLOCK,
            DxgiFormat::BC7_UNorm => vk::Format::BC7_UNORM_BLOCK,
            DxgiFormat::BC7_UNorm_sRGB => vk::Format::BC7_SRGB_BLOCK,
            _ => {
                return Err(anyhow!("Format is not supported"));
            }
        };

        let block_size = match format {
            vk::Format::BC1_RGBA_UNORM_BLOCK | vk::Format::BC4_SNORM_BLOCK | vk::Format::BC4_UNORM_BLOCK => 8,
            _ => 16,
        };

        let mut width = dds_file.get_width();
        let mut height = dds_file.get_height();
        let mut buffer_offset: u64 = 0;
        let regions: Vec<_> = (0..dds_file.get_num_mipmap_levels())
            .map(|mip_level| {
                let region = vk::BufferImageCopy::default()
                    .buffer_offset(buffer_offset)
                    .buffer_row_length(0)
                    .buffer_image_height(0)
                    .image_subresource(vk::ImageSubresourceLayers {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        mip_level,
                        base_array_layer: 0,
                        layer_count: 1,
                    })
                    .image_offset(vk::Offset3D { x: 0, y: 0, z: 0 })
                    .image_extent(vk::Extent3D {
                        width,
                        height,
                        depth: 1,
                    });

                buffer_offset += (width.div_ceil(4) * height.div_ceil(4) * block_size) as u64;

                width = u32::max(1, width / 2);
                height = u32::max(1, height / 2);
                region
            })
            .collect();

        let width = dds_file.get_width();
        let height = dds_file.get_height();
        let mip_levels = dds_file.get_num_mipmap_levels();

        Ok(Self {
            data: dds_file.data,
            width,
            height,
            mip_levels,
            format,
            regions,
        })
    }

    fn from_png<P: AsRef<Path> + Debug>(path: P) -> Result<Self> {
        let mut decoder = png::Decoder::new(File::open(&path)?);
        decoder.set_transformations(Transformations::ALPHA);
        let mut reader = decoder.read_info()?;
        let mut pixels = vec![0; reader.output_buffer_size()];

        let frame = reader.next_frame(&mut pixels)?;
        let width = frame.width;
        let height = frame.height;
        pixels.resize(frame.buffer_size(), 0);

        let regions = vec![
            vk::BufferImageCopy::default()
                .image_subresource(vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level: 0,
                    base_array_layer: 0,
                    layer_count: 1,
                })
                .image_offset(vk::Offset3D { x: 0, y: 0, z: 0 })
                .image_extent(vk::Extent3D {
                    width: width as u32,
                    height: height as u32,
                    depth: 1,
                }),
        ];

        let format = if path.as_ref().to_str().unwrap().contains("basecolor") {
            vk::Format::R8G8B8A8_SRGB
        } else {
            vk::Format::R8G8B8A8_UNORM
        };

        Ok(Self {
            data: pixels,
            width: width as u32,
            height: height as u32,
            mip_levels: 1,
            format,
            regions,
        })
    }

    fn from_ktx2<P: AsRef<Path> + Debug>(path: P) -> Result<Self> {
        let mut new_tex = std::ptr::null_mut();
        let cubemap_tex_data = std::fs::read(&path)?;
        let result = unsafe {
            ktxvulkan_sys::ktxTexture2_CreateFromMemory(
                cubemap_tex_data.as_ptr(),
                cubemap_tex_data.len(),
                // i32 on windows and u32 on linux :)
                ktxvulkan_sys::ktxTextureCreateFlagBits_KTX_TEXTURE_CREATE_LOAD_IMAGE_DATA_BIT as u32,
                &mut new_tex,
            )
        };
        assert_eq!(result, ktxvulkan_sys::ktx_error_code_e_KTX_SUCCESS);

        let texture_data = (unsafe { *new_tex }).pData;
        let texture_size = (unsafe { *new_tex }).dataSize;
        let format = vk::Format::from_raw((unsafe { *new_tex }).vkFormat as i32);
        let width = (unsafe { *new_tex }).baseWidth;
        let height = (unsafe { *new_tex }).baseHeight;
        let mip_levels = (unsafe { *new_tex }).numLevels;

        let pixels = unsafe { std::slice::from_raw_parts(texture_data, texture_size).to_vec() };

        let mut mip_width = width;
        let mut mip_height = height;
        let regions: Vec<_> = (0..mip_levels)
            .map(|mip_level| {
                let mut offset = 0;
                unsafe { ktxvulkan_sys::ktxTexture2_GetImageOffset(new_tex, mip_level, 0, 0, &mut offset) };
                let region = vk::BufferImageCopy::default()
                    .buffer_offset(offset as u64)
                    .buffer_row_length(0)
                    .buffer_image_height(0)
                    .image_subresource(vk::ImageSubresourceLayers {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        mip_level,
                        base_array_layer: 0,
                        layer_count: 1,
                    })
                    .image_offset(vk::Offset3D { x: 0, y: 0, z: 0 })
                    .image_extent(vk::Extent3D {
                        width: mip_width,
                        height: mip_height,
                        depth: 1,
                    });

                mip_width = u32::max(1, mip_width / 2);
                mip_height = u32::max(1, mip_height / 2);
                region
            })
            .collect();

        unsafe { ktxvulkan_sys::ktxTexture2_Destroy(new_tex) };

        Ok(Self {
            data: pixels,
            width: width as u32,
            height: height as u32,
            mip_levels,
            format,
            regions,
        })
    }
}

pub fn load_pixel_data<P: AsRef<Path> + Debug>(path: P) -> Result<PixelData> {
    let _span = tracy_client::span!("Loading pixels");
    let ext = path
        .as_ref()
        .extension()
        .context("File doesn't have an extenstion")?
        .to_str()
        .context("File name is not valid")?;

    match ext {
        "jpg" | "jpeg" => {
            let new_path = path.as_ref().with_extension("dds");
            PixelData::from_dds(&new_path).or_else(|_| PixelData::from_jpeg(&path))
        }
        "dds" => PixelData::from_dds(&path),
        "png" => {
            let new_path = path.as_ref().with_extension("dds");
            PixelData::from_dds(&new_path).or_else(|_| PixelData::from_png(&path))
        }
        "ktx2" => PixelData::from_ktx2(&path),
        _ => Err(anyhow!("Unsupported image format")),
    }
}

pub(crate) fn load_image<'d>(
    device: LDeviceRef<'d>,
    staging_buffer: &LMappedBuffer,
    pixel_data: &PixelData,
) -> Result<LImageWithView<'d>> {
    let _span = tracy_client::span!("Loading image");
    let image = unsafe {
        device.create_image_with_view_and_mips(
            pixel_data.width,
            pixel_data.height,
            pixel_data.format,
            vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST,
            pixel_data.mip_levels,
        )?
    };

    unsafe {
        std::ptr::copy_nonoverlapping(
            pixel_data.data.as_ptr(),
            staging_buffer.memory_map.cast(),
            pixel_data.data.len(),
        )
    };

    let cmd_buffer = unsafe { device.begin_single_time_command() }?;

    unsafe {
        cmd_buffer.image_barrier(&[image_barrier!(
            image: image.image.inner,
            access: empty => TRANSFER_WRITE,
            layout: UNDEFINED => TRANSFER_DST_OPTIMAL,
            stage: NONE => TRANSFER,
            aspect: COLOR
        )])
    };

    unsafe {
        cmd_buffer.copy_buffer_to_image(
            staging_buffer.buffer.inner,
            image.image.inner,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            &pixel_data.regions,
        )
    };

    unsafe {
        cmd_buffer.image_barrier(&[image_barrier!(
            image: image.image.inner,
            access: TRANSFER_WRITE => SHADER_SAMPLED_READ,
            layout: TRANSFER_DST_OPTIMAL => SHADER_READ_ONLY_OPTIMAL,
            stage: TRANSFER => ALL_GRAPHICS,
            aspect: COLOR
        )])
    };
    unsafe { device.end_single_time_command(cmd_buffer) }?;

    Ok(image)
}
