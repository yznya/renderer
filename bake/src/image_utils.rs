use ash::vk;
use std::ffi::CString;

use anyhow::Result;

pub(crate) unsafe fn write_cubemap_to_ktx_file(
    face_size: u32,
    format: vk::Format,
    output_file: &str,
    transcode: bool,
    num_levels: u32,
    mut fill_bytes: impl FnMut(usize, &dyn Fn(u32) -> *mut u8),
) -> Result<()> {
    let info = ktxvulkan_sys::ktxTextureCreateInfo {
        glInternalformat: 0,
        vkFormat: format.as_raw() as u32,
        baseWidth: face_size,
        baseHeight: face_size,
        baseDepth: 1,
        numDimensions: 2,
        numLevels: num_levels,
        numLayers: 1,
        numFaces: 6,
        generateMipmaps: false,
        isArray: false,
        pDfd: std::ptr::null::<u32>() as *mut u32,
    };

    let mut new_tex = std::ptr::null_mut();
    unsafe {
        ktxvulkan_sys::ktxTexture2_Create(
            &info,
            ktxvulkan_sys::ktxTextureCreateStorageEnum_KTX_TEXTURE_CREATE_ALLOC_STORAGE,
            &mut new_tex,
        );

        for face in 0..6 {
            fill_bytes(face, &|level: u32| {
                let mut offset = 0;
                ktxvulkan_sys::ktxTexture2_GetImageOffset(new_tex, level, 0, face as u32, &mut offset);
                (*new_tex).pData.add(offset).cast()
            });
        }

        if transcode {
            transcode_ktx(new_tex);
        }

        (*(*new_tex).vtbl).WriteToNamedFile.unwrap()(new_tex.cast(), CString::new(output_file)?.as_ptr());

        (*(*new_tex).vtbl).Destroy.unwrap()(new_tex.cast());
    }

    Ok(())
}

unsafe fn transcode_ktx(new_tex: *mut ktxvulkan_sys::ktxTexture2) {
    unsafe {
        let res = ktxvulkan_sys::ktxTexture2_CompressBasis(new_tex, 0);
        assert_eq!(res, 0);
        let res =
            ktxvulkan_sys::ktxTexture2_TranscodeBasis(new_tex, ktxvulkan_sys::ktx_transcode_fmt_e_KTX_TTF_BC7_RGBA, 0);
        assert_eq!(res, 0);
    }
}

pub(crate) unsafe fn write_image_to_ktx_file(
    width: u32,
    height: u32,
    format: vk::Format,
    output_file: &str,
    transcode: bool,
    num_levels: u32,
    fill_bytes: impl FnOnce(&dyn Fn(u32) -> *mut u8),
) -> Result<()> {
    let info = ktxvulkan_sys::ktxTextureCreateInfo {
        glInternalformat: 0,
        vkFormat: format.as_raw() as u32,
        baseWidth: width,
        baseHeight: height,
        baseDepth: 1,
        numDimensions: 2,
        numLevels: num_levels,
        numLayers: 1,
        numFaces: 1,
        generateMipmaps: false,
        isArray: false,
        pDfd: std::ptr::null::<u32>() as *mut u32,
    };

    let mut new_tex = std::ptr::null_mut();
    unsafe {
        ktxvulkan_sys::ktxTexture2_Create(
            &info,
            ktxvulkan_sys::ktxTextureCreateStorageEnum_KTX_TEXTURE_CREATE_ALLOC_STORAGE,
            &mut new_tex,
        );

        fill_bytes(&|level: u32| {
            let mut offset = 0;
            ktxvulkan_sys::ktxTexture2_GetImageOffset(new_tex, level, 0, 0, &mut offset);
            (*new_tex).pData.add(offset).cast()
        });

        if transcode {
            transcode_ktx(new_tex);
        }

        (*(*new_tex).vtbl).WriteToNamedFile.unwrap()(new_tex.cast(), CString::new(output_file)?.as_ptr());

        (*(*new_tex).vtbl).Destroy.unwrap()(new_tex.cast());
    }

    Ok(())
}
