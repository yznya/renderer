use ash::vk;
use glam::{IVec4, Vec4};

pub fn clear_color(color: Vec4) -> vk::ClearValue {
    vk::ClearValue {
        color: vk::ClearColorValue {
            float32: color.to_array(),
        },
    }
}

pub fn clear_color_int(color: IVec4) -> vk::ClearValue {
    vk::ClearValue {
        color: vk::ClearColorValue {
            int32: color.to_array(),
        },
    }
}

pub fn clear_depth(depth: f32, stencil: u32) -> vk::ClearValue {
    vk::ClearValue {
        depth_stencil: vk::ClearDepthStencilValue { depth, stencil },
    }
}
