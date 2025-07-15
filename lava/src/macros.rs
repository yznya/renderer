#[macro_export]
macro_rules! with {
    ($t:ty => {$($i:ident)|+}) => {
        $(<$t>::$i)|+
    };
}

#[macro_export]
macro_rules! handle_empty_enum {
    ($t:ty, empty) => {
        <$t>::empty()
    };
    ($t:ty, $($i:ident)|+) => {
        $crate::with!($t => {$($i)|+})
    };
}

#[macro_export]
macro_rules! image_barrier {
    (
        image: $image:expr,
        access: $($src_access:tt)|+ => $($dst_access:ident)|+,
        layout: $src_layout:ident => $dst_layout:ident,
        stage: $($src_stage_mask:ident)|+ => $($dst_stage_mask:ident)|+,
        aspect: $aspect_mask:ident
    ) => {
        vk::ImageMemoryBarrier2::default()
            .image($image)
            .src_access_mask($crate::handle_empty_enum!(vk::AccessFlags2, $($src_access)|+))
            .dst_access_mask($crate::handle_empty_enum!(vk::AccessFlags2, $($dst_access)|+))
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .old_layout($crate::handle_empty_enum!(vk::ImageLayout, $src_layout))
            .new_layout($crate::handle_empty_enum!(vk::ImageLayout, $dst_layout))
            .src_stage_mask($crate::handle_empty_enum!(vk::PipelineStageFlags2, $($src_stage_mask)|+))
            .dst_stage_mask($crate::handle_empty_enum!(vk::PipelineStageFlags2, $($dst_stage_mask)|+))
            .subresource_range(
                vk::ImageSubresourceRange::default()
                    .aspect_mask(vk::ImageAspectFlags::$aspect_mask)
                    .level_count(vk::REMAINING_MIP_LEVELS)
                    .layer_count(vk::REMAINING_ARRAY_LAYERS),
            )
    };
}
