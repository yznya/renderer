use crate::device::LDeviceRef;
use ash::vk;

pub struct Program<'d> {
    pub descriptor_set_layout: vk::DescriptorSetLayout,
    pub descriptor_update_template: vk::DescriptorUpdateTemplate,
    pub pipeline_layout: vk::PipelineLayout,
    pub pipeline: vk::Pipeline,
    pub local_size_x: u32,
    pub local_size_y: u32,
    pub local_size_z: u32,
    pub device: LDeviceRef<'d>,
}

impl Drop for Program<'_> {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_pipeline(self.pipeline, None);
            self.device.destroy_pipeline_layout(self.pipeline_layout, None);
            self.device
                .destroy_descriptor_update_template(self.descriptor_update_template, None);
            self.device
                .destroy_descriptor_set_layout(self.descriptor_set_layout, None);
        };
    }
}
