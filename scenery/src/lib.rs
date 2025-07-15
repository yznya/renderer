pub mod camera;
pub mod mesh;
pub mod scene;
pub mod vertex;

#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct RendererConfig {
    pub enable_vsync: bool,
    pub lod_enabled: bool,
    pub cull_enabled: bool,
    pub fast_mode: bool,
    pub fullscreen: bool,
    pub secondary_monitor: bool,
}
