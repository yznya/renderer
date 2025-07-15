#![allow(clippy::too_many_arguments)]
use anyhow::{Context, Result};
use lava::{device::LDevice, logger::init_logger};
use scenery::scene::Scene;
use scenery::{RendererConfig, scene::EnvMap};
use std::{
    fs::File,
    time::{Duration, Instant},
};
#[cfg(feature = "profile_allocations")]
use tracy_client::ProfiledAllocator;
use winit::{
    event_loop::{ControlFlow, EventLoop},
    platform::pump_events::{EventLoopExtPumpEvents, PumpStatus},
};

use potato::app::App;

#[cfg(feature = "profile_allocations")]
#[global_allocator]
static GLOBAL: ProfiledAllocator<std::alloc::System> = ProfiledAllocator::new(std::alloc::System, 100);

fn main() -> Result<()> {
    init_logger();

    let args = std::env::args().collect::<Vec<String>>();
    if args.len() < 4 {
        println!("Usage: potato <config> <scene>");
        return Ok(());
    }
    let config = serde_json::from_reader::<_, RendererConfig>(File::open(&args[1])?)?;
    let env_map = serde_json::from_reader::<_, EnvMap>(File::open(&args[2])?)?;

    let ext = args[3].rsplit_once(".");
    let scene = match ext.context("File has no extension")?.1 {
        "gltf" => Scene::load_gltf_file(&args[3], env_map)?,
        "ptto" => Scene::load_ptto_file(&args[3])?,
        _ => Scene::load_obj_file(&args[3], env_map)?,
    };

    let device = Box::new(LDevice::create_without_surface()?);

    // TODO: add constructor
    let mut app = App {
        device: device.device_ref(),
        renderer: None,
        window: None,
        start: Instant::now(),
        cpu_time: 0.0,
        config,
        scene,
        minimized: false,
        input: Default::default(),
        delta_time: 1.0,
        camera_speed: 1.0,
        show_debug_menu: false,
        should_grab_mouse: false,
        selected_object: None,
        changed_this_frame: Default::default(),
    };

    let mut event_loop = EventLoop::new()?;
    event_loop.set_control_flow(ControlFlow::Poll);

    loop {
        let status = event_loop.pump_app_events(Some(Duration::ZERO), &mut app);
        if let PumpStatus::Exit(_) = status {
            break;
        }

        app.render()?;
    }

    Ok(())
}
