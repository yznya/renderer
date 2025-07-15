use anyhow::Result;
use egui::ScrollArea;
use egui::{DragValue, RichText, SidePanel, panel::Side};
use egui_ltreeview::{NodeBuilder, TreeView, TreeViewBuilder};
use glam::{Mat4, Vec2};
use lava::device::LDeviceRef;
use scenery::RendererConfig;
use scenery::scene::Input;
use scenery::scene::Scene;
use std::time::Instant;
use winit::event::{ElementState, MouseButton};
use winit::{
    application::ApplicationHandler,
    dpi::PhysicalSize,
    event::{DeviceEvent, WindowEvent},
    event_loop::ActiveEventLoop,
    keyboard::{KeyCode, PhysicalKey},
    raw_window_handle::HasDisplayHandle,
    window::{CursorGrabMode, Window, WindowId},
};

use renderer::Renderer;

use crate::egui_widgets::{quat_drag, stats_text, vec3_drag};

pub const MAX_SCENE_DEPTH: usize = 10;

pub struct App<'d> {
    pub device: LDeviceRef<'d>,
    pub window: Option<Window>,
    pub renderer: Option<Renderer<'d>>,
    pub config: RendererConfig,
    pub scene: Scene,
    pub start: Instant,
    pub input: Input,
    pub cpu_time: f64,
    pub minimized: bool,
    pub delta_time: f32,
    pub camera_speed: f32,
    pub show_debug_menu: bool,
    pub should_grab_mouse: bool,
    pub selected_object: Option<u32>,
    pub changed_this_frame: [Vec<u32>; MAX_SCENE_DEPTH],
}

impl App<'_> {
    pub fn render(&mut self) -> Result<()> {
        let _span = tracy_client::span!();
        if self.renderer.is_none() {
            return Ok(());
        }
        let Some(renderer) = self.renderer.as_mut() else {
            return Ok(());
        };

        let frame_start = Instant::now();

        renderer.begin_frame(self.window.as_ref().unwrap());
        if self.show_debug_menu {
            self.display_ui();
        }

        self.scene.update(&self.input, self.delta_time);
        self.recalculate_transform_matrix();
        self.input.mouse = Vec2::ZERO;

        let renderer = self.renderer.as_mut().unwrap();
        let window = self.window.as_mut().unwrap();

        renderer.render(window, self.start, &mut self.scene, &mut self.changed_this_frame)?;
        let frame_end = Instant::now();

        self.cpu_time = self.cpu_time * 0.99 + (frame_end - frame_start).as_secs_f64() * 0.01;
        self.delta_time = (frame_end - frame_start).as_secs_f32();
        Ok(())
    }

    fn display_ui(&mut self) {
        let renderer = self.renderer.as_mut().unwrap();

        let _egui_span = tracy_client::span!("EGUI");

        SidePanel::new(Side::Left, "main menu")
            .default_width(600.0)
            .min_width(200.0)
            .max_width(600.0)
            .resizable(true)
            .show(&renderer.egui(), |ui| {
                ScrollArea::new([false, true]).show(ui, |ui| {
                    ui.collapsing("Stats", |ui| {
                        stats_text(ui, "CPU:", &format!("{:.3}ms", self.cpu_time * 1e3));
                        stats_text(ui, "Frame/S:", &format!("{:.3}", 1.0 / self.cpu_time));
                        stats_text(
                            ui,
                            "Triangles:",
                            &format!("{:.3}M", renderer.gpu_stats.triangles * 1e-6),
                        );
                        ui.checkbox(&mut renderer.config.lod_enabled, RichText::new("Lod").strong());
                        ui.checkbox(&mut renderer.config.cull_enabled, RichText::new("Cull").strong());
                    });

                    ui.collapsing("Camera", |ui| {
                        vec3_drag(ui, "Camera Position: ", &mut self.scene.camera.position);
                        vec3_drag(ui, "Camera Direction: ", &mut self.scene.camera.direction);

                        ui.horizontal(|ui| {
                            ui.label(RichText::new("Camera Speed: ").strong());
                            ui.add_sized([40.0, 20.0], DragValue::new(&mut self.camera_speed).speed(0.01));
                        });
                    });

                    ui.collapsing("GPU:", |ui| {
                        renderer.display_ui(ui);
                    });

                    ui.collapsing("Scene:", |ui| {
                        let id = ui.make_persistent_id("some name");
                        let (_response, actions) = TreeView::new(id).show(ui, |builder| {
                            for node in 0..self.scene.heirarchy.len() {
                                if self.scene.heirarchy[node].parent == -1 {
                                    draw_scene(&self.scene, node, builder);
                                }
                            }
                        });
                        for action in &actions {
                            if let egui_ltreeview::Action::SetSelected(items) = action {
                                self.selected_object = items.first().map(|&id| id as u32);
                            }
                        }
                    });
                })
            });

        SidePanel::new(Side::Right, "Right Panel")
            .min_width(200.0)
            .max_width(400.0)
            .resizable(true)
            .show(&renderer.egui(), |ui| {
                let Some(current_node) = self.selected_object else {
                    return;
                };

                let node_scale = &mut self.scene.scale[current_node as usize];
                let node_rotation = &mut self.scene.rotation[current_node as usize];
                let node_translation = &mut self.scene.translation[current_node as usize];
                let mut changed = false;

                changed |= vec3_drag(ui, "Scale: ", node_scale);
                changed |= quat_drag(ui, "Rotation: ", node_rotation);
                changed |= vec3_drag(ui, "Translation: ", node_translation);

                if changed {
                    self.mark_node_changed(current_node as usize);
                }
            });
    }

    pub fn mark_node_changed(&mut self, node_index: usize) {
        let h = &self.scene.heirarchy[node_index];
        self.changed_this_frame[h.level as usize].push(node_index as u32);

        let mut child = h.first_child;
        while child != -1 {
            self.mark_node_changed(child as usize);
            child = self.scene.heirarchy[child as usize].next_sibling;
        }
    }

    pub fn recalculate_transform_matrix(&mut self) {
        for &n in &self.changed_this_frame[0] {
            let local_transform = Mat4::from_scale_rotation_translation(
                self.scene.scale[n as usize],
                self.scene.rotation[n as usize],
                self.scene.translation[n as usize],
            );
            self.scene.global_transforms[n as usize] = local_transform;
        }

        for changes in &mut self.changed_this_frame[1..] {
            for n in changes.iter() {
                let local_transform = Mat4::from_scale_rotation_translation(
                    self.scene.scale[*n as usize],
                    self.scene.rotation[*n as usize],
                    self.scene.translation[*n as usize],
                );
                let h = &self.scene.heirarchy[*n as usize];
                self.scene.global_transforms[*n as usize] =
                    self.scene.global_transforms[h.parent as usize] * local_transform;
            }
        }
    }
}

fn draw_scene(scene: &Scene, current_node: usize, ui: &mut TreeViewBuilder<'_, usize>) {
    let node_name = scene.node_names[current_node].as_str();

    let node = &scene.heirarchy[current_node];
    if node.first_child == -1 {
        ui.leaf(current_node, node_name);
    } else {
        ui.node(
            NodeBuilder::dir(current_node)
                .default_open(false)
                .activatable(true)
                .label(node_name),
        );
        let mut child = node.first_child;
        while child != -1 {
            draw_scene(scene, child as usize, ui);
            child = scene.heirarchy[child as usize].next_sibling;
        }
        ui.close_dir();
    }
}

impl ApplicationHandler for App<'_> {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let _span = tracy_client::span!("App resumed");
        let mut attributes = Window::default_attributes().with_inner_size(PhysicalSize::new(1920, 1080));

        attributes = if self.config.fullscreen {
            let monitor = if self.config.secondary_monitor {
                // TODO: this breaks on linux
                event_loop.available_monitors().nth(0)
            } else {
                event_loop.primary_monitor()
            };
            let size = monitor.as_ref().unwrap().size();
            attributes
                .with_fullscreen(Some(winit::window::Fullscreen::Borderless(monitor)))
                .with_inner_size(size)
        } else {
            attributes
        };

        let window = event_loop.create_window(attributes).unwrap();
        self.device
            .set_surface(&window.display_handle().unwrap(), &window)
            .unwrap();

        let mut renderer =
            Renderer::new(self.device, &window, self.config.clone()).expect("Failed to initialize renderer");

        renderer.set_scene(&self.scene).expect("Failed to set scene");

        self.window = Some(window);
        self.renderer = Some(renderer);
    }

    fn device_event(&mut self, _event_loop: &ActiveEventLoop, _device_id: winit::event::DeviceId, event: DeviceEvent) {
        if let DeviceEvent::MouseMotion { delta } = event {
            let (x, y) = delta;
            let x = -(x as f32 * 0.03).to_radians();
            let y = (y as f32 * 0.03).to_radians();
            if self.should_grab_mouse {
                self.input.mouse += Vec2::new(x, y);
            }
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _: WindowId, event: WindowEvent) {
        let Some(ref window) = self.window else { return };
        let Some(ref mut renderer) = self.renderer else { return };

        renderer.handle_events(window, &event);

        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            WindowEvent::RedrawRequested => {
                self.window.as_ref().unwrap().request_redraw();
            }
            WindowEvent::Resized(size) => {
                self.minimized = size.width == 0 || size.height == 0;
            }
            WindowEvent::MouseInput {
                device_id: _,
                state,
                button,
            } => {
                if button == MouseButton::Right && state == ElementState::Pressed {
                    self.should_grab_mouse = true;
                    window.set_cursor_grab(CursorGrabMode::Confined).unwrap();
                }

                if button == MouseButton::Right && state == ElementState::Released {
                    self.should_grab_mouse = false;
                    window.set_cursor_grab(CursorGrabMode::None).unwrap();
                }
            }
            WindowEvent::KeyboardInput {
                device_id: _,
                event,
                is_synthetic: _,
            } => {
                if event.physical_key == KeyCode::Backquote && event.state.is_pressed() {
                    self.show_debug_menu = !self.show_debug_menu;
                }

                let change = if event.state.is_pressed() { 1.0 } else { 0.0 };

                if !self.show_debug_menu
                    && let PhysicalKey::Code(code) = event.physical_key
                {
                    match code {
                        KeyCode::KeyW => {
                            self.input.forward = -self.camera_speed * change;
                        }
                        KeyCode::KeyS => {
                            self.input.forward = self.camera_speed * change;
                        }
                        KeyCode::KeyA => {
                            self.input.strafe = self.camera_speed * change;
                        }
                        KeyCode::KeyD => {
                            self.input.strafe = -self.camera_speed * change;
                        }
                        KeyCode::KeyQ => {
                            self.input.fly = -self.camera_speed * change;
                        }
                        KeyCode::KeyE => {
                            self.input.fly = self.camera_speed * change;
                        }
                        _ => {}
                    }
                }
            }
            _ => (),
        }
    }
}
