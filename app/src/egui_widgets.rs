use egui::RichText;
use glam::{Quat, Vec3};

pub fn vec3_drag(ui: &mut egui::Ui, label: &str, value: &mut Vec3) -> bool {
    let mut changed = false;
    ui.horizontal(|ui| {
        ui.label(label);
        changed |= ui
            .add_sized([40.0, 20.0], egui::DragValue::new(&mut value.x).speed(0.001))
            .changed();
        changed |= ui
            .add_sized([40.0, 20.0], egui::DragValue::new(&mut value.y).speed(0.001))
            .changed();
        changed |= ui
            .add_sized([40.0, 20.0], egui::DragValue::new(&mut value.z).speed(0.001))
            .changed();
    });

    changed
}

pub fn quat_drag(ui: &mut egui::Ui, label: &str, value: &mut Quat) -> bool {
    let mut changed = false;
    ui.horizontal(|ui| {
        ui.label(label);
        changed |= ui
            .add_sized([40.0, 20.0], egui::DragValue::new(&mut value.x).speed(0.001))
            .changed();
        changed |= ui
            .add_sized([40.0, 20.0], egui::DragValue::new(&mut value.y).speed(0.001))
            .changed();
        changed |= ui
            .add_sized([40.0, 20.0], egui::DragValue::new(&mut value.z).speed(0.001))
            .changed();
        changed |= ui
            .add_sized([40.0, 20.0], egui::DragValue::new(&mut value.w).speed(0.001))
            .changed();
    });

    changed
}

pub fn stats_text(ui: &mut egui::Ui, label: &str, value: &str) {
    ui.horizontal(|ui| {
        ui.label(RichText::new(label).strong());
        ui.label(value);
    });
}
