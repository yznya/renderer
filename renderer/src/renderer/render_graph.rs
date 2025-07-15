use super::{GpuResources, pass::Pass};
use anyhow::Result;
use egui::{RichText, Ui};
use egui_plot::{Line, Plot, PlotPoints};
use lava::{
    device::LDeviceRef,
    gpu_stats::{GPU_TIME_GRAPH_PERIOD, GpuStats},
    resources::LCommandBuffer,
};
use scenery::scene::Scene;

pub(crate) struct RenderPass {
    pub(crate) pass: Box<dyn Pass>,
    pub(crate) enabled: bool,
    pub(crate) index: usize,
}

#[derive(Default)]
pub(crate) struct RenderGraph {
    pub(crate) passes: Vec<RenderPass>,
}

impl RenderGraph {
    pub(crate) fn new() -> Self {
        Self { passes: Vec::new() }
    }

    pub(crate) fn add_pass(&mut self, pass: impl Pass + 'static) {
        let index = self.passes.len();
        self.passes.push(RenderPass {
            pass: Box::new(pass),
            enabled: true,
            index,
        });
    }

    pub(crate) fn render(
        &self,
        device: LDeviceRef,
        gpu_resources: &GpuResources,
        gpu_stats: &mut GpuStats,
        cmd: &LCommandBuffer,
        image_index: u32,
        scene: &Scene,
        frame_index: usize,
    ) -> Result<()> {
        for pass in &self.passes {
            if pass.enabled {
                let span = gpu_stats.begin_span(cmd, pass.pass.name(), pass.index, frame_index)?;
                let win_res = gpu_resources.window_resources.as_ref().unwrap();
                let scene_resources = gpu_resources.scene_resources.as_ref().unwrap();

                unsafe {
                    pass.pass.record_cmd_buffer(
                        device,
                        cmd,
                        image_index,
                        frame_index,
                        win_res,
                        scene_resources,
                        gpu_resources,
                        scene,
                    )
                }?;
                gpu_stats.end_span(cmd, span, frame_index);
            }
        }

        Ok(())
    }

    pub(crate) fn display_ui(&mut self, ui: &mut Ui, gpu_stats: &GpuStats) {
        ui.horizontal(|ui| {
            ui.label(RichText::new("GPU time: ").strong());
            ui.label(format!("{:.3}ms", gpu_stats.total_gpu_time * 1e-6));
        });

        Plot::new("my_plot")
            .view_aspect(2.0)
            .auto_bounds([true, false])
            .default_y_bounds(0.0, (gpu_stats.total_gpu_time * 1e-6).ceil())
            .default_x_bounds(0.1, 4.9)
            .show(ui, |plot_ui| {
                plot_ui.line(build_plot_line(
                    &gpu_stats.total_gpu_time_history,
                    gpu_stats.gpu_time_history_begin,
                ))
            });

        for pass in &mut self.passes {
            let time = gpu_stats.results[pass.index];
            ui.horizontal(|ui| {
                ui.horizontal(|ui| ui.checkbox(&mut pass.enabled, pass.pass.name()));
                ui.label(format!("{:.3}ms", time * 1e-6));
            });

            Plot::new(pass.pass.name())
                .view_aspect(2.0)
                .auto_bounds([true, false])
                .default_y_bounds(0.0, (time * 2e-4).ceil() / 100.0)
                .default_x_bounds(0.1, 4.9)
                .show(ui, |plot_ui| {
                    plot_ui.line(build_plot_line(
                        &gpu_stats.gpu_time_history[pass.index],
                        gpu_stats.gpu_time_history_begin,
                    ))
                });
        }
    }
}

fn build_plot_line(graph: &[f64], start: usize) -> Line<'_> {
    let gpu_time_plot: PlotPoints = (0..graph.len())
        .map(|i| {
            let x = i as f64 * GPU_TIME_GRAPH_PERIOD;
            [x, graph[(start + i) % graph.len()] * 1e-6]
        })
        .collect();

    Line::new("time", gpu_time_plot)
}
