use std::{
    cmp::{max, min},
    time::{Duration, Instant},
};

use crate::{MAX_FRAMES_IN_FLIGHT, resources::LCommandBuffer};

use super::device::LDeviceRef;
use anyhow::{Result, anyhow};
use ash::vk;

pub const QUERY_POOL_SIZE: usize = u8::MAX as usize * 2;
pub const GPU_TIME_GRAPH_DURATION: f64 = 5.0;
pub const GPU_TIME_GRAPH_PERIOD: f64 = 0.1;

#[derive(Copy, Clone, PartialEq, Default, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
struct QueryResult {
    result: u64,
    status: u64,
}

struct GpuSpan {
    index: usize,
    inner: tracy_client::GpuSpan,
}

// TODO: this type is becoming too big, it's now responsible for quering gpu stats and keeping historical data
pub struct GpuStats<'d> {
    pub triangles: f64,
    // TODO: these sizes are known at compile time, we can use arrays
    timestamp_query_pool: Vec<vk::QueryPool>,
    pipeline_stats_query_pool: vk::QueryPool,
    spans: slab::Slab<GpuSpan>,
    gpu_ctx: tracy_client::GpuContext,
    // TODO: one data Vec is enough
    data: Vec<Vec<QueryResult>>,
    begin: Vec<usize>,
    end: Vec<usize>,
    device: LDeviceRef<'d>,
    // TODO: instead of these being public we can access it from methods
    pub results: Vec<f64>,
    pub total_gpu_time: f64,
    pub total_gpu_time_history: Vec<f64>,
    pub gpu_time_history: Vec<Vec<f64>>,
    pub gpu_time_history_begin: usize,
    last_update_time: Instant,

    span_maps: Vec<Vec<u16>>,
}

impl<'d> GpuStats<'d> {
    pub fn new(device: LDeviceRef<'d>, passes_count: usize) -> Result<Self> {
        let timestamp_query_pool = unsafe {
            (0..MAX_FRAMES_IN_FLIGHT)
                .map(|_| {
                    device.create_query_pool(
                        &vk::QueryPoolCreateInfo::default()
                            .query_type(vk::QueryType::TIMESTAMP)
                            .query_count(QUERY_POOL_SIZE as u32),
                        None,
                    )
                })
                .collect::<Result<Vec<_>, _>>()?
        };

        let pipeline_stats_query_pool = unsafe {
            device.create_query_pool(
                &vk::QueryPoolCreateInfo::default()
                    .query_type(vk::QueryType::PIPELINE_STATISTICS)
                    .pipeline_statistics(vk::QueryPipelineStatisticFlags::CLIPPING_INVOCATIONS)
                    .query_count(128),
                None,
            )?
        };
        let cmd = unsafe { device.begin_single_time_command()? };
        unsafe { cmd.reset_query_pool(timestamp_query_pool[0], 0, QUERY_POOL_SIZE as u32) };
        unsafe { cmd.write_timestamp(vk::PipelineStageFlags::BOTTOM_OF_PIPE, timestamp_query_pool[0], 0) };
        unsafe { device.end_single_time_command(cmd)? };

        let mut data = [0u64; 1];
        unsafe {
            device.get_query_pool_results(
                timestamp_query_pool[0],
                0,
                &mut data,
                vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT,
            )?
        }

        let cmd = unsafe { device.begin_single_time_command()? };
        for query_pool in &timestamp_query_pool {
            unsafe { cmd.reset_query_pool(*query_pool, 0, QUERY_POOL_SIZE as u32) };
        }
        unsafe { device.end_single_time_command(cmd)? };

        let gpu_ctx = tracy_client::Client::running().unwrap().new_gpu_context(
            Some("Vulkan"),
            tracy_client::GpuContextType::Vulkan,
            data[0] as i64,
            device.props().limits.timestamp_period,
        )?;

        Ok(Self {
            timestamp_query_pool,
            pipeline_stats_query_pool,
            gpu_ctx,
            triangles: 0.0,
            data: vec![vec![Default::default(); QUERY_POOL_SIZE]; MAX_FRAMES_IN_FLIGHT],
            spans: slab::Slab::with_capacity(u16::MAX as usize),
            begin: vec![0; MAX_FRAMES_IN_FLIGHT],
            end: vec![0; MAX_FRAMES_IN_FLIGHT],
            device,
            results: vec![0.0; passes_count],
            span_maps: vec![vec![0; QUERY_POOL_SIZE / 2]; MAX_FRAMES_IN_FLIGHT],
            last_update_time: Instant::now(),
            total_gpu_time: 0.0,
            total_gpu_time_history: vec![0.0; (GPU_TIME_GRAPH_DURATION / GPU_TIME_GRAPH_PERIOD).ceil() as usize],
            gpu_time_history: vec![
                vec![0.0; (GPU_TIME_GRAPH_DURATION / GPU_TIME_GRAPH_PERIOD).ceil() as usize];
                passes_count
            ],
            gpu_time_history_begin: 0,
        })
    }

    pub fn begin_span(
        &mut self,
        cmd: &LCommandBuffer,
        name: &'static str,
        pass_index: usize,
        frame_index: usize,
    ) -> Result<usize> {
        if self.end[frame_index] == self.span_maps[frame_index].len() - 1 && self.begin[frame_index] == 0 {
            return Err(anyhow!(
                "query pool is full, consider increasing the size of the query pool"
            ));
        }

        let inner_span = self.gpu_ctx.span_alloc(name, "function", "file", 1)?;
        let index = self.spans.insert(GpuSpan {
            index: pass_index,
            inner: inner_span,
        });

        assert!(self.spans.len() <= (u16::MAX as usize));
        self.span_maps[frame_index][self.end[frame_index]] = index as u16;

        unsafe {
            cmd.write_timestamp(
                vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                self.timestamp_query_pool[frame_index],
                (self.end[frame_index] as u32) * 2,
            )
        };

        let ret = self.end[frame_index];

        self.end[frame_index] = (self.end[frame_index] + 1) % (QUERY_POOL_SIZE / 2);

        Ok(ret)
    }

    pub fn end_span(&mut self, cmd: &LCommandBuffer, index: usize, frame_index: usize) {
        let span_index = self.span_maps[frame_index][index];
        let span = &mut self.spans[span_index as usize];
        span.inner.end_zone();
        unsafe {
            cmd.write_timestamp(
                vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                self.timestamp_query_pool[frame_index],
                (index as u32) * 2 + 1,
            )
        };
    }

    pub unsafe fn begin_pipeline_stats_query(&self, cmd: &LCommandBuffer) {
        unsafe {
            cmd.reset_query_pool(self.pipeline_stats_query_pool, 0, 128);
            cmd.begin_query(self.pipeline_stats_query_pool, 0, vk::QueryControlFlags::empty());
        }
    }

    pub unsafe fn end_pipeline_stats_query(&self, cmd: &LCommandBuffer) {
        unsafe {
            cmd.end_query(self.pipeline_stats_query_pool, 0);
        }
    }

    pub unsafe fn update_gpu_time(&mut self, cmd: &LCommandBuffer, frame_index: usize) -> Result<()> {
        let begin = self.begin[frame_index];
        let mut end = self.end[frame_index];

        if begin == end {
            return Ok(());
        }

        if begin > end {
            // end has wrapped around
            end = QUERY_POOL_SIZE / 2;
        }

        let data_begin = begin * 2;
        let data_end = end * 2;
        let _ = unsafe {
            self.device.get_query_pool_results(
                self.timestamp_query_pool[frame_index],
                (self.begin[frame_index] * 2) as u32,
                &mut self.data[frame_index][data_begin..data_end],
                vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WITH_AVAILABILITY,
            )
        };

        let mut i = self.begin[frame_index];
        while i < end {
            let span_begin = self.data[frame_index][i * 2];
            let span_end = self.data[frame_index][i * 2 + 1];
            let span_index = self.span_maps[frame_index][i];

            if span_begin.status == 0 || span_end.status == 0 || !self.spans.contains(span_index as usize) {
                break;
            }

            let span = self.spans.remove(span_index as usize);

            let s = min(span_begin.result, span_end.result);
            let e = max(span_begin.result, span_end.result);
            span.inner.upload_timestamp_start(s as i64);
            span.inner.upload_timestamp_end(e as i64);
            let acc = &mut self.results[span.index];
            *acc = *acc * 0.99 + ((e - s) as f32 * self.device.props().limits.timestamp_period) as f64 * 0.01;

            i += 1;
        }

        if i != begin {
            unsafe {
                cmd.reset_query_pool(
                    self.timestamp_query_pool[frame_index],
                    (begin * 2) as u32,
                    ((i - begin) * 2) as u32,
                )
            };
        }

        self.begin[frame_index] = i % (QUERY_POOL_SIZE / 2);

        Ok(())
    }

    pub unsafe fn update_pipeline_stats(&mut self) -> Result<()> {
        let mut data: [u64; 1] = [0];

        if unsafe {
            self.device
                .get_query_pool_results(
                    self.pipeline_stats_query_pool,
                    0,
                    &mut data,
                    vk::QueryResultFlags::TYPE_64,
                )
                .is_err()
        } {
            return Ok(());
        }

        self.triangles = data[0] as f64;
        Ok(())
    }

    pub fn update(&mut self) {
        self.total_gpu_time = self.results.iter().sum();

        let now = Instant::now();
        if (now - self.last_update_time) > Duration::from_secs_f64(GPU_TIME_GRAPH_PERIOD) {
            self.total_gpu_time_history[self.gpu_time_history_begin] = self.total_gpu_time;

            for (i, result) in self.results.iter().enumerate() {
                self.gpu_time_history[i][self.gpu_time_history_begin] = *result;
            }

            self.gpu_time_history_begin = (self.gpu_time_history_begin + 1) % self.total_gpu_time_history.len();
            self.last_update_time = now;
        }
    }
}

impl Drop for GpuStats<'_> {
    fn drop(&mut self) {
        self.timestamp_query_pool.drain(..).for_each(|query_pool| {
            unsafe { self.device.destroy_query_pool(query_pool, None) };
        });
        unsafe { self.device.destroy_query_pool(self.pipeline_stats_query_pool, None) };
    }
}
