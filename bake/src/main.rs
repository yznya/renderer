use anyhow::Result;
use ash::vk;
use brdf::Distributions;
use clap::{Parser, Subcommand};
use lava::device::LDevice;
use lava::logger::init_logger;
use scenery::scene::EnvMap;

use crate::brdf::{compute_brdf_lut, compute_env_map};
use crate::cubemaps::convert_hdr_cubemap_to_ktx;
use crate::ptto_converter::convert_to_ptto;

mod brdf;
mod cubemaps;
mod image_utils;
mod ptto_converter;

#[derive(Debug, Parser)]
#[command(name = "bake")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    BrdfLut {
        #[arg(long)]
        output: String,
    },
    EnvMap {
        #[arg(long)]
        input: String,
        #[arg(long)]
        lambertian: String,
        #[arg(long)]
        ggx: String,
        #[arg(long)]
        charlie: String,
    },
    Cube2Ktx {
        #[arg(long)]
        input: String,
        #[arg(long)]
        output: String,
    },
    Ptto {
        #[arg(long)]
        model: String,
        #[arg(long)]
        skybox: String,
        #[arg(long)]
        brdf_lut: String,
        #[arg(long)]
        lambertian: String,
        #[arg(long)]
        ggx: String,
        #[arg(long)]
        charlie: String,
        #[arg(long)]
        output: String,
    },
}

fn main() -> Result<()> {
    init_logger();

    let cli = Cli::parse();

    let owned_device = LDevice::create_without_surface()?;
    let device = owned_device.device_ref();

    let info = vk::CommandPoolCreateInfo::default()
        .flags(vk::CommandPoolCreateFlags::TRANSIENT)
        .queue_family_index(device.compute_queue_index());
    let command_pool = unsafe { device.create_command_pool(&info, None) }?;

    match cli.command {
        Command::BrdfLut { output: output_file } => {
            compute_brdf_lut(device, command_pool, &output_file)?;
        }
        Command::EnvMap {
            input: input_file,
            lambertian: output_file_lambertian,
            ggx: output_file_ggx,
            charlie: output_file_charlie,
        } => {
            log::info!("Processing Lambertian");
            compute_env_map(device, Distributions::Lambertian, &input_file, &output_file_lambertian)?;
            log::info!("Processing GGX");
            compute_env_map(device, Distributions::Ggx, &input_file, &output_file_ggx)?;
            log::info!("Processing Charlie");
            compute_env_map(device, Distributions::Charlie, &input_file, &output_file_charlie)?;
        }
        Command::Cube2Ktx {
            input: input_file,
            output: output_file,
        } => {
            convert_hdr_cubemap_to_ktx(&input_file, &output_file)?;
        }
        Command::Ptto {
            model,
            skybox,
            brdf_lut,
            lambertian,
            ggx,
            charlie,
            output,
        } => {
            let env_map = EnvMap {
                skybox,
                brdf_lut,
                lambertian,
                ggx,
                charlie,
            };
            convert_to_ptto(&model, env_map, &output)?;
        }
    }

    // TODO: Drop for vulkan resources
    unsafe { device.destroy_command_pool(command_pool, None) };
    Ok(())
}
