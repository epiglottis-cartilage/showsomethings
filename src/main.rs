use std::sync::Arc;

use winit::application::ApplicationHandler;
use winit::event::{self};
use winit::event_loop::{self, EventLoop};
use winit::window::{self, Window};

use vulkano::VulkanLibrary;
use vulkano::buffer::{Buffer, BufferCreateInfo, BufferUsage};
use vulkano::command_buffer::allocator::{
    StandardCommandBufferAllocator, StandardCommandBufferAllocatorCreateInfo,
};
use vulkano::command_buffer::{
    AutoCommandBufferBuilder, ClearColorImageInfo, CommandBufferUsage, CopyBufferInfo,
    CopyImageToBufferInfo,
};
use vulkano::descriptor_set::allocator::StandardDescriptorSetAllocator;
use vulkano::descriptor_set::{DescriptorSet, WriteDescriptorSet};
use vulkano::device::{self, Device};
use vulkano::format::{ClearColorValue, Format};
use vulkano::image::{Image, ImageCreateInfo, ImageType, ImageUsage};
use vulkano::instance::{Instance, InstanceCreateFlags, InstanceCreateInfo};
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator};
use vulkano::pipeline::compute::ComputePipelineCreateInfo;
use vulkano::pipeline::layout::{PipelineDescriptorSetLayoutCreateInfo, PipelineLayoutCreateInfo};
use vulkano::pipeline::{ComputePipeline, PipelineLayout, PipelineShaderStageCreateInfo};
use vulkano::pipeline::{Pipeline, PipelineBindPoint};
use vulkano::sync::{self, GpuFuture};

use image::{ImageBuffer, Rgba};

mod error;
pub use error::Result;

#[derive(Default)]
struct App {
    window: Option<Window>,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &event_loop::ActiveEventLoop) {
        self.window = Some(
            event_loop
                .create_window(window::WindowAttributes::default())
                .unwrap(),
        );
    }

    fn window_event(
        &mut self,
        event_loop: &event_loop::ActiveEventLoop,
        _window_id: window::WindowId,
        event: event::WindowEvent,
    ) {
        match event {
            event::WindowEvent::CloseRequested => {
                println!("The close button was pressed; stopping");
                event_loop.exit();
            }
            event::WindowEvent::RedrawRequested => {
                self.window.as_ref().unwrap().request_redraw();
            }
            _ => (),
        }
    }
}

fn main() -> Result<()> {
    // let event_loop = EventLoop::new().unwrap();
    // event_loop.set_control_flow(event_loop::ControlFlow::Poll);
    // let mut app = App::default();
    // event_loop.run_app(&mut app)?;
    compute_pipeline()
}
fn compute_pipeline() -> Result<()> {
    let library = VulkanLibrary::new()?;
    let instance = Instance::new(
        library,
        InstanceCreateInfo {
            flags: InstanceCreateFlags::ENUMERATE_PORTABILITY,
            ..Default::default()
        },
    )?;
    let devices = instance.enumerate_physical_devices()?.collect::<Vec<_>>();
    for dev in &devices {
        println!("{:?}", dev.properties().device_name);
    }
    let device_phy = devices.into_iter().next().unwrap();

    let queue_family_idx = device_phy
        .queue_family_properties()
        .iter()
        .enumerate()
        .position(|(_i, prop)| prop.queue_flags.contains(device::QueueFlags::GRAPHICS))
        .ok_or("queue family not suitable")?;

    let (device, mut queues) = Device::new(
        device_phy,
        device::DeviceCreateInfo {
            queue_create_infos: vec![device::QueueCreateInfo {
                queue_family_index: queue_family_idx as _,
                ..Default::default()
            }],
            enabled_extensions: device::DeviceExtensions {
                khr_storage_buffer_storage_class: true,
                ..Default::default()
            },
            ..Default::default()
        },
    )?;

    let queue = queues.next().unwrap();

    let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));

    let image_size = [1920, 1080, 1];
    let image = Image::new(
        memory_allocator.clone(),
        ImageCreateInfo {
            image_type: ImageType::Dim2d,
            format: Format::R16G16B16A16_UNORM,
            extent: image_size,
            usage: ImageUsage::TRANSFER_DST | ImageUsage::TRANSFER_SRC,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
            ..Default::default()
        },
    )?;

    let buf = Buffer::from_iter(
        memory_allocator.clone(),
        BufferCreateInfo {
            usage: BufferUsage::TRANSFER_DST,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_HOST
                | MemoryTypeFilter::HOST_RANDOM_ACCESS,
            ..Default::default()
        },
        (0..image_size.iter().product::<u32>() * 4).map(|_| 0u16),
    )
    .expect("failed to create buffer");

    let command_buffer_allocator = Arc::new(StandardCommandBufferAllocator::new(
        device.clone(),
        StandardCommandBufferAllocatorCreateInfo::default(),
    ));
    let mut command_buffer_builder = AutoCommandBufferBuilder::primary(
        command_buffer_allocator.clone(),
        queue.queue_family_index(),
        CommandBufferUsage::OneTimeSubmit,
    )?;

    command_buffer_builder
        .clear_color_image(ClearColorImageInfo {
            clear_value: ClearColorValue::Float([137. / 256., 100. / 256., 204. / 256., 0.5]),
            ..ClearColorImageInfo::image(image.clone())
        })?
        .copy_image_to_buffer(CopyImageToBufferInfo::image_buffer(
            image.clone(),
            buf.clone(),
        ))?;

    let command_buffer = command_buffer_builder.build()?;

    let future = sync::now(device.clone())
        .then_execute(queue.clone(), command_buffer)?
        .then_signal_fence_and_flush()?;

    future.wait(None)?;
    let buffer_content = buf.read()?;
    let image =
        ImageBuffer::<Rgba<u16>, _>::from_raw(image_size[0], image_size[1], &buffer_content[..])
            .unwrap();
    image.save("image.tif")?;
    Ok(())
}
