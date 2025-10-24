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
use vulkano::command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage, CopyBufferInfo};
use vulkano::descriptor_set::allocator::StandardDescriptorSetAllocator;
use vulkano::descriptor_set::{DescriptorSet, WriteDescriptorSet};
use vulkano::device::{self, Device};
use vulkano::instance::{Instance, InstanceCreateFlags, InstanceCreateInfo};
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator};
use vulkano::pipeline::compute::ComputePipelineCreateInfo;
use vulkano::pipeline::layout::{PipelineDescriptorSetLayoutCreateInfo, PipelineLayoutCreateInfo};
use vulkano::pipeline::{ComputePipeline, PipelineLayout, PipelineShaderStageCreateInfo};
use vulkano::pipeline::{Pipeline, PipelineBindPoint};
use vulkano::sync::{self, GpuFuture};

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

    let data_iter = 0..65536u32;
    let data_buffer = Buffer::from_iter(
        memory_allocator.clone(),
        BufferCreateInfo {
            usage: BufferUsage::STORAGE_BUFFER,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
            ..Default::default()
        },
        data_iter,
    )
    .expect("failed to create buffer");
    mod cs {
        vulkano_shaders::shader! {
            ty: "compute",
            path: "./shader/mul114514",
        }
    }
    let shader = cs::load(device.clone()).expect("failed to create shader module");
    let cs = shader.entry_point("main").unwrap();
    let stage = PipelineShaderStageCreateInfo::new(cs);
    let layout = PipelineLayout::new(
        device.clone(),
        PipelineDescriptorSetLayoutCreateInfo::from_stages([&stage])
            .into_pipeline_layout_create_info(device.clone())
            .unwrap(),
    )?;
    let compute_pipeline = ComputePipeline::new(
        device.clone(),
        None,
        ComputePipelineCreateInfo::stage_layout(stage, layout),
    )
    .expect("failed to create compute pipeline");

    let descriptor_set_allocator = Arc::new(StandardDescriptorSetAllocator::new(
        device.clone(),
        Default::default(),
    ));

    let pipeline_layout = compute_pipeline.layout();
    let descriptor_set_layouts = pipeline_layout.set_layouts();

    let descriptor_set_layout_index = 0;
    let descriptor_set_layout = descriptor_set_layouts
        .get(descriptor_set_layout_index)
        .unwrap();
    let descriptor_set = DescriptorSet::new(
        descriptor_set_allocator,
        descriptor_set_layout.clone(),
        [WriteDescriptorSet::buffer(0, data_buffer.clone())], // 0 is the binding
        [],
    )?;

    let command_buffer_allocator = Arc::new(StandardCommandBufferAllocator::new(
        device.clone(),
        StandardCommandBufferAllocatorCreateInfo::default(),
    ));
    let mut command_buffer_builder = AutoCommandBufferBuilder::primary(
        command_buffer_allocator.clone(),
        queue.queue_family_index(),
        CommandBufferUsage::OneTimeSubmit,
    )?;

    let work_group_counts = [1024, 1, 1];

    unsafe {
        command_buffer_builder
            .bind_pipeline_compute(compute_pipeline.clone())?
            .bind_descriptor_sets(
                PipelineBindPoint::Compute,
                compute_pipeline.layout().clone(),
                descriptor_set_layout_index as u32,
                descriptor_set,
            )?
            .dispatch(work_group_counts)?
    };

    let command_buffer = command_buffer_builder.build()?;

    let future = sync::now(device.clone())
        .then_execute(queue.clone(), command_buffer)
        .unwrap()
        .then_signal_fence_and_flush()
        .unwrap();
    future.wait(None).unwrap();
    let content = data_buffer.read().unwrap();
    for (n, val) in content.iter().enumerate() {
        assert_eq!(*val, (n as u32).wrapping_mul(114514));
    }

    println!("Everything succeeded!");

    Ok(())
}
