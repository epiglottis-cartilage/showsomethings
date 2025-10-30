#![feature(try_trait_v2)]
#![feature(impl_trait_in_bindings)]
#![feature(iterator_try_collect)]
use std::sync::Arc;

use vulkano::{
    Validated, VulkanError, VulkanLibrary,
    buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage, Subbuffer},
    command_buffer::{
        AutoCommandBufferBuilder, CommandBufferUsage, PrimaryAutoCommandBuffer,
        RenderPassBeginInfo, SubpassBeginInfo, SubpassContents,
        allocator::StandardCommandBufferAllocator,
    },
    device::{
        self, Device, DeviceCreateInfo, DeviceExtensions, Queue, QueueCreateInfo,
        physical::PhysicalDevice,
    },
    image::{Image, ImageUsage, view::ImageView},
    instance::{Instance, InstanceCreateFlags, InstanceCreateInfo},
    memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator},
    pipeline::{
        GraphicsPipeline, PipelineLayout, PipelineShaderStageCreateInfo,
        graphics::{
            GraphicsPipelineCreateInfo,
            color_blend::{ColorBlendAttachmentState, ColorBlendState},
            input_assembly::InputAssemblyState,
            multisample::MultisampleState,
            rasterization::RasterizationState,
            vertex_input::{Vertex, VertexDefinition},
            viewport::{Viewport, ViewportState},
        },
        layout::PipelineDescriptorSetLayoutCreateInfo,
    },
    render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass, Subpass},
    shader::ShaderModule,
    swapchain::{self, Surface, Swapchain, SwapchainCreateInfo, SwapchainPresentInfo},
    sync::{GpuFuture, future::FenceSignalFuture},
};
use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    window::Window,
};

mod error;
pub use error::Result;

use crate::error::ErrorLogger;
mod shader;
fn main() -> Result<()> {
    env_logger::init();
    let event_loop = EventLoop::new().log()?;
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = App::default();
    event_loop.run_app(&mut app).log()?;
    Ok(())
}

#[derive(Default)]
struct App {
    recreate_swapchain: bool,
    window_resize: bool,
    window: Option<Arc<Window>>,
    renderer: Option<RendererContext>,
}
impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window = Arc::new(
            event_loop
                .create_window(
                    Window::default_attributes()
                        .with_title("ShowSomething")
                        .with_transparent(true),
                )
                .log()
                .unwrap(),
        );
        let render = RendererContext::new_with(event_loop, window.clone())
            .log()
            .unwrap();

        self.window = Some(window);
        self.renderer = Some(render);
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: winit::window::WindowId,
        event: WindowEvent,
    ) {
        match event {
            WindowEvent::CloseRequested => {
                log::info!("The close button was pressed; stopping");
                event_loop.exit();
            }
            WindowEvent::Resized(_) => {
                self.window_resize = true;
            }
            WindowEvent::RedrawRequested => {
                self.renderer
                    .as_mut()
                    .unwrap()
                    .update(
                        self.window.as_ref().unwrap(),
                        &mut self.recreate_swapchain,
                        &mut self.window_resize,
                    )
                    .unwrap();
            }
            _ => {}
        }
    }
}
#[allow(dead_code)]
struct RendererContext {
    instance: Arc<Instance>,
    device: Arc<Device>,
    queue: Arc<Queue>,
    render_pass: Arc<RenderPass>,
    swapchain: Arc<Swapchain>,
    framebuffers: Vec<Arc<Framebuffer>>,
    vertex_buffer: Subbuffer<[Vertex2D]>,
    viewport: Viewport,
    vs: Arc<ShaderModule>,
    fs: Arc<ShaderModule>,
    command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    command_buffers: Vec<Arc<PrimaryAutoCommandBuffer>>,
    fences: Vec<
        Option<
            Arc<
                FenceSignalFuture<
                    swapchain::PresentFuture<
                        vulkano::command_buffer::CommandBufferExecFuture<
                            vulkano::sync::future::JoinFuture<
                                Box<dyn GpuFuture>,
                                swapchain::SwapchainAcquireFuture,
                            >,
                        >,
                    >,
                >,
            >,
        >,
    >,
    previous_fence_i: u32,
}
impl RendererContext {
    fn new_with(event_loop: &ActiveEventLoop, window: Arc<Window>) -> Result<Self> {
        let library = VulkanLibrary::new().log()?;
        let enabled_extensions = Surface::required_extensions(event_loop).log()?;

        let instance = Instance::new(
            library,
            InstanceCreateInfo {
                flags: InstanceCreateFlags::ENUMERATE_PORTABILITY,
                enabled_extensions,
                ..Default::default()
            },
        )
        .log()?;

        let device_extensions = DeviceExtensions {
            khr_swapchain: true,
            ..DeviceExtensions::empty()
        };
        let surface = Surface::from_window(instance.clone(), window.clone()).log()?;

        let (physical_device, queue_family_index) =
            select_physical_device(&instance, &surface, &device_extensions)?;

        let (device, mut queues) = Device::new(
            physical_device.clone(),
            DeviceCreateInfo {
                queue_create_infos: vec![QueueCreateInfo {
                    queue_family_index,
                    ..Default::default()
                }],
                enabled_extensions: device_extensions, // new
                ..Default::default()
            },
        )
        .log()?;

        let queue = queues.next().log().unwrap();

        let (swapchain, images) = {
            let caps = physical_device
                .surface_capabilities(&surface, Default::default())
                .log()?;

            let dimensions = window.inner_size();
            let composite_alpha = caps
                .supported_composite_alpha
                .into_iter()
                .find(|&alpha| match alpha {
                    swapchain::CompositeAlpha::PreMultiplied => true,
                    swapchain::CompositeAlpha::PostMultiplied => true,
                    _ => false,
                })
                .unwrap();
            let image_format = physical_device
                .surface_formats(&surface, Default::default())
                .log()?[0]
                .0;

            Swapchain::new(
                device.clone(),
                surface,
                SwapchainCreateInfo {
                    min_image_count: caps.min_image_count,
                    image_format,
                    image_extent: dimensions.into(),
                    image_usage: ImageUsage::COLOR_ATTACHMENT,
                    composite_alpha,
                    ..Default::default()
                },
            )
            .log()?
        };

        let render_pass = get_render_pass(device.clone(), swapchain.clone())?;
        let framebuffers = get_framebuffers(&images, render_pass.clone())?;

        let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));

        let vertex_buffer = Buffer::from_iter(
            memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::VERTEX_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            [
                Vertex2D {
                    position: [-0.5, -0.5],
                },
                Vertex2D {
                    position: [0., 0.5],
                },
                Vertex2D {
                    position: [0.5, 0.],
                },
                Vertex2D {
                    position: [-1.3, -1.3],
                },
                Vertex2D {
                    position: [-0.8, -0.6],
                },
                Vertex2D {
                    position: [-0.6, -0.8],
                },
            ],
        )
        .log()?;
        let image_size = [1920, 1080, 1];
        let viewport = Viewport {
            offset: [0.0, 0.0],
            extent: [image_size[0] as f32, image_size[1] as f32],
            depth_range: 0.0..=1.0,
        };

        let vs = shader::vs::load(device.clone()).log()?;
        let fs = shader::fs::load(device.clone()).log()?;
        let pipeline = get_pipeline(
            device.clone(),
            vs.clone(),
            fs.clone(),
            render_pass.clone(),
            viewport.clone(),
        )?;

        let command_buffer_allocator = Arc::new(StandardCommandBufferAllocator::new(
            device.clone(),
            Default::default(),
        ));
        let command_buffers = get_command_buffers(
            &command_buffer_allocator,
            &queue,
            &pipeline,
            &framebuffers,
            &vertex_buffer,
        );

        Ok(Self {
            instance,
            device,
            queue,
            swapchain,
            render_pass,
            framebuffers,
            vertex_buffer,
            viewport,
            vs,
            fs,
            command_buffer_allocator,
            command_buffers,
            fences: vec![None; images.len()],
            previous_fence_i: 0,
        })
    }
    fn update(
        &mut self,
        window: &Arc<Window>,
        swapchain_recreate: &mut bool,
        winodw_resize: &mut bool,
    ) -> Result<()> {
        if *swapchain_recreate || *winodw_resize {
            *swapchain_recreate = false;
            let new_dimensions = window.inner_size();

            let (new_swapchain, new_images) = self
                .swapchain
                .recreate(SwapchainCreateInfo {
                    image_extent: new_dimensions.into(),
                    ..self.swapchain.create_info()
                })
                .log()?;

            self.swapchain = new_swapchain;
            let new_framebuffers = get_framebuffers(&new_images, self.render_pass.clone())?;

            if *winodw_resize {
                *winodw_resize = false;
                self.viewport.extent = new_dimensions.into();
                let new_pipeline = get_pipeline(
                    self.device.clone(),
                    self.vs.clone(),
                    self.fs.clone(),
                    self.render_pass.clone(),
                    self.viewport.clone(),
                )?;
                self.command_buffers = get_command_buffers(
                    &self.command_buffer_allocator,
                    &self.queue,
                    &new_pipeline,
                    &new_framebuffers,
                    &self.vertex_buffer,
                );
            }

            let (image_i, suboptimal, acquire_future) =
                match swapchain::acquire_next_image(self.swapchain.clone(), None)
                    .map_err(Validated::unwrap)
                {
                    Ok(r) => r,
                    Err(VulkanError::OutOfDate) => {
                        *swapchain_recreate = true;
                        return Ok(());
                    }
                    Err(e) => panic!("failed to acquire next image: {e}"),
                };

            if suboptimal {
                *swapchain_recreate = true;
            }
            // wait for the fence related to this image to finish (normally this would be the oldest fence)
            if let Some(image_fence) = &self.fences[image_i as usize] {
                image_fence.wait(None).unwrap();
            }

            let previous_future = match self.fences[self.previous_fence_i as usize].clone() {
                // Create a NowFuture
                None => {
                    let mut now = vulkano::sync::now(self.device.clone());
                    now.cleanup_finished();

                    now.boxed()
                }
                // Use the existing FenceSignalFuture
                Some(fence) => fence.boxed(),
            };

            let future = previous_future
                .join(acquire_future)
                .then_execute(
                    self.queue.clone(),
                    self.command_buffers[image_i as usize].clone(),
                )
                .unwrap()
                .then_swapchain_present(
                    self.queue.clone(),
                    SwapchainPresentInfo::swapchain_image_index(self.swapchain.clone(), image_i),
                )
                .then_signal_fence_and_flush();

            self.fences[image_i as usize] = match future.map_err(Validated::unwrap) {
                Ok(value) => Some(Arc::new(value)),
                Err(VulkanError::OutOfDate) => {
                    *swapchain_recreate = true;
                    None
                }
                Err(e) => {
                    println!("failed to flush future: {e}");
                    None
                }
            };

            self.previous_fence_i = image_i;
        }
        Ok(())
    }
}

pub fn select_physical_device(
    instance: &Arc<Instance>,
    surface: &Arc<Surface>,
    device_extensions: &DeviceExtensions,
) -> Result<(Arc<PhysicalDevice>, u32)> {
    use vulkano::device::QueueFlags;
    use vulkano::device::physical::PhysicalDeviceType;

    instance
        .enumerate_physical_devices()
        .log()?
        .filter(|p| p.supported_extensions().contains(device_extensions))
        .filter_map(|p| {
            p.queue_family_properties()
                .iter()
                .enumerate()
                .position(|(i, q)| {
                    q.queue_flags.contains(QueueFlags::GRAPHICS)
                        && p.surface_support(i as u32, surface).unwrap_or(false)
                })
                .map(|q| (p, q as u32))
        })
        .min_by_key(|(p, _)| match p.properties().device_type {
            PhysicalDeviceType::DiscreteGpu => 0,
            PhysicalDeviceType::IntegratedGpu => 1,
            PhysicalDeviceType::VirtualGpu => 2,
            PhysicalDeviceType::Cpu => 3,
            _ => 4,
        })
        .log()
}

fn get_render_pass(device: Arc<Device>, swapchain: Arc<Swapchain>) -> Result<Arc<RenderPass>> {
    vulkano::single_pass_renderpass!(
        device,
        attachments: {
            color: {
                format: swapchain.image_format(), // set the format the same as the swapchain
                samples: 1,
                load_op: Clear,
                store_op: Store,
            },
        },
        pass: {
            color: [color],
            depth_stencil: {},
        },
    )
    .log()
    .map_err(Into::into)
}

fn get_framebuffers(
    images: &[Arc<Image>],
    render_pass: Arc<RenderPass>,
) -> Result<Vec<Arc<Framebuffer>>> {
    images
        .iter()
        .map(|image| {
            let view = ImageView::new_default(image.clone()).unwrap();
            Framebuffer::new(
                render_pass.clone(),
                FramebufferCreateInfo {
                    attachments: vec![view],
                    ..Default::default()
                },
            )
            .log()
        })
        .try_collect()
        .map_err(Into::into)
}

fn get_pipeline(
    device: Arc<Device>,
    vs: Arc<ShaderModule>,
    fs: Arc<ShaderModule>,
    render_pass: Arc<RenderPass>,
    viewport: Viewport,
) -> Result<Arc<GraphicsPipeline>> {
    let vs = vs.entry_point("main").log()?;
    let fs = fs.entry_point("main").log()?;

    let vertex_input_state = Vertex2D::per_vertex().definition(&vs).unwrap();

    let stages = [
        PipelineShaderStageCreateInfo::new(vs),
        PipelineShaderStageCreateInfo::new(fs),
    ];

    let layout = PipelineLayout::new(
        device.clone(),
        PipelineDescriptorSetLayoutCreateInfo::from_stages(&stages)
            .into_pipeline_layout_create_info(device.clone())
            .log()?,
    )?;

    let subpass = Subpass::from(render_pass.clone(), 0).log()?;

    GraphicsPipeline::new(
        device.clone(),
        None,
        GraphicsPipelineCreateInfo {
            stages: stages.into_iter().collect(),
            vertex_input_state: Some(vertex_input_state),
            input_assembly_state: Some(InputAssemblyState::default()),
            viewport_state: Some(ViewportState {
                viewports: [viewport].into_iter().collect(),
                ..Default::default()
            }),
            rasterization_state: Some(RasterizationState::default()),
            multisample_state: Some(MultisampleState::default()),
            color_blend_state: Some(ColorBlendState::with_attachment_states(
                subpass.num_color_attachments(),
                ColorBlendAttachmentState::default(),
            )),
            subpass: Some(subpass.into()),
            ..GraphicsPipelineCreateInfo::layout(layout)
        },
    )
    .map_err(Into::into)
}

fn get_command_buffers(
    command_buffer_allocator: &Arc<StandardCommandBufferAllocator>,
    queue: &Arc<device::Queue>,
    pipeline: &Arc<GraphicsPipeline>,
    framebuffers: &[Arc<Framebuffer>],
    vertex_buffer: &Subbuffer<[Vertex2D]>,
) -> Vec<Arc<PrimaryAutoCommandBuffer>> {
    framebuffers
        .iter()
        .map(|framebuffer| {
            let mut builder = AutoCommandBufferBuilder::primary(
                command_buffer_allocator.clone(),
                queue.queue_family_index(),
                CommandBufferUsage::MultipleSubmit,
            )
            .unwrap();

            unsafe {
                builder
                    .begin_render_pass(
                        RenderPassBeginInfo {
                            clear_values: vec![Some([1.0, 1.0, 1.0, 0.1].into())],
                            ..RenderPassBeginInfo::framebuffer(framebuffer.clone())
                        },
                        SubpassBeginInfo {
                            contents: SubpassContents::Inline,
                            ..Default::default()
                        },
                    )
                    .unwrap()
                    .bind_pipeline_graphics(pipeline.clone())
                    .unwrap()
                    .bind_vertex_buffers(0, vertex_buffer.clone())
                    .unwrap()
                    .draw(vertex_buffer.len() as u32, 1, 0, 0)
                    .unwrap()
                    .end_render_pass(Default::default())
                    .unwrap()
            };

            builder.build().unwrap()
        })
        .collect()
}

#[derive(BufferContents, Vertex)]
#[repr(C)]
struct Vertex2D {
    #[format(R32G32_SFLOAT)]
    position: [f32; 2],
}
