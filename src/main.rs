#![feature(iterator_try_collect)]
#![feature(lazy_type_alias)]
#![allow(incomplete_features)]
use glam::{Mat4, vec3};
use std::sync::Arc;
use std::time;
use vulkano::{
    Validated, VulkanError, VulkanLibrary,
    buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage, Subbuffer},
    command_buffer::{
        AutoCommandBufferBuilder, CommandBufferUsage, PrimaryAutoCommandBuffer,
        RenderPassBeginInfo, SubpassBeginInfo, SubpassContents,
        allocator::{CommandBufferAllocator, StandardCommandBufferAllocator},
    },
    descriptor_set::{
        DescriptorSet, WriteDescriptorSet,
        allocator::{
            DescriptorSetAllocator, StandardDescriptorSetAllocator,
            StandardDescriptorSetAllocatorCreateInfo,
        },
    },
    device::{
        Device, DeviceCreateInfo, DeviceExtensions, Queue, QueueCreateInfo,
        physical::PhysicalDevice,
    },
    image::{Image, ImageUsage, view::ImageView},
    instance::{
        Instance, InstanceCreateFlags, InstanceCreateInfo,
        debug::{
            DebugUtilsMessageSeverity, DebugUtilsMessenger, DebugUtilsMessengerCallback,
            DebugUtilsMessengerCreateInfo,
        },
    },
    memory::allocator::{
        AllocationCreateInfo, MemoryAllocator, MemoryTypeFilter, StandardMemoryAllocator,
    },
    pipeline::{
        GraphicsPipeline, Pipeline, PipelineBindPoint, PipelineLayout,
        PipelineShaderStageCreateInfo,
        graphics::{
            GraphicsPipelineCreateInfo,
            color_blend::{ColorBlendAttachmentState, ColorBlendState},
            input_assembly::{InputAssemblyState, PrimitiveTopology},
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
    dpi::LogicalSize,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    window::Window,
};

mod error;
use crate::error::ErrorLogger;
pub use error::Result;
mod shader;
fn main() -> Result<()> {
    env_logger::init();
    let event_loop = EventLoop::new().log()?;
    event_loop.set_control_flow(ControlFlow::wait_duration(time::Duration::from_millis(
        1000 / 144,
    )));
    let mut app = App::new();
    event_loop.run_app(&mut app).log()?;
    Ok(())
}

struct App {
    window: Option<Arc<Window>>,
    renderer: Option<RendererContext>,
    frame_interval: time::Duration,
}
impl App {
    fn new() -> Self {
        Self {
            frame_interval: time::Duration::from_secs_f32(1. / 144.),
            window: None,
            renderer: None,
        }
    }
}
impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window = Arc::new(
            event_loop
                .create_window(
                    Window::default_attributes()
                        .with_title("ShowSomething")
                        .with_transparent(true)
                        // .with_resizable(false)
                        .with_inner_size(LogicalSize::new(720, 720)),
                )
                .log()
                .unwrap(),
        );
        let render = RendererContext::new_with(event_loop, window.clone())
            .log()
            .unwrap();

        window.request_redraw();
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
                log::trace!("Resized");
                self.renderer.as_mut().unwrap().window_resize = true;
            }
            WindowEvent::RedrawRequested => {
                self.renderer
                    .as_mut()
                    .unwrap()
                    .update(self.window.as_ref().unwrap())
                    .unwrap();
                event_loop.set_control_flow(ControlFlow::wait_duration(self.frame_interval));
                self.window.as_ref().unwrap().request_redraw();
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
    memory_allocator: Arc<dyn MemoryAllocator>,
    command_buffer_allocator: Arc<dyn CommandBufferAllocator>,
    descriptor_set_allocator: Arc<dyn DescriptorSetAllocator>,
    render_pass: Arc<RenderPass>,
    swapchain: Arc<Swapchain>,
    framebuffers: Vec<Arc<Framebuffer>>,
    vertex_buffer: Subbuffer<[Vertex3D]>,
    descriptor_sets: Vec<Arc<DescriptorSet>>,
    uniform_buffers: Vec<Subbuffer<shader::vs::Data>>,
    viewport: Viewport,
    vs: Arc<ShaderModule>,
    fs: Arc<ShaderModule>,
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
    pub swapchain_recreate: bool,
    pub window_resize: bool,
    start_time: time::Instant,
    world: Mat4,
    view: Mat4,
    proj: Mat4,
    _debug_handle: DebugUtilsMessenger,
}
impl RendererContext {
    fn new_with(event_loop: &ActiveEventLoop, window: Arc<Window>) -> Result<Self> {
        let library = VulkanLibrary::new().log()?;
        let mut enabled_extensions = Surface::required_extensions(event_loop).log()?;
        enabled_extensions.ext_debug_utils = true;
        let instance = Instance::new(
            library,
            InstanceCreateInfo {
                flags: InstanceCreateFlags::ENUMERATE_PORTABILITY,
                enabled_extensions,
                ..Default::default()
            },
        )
        .log()?;

        let debug_handle = DebugUtilsMessenger::new(
            instance.clone(),
            DebugUtilsMessengerCreateInfo::user_callback(unsafe {
                DebugUtilsMessengerCallback::new(|severity, r#type, data| match severity {
                    DebugUtilsMessageSeverity::ERROR => {
                        log::error!("{:?}-{:?}", r#type, data.message)
                    }
                    DebugUtilsMessageSeverity::WARNING => {
                        log::warn!("{:?}-{:?}", r#type, data.message)
                    }
                    DebugUtilsMessageSeverity::INFO => {
                        log::info!("{:?}-{:?}", r#type, data.message)
                    }
                    DebugUtilsMessageSeverity::VERBOSE => {
                        log::debug!("{:?}-{:?}", r#type, data.message)
                    }
                    _ => unreachable!(),
                })
            }),
        )
        .log()?;

        let device_extensions = DeviceExtensions {
            khr_swapchain: true,
            ..DeviceExtensions::empty()
        };
        let surface = Surface::from_window(instance.clone(), window.clone()).log()?;

        let (physical_device, queue_family_index) =
            Self::select_physical_device(&instance, &surface, &device_extensions)?;

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

        let queue = queues.next().log()?;

        let (swapchain, images): (Arc<Swapchain>, Vec<Arc<Image>>) = {
            let caps = physical_device
                .surface_capabilities(&surface, Default::default())
                .log()?;

            let dimensions = window.inner_size();
            let composite_alpha = caps
                .supported_composite_alpha
                .into_iter()
                .map(|alpha| {
                    (
                        match alpha {
                            swapchain::CompositeAlpha::Inherit => false,
                            swapchain::CompositeAlpha::Opaque => false,
                            swapchain::CompositeAlpha::PreMultiplied => true,
                            swapchain::CompositeAlpha::PostMultiplied => true,
                            _ => unreachable!(),
                        },
                        alpha,
                    )
                })
                .max_by_key(|(s, _)| *s)
                .unwrap()
                .1;
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
                    present_mode: swapchain::PresentMode::Fifo,
                    ..Default::default()
                },
            )
            .log()?
        };

        let render_pass = Self::get_render_pass(device.clone(), swapchain.clone())?;
        let framebuffers = Self::get_framebuffers(&images, render_pass.clone())?;

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
                Vertex3D {
                    position: [0.0, 1.0, 2.0],
                },
                Vertex3D {
                    position: [-1.0, -1.0, 0.0],
                },
                Vertex3D {
                    position: [1.0, -1.0, 0.0],
                },
                Vertex3D {
                    position: [0.0, -2.0, 5.0],
                },
            ],
        )
        .log()?;

        let uniform_buffers = (0..swapchain.image_count())
            .map(|_| {
                Buffer::new_sized(
                    memory_allocator.clone(),
                    BufferCreateInfo {
                        usage: BufferUsage::UNIFORM_BUFFER,
                        ..Default::default()
                    },
                    AllocationCreateInfo {
                        memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                            | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                        ..Default::default()
                    },
                )
                .log()
            })
            .try_collect::<Vec<_>>()?;

        let viewport = Viewport {
            offset: [0.0, 0.0],
            extent: window.inner_size().into(),
            depth_range: 0.0..=1.0,
        };

        let vs = shader::vs::load(device.clone()).log()?;
        let fs = shader::fs::load(device.clone()).log()?;

        let pipeline = Self::get_pipeline(
            device.clone(),
            vs.clone(),
            fs.clone(),
            render_pass.clone(),
            viewport.clone(),
        )
        .log()?;

        let descriptor_set_allocator = Arc::new(StandardDescriptorSetAllocator::new(
            device.clone(),
            StandardDescriptorSetAllocatorCreateInfo {
                ..Default::default()
            },
        ));
        let descriptor_set_layouts = pipeline.layout().set_layouts();
        let descriptor_set_layout_index = 0;
        let descriptor_set_layout = descriptor_set_layouts[descriptor_set_layout_index].clone();
        let descriptor_sets = uniform_buffers
            .iter()
            .map(|buf| {
                DescriptorSet::new(
                    descriptor_set_allocator.clone(),
                    descriptor_set_layout.clone(),
                    [WriteDescriptorSet::buffer(0, buf.clone())], // 0 is the binding
                    [],
                )
            })
            .try_collect::<Vec<_>>()
            .log()?;

        let command_buffer_allocator: Arc<dyn CommandBufferAllocator> = Arc::new(
            StandardCommandBufferAllocator::new(device.clone(), Default::default()),
        );
        let command_buffers = Self::get_command_buffers(
            &command_buffer_allocator,
            &queue,
            &pipeline,
            &framebuffers,
            &vertex_buffer,
            &descriptor_sets,
        )?;

        let view = Mat4::look_to_rh(vec3(0.0, -0.8, 2.0), vec3(0., 0., -1.), vec3(0., 1., 0.));
        let aspect_ratio = swapchain.image_extent()[0] as f32 / swapchain.image_extent()[1] as f32;
        let proj = Mat4::perspective_rh_gl(std::f32::consts::FRAC_PI_2, aspect_ratio, 0.01, 10.0);

        Ok(Self {
            instance,
            device,
            queue,
            command_buffer_allocator,
            memory_allocator,
            descriptor_set_allocator,
            swapchain,
            render_pass,
            framebuffers,
            descriptor_sets,
            vertex_buffer,
            uniform_buffers,
            viewport,
            vs,
            fs,
            command_buffers,
            fences: vec![None; images.len()],
            previous_fence_i: 0,
            world: Mat4::IDENTITY,
            view,
            proj,
            swapchain_recreate: false,
            window_resize: false,
            start_time: time::Instant::now(),
            _debug_handle: debug_handle,
        })
    }
    fn update(&mut self, window: &Arc<Window>) -> Result<()> {
        log::trace!("Rendering new frame");
        if self.swapchain_recreate || self.window_resize {
            self.swapchain_recreate = false;
            let new_dimensions = window.inner_size();

            let (new_swapchain, new_images) = self
                .swapchain
                .recreate(SwapchainCreateInfo {
                    image_extent: new_dimensions.into(),
                    ..self.swapchain.create_info()
                })
                .log()?;

            self.swapchain = new_swapchain;
            let new_framebuffers = Self::get_framebuffers(&new_images, self.render_pass.clone())?;

            if self.window_resize {
                self.window_resize = false;

                self.viewport.extent = new_dimensions.into();
                let new_aspect_ratio = new_dimensions.width as f32 / new_dimensions.height as f32;
                self.proj = Mat4::perspective_rh_gl(
                    std::f32::consts::FRAC_PI_2,
                    new_aspect_ratio,
                    0.1,   // 近平面（避免模型被裁剪）
                    100.0, // 远平面
                );

                let new_pipeline = Self::get_pipeline(
                    self.device.clone(),
                    self.vs.clone(),
                    self.fs.clone(),
                    self.render_pass.clone(),
                    self.viewport.clone(),
                )?;

                self.command_buffers = Self::get_command_buffers(
                    &self.command_buffer_allocator,
                    &self.queue,
                    &new_pipeline,
                    &new_framebuffers,
                    &self.vertex_buffer,
                    &self.descriptor_sets,
                )?;
            }
        }
        let elapsed = self.start_time.elapsed().as_secs_f32();
        self.world = Mat4::from_rotation_y(elapsed * 2.0);

        let (image_i, suboptimal, acquire_future) =
            match swapchain::acquire_next_image(self.swapchain.clone(), None)
                .map_err(Validated::unwrap)
            {
                Ok(r) => r,
                Err(VulkanError::OutOfDate) => {
                    self.swapchain_recreate = true;
                    return Ok(());
                }
                Err(e) => panic!("failed to acquire next image: {e}"),
            };

        if suboptimal {
            self.swapchain_recreate = true;
        }
        // wait for the fence related to this image to finish (normally this would be the oldest fence)
        if let Some(image_fence) = &self.fences[image_i as usize] {
            image_fence.wait(None).log()?;
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

        let uniform_buffer = &mut self.uniform_buffers[image_i as usize];
        *uniform_buffer.write()? = shader::vs::Data {
            world: self.world.to_cols_array_2d(),
            view: self.view.to_cols_array_2d(),
            proj: self.proj.to_cols_array_2d(),
        };

        let future = previous_future
            .join(acquire_future)
            .then_execute(
                self.queue.clone(),
                self.command_buffers[image_i as usize].clone(),
            )
            .log()?
            .then_swapchain_present(
                self.queue.clone(),
                SwapchainPresentInfo::swapchain_image_index(self.swapchain.clone(), image_i),
            )
            .then_signal_fence_and_flush();

        self.fences[image_i as usize] = match future.map_err(Validated::unwrap) {
            Ok(value) => Some(Arc::new(value)),
            Err(VulkanError::OutOfDate) => {
                self.swapchain_recreate = true;
                None
            }
            Err(e) => {
                println!("failed to flush future: {e}");
                None
            }
        };
        self.previous_fence_i = image_i;
        Ok(())
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
                let view = ImageView::new_default(image.clone()).log()?;
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

        let vertex_input_state = Vertex3D::per_vertex().definition(&vs).log()?;

        let stages = [
            PipelineShaderStageCreateInfo::new(vs),
            PipelineShaderStageCreateInfo::new(fs),
        ];

        let layout = PipelineLayout::new(
            device.clone(),
            PipelineDescriptorSetLayoutCreateInfo::from_stages(&stages)
                .into_pipeline_layout_create_info(device.clone())
                .log()?,
        )
        .log()?;

        let subpass = Subpass::from(render_pass.clone(), 0).log()?;

        let pipeline = GraphicsPipeline::new(
            device.clone(),
            None,
            GraphicsPipelineCreateInfo {
                stages: stages.into_iter().collect(),
                vertex_input_state: Some(vertex_input_state),
                input_assembly_state: Some(InputAssemblyState {
                    topology: PrimitiveTopology::TriangleStrip,
                    ..Default::default()
                }),
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
        .log()?;
        Ok(pipeline)
    }

    fn get_command_buffers(
        command_buffer_allocator: &Arc<dyn CommandBufferAllocator>,
        queue: &Arc<Queue>,
        pipeline: &Arc<GraphicsPipeline>,
        framebuffers: &[Arc<Framebuffer>],
        vertex_buffer: &Subbuffer<[Vertex3D]>,
        descriptor_sets: &[Arc<DescriptorSet>],
    ) -> Result<Vec<Arc<PrimaryAutoCommandBuffer>>> {
        framebuffers
            .iter()
            .zip(descriptor_sets.iter())
            .map(|(framebuffer, descriptor_set)| {
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
                                clear_values: vec![Some([1.0, 1.0, 1.0, 0.0].into())],
                                ..RenderPassBeginInfo::framebuffer(framebuffer.clone())
                            },
                            SubpassBeginInfo {
                                contents: SubpassContents::Inline,
                                ..Default::default()
                            },
                        )?
                        .bind_pipeline_graphics(pipeline.clone())?
                        .bind_vertex_buffers(0, vertex_buffer.clone())?
                        .bind_descriptor_sets(
                            PipelineBindPoint::Graphics,
                            pipeline.layout().clone(),
                            0,
                            descriptor_set.clone(),
                        )?
                        .draw(vertex_buffer.len() as u32, 1, 0, 0)?
                        .end_render_pass(Default::default())?
                };
                builder.build()
            })
            .try_collect()
            .log()
            .map_err(Into::into)
    }
}

#[derive(BufferContents, Vertex)]
#[repr(C)]
struct Vertex2D {
    #[format(R32G32_SFLOAT)]
    position: [f32; 2],
}
impl Vertex2D {
    #[allow(unused)]
    const NONE: Self = Self {
        position: unsafe { std::mem::transmute([-1i32; 2]) },
    };
}

#[derive(BufferContents, Vertex)]
#[repr(C)]
struct Vertex3D {
    #[format(R32G32B32_SFLOAT)]
    position: [f32; 3],
}
impl Vertex3D {
    #[allow(unused)]
    const NONE: Self = Self {
        position: unsafe { std::mem::transmute([-1i32; 3]) },
    };
}
