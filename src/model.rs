use std::{collections::HashMap, path::Path, sync::Arc};

use crate::{Result, error::ErrorLogger};
use glam::Mat4;
use vulkano::{
    buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer},
    memory::allocator::{AllocationCreateInfo, MemoryAllocator, MemoryTypeFilter},
    pipeline::graphics::input_assembly::PrimitiveTopology,
};

#[derive(Clone, Debug)]
pub struct Primitive {
    pub vertex_buf: Subbuffer<[VertexInput]>,
    pub indices: Subbuffer<[u32]>,
    pub topo: PrimitiveTopology,
    pub material: Arc<Material>,
}
type Texture = Subbuffer<[u8]>;
#[derive(Debug)]
pub struct Mesh {
    pub name: String,
    pub primitives: Vec<Arc<Primitive>>,
}
#[derive(Clone, Debug)]
pub struct Material {
    pub base_color_texture: Option<Arc<Texture>>,
    pub base_color_factor: [f32; 4],
}
const DEFAULT_MATERIAL: std::cell::LazyCell<Arc<Material>> = std::cell::LazyCell::new(|| {
    Arc::new(Material {
        base_color_texture: None,
        base_color_factor: [1.0, 1.0, 1.0, 1.0],
    })
});
#[derive(Debug)]
pub struct Node {
    pub name: String,
    pub mesh: Option<Mesh>,
    pub children: Vec<Node>,
    pub transform: Mat4,
}
#[derive(Debug)]
pub struct Model {
    pub nodes: Vec<Node>,
}
impl Model {
    pub fn load<P>(path: P, memory_allocator: Arc<dyn MemoryAllocator>) -> Result<Vec<Self>>
    where
        P: AsRef<Path>,
    {
        let (document, buffers, images) = gltf::import(path).log()?;

        let mut vertexs = HashMap::with_capacity(buffers.len());
        let mut materials = HashMap::with_capacity(document.materials().len() + 1);
        materials.insert(None, DEFAULT_MATERIAL.clone());
        let mut textures = HashMap::with_capacity(images.len());

        document
            .scenes()
            .map(|s| {
                Self::load_scene(
                    s,
                    &mut vertexs,
                    &mut materials,
                    &mut textures,
                    &buffers,
                    &images,
                    &memory_allocator,
                )
            })
            .try_collect()
            .log()
    }
    fn load_scene(
        scene: gltf::Scene,
        primitives: &mut HashMap<usize, Arc<Primitive>>,
        materials: &mut HashMap<Option<usize>, Arc<Material>>,
        textures: &mut HashMap<usize, Arc<Texture>>,
        buffers: &[gltf::buffer::Data],
        images: &[gltf::image::Data],
        memory_allocator: &Arc<dyn MemoryAllocator>,
    ) -> Result<Self> {
        log::info!("Loading scene {:?}", scene.name());
        let nodes = scene
            .nodes()
            .map(|n| {
                Self::load_node(
                    n,
                    primitives,
                    materials,
                    textures,
                    buffers,
                    images,
                    memory_allocator,
                    Mat4::IDENTITY,
                )
            })
            .try_collect()?;
        Ok(Self { nodes })
    }
    fn load_node(
        node: gltf::Node,
        primitives: &mut HashMap<usize, Arc<Primitive>>,
        materials: &mut HashMap<Option<usize>, Arc<Material>>,
        textures: &mut HashMap<usize, Arc<Texture>>,
        buffers: &[gltf::buffer::Data],
        images: &[gltf::image::Data],
        memory_allocator: &Arc<dyn MemoryAllocator>,
        transform: Mat4,
    ) -> Result<Node> {
        log::debug!("Loading node {:?}", node.name());
        let children = node
            .children()
            .map(|n| {
                let transform = Mat4::from_cols_array_2d(&n.transform().matrix()) * transform;
                Self::load_node(
                    n,
                    primitives,
                    materials,
                    textures,
                    buffers,
                    images,
                    memory_allocator,
                    transform,
                )
            })
            .try_collect()?;
        let mesh = match node.mesh() {
            None => None,
            Some(m) => Some(Self::load_mesh(
                m,
                primitives,
                materials,
                textures,
                buffers,
                images,
                memory_allocator,
            )?),
        };
        Ok(Node {
            name: node.name().unwrap_or_default().to_string(),
            mesh,
            children,
            transform,
        })
    }
    fn load_mesh(
        mesh: gltf::Mesh,
        primitives: &mut HashMap<usize, Arc<Primitive>>,
        materials: &mut HashMap<Option<usize>, Arc<Material>>,
        textures: &mut HashMap<usize, Arc<Texture>>,
        buffers: &[gltf::buffer::Data],
        images: &[gltf::image::Data],
        memory_allocator: &Arc<dyn MemoryAllocator>,
    ) -> Result<Mesh> {
        log::debug!("Loading mesh {:?}", mesh.name());
        let mesh = Mesh {
            name: mesh.name().unwrap_or_default().to_string(),
            primitives: mesh
                .primitives()
                .map(|p| {
                    Self::load_primitive(
                        p,
                        primitives,
                        materials,
                        textures,
                        buffers,
                        images,
                        memory_allocator,
                    )
                })
                .try_collect()?,
        };
        Ok(mesh)
    }
    fn load_primitive(
        primitive: gltf::Primitive,

        primitives: &mut HashMap<usize, Arc<Primitive>>,
        materials: &mut HashMap<Option<usize>, Arc<Material>>,
        textures: &mut HashMap<usize, Arc<Texture>>,

        buffers: &[gltf::buffer::Data],
        images: &[gltf::image::Data],
        memory_allocator: &Arc<dyn MemoryAllocator>,
    ) -> Result<Arc<Primitive>> {
        use std::collections::hash_map::Entry;
        use std::iter::repeat_n;
        let material = Self::load_materials(
            primitive.material(),
            materials,
            textures,
            images,
            memory_allocator,
        )?;

        match primitives.entry(primitive.index()) {
            Entry::Occupied(e) => Ok(e.get().to_owned()),
            Entry::Vacant(entry) => {
                let reader = primitive.reader(|buffer| Some(&buffers[buffer.index()]));

                let positons = reader.read_positions().ok_or("no positions")?;
                let n = positons.len();

                let normals: Box<dyn ExactSizeIterator<Item = _>> = reader
                    .read_normals()
                    .map(|x| Box::new(x) as _)
                    .unwrap_or(Box::new(repeat_n([0f32, 0., 1.], n)));

                let uv: Box<dyn ExactSizeIterator<Item = _>> = reader
                    .read_tex_coords(0)
                    .map(|x| Box::new(x.into_f32()) as _)
                    .unwrap_or(Box::new(repeat_n([0f32, 0.], n)));

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
                    positons
                        .zip(normals)
                        .zip(uv)
                        .map(|((position, normal), uv)| VertexInput {
                            position,
                            normal,
                            uv,
                        }),
                )?;

                let indices_buffer = Buffer::from_iter(
                    memory_allocator.clone(),
                    BufferCreateInfo {
                        usage: BufferUsage::INDEX_BUFFER,
                        ..Default::default()
                    },
                    AllocationCreateInfo {
                        memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                            | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                        ..Default::default()
                    },
                    reader.read_indices().ok_or("no indices")?.into_u32(),
                )?;
                use gltf::mesh::Mode;
                let topo = match primitive.mode() {
                    Mode::Triangles => PrimitiveTopology::TriangleList,
                    Mode::TriangleStrip => PrimitiveTopology::TriangleStrip,
                    _ => unimplemented!(),
                };
                let vertex = Arc::new(Primitive {
                    vertex_buf: vertex_buffer,
                    indices: indices_buffer,
                    topo,
                    material: material,
                });
                entry.insert_entry(vertex.clone());
                Ok(vertex)
            }
        }
    }
    fn load_materials(
        material: gltf::Material,
        materials: &mut HashMap<Option<usize>, Arc<Material>>,
        textures: &mut HashMap<usize, Arc<Texture>>,
        images: &[gltf::image::Data],
        memory_allocator: &Arc<dyn MemoryAllocator>,
    ) -> Result<Arc<Material>> {
        use std::collections::hash_map::Entry;
        match materials.entry(material.index()) {
            Entry::Occupied(x) => Ok(x.get().clone()),
            Entry::Vacant(entry) => {
                let pbr = material.pbr_metallic_roughness();
                let base_color_factor = pbr.base_color_factor();
                let base_color_texture = match pbr.base_color_texture() {
                    None => None,
                    Some(t) => Some(Self::load_texture(
                        t.texture(),
                        textures,
                        images,
                        memory_allocator,
                    )?),
                };
                let material = Arc::new(Material {
                    base_color_factor,
                    base_color_texture,
                });
                entry.insert(material.clone());
                Ok(material)
            }
        }
    }
    fn load_texture(
        texture: gltf::Texture,
        textures: &mut HashMap<usize, Arc<Texture>>,
        images: &[gltf::image::Data],
        memory_allocator: &Arc<dyn MemoryAllocator>,
    ) -> Result<Arc<Texture>> {
        use std::collections::hash_map::Entry;
        match textures.entry(texture.index()) {
            Entry::Occupied(x) => Ok(x.get().clone()),
            Entry::Vacant(entry) => {
                let data = &images[texture.index()];
                let image_buffer = Arc::new(Buffer::from_iter(
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
                    data.pixels.iter().copied(),
                )?);
                entry.insert(image_buffer.clone());
                Ok(image_buffer)
            }
        }
    }
}

#[derive(
    vulkano::buffer::BufferContents, vulkano::pipeline::graphics::vertex_input::Vertex, Clone, Debug,
)]
#[repr(C)]
pub struct VertexInput {
    #[format(R32G32B32_SFLOAT)]
    pub position: [f32; 3],
    #[format(R32G32B32_SFLOAT)]
    pub normal: [f32; 3],
    #[format(R32G32_SFLOAT)]
    pub uv: [f32; 2],
}

impl VertexInput {
    #[allow(unused)]
    const NONE: Self = unsafe { std::mem::transmute([0xffu8; size_of::<Self>()]) };
}
