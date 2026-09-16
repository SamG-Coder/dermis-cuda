// This WGSL is deliberately only an unlit raster/presentation bridge.
// The .cu files own shape generation, face materials, lighting and diffusion.
export const RASTER=`
struct Frame{r:array<vec4<f32>,16>};
@group(0) @binding(0) var<storage,read> positions:array<vec4<f32>>;
@group(0) @binding(1) var<storage,read> normals:array<vec4<f32>>;
@group(0) @binding(2) var<uniform> frame:Frame;
struct Vertex{@builtin(position) clip:vec4<f32>,@location(0) p:vec3<f32>,@location(1) n:vec3<f32>,@location(2) @interpolate(flat) material:f32,@location(3) bary:vec3<f32>};
fn project(p:vec3<f32>,first:u32)->vec4<f32>{let q=vec4<f32>(p,1);return vec4<f32>(dot(frame.r[first],q),dot(frame.r[first+1],q),dot(frame.r[first+2],q),dot(frame.r[first+3],q));}
@vertex fn mainVertex(@builtin(vertex_index) i:u32)->Vertex{var o:Vertex;let p=positions[i];o.clip=project(p.xyz,0u);o.p=p.xyz;o.n=normals[i].xyz;o.material=p.w;o.bary=vec3<f32>(0);o.bary[i%3u]=1;return o;}
struct GBuffer{@location(0) p:vec4<f32>,@location(1) n:vec4<f32>};
@fragment fn mainFragment(v:Vertex)->GBuffer{var o:GBuffer;o.p=vec4<f32>(v.p,v.material);o.n=vec4<f32>(normalize(v.n),min(v.bary.x,min(v.bary.y,v.bary.z)));return o;}
@vertex fn shadowVertex(@builtin(vertex_index) i:u32)->@builtin(position) vec4<f32>{return project(positions[i].xyz,5u);}
@fragment fn shadowFragment(@builtin(position) p:vec4<f32>)->@location(0) f32{return p.z;}
`;
export const PRESENT=`
@group(0) @binding(0) var<storage,read> pixels:array<vec4<f32>>;
struct Info{size:vec4<u32>}; @group(0) @binding(1) var<uniform> info:Info;
@vertex fn vertex(@builtin(vertex_index) i:u32)->@builtin(position) vec4<f32>{let x=f32((i<<1u)&2u);let y=f32(i&2u);return vec4<f32>(x*2-1,1-y*2,0,1);}
@fragment fn fragment(@builtin(position) p:vec4<f32>)->@location(0) vec4<f32>{let xy=min(vec2<u32>(p.xy),info.size.xy-vec2<u32>(1));return pixels[xy.y*info.size.x+xy.x];}
`;
