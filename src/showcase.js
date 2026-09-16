import {PortraitEngine} from './engine.js';
const film=document.querySelector('#film'),ctx=film.getContext('2d',{willReadFrequently:true}),button=document.querySelector('#start'),status=document.querySelector('#status');
const schedule=callback=>globalThis.scheduler?.postTask?globalThis.scheduler.postTask(callback):setTimeout(callback,0);
const chapters=[
 ['THE HUMAN SURFACE','A live computational portrait.','100,000 scalp strands. Living portrait enabled.'],
 ['BUILT FROM ANATOMY','MakeHuman CC0 provides the anatomical base.','The rest mesh is subdivided; CUDA deforms the face.'],
 ['MAXIMUM FIBER LENGTH','Length 1.80 / breeze 1.00 / pigment 1.00','Individual strand geometry follows the scalp.'],
 ['A LIVING PORTRAIT','Blinking and subtle gaze motion are evaluated live.','Eight spatial samples share the same moving pose.'],
 ['SKIN / MICROSTRUCTURE','Procedural pores, pigment and regional roughness.','realism.cu evaluates the material in object space.'],
 ['EYES / IRIS DETAIL','Hazel iris, pupil radius 0.33.','Iris fibers, crypts, outer ring and sclera variation.'],
 ['FORM / EXPRESSION','Smile, brow and gaze deform in real time.','Authored eye-closure targets drive the eyelids.'],
 ['UNDER THE SURFACE','Surface normals expose the shape and pore detail.','The same live geometry; a diagnostic shading view.'],
 ['MATERIAL / ALBEDO','Base colour without the studio illumination.','No photographic skin or eye textures are used.'],
 ['LIGHT / REFLECTION','Specular-only view isolates material highlights.','Skin roughness and strand direction shape reflections.'],
 ['CUDA TO THE BROWSER','CUDA WebShader compiles the .cu source to WGSL.','WebGPU runs compute, projects geometry and presents pixels.'],
 ['EXPLORE THE SOURCE','Public code. Automated checks. GitHub Pages.','SamG-Coder / dermis-cuda    •    WebGPU required.']
];
function text(s,x,y,size=24,color='#d8e3e0'){ctx.fillStyle=color;ctx.font=`${size>=40?600:400} ${size}px system-ui`;ctx.fillText(s,x,y);}
ctx.fillStyle='#0a111b';ctx.fillRect(0,0,1920,1080);text('DERMIS',88,200,92);text('A live portrait, authored in CUDA.',90,268,32);text('Record the real renderer at Extreme quality.',90,340);
button.onclick=async()=>{button.disabled=true;status.textContent='Compiling shaders and allocating Extreme geometry…';let e,recorder,failure='';
try{
 const load=async p=>{const r=await fetch(p);if(!r.ok)throw Error(p+': '+r.status);return r.text();};
 const [human,realism,appearance,asset]=await Promise.all(['kernels/human.cu','kernels/realism.cu','kernels/appearance.cu','assets/male-anatomy.json'].map(load));
 e=await PortraitEngine.create({human,appearance:realism+'\n'+appearance,anatomy:JSON.parse(asset),config:{quality:'extreme',melanin:0,roughness:.57,scattering:.55,detail:.8,iris:2,pupil:.33,hairMelanin:1,hairRoughness:1,hairLength:1.8,breeze:1,animation:true,distance:6,targetY:.15}});
 e.resize(1920,1080);e.renderFrame(0);await e.rt.idle();
 // Pack display pixels on the GPU, then reuse a small readback buffer for capture.
 const device=e.device,bytes=1920*1080*4;
 const packed=device.createBuffer({size:bytes,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC});
 const staging=device.createBuffer({size:bytes,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
 const module=device.createShaderModule({code:'@group(0) @binding(0) var<storage,read> src:array<vec4f>; @group(0) @binding(1) var<storage,read_write> dst:array<u32>; @compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id:vec3u){if(id.x<arrayLength(&dst)){dst[id.x]=pack4x8unorm(src[id.x]);}}'});
 const pipeline=await device.createComputePipelineAsync({layout:'auto',compute:{module,entryPoint:'main'}});
 const bind=device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:e.output.gpuBuffer}},{binding:1,resource:{buffer:packed}}]});
 const chunks=[],stream=film.captureStream(30),mime=['video/webm;codecs=vp8','video/webm;codecs=vp9'].find(x=>MediaRecorder.isTypeSupported(x));
 recorder=new MediaRecorder(stream,{mimeType:mime,videoBitsPerSecond:14000000});recorder.ondataavailable=event=>{if(event.data.size)chunks.push(event.data);};
 recorder.onstop=()=>{const url=URL.createObjectURL(new Blob(chunks,{type:mime})),save=document.querySelector('#save');save.href=url;save.download=failure?'DERMIS-incomplete-recording.webm':'DERMIS-live-showcase.webm';save.hidden=false;if(!failure)save.click();stream.getTracks().forEach(t=>t.stop());status.textContent=failure?'Recording stopped: '+failure:'Complete — 120 seconds of live rendering recorded.';button.disabled=false;packed.destroy();staging.destroy();e.dispose();};
 recorder.start(1000);const start=performance.now();let frames=0,last=0,frameCount=0,fps=0;
 const frame=async()=>{try{
 const t=(performance.now()-start)/1000;if(t>=120){recorder.stop();return;}
 const chapter=Math.min(11,Math.floor(t/10)),s=(t%10)/10;
 let yaw=.35+.24*Math.sin(t*.18),pitch=.035,distance=6,targetY=.18,view=0;
 if(chapter===1){yaw=.65+s*.85;distance=5.7;}
 if(chapter===2){yaw=1.25-s*.9;pitch=.32;distance=4.6;targetY=.6;}
 if(chapter===3){yaw=.15;distance=4.6;targetY=.3;}
 if(chapter===4){yaw=.22;distance=3.4;targetY=.12;}
 if(chapter===5){yaw=.05;distance=2.6;targetY=.335;}
 if(chapter===6){yaw=.3;distance=4.7;}
 if(chapter===7)view=2;if(chapter===8)view=1;if(chapter===9)view=5;
 if(chapter===10){yaw=-.8+s*1.6;distance=5.8;}
 e.update({orbitYaw:yaw,orbitPitch:pitch,distance,targetY,view,smile:chapter===6?.55*(.5+.5*Math.sin(s*Math.PI*2)):.08,brow:chapter===6?.4*Math.sin(s*Math.PI*2):0});
 e.renderFrame(t);if(e.fatal)throw e.fatal;
 const encoder=device.createCommandEncoder(),pass=encoder.beginComputePass();pass.setPipeline(pipeline);pass.setBindGroup(0,bind);pass.dispatchWorkgroups(Math.ceil(1920*1080/64));pass.end();encoder.copyBufferToBuffer(packed,0,staging,0,bytes);device.queue.submit([encoder.finish()]);
 await staging.mapAsync(GPUMapMode.READ);
 ctx.putImageData(new ImageData(new Uint8ClampedArray(staging.getMappedRange()),1920,1080),0,0);staging.unmap();
 const gradient=ctx.createLinearGradient(0,0,690,0);gradient.addColorStop(0,'rgba(6,12,20,.97)');gradient.addColorStop(1,'rgba(6,12,20,0)');ctx.fillStyle=gradient;ctx.fillRect(0,0,690,1080);
 ctx.fillStyle='rgba(6,12,20,.9)';ctx.fillRect(0,884,1920,196);
 text('DERMIS',64,90,42);text('CUDA HUMAN LABORATORY',66,125,16,'#9facb4');
 text(`${String(chapter+1).padStart(2,'0')} / 12`,65,242,20,'#b9d4bd');text(chapters[chapter][0],65,292,27);
 text('EXTREME',65,360,18,'#b9d4bd');text('100,000 scalp strands',65,394,22);text('Length 1.80 · Breeze 1.00',65,434,20);text('Living portrait · ON',65,472,20);
 text(chapters[chapter][1],65,950,30);text(chapters[chapter][2],65,997,24,'#a9b9c1');
 frameCount++;frames++;if(t-last>=1){fps=frameCount/(t-last);last=t;frameCount=0;}
 text(`LIVE GPU CAPTURE  /  ${fps.toFixed(0)} FPS  /  1920 × 1080  /  8 samples`,1120,64,17,'#b9d4bd');
 ctx.fillStyle='#a7c4a9';ctx.fillRect(0,1075,1920*t/120,5);
 status.textContent=`Recording ${Math.floor(t)} / 120 seconds · ${chapters[chapter][0]}`;
 schedule(frame);
 }catch(error){failure=error.message;recorder.stop();}};
 schedule(frame);
}catch(error){status.textContent='Unable to record: '+error.message;button.disabled=false;e?.dispose();}};

