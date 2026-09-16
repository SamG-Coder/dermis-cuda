import {GpuRuntime} from '../vendor/cuda-webshader/src/runtime/runtime.js';
import {geometryLayout,DEFAULTS,validateConfig} from './config.js';
import {multiply,lookAt,perspective,ortho,unit,halton} from './math.js';
import {RASTER,PRESENT} from './bridge.js';
import {buildAnatomy} from './anatomy.js';
export class PortraitEngine{
 static async create({canvas=null,human,appearance,anatomy=null,device=null,adapter=null,progress=()=>{},config={}}={}){
  const rt=await GpuRuntime.create({device,adapter,onError:e=>console.error(e)});const engine=new PortraitEngine(rt,canvas);engine.anatomySource=anatomy;try{await engine.init(human,appearance,validateConfig(config),progress);return engine;}catch(e){engine.dispose();throw e;}
 }
 constructor(rt,canvas){this.rt=rt;this.device=rt.device;this.canvas=canvas;this.resources=[];this.textures=[];this.k={};this.samples=0;this.frames=0;this.width=0;this.height=0;this.time=0;this.geometryDirty=true;this.sceneDirty=true;this.disposed=false;this.fatal=null;rt.onError=err=>{this.fatal=err;console.error(err);};}
 buffer(bytes,label,usage=0){const r=this.rt.createBuffer(bytes,{label,usage});this.resources.push(r);return r;}
 texture(width,height,format,label){const t=this.device.createTexture({label,size:[width,height],format,usage:GPUTextureUsage.RENDER_ATTACHMENT|GPUTextureUsage.COPY_SRC});this.textures.push(t);return t;}
 async init(human,appearance,config,progress){
  this.config=config;this.sources={human,appearance};
  for(const entry of ['generateSurface','generateFibers','surfaceNormals','shadePortrait','diffuseSkin','finishPortrait']){progress(`Compiling ${entry}`);this.k[entry]=await this.rt.kernel((entry.startsWith('generate')||entry==='surfaceNormals')?human:human+'\n'+appearance,{entry,workgroupSize:[64,1,1]});}
  if(this.anatomySource)this.k.generateAnatomy=await this.rt.kernel(human,{entry:'generateAnatomy',workgroupSize:[64,1,1]});
  this.settingBuffer=this.buffer(48,'Portrait shape and animation controls');this.sceneBuffer=this.buffer(256,'Camera, studio lighting and appearance',GPUBufferUsage.UNIFORM);
  const module=this.device.createShaderModule({label:'Unlit raster bridge',code:RASTER});
  const vertex={module,entryPoint:'mainVertex'};
  const layout=this.device.createBindGroupLayout({entries:[{binding:0,visibility:GPUShaderStage.VERTEX,buffer:{type:'read-only-storage'}},{binding:1,visibility:GPUShaderStage.VERTEX,buffer:{type:'read-only-storage'}},{binding:2,visibility:GPUShaderStage.VERTEX,buffer:{type:'uniform'}}]});
  this.rasterLayout=layout;const pipelineLayout=this.device.createPipelineLayout({bindGroupLayouts:[layout]});
  this.mainPipeline=await this.device.createRenderPipelineAsync({label:'CUDA surfaces to G-buffer',layout:pipelineLayout,vertex,fragment:{module,entryPoint:'mainFragment',targets:[{format:'rgba32float'},{format:'rgba32float'}]},primitive:{topology:'triangle-list',cullMode:'none'},depthStencil:{format:'depth32float',depthWriteEnabled:true,depthCompare:'less'}});
  this.shadowPipeline=await this.device.createRenderPipelineAsync({label:'CUDA surface shadow projection',layout:pipelineLayout,vertex:{module,entryPoint:'shadowVertex'},fragment:{module,entryPoint:'shadowFragment',targets:[{format:'r32float'}]},primitive:{topology:'triangle-list',cullMode:'none'},depthStencil:{format:'depth32float',depthWriteEnabled:true,depthCompare:'less',depthBias:1,depthBiasSlopeScale:1}});
  if(this.canvas){this.context=this.canvas.getContext('webgpu');if(!this.context)throw Error('The browser could not create a WebGPU canvas.');this.format=navigator.gpu.getPreferredCanvasFormat();this.context.configure({device:this.device,format:this.format,alphaMode:'opaque'});
   const present=this.device.createShaderModule({label:'Display CUDA-computed pixels',code:PRESENT});this.presentPipeline=await this.device.createRenderPipelineAsync({layout:'auto',vertex:{module:present,entryPoint:'vertex'},fragment:{module:present,entryPoint:'fragment',targets:[{format:this.format}]},primitive:{topology:'triangle-list'}});this.presentInfo=this.buffer(16,'Presentation dimensions',GPUBufferUsage.UNIFORM);
  }
  await this.allocateGeometry();this.resize(768,896);progress('Generating the portrait');this.generate();await this.rt.idle();
 }
 async allocateGeometry(){
  if(this.positions){await this.rt.idle();this.rt.destroyBuffer(this.positions);this.rt.destroyBuffer(this.normals);}
  let rest=null;
  if(this.anatomySource){rest=buildAnatomy(this.anatomySource,this.config.quality);this.layout=rest.layout;for(const b of this.anatomyBuffers??[])this.rt.destroyBuffer(b);this.anatomyBuffers=[rest.positions,rest.normals,rest.roots,rest.closure].map((data,i)=>{const b=this.buffer(data.byteLength,'Anatomical rest data '+i);this.rt.write(b,data);return b;});}
  else this.layout=geometryLayout(this.config.quality);this.positions=this.buffer(this.layout.vertices*16,'CUDA generated surface positions');this.normals=this.buffer(this.layout.vertices*16,'CUDA generated normals and hair tangents');
  this.rasterBindings=this.device.createBindGroup({layout:this.rasterLayout,entries:[{binding:0,resource:{buffer:this.positions.gpuBuffer}},{binding:1,resource:{buffer:this.normals.gpuBuffer}},{binding:2,resource:{buffer:this.sceneBuffer.gpuBuffer}}]});
  if(this.anatomySource){this.anatomyGenerators=[];const max=this.device.limits.maxComputeWorkgroupsPerDimension*64;for(let first=0;first<this.layout.vertices;first+=max){const count=Math.min(max,this.layout.vertices-first);this.anatomyGenerators.push({count,binding:this.k.generateAnatomy.bind({settings:this.settingBuffer,rest:this.anatomyBuffers[0],directions:this.anatomyBuffers[1],roots:this.anatomyBuffers[2],closure:this.anatomyBuffers[3],positions:this.positions,normals:this.normals},{first,count})});}this.geometryDirty=true;return;}
  this.generators=this.layout.layout.map(part=>({part,binding:part.fiber?this.k.generateFibers.bind({settings:this.settingBuffer,positions:this.positions,normals:this.normals},{kind:part.kind,strands:part.nu,segments:part.nv,first:part.first,count:part.count}):this.k.generateSurface.bind({settings:this.settingBuffer,positions:this.positions,normals:this.normals},{kind:part.kind,nu:part.nu,nv:part.nv,first:part.first,count:part.count})}));for(const item of this.generators)if(!item.part.fiber){const p=item.part;item.normalBinding=this.k.surfaceNormals.bind({settings:this.settingBuffer,positions:this.positions,normals:this.normals},{kind:p.kind,nu:p.nu,nv:p.nv,first:p.first,count:p.count});}this.geometryDirty=true;
 }
 resize(w,h){
  this.displayAspect=w/h;
  w=Math.max(64,Math.ceil(w/16)*16);h=Math.max(64,Math.round(h));if(this.width===w&&this.height===h){this.reset();return;}
  if(w*h>3840*2160)throw Error('Render target is too large.');
  for(const r of this.frameBuffers??[])this.rt.destroyBuffer(r);for(const t of this.frameTextures??[])t.destroy();
  this.width=w;this.height=h;if(this.canvas){this.canvas.width=w;this.canvas.height=h;}this.frameBuffers=[];this.frameTextures=[];
  const fb=(n,label)=>{const b=this.buffer(n,label);this.frameBuffers.push(b);return b;},ft=(x,y,format,label)=>{const t=this.texture(x,y,format,label);this.frameTextures.push(t);return t;};
  const bytes=w*h*16;
  this.gPosition=ft(w,h,'rgba32float','Position / material attachment');this.gNormal=ft(w,h,'rgba32float','Normal / barycentric attachment');this.depth=ft(w,h,'depth32float','Portrait depth');
  this.screenPositions=fb(bytes,'Raster positions for CUDA shading');this.screenNormals=fb(bytes,'Raster normals for CUDA shading');this.diffuse=fb(bytes,'CUDA diffuse lighting');this.specular=fb(bytes,'CUDA specular lighting');this.scatterX=fb(bytes,'CUDA horizontal skin diffusion');this.scatterY=fb(bytes,'CUDA vertical skin diffusion');this.history=fb(bytes,'Accumulated portrait');this.output=fb(bytes,'Display-ready CUDA colour');
  this.shadowSize=['high','stress','extreme'].includes(this.config.quality)?2048:1024;
  this.shadowColour=ft(this.shadowSize,this.shadowSize,'r32float','Light-space depth values');this.shadowDepth=ft(this.shadowSize,this.shadowSize,'depth32float','Light-space depth test');this.shadowBuffer=fb(this.shadowSize*this.shadowSize*4,'Shadow map for CUDA evaluation');
  this.shade=this.k.shadePortrait.bind({position:this.screenPositions,normal:this.screenNormals,shadow:this.shadowBuffer,scene:this.sceneBuffer,settings:this.settingBuffer,diffuse:this.diffuse,specular:this.specular},{width:w,height:h,shadowSize:this.shadowSize,view:this.config.view});
  this.horizontal=this.k.diffuseSkin.bind({input:this.diffuse,position:this.screenPositions,output:this.scatterX},{width:w,height:h,vertical:0,strength:this.config.scattering,scale:1});
  this.vertical=this.k.diffuseSkin.bind({input:this.scatterX,position:this.screenPositions,output:this.scatterY},{width:w,height:h,vertical:1,strength:this.config.scattering,scale:1});
  this.finish=this.k.finishPortrait.bind({diffuse:this.scatterY,specular:this.specular,history:this.history,output:this.output},{width:w,height:h,samples:0,exposure:this.config.exposure,view:this.config.view});
  if(this.context){this.rt.write(this.presentInfo,new Uint32Array([w,h,0,0]));this.presentGroup=this.device.createBindGroup({layout:this.presentPipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:this.output.gpuBuffer}},{binding:1,resource:{buffer:this.presentInfo.gpuBuffer}}]});}
  this.reset();
 }
 reset(){this.samples=0;this.sceneDirty=true;}
 update(patch){const next=validateConfig({...this.config,...patch});if(next.quality!==this.config.quality)throw Error('Use setQuality() to change geometry allocation.');const keys=['jaw','nose','smile','blink','brow','gazeX','gazeY','hairLength','breeze','animation'];if(keys.some(k=>next[k]!==this.config[k]))this.geometryDirty=true;this.config=next;this.reset();}
 async setQuality(quality){if(quality===this.config.quality)return;await this.rt.idle();this.config=validateConfig({...this.config,quality});await this.allocateGeometry();const w=this.width,h=this.height,aspect=this.displayAspect;this.width=0;this.resize(w,h);this.displayAspect=aspect;}
 settings(time=0){const c=this.config;return new Float32Array([c.jaw,c.nose,c.smile,c.blink,c.gazeX,c.gazeY,c.hairLength,time,c.breeze,c.brow,c.animation?1:0,0]);}
 generate(time=this.time,full=true){this.rt.write(this.settingBuffer,this.settings(time));const batch=this.rt.batch({label:'CUDA anatomical surfaces and individual fibers'});if(this.anatomySource){for(const item of this.anatomyGenerators)batch.dispatch(item.binding,[Math.ceil(item.count/64)]);batch.submit();this.geometryDirty=false;return;}for(const {part,binding,normalBinding} of this.generators){if(full||[1,2,3,4,14,15,16,17].includes(part.kind)||(part.kind===13&&this.config.breeze>0)){const groups=Math.ceil(part.count/64),limit=this.device.limits.maxComputeWorkgroupsPerDimension;batch.dispatch(binding,[Math.min(groups,limit),Math.ceil(groups/limit)]);if(normalBinding)batch.dispatch(normalBinding,[Math.ceil(part.count/64)]);}}batch.submit();this.geometryDirty=false;}
 updateScene(){const c=this.config,a=c.orbitYaw,b=c.orbitPitch,d=c.distance,target=[0,c.targetY,.07],eye=[d*Math.sin(a)*Math.cos(b),target[1]+d*Math.sin(b),target[2]+d*Math.cos(a)*Math.cos(b)];
  const view=lookAt(eye,target),projection=perspective(32,this.displayAspect??this.width/this.height);if(this.samples>0){projection[2]-=(halton(this.samples,2)-.5)*2/this.width;projection[6]-=(halton(this.samples,3)-.5)*2/this.height;}
  const angle=c.lightAngle*Math.PI/180,key=unit([Math.sin(angle)*.9,.66,Math.cos(angle)*.9]);const lp=key.map(x=>x*7);lp[1]+=.1;
  const shadow=multiply(ortho(4.6,.1,16),lookAt(lp,[0,.05,0]));const scene=new Float32Array(64);scene.set(multiply(projection,view),0);scene.set([...eye,0],16);scene.set(shadow,20);scene.set([c.melanin,c.roughness,c.scattering,c.detail],36);scene.set([c.iris,c.hairMelanin,c.hairRoughness,c.exposure],40);scene.set([c.pupil,this.anatomySource?1:0,c.hair?1:0,0],44);scene.set([...key,c.lightPower],48);this.rt.write(this.sceneBuffer,scene);this.lastScene=scene;
 }
 drawObjects(pass){pass.setBindGroup(0,this.rasterBindings);if(this.anatomySource){pass.draw(this.config.hair?this.layout.vertices:this.layout.hairFirst);return;}if(this.config.hair)pass.draw(this.layout.vertices);else{for(const part of this.layout.layout)if(part.kind!==12&&part.kind!==13)pass.draw(part.count,1,part.first);}}
 renderFrame(time=this.time,{motion=false}={}){
  const moving=motion||this.config.animation;
  const count=moving?{preview:2,balanced:4,high:8,stress:8,extreme:8}[this.config.quality]:1;
  if(moving)this.samples=0;
  for(let i=0;i<count;i++)this.render(time,{present:i===count-1,temporalSample:i>0});
  this.frameSamples=count;
 }
 render(time=this.time,{present=true,temporalSample=false}={}){
  if(this.fatal)throw this.fatal;if(this.disposed)throw Error('Renderer disposed.');this.time=time;
  if(this.config.animation&&!temporalSample)this.samples=0;if(this.geometryDirty)this.generate(time);else if(this.config.animation&&!temporalSample)this.generate(time,false);else this.rt.write(this.settingBuffer,this.settings(time));this.updateScene();
  const e=this.device.createCommandEncoder({label:'Surface raster + GPU-only attachment transfers'});
  // Shadow is stable across temporal camera jitter; regenerate only for an authored change / animation.
  if(this.sceneDirty||this.config.animation){const s=e.beginRenderPass({label:'Portrait shadow map',colorAttachments:[{view:this.shadowColour.createView(),clearValue:{r:1,g:0,b:0,a:0},loadOp:'clear',storeOp:'store'}],depthStencilAttachment:{view:this.shadowDepth.createView(),depthClearValue:1,depthLoadOp:'clear',depthStoreOp:'store'}});s.setPipeline(this.shadowPipeline);this.drawObjects(s);s.end();e.copyTextureToBuffer({texture:this.shadowColour},{buffer:this.shadowBuffer.gpuBuffer,bytesPerRow:this.shadowSize*4},[this.shadowSize,this.shadowSize]);}
  const p=e.beginRenderPass({label:'Portrait geometry buffers',colorAttachments:[this.gPosition,this.gNormal].map(t=>({view:t.createView(),clearValue:{r:0,g:0,b:0,a:0},loadOp:'clear',storeOp:'store'})),depthStencilAttachment:{view:this.depth.createView(),depthClearValue:1,depthLoadOp:'clear',depthStoreOp:'store'}});p.setPipeline(this.mainPipeline);this.drawObjects(p);p.end();
  e.copyTextureToBuffer({texture:this.gPosition},{buffer:this.screenPositions.gpuBuffer,bytesPerRow:this.width*16},[this.width,this.height]);e.copyTextureToBuffer({texture:this.gNormal},{buffer:this.screenNormals.gpuBuffer,bytesPerRow:this.width*16},[this.width,this.height]);this.device.queue.submit([e.finish()]);
  const scale=Math.max(.65,Math.min(2.5,this.height/850*6.4/this.config.distance));const groups=[Math.ceil(this.width*this.height/64)];const batch=this.rt.batch({label:'CUDA skin / iris / hair shading and diffusion'});
  batch.dispatch(this.shade.setScalars({view:this.config.view}),groups).dispatch(this.horizontal.setScalars({strength:this.config.scattering,scale}),groups).dispatch(this.vertical.setScalars({strength:this.config.scattering,scale}),groups).dispatch(this.finish.setScalars({samples:this.samples,exposure:this.config.exposure,view:this.config.view}),groups).submit();
  if(present&&this.context){const q=this.device.createCommandEncoder();const pass=q.beginRenderPass({colorAttachments:[{view:this.context.getCurrentTexture().createView(),loadOp:'clear',storeOp:'store',clearValue:{r:0,g:0,b:0,a:1}}]});pass.setPipeline(this.presentPipeline);pass.setBindGroup(0,this.presentGroup);pass.draw(3);pass.end();this.device.queue.submit([q.finish()]);}
  this.samples++;this.frames++;this.sceneDirty=false;
 }
 async pixels(){return this.rt.read(this.output,Float32Array);}
 stats(){return {width:this.width,height:this.height,triangles:this.layout.triangles,vertices:this.layout.vertices,strands:this.layout.strands,scalpStrands:this.layout.scalpStrands,samples:this.samples,frames:this.frames,backend:this.rt.describe(),readbackBytes:this.rt.stats.readbackBytes,bufferMiB:[...this.rt.buffers].reduce((n,r)=>n+r.size,0)/1048576};}
 dispose(){if(this.disposed)return;this.disposed=true;for(const t of this.textures)t.destroy();this.context?.unconfigure();this.rt.dispose();}
}
