import {readFile,writeFile,mkdir} from 'node:fs/promises';
import {nativeGpu} from './native.mjs';import {PortraitEngine} from '../src/engine.js';import {DEFAULTS} from '../src/config.js';import {writePNG} from './png.mjs';
const checks=[];let engine;
function check(name,value,extra={}){const result={name,passed:!!value,...extra};checks.push(result);console.log((value?'PASS ':'FAIL ')+name);if(!value)throw Error(name);}
const mae=(a,b,mask=null)=>{let s=0,n=0;for(let i=0;i<a.length;i+=4)if(!mask||mask(i/4)){for(let c=0;c<3;c++){s+=Math.abs(a[i+c]-b[i+c]);n++;}}return s/Math.max(n,1);};
try{
 await nativeGpu();const human=await readFile(new URL('../kernels/human.cu',import.meta.url),'utf8'),appearance=(await readFile(new URL('../kernels/realism.cu',import.meta.url),'utf8'))+'\n'+(await readFile(new URL('../kernels/appearance.cu',import.meta.url),'utf8'));engine=await PortraitEngine.create({human,appearance,config:{quality:'preview'},progress:console.log});engine.resize(256,320);
 check('All six CUDA kernels create native WebGPU compute pipelines',Object.keys(engine.k).length===6);
 const vertices=await engine.rt.read(engine.positions);check('Complete Preview surface and hair position buffer is finite',vertices.every(Number.isFinite));
 let outOfBounds=0;const materials=new Set();for(let i=0;i<vertices.length;i+=4){if(Math.abs(vertices[i])>3||vertices[i+1]<-3||vertices[i+1]>2.2||Math.abs(vertices[i+2])>2)outOfBounds++;materials.add(Math.floor(vertices[i+3]));}
 check('Complete generated mesh has plausible bounded dimensions',outOfBounds===0);check('Geometry contains skin, lips, both eyes, scalp, hair, brows, lashes and clothing',[1,2,3,4,7,8,9,10,11,13].every(x=>materials.has(x)));
 const normals=await engine.rt.read(engine.normals);check('Complete generated normal/tangent buffer is finite',normals.every(Number.isFinite));let normalized=0;for(let i=0;i<normals.length;i+=4){const len=Math.hypot(normals[i],normals[i+1],normals[i+2]);if(Math.abs(len-1)<.015)normalized++;}check('More than 99% of normals/tangents are unit length',normalized/(normals.length/4)>.99,{ratio:normalized/(normals.length/4)});
 const readBefore=engine.rt.stats.readbackBytes;engine.render();await engine.rt.idle();check('Ordinary render path has zero GPU-to-CPU readback',engine.rt.stats.readbackBytes===readBefore);
 let base=await engine.pixels();check('Actual raster + CUDA shading produces finite bounded pixels',base.every(x=>Number.isFinite(x)&&x>=0&&x<=1));
 const gp=await engine.rt.read(engine.screenPositions);const ids=Array.from({length:256*320},(_,i)=>Math.floor(gp[i*4+3]));const skinMask=i=>ids[i]===1,eyeMask=i=>ids[i]===3||ids[i]===4,hairMask=i=>ids[i]===10;
 check('Rasterized image includes a face rather than only a background',ids.filter(x=>x===1).length>10000);check('Both actual eye surfaces are visible',ids.includes(3)&&ids.includes(4));check('CUDA-generated strand hair reaches the render target',ids.filter(x=>x===10).length>100);
 engine.reset();engine.render();await engine.rt.idle();const repeat=await engine.pixels();check('Resetting accumulation reproduces the same static image',mae(base,repeat)<.000002);
 const render=async patch=>{engine.update(patch);engine.render();await engine.rt.idle();if(engine.fatal)throw engine.fatal;return engine.pixels();};
 const albedo=await render({view:1});check('Albedo inspection differs from lit beauty',mae(base,albedo)>.025);
 const dark=await render({melanin:.9});check('Skin pigment affects CUDA-evaluated skin colour',mae(albedo,dark,skinMask)>.09);
 engine.update({melanin:DEFAULTS.melanin});const iris0=await render({iris:0});const iris3=await render({iris:3});check('Changing iris pigment changes actual eye pixels',mae(iris0,iris3,eyeMask)>.003);
 const pupilSmall=await render({pupil:.22});const pupilLarge=await render({pupil:.52});check('Pupil radius changes actual eye pixels',mae(pupilSmall,pupilLarge,eyeMask)>.005);
 const hairDark=await render({hairMelanin:1});const hairLight=await render({hairMelanin:0});check('Hair pigmentation changes individual strand shading',mae(hairDark,hairLight,hairMask)>.05);
 await render({...DEFAULTS,quality:'preview',view:0,scattering:0});const unscattered=await engine.pixels();const scattered=await render({scattering:1});check('Subsurface diffusion changes lit skin pixels',mae(unscattered,scattered,skinMask)>.0001);
 const specSmooth=await render({view:5,roughness:.28});const specRough=await render({roughness:.65});check('Skin roughness changes the computed specular response',mae(specSmooth,specRough,skinMask)>.001);
 const shadow1=await render({view:3,lightAngle:-50});const shadow2=await render({lightAngle:50});check('Moving the key light changes real surface shadows',mae(shadow1,shadow2,skinMask)>.01);
 const eyePart=engine.layout.layout.find(p=>p.kind===1);const openGeometry=await engine.rt.read(engine.positions,Float32Array,eyePart.count*16,eyePart.first*16);
 await render({...DEFAULTS,quality:'preview',blink:.98});const closedGeometry=await engine.rt.read(engine.positions,Float32Array,eyePart.count*16,eyePart.first*16);
 const span=a=>{let lo=Infinity,hi=-Infinity;for(let i=0;i<a.length;i+=4){lo=Math.min(lo,a[i+1]);hi=Math.max(hi,a[i+1]);}return hi-lo;};check('Blink moves real eyelid / eye geometry',span(closedGeometry)<span(openGeometry)*.5);
 const noHair=await render({...DEFAULTS,quality:'preview',hair:false});check('Scalp-hair visibility affects the rendered image',mae(base,noHair)>.005);
 const statsBefore=engine.rt.stats.pipelineCompiles;await render({exposure:.8});check('Material parameter edits do not recompile CUDA',engine.rt.stats.pipelineCompiles===statsBefore);
 const shapeBefore=engine.rt.stats.dispatches;await render({animation:true});engine.render(3.7);await engine.rt.idle();check('Live portrait dispatches CUDA eye and lid animation',engine.rt.stats.dispatches>shapeBefore+5);
 engine.update({...DEFAULTS,quality:'preview',animation:false});await engine.setQuality('high');engine.render();await engine.rt.idle();const high=await engine.pixels();check('High geometry with 22,000 scalp strands actually allocates and renders',engine.layout.scalpStrands===22000&&high.every(Number.isFinite));
 await engine.setQuality('balanced');engine.update({...DEFAULTS,quality:'balanced'});engine.resize(768,960);
 for(let i=0;i<8;i++){const start=performance.now();engine.render();await engine.rt.idle();console.log('Portrait sample',i+1,Math.round(performance.now()-start)+' ms (software WebGPU)');}
 const portrait=await engine.pixels();await writePNG(new URL('../docs/portrait.png',import.meta.url),portrait,engine.width,engine.height);check('Balanced portrait captured from the actual CUDA output buffer',portrait.every(Number.isFinite));
 const portraitStats=engine.stats();
 engine.update({orbitYaw:.58,orbitPitch:.025,distance:6.4,targetY:.12});for(let i=0;i<4;i++){engine.render();await engine.rt.idle();}await writePNG(new URL('../docs/three-quarter.png',import.meta.url),await engine.pixels(),engine.width,engine.height);
 engine.update({orbitYaw:0,orbitPitch:0,distance:2.5,targetY:.335});engine.resize(960,480);for(let i=0;i<4;i++){engine.render();await engine.rt.idle();}await writePNG(new URL('../docs/eyes.png',import.meta.url),await engine.pixels(),engine.width,engine.height);
 check('Close-up and three-quarter views render without GPU validation errors',!engine.fatal);
 await writeFile(new URL('../docs/gpu-validation.json',import.meta.url),JSON.stringify({date:new Date().toISOString(),passed:checks.filter(x=>x.passed).length,total:checks.length,backend:engine.rt.describe(),portrait:portraitStats,checks,limitations:['Dawn/SwiftShader is a software WebGPU adapter, not physical NVIDIA hardware.','All readbacks above were explicit validation/image captures; the normal render loop has none.','Native WebGPU source/pipeline validation is not an end-to-end browser presentation test.','No native nvcc/CUDA backend benchmark was performed.']},null,2));
}catch(e){console.error(e);await writeFile(new URL('../docs/gpu-validation.json',import.meta.url),JSON.stringify({passed:checks.filter(x=>x.passed).length,total:checks.length,checks,error:e.message},null,2));process.exitCode=1;}finally{engine?.dispose();}
process.exit(process.exitCode||0);
