import test from 'node:test';import assert from 'node:assert/strict';import {readFile} from 'node:fs/promises';
import {compile} from '../vendor/cuda-webshader/src/compiler/compiler.js';import {executeCPU} from '../vendor/cuda-webshader/src/compiler/cpu-oracle.js';import {DEFAULTS,validateConfig,geometryLayout,RANGES} from '../src/config.js';import {multiply,lookAt,perspective,ortho,halton} from '../src/math.js';
const human=await readFile(new URL('../kernels/human.cu',import.meta.url),'utf8'),appearance=(await readFile(new URL('../kernels/realism.cu',import.meta.url),'utf8'))+'\n'+(await readFile(new URL('../kernels/appearance.cu',import.meta.url),'utf8'));
for(const entry of ['generateAnatomy','generateSurface','generateFibers','surfaceNormals','shadePortrait','diffuseSkin','finishPortrait'])test('CUDA compiles: '+entry,()=>{const c=compile((entry.startsWith('generate')||entry==='surfaceNormals')?human:human+'\n'+appearance,{entry,workgroupSize:[64,1,1]});assert.ok(c.wgsl.includes('@compute'));assert.ok(c.metadata.bindings.length<=8);});
test('Default look validates and survives a JSON round trip',()=>assert.deepEqual(validateConfig(JSON.parse(JSON.stringify(DEFAULTS))),DEFAULTS));
test('Settings reject NaN, Infinity and non-numeric controls',()=>{for(const v of [NaN,Infinity,'0.3',null])assert.throws(()=>validateConfig({melanin:v}));});
test('Every numeric setting enforces declared bounds',()=>{for(const [k,[min,max]]of Object.entries(RANGES)){assert.throws(()=>validateConfig({[k]:min-1}));assert.throws(()=>validateConfig({[k]:max+1}));}});
test('Mode selections must be integer IDs',()=>{assert.throws(()=>validateConfig({iris:1.2}));assert.throws(()=>validateConfig({view:3.1}));});
test('Boolean settings cannot smuggle truthy strings',()=>{assert.throws(()=>validateConfig({hair:'false'}));assert.throws(()=>validateConfig({animation:1}));});
test('Unknown look fields do not become executable configuration',()=>{const c=validateConfig({source:'alert(1)',__proto__:{injected:true}});assert.equal(c.source,undefined);assert.equal(c.injected,undefined);});
test('Unknown geometry presets fail explicitly',()=>{assert.throws(()=>geometryLayout('ultra'));assert.throws(()=>validateConfig({quality:'fake'}));});
for(const q of ['preview','balanced','high','stress','extreme'])test(q+': mesh ranges fit binding and dispatch limits',()=>{const a=geometryLayout(q);let next=0;for(const p of a.layout){assert.equal(p.first,next);assert.equal(p.count%3,0);assert.ok(Math.ceil(p.count/64)<=65535*65535);next+=p.count;}assert.equal(a.vertices,next);assert.equal(a.triangles*3,next);assert.ok(next*16<256*1048576);assert.ok(a.strands>a.scalpStrands);});
test('Quality increases real triangles and actual scalp strand counts',()=>{const a=geometryLayout('preview'),b=geometryLayout('balanced'),c=geometryLayout('high');assert.ok(a.vertices<b.vertices&&b.vertices<c.vertices);assert.deepEqual([a.scalpStrands,b.scalpStrands,c.scalpStrands],[4500,12000,22000]);});
test('View matrix projects its eye to the origin',()=>{const e=[2,3,5],v=lookAt(e,[0,0,0]);for(let i=0;i<3;i++)assert.ok(Math.abs(v[i*4]*e[0]+v[i*4+1]*e[1]+v[i*4+2]*e[2]+v[i*4+3])<1e-6);});
test('Projection and shadow matrices are finite',()=>{assert.ok(multiply(perspective(32,1.4),lookAt([1,2,6],[0,0,0])).every(Number.isFinite));assert.ok(ortho(4.6).every(Number.isFinite));});
test('Temporal camera jitter is bounded and non-repeating in first 24 samples',()=>{const samples=Array.from({length:24},(_,i)=>[halton(i+1,2),halton(i+1,3)]);assert.ok(samples.flat().every(x=>x>=0&&x<1));assert.equal(new Set(samples.map(String)).size,24);});
const settings=new Float32Array([1,1,0,0,0,0,1,0,0,0,0,0]);
const probeSource=human+'\n'+appearance+`\n__global__ void referenceProbe(const float4* settings,float4* out){unsigned int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=16u)return;float3 p=make_float3(0.0f,0.0f,0.0f);if(i<4u){p=eyePoint(.5f,float(i%2u),1.0f,i>=2u?.98f:0.0f,0.0f,0.0f);}if(i==4u)p=headPoint(.15f,.65f,1.0f,0.0f,1.0f);if(i==5u)p=headPoint(.85f,.65f,1.0f,0.0f,1.0f);if(i==6u)p=skinAlbedo(make_float3(.3f,0.0f,.62f),0.0f,1.0f);if(i==7u)p=skinAlbedo(make_float3(.3f,0.0f,.62f),1.0f,1.0f);if(i==8u)p=fiberCenter(73u,.8f,1.0f,1.0f,0.0f);if(i==9u)p=fiberCenter(73u,.8f,1.0f,2.0f,0.0f);if(i==10u)p=fiberCenter(73u,.8f,1.0f,2.0f,1.0f);if(i==11u)p=poreField(.31f,.18f);if(i==12u){float4 a=poseA(settings);p=make_float3(a.w,a.z,1.0f);}if(i==13u)p=eyeAlbedo(make_float3(.325f,.335f,.65f),1.0f,0.0f,0.0f,1,.34f);if(i==14u)p=make_float3(frontZ(0.0f,-.08f,1.0f,0.0f,.8f),frontZ(0.0f,-.08f,1.0f,0.0f,1.2f),0.0f);if(i==15u)p=make_float3(rnd(91u),rnd(91u),rnd(92u));out[i]=make_float4(p.x,p.y,p.z,1.0f);}`;
const probe=compile(probeSource,{entry:'referenceProbe',workgroupSize:[64,1,1]});function values(s=settings){const out=new Float32Array(64);executeCPU(probe,{settings:s,out},{},[1]);return Array.from({length:16},(_,i)=>Array.from(out.slice(i*4,i*4+4)));}
const results=values();
test('Fiber generation uses both grid dimensions without repeating or dropping strands',()=>{
 const art=compile(human,{entry:'generateFibers',workgroupSize:[64,1,1]}),args={kind:13,strands:3,segments:8,first:0,count:128};
 const run=grid=>{const positions=new Float32Array(512),normals=new Float32Array(512);executeCPU(art,{settings,positions,normals},args,grid);return positions;};
 assert.deepEqual(run([2,1]),run([1,2]));
 assert.equal(geometryLayout('extreme').scalpStrands,100000);
});
test('Material detail fades below pixel size and preserves normals when disabled',()=>{
 const art=compile(human+'\n'+appearance+`\n__global__ void detailProbe(float4* out){
 unsigned int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=64u)return;
 float3 p=make_float3(float(i)*.011f,.12f,.6f),n=make_float3(0.0f,0.0f,1.0f);
 float3 off=skinDetailNormal(p,n,0.0f,0.0f),far=skinDetailNormal(p,n,2.0f,.1f),near=skinDetailNormal(p,n,2.0f,0.0f);
 out[i*2u]=make_float4(dot3(sub(off,n),sub(off,n)),dot3(sub(far,n),sub(far,n)),dot3(near,near),skinDetailRoughness(p,.57f,2.0f));
 out[i*2u+1u]=make_float4(detailFilter(155.0f,0.0f),detailFilter(155.0f,.1f),near.x,near.y);
 }`,{entry:'detailProbe',workgroupSize:[64,1,1]});
 const out=new Float32Array(512);executeCPU(art,{out},{},[1]);assert.ok(out.every(Number.isFinite));
 for(let i=0;i<64;i++){const k=i*8;assert.ok(out[k]<1e-10&&out[k+1]<1e-10);assert.ok(Math.abs(out[k+2]-1)<1e-5);assert.ok(out[k+3]>=.24&&out[k+3]<=.78);assert.equal(out[k+4],1);assert.equal(out[k+5],0);}
 assert.ok(out.some((v,i)=>i%8===6&&Math.abs(v)>.001));
});
test('Iris polar seam is continuous for all colours and remains finite at distance',()=>{
 const art=compile(human+'\n'+appearance+`\n__global__ void irisProbe(float4* out){
 unsigned int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=8u)return;
 float3 a=eyeAlbedoFiltered(make_float3(.27f,.335001f,.6f),1.0f,0.0f,0.0f,int(i%4u),.34f,i<4u?0.0f:.02f);
 float3 b=eyeAlbedoFiltered(make_float3(.27f,.334999f,.6f),1.0f,0.0f,0.0f,int(i%4u),.34f,i<4u?0.0f:.02f);
 out[i]=make_float4(a.x,a.y,a.z,dot3(sub(a,b),sub(a,b)));
 }`,{entry:'irisProbe',workgroupSize:[64,1,1]});
 const out=new Float32Array(32);executeCPU(art,{out},{},[1]);assert.ok(out.every(Number.isFinite));
 for(let i=0;i<8;i++){assert.ok(out[i*4+3]<1e-5);assert.ok(out.slice(i*4,i*4+3).every(v=>v>=0&&v<=1));}
});
test('Anatomy and material reference samples remain finite',()=>assert.ok(results.flat().every(Number.isFinite)));
test('Closing the eye reduces the actual aperture height, not just colour',()=>{const open=results[1][1]-results[0][1],closed=results[3][1]-results[2][1];assert.ok(open>.08);assert.ok(closed>0&&closed<open*.04);});
test('Neutral head has coherent left/right surface proportions',()=>{assert.ok(Math.abs(results[4][0]+results[5][0])<1e-5);assert.ok(Math.abs(results[4][1]-results[5][1])<1e-6);assert.ok(Math.abs(results[4][2]-results[5][2])<1e-5);});
test('Increasing skin pigment reduces computed linear reflectance',()=>{for(let i=0;i<3;i++)assert.ok(results[6][i]>results[7][i]);});
test('Hair at zero breeze is independent of time',()=>assert.deepEqual(results[8],results[9]));
test('Breeze changes actual strand geometry',()=>assert.notDeepEqual(results[9],results[10]));
test('The iris centre is a dark pupil',()=>assert.ok(results[13].slice(0,3).every(x=>x<.01)));
test('Nose projection is an actual geometry parameter',()=>assert.ok(results[14][1]>results[14][0]+.05));
test('Hair and microdetail seeds are reproducible',()=>{assert.equal(results[15][0],results[15][1]);assert.notEqual(results[15][0],results[15][2]);});
test('Autonomous blink is evaluated by CUDA pose code',()=>{const a=settings.slice();a[7]=3.7;a[10]=1;assert.ok(values(a)[12][0]>.96);a[7]=1;assert.ok(values(a)[12][0]<.001);});
test('Padded geometry lanes do not write past the requested range',()=>{const art=compile(human,{entry:'generateSurface',workgroupSize:[64,1,1]}),positions=new Float32Array(40).fill(999),normals=new Float32Array(40).fill(999);executeCPU(art,{settings,positions,normals},{kind:1,nu:2,nv:2,first:2,count:3},[1]);assert.ok(positions.slice(0,8).every(x=>x===999));assert.ok(positions.slice(20).every(x=>x===999));assert.ok(positions.slice(8,20).every(Number.isFinite));});
test('Zero diffusion is exactly an identity operation',()=>{const art=compile(human+'\n'+appearance,{entry:'diffuseSkin',workgroupSize:[64,1,1]}),input=new Float32Array([.3,.2,.1,1,.2,.4,.6,1]),position=new Float32Array(8),output=new Float32Array(8);executeCPU(art,{input,position,output},{width:2,height:1,vertical:0,strength:0,scale:1},[1]);assert.deepEqual(output,input);});
test('Diffusion preserves a uniform skin field',()=>{const art=compile(human+'\n'+appearance,{entry:'diffuseSkin',workgroupSize:[64,1,1]}),input=new Float32Array(16),position=new Float32Array(16),output=new Float32Array(16);for(let i=0;i<4;i++)input.set([.3,.2,.1,1],i*4);executeCPU(art,{input,position,output},{width:4,height:1,vertical:0,strength:1,scale:1},[1]);for(let i=0;i<16;i++)assert.ok(Math.abs(input[i]-output[i])<1e-6);});
// Host state tests: the source methods, without claiming a GPU/render playthrough.
const {PortraitEngine}=await import('../src/engine.js');
test('Stopping animation schedules restoration of the authored facial pose',()=>{const e=Object.create(PortraitEngine.prototype);e.config={...DEFAULTS,animation:true};e.geometryDirty=false;e.samples=7;e.update({animation:false});assert.equal(e.geometryDirty,true);assert.equal(e.samples,0);assert.equal(e.config.animation,false);});
test('Appearance-only edits retain GPU geometry and reset accumulation',()=>{const e=Object.create(PortraitEngine.prototype);e.config={...DEFAULTS};e.geometryDirty=false;e.samples=9;e.update({melanin:.7});assert.equal(e.geometryDirty,false);assert.equal(e.samples,0);assert.equal(e.config.melanin,.7);});
test('Malformed engine edits preserve valid state',()=>{const e=Object.create(PortraitEngine.prototype);e.config={...DEFAULTS};const before={...e.config};assert.throws(()=>e.update({jaw:NaN}));assert.deepEqual(e.config,before);});

test('Aligned render buffers preserve the displayed viewport aspect ratio',()=>{
 const e=Object.create(PortraitEngine.prototype);e.config={...DEFAULTS};e.samples=0;
 e.width=640;e.height=547;e.displayAspect=637/547;e.rt={write(){}};
 e.updateScene();const expected=multiply(perspective(32,637/547),lookAt([
  DEFAULTS.distance*Math.sin(DEFAULTS.orbitYaw)*Math.cos(DEFAULTS.orbitPitch),
  DEFAULTS.targetY+DEFAULTS.distance*Math.sin(DEFAULTS.orbitPitch),
  .07+DEFAULTS.distance*Math.cos(DEFAULTS.orbitYaw)*Math.cos(DEFAULTS.orbitPitch)
 ],[0,DEFAULTS.targetY,.07]));
 for(let i=0;i<16;i++)assert.ok(Math.abs(e.lastScene[i]-expected[i])<1e-6);
 e.resize(638,547);assert.equal(e.displayAspect,638/547);assert.equal(e.width,640);assert.equal(e.samples,0);
});

test('Adult proportions retain a tapered jaw and matching feature coordinates',()=>{
 const art=compile(human+`\n__global__ void proportions(float4* out){
  unsigned int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>0u)return;
  float3 p=make_float3(.325f,.335f,.65f),q=anatomyPoint(portraitPoint(p));
  out[0]=make_float4(q.x,q.y,q.z,headWidth(-.70f,1.0f)/headWidth(.28f,1.0f));
  float3 top=portraitPoint(headPoint(0.0f,1.0f,1.0f,0.0f,1.0f));
  float3 chin=portraitPoint(headPoint(0.0f,0.0f,1.0f,0.0f,1.0f));
  out[1]=make_float4(top.y-chin.y,neckPoint(.25f,1.0f).x,0.0f,0.0f);
 }`,{entry:'proportions',workgroupSize:[64,1,1]});
 const out=new Float32Array(8);executeCPU(art,{out},{},[1]);
 assert.ok(Math.abs(out[0]-.325)<1e-6&&Math.abs(out[1]-.335)<1e-6&&Math.abs(out[2]-.65)<1e-6);
 assert.ok(out[3]>.64&&out[3]<.73);assert.ok(out[4]>2.45&&out[4]<2.58);assert.ok(out[5]>.35&&out[5]<.40);
});

test('Scalp closes at one crown point and strands start on their authored roots',()=>{
 const art=compile(human+`\n__global__ void scalpProbe(float4* out){
  unsigned int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=64u)return;
  float3 crown=scalpPoint(float(i)/64.0f,0.0f);
  float theta=2.0f*PI*rnd(i*17u+67u),v=.00001f+.9999f*rnd(i*29u+991u);
  float3 root=scalpPoint(theta/(2.0f*PI),v),strand=fiberCenter(i,0.0f,.65f,0.0f,0.0f);
  float3 delta=sub(root,strand);out[i]=make_float4(crown.x,crown.y,crown.z,dot3(delta,delta));
 }`,{entry:'scalpProbe',workgroupSize:[64,1,1]});
 const out=new Float32Array(256);executeCPU(art,{out},{},[1]);
 for(let i=0;i<64;i++){
  assert.ok(Math.abs(out[i*4])<1e-6);assert.ok(Math.abs(out[i*4+1]-1.632)<1e-5);
  assert.ok(Math.abs(out[i*4+2]-.045)<1e-6);assert.ok(out[i*4+3]<1e-9);
 }
});

const {StressRun}=await import('../src/stress.js');
test('Moving frames use spatial samples at a single pose and only present the final sample',()=>{
 const e=Object.create(PortraitEngine.prototype);e.config={...DEFAULTS,quality:'high',animation:true};e.samples=24;
 const calls=[];e.render=(time,options)=>calls.push({time,...options});e.renderFrame(3.7);
 assert.equal(calls.length,8);assert.ok(calls.every(c=>c.time===3.7));assert.equal(calls.filter(c=>c.present).length,1);
 assert.equal(calls[0].temporalSample,false);assert.equal(calls[7].temporalSample,true);assert.equal(calls[7].present,true);
 e.config.animation=false;calls.length=0;e.renderFrame(4,{motion:true});assert.equal(calls.length,8);
 calls.length=0;e.renderFrame(4);assert.equal(calls.length,1);
});
test('Stress metrics include elapsed time and stop at the requested duration',()=>{
 const run=new StressRun(1,0);assert.equal(run.record(8,20),false);assert.equal(run.record(12,1000),true);
 const s=run.summary(9000);assert.equal(s.reason,'completed');assert.equal(s.frames,2);assert.equal(s.fps,2);assert.equal(s.meanRenderMs,10);assert.equal(s.p95RenderMs,12);assert.equal(s.elapsedSeconds,1);
 run.record(999,9000);assert.equal(run.summary().frames,2);
});
test('Stopping a stress run freezes its report and handles no completed frames',()=>{
 const run=new StressRun(15,100);run.stop('tab hidden',200);const s=run.summary(9999);
 assert.equal(s.elapsedSeconds,.1);assert.equal(s.reason,'tab hidden');assert.equal(s.fps,0);assert.equal(s.p95RenderMs,0);
 run.stop('completed',16000);assert.equal(run.summary().reason,'tab hidden');assert.throws(()=>new StressRun(0));
});

test('Anatomical deformation closes the eyelid without shrinking the eyeball',()=>{
 const art=compile(human,{entry:'generateAnatomy',workgroupSize:[64,1,1]}),s=settings.slice();s[3]=.98;
 const rest=new Float32Array([.325,.405,.61,1,.325,.405,.61,4]),directions=new Float32Array([0,0,1,0,0,0,1,0]),roots=new Float32Array([0,-.07,.01,0,0,0,0,0]),closure=directions.slice(),positions=new Float32Array(8),normals=new Float32Array(8);
 executeCPU(art,{settings:s,rest,directions,roots,closure,positions,normals},{first:0,count:2},[1]);
 assert.ok(positions[1]<.34);assert.ok(Math.abs(positions[5]-.405)<1e-6);assert.ok(normals.every(Number.isFinite));
 positions.fill(99);executeCPU(art,{settings:s,rest,directions,roots,closure,positions,normals},{first:1,count:1},[1]);assert.ok(positions.slice(0,4).every(x=>x===99));
});

test('Bundled anatomical rest mesh and generated Preview fibers are finite and contiguous',async()=>{
 const {buildAnatomy}=await import('../src/anatomy.js'),asset=JSON.parse(await readFile(new URL('../assets/male-anatomy.json',import.meta.url),'utf8'));
 const a=buildAnatomy(asset,'preview');assert.equal(a.layout.scalpStrands,4500);assert.equal(a.positions.length,a.layout.vertices*4);
 assert.ok(a.positions.every(Number.isFinite));assert.ok(a.normals.every(Number.isFinite));assert.ok(a.roots.every(Number.isFinite));
 assert.ok(a.layout.hairFirst>a.layout.bodyCount);assert.ok(a.positions.slice(a.layout.hairFirst*4).some((v,i)=>i%4===3&&v>=10));
 for(let i=0;i<a.normals.length;i+=4)assert.ok(Math.abs(Math.hypot(...a.normals.slice(i,i+3))-1)<.01);
});
