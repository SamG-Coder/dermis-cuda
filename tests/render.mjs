import {readFile,writeFile,mkdir} from 'node:fs/promises';import {nativeGpu} from './native.mjs';import {PortraitEngine} from '../src/engine.js';
await nativeGpu();let engine;try{
const human=await readFile(new URL('../kernels/human.cu',import.meta.url),'utf8'),appearance=(await readFile(new URL('../kernels/realism.cu',import.meta.url),'utf8'))+'\n'+(await readFile(new URL('../kernels/appearance.cu',import.meta.url),'utf8'));
engine=await PortraitEngine.create({human,appearance,progress:console.log,config:{quality:process.env.QUALITY||'preview'}});engine.resize(Number(process.env.WIDTH||640),Number(process.env.HEIGHT||800));
if(process.env.CONFIG)engine.update(JSON.parse(process.env.CONFIG));
for(let i=0;i<Number(process.env.FRAMES||4);i++){const start=performance.now();engine.render();await engine.rt.idle();console.log('frame',i,(performance.now()-start).toFixed(0)+'ms');}
const pixels=await engine.pixels();await mkdir(new URL('../docs/',import.meta.url),{recursive:true});await writeFile(new URL('../docs/'+(process.env.NAME||'portrait')+'.rgba',import.meta.url),Buffer.from(pixels.buffer));await writeFile(new URL('../docs/'+(process.env.NAME||'portrait')+'.json',import.meta.url),JSON.stringify(engine.stats(),null,2));console.log(engine.stats());
}finally{engine?.dispose();}process.exit(0);
