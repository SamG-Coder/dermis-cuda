import {readFile,mkdir,writeFile} from 'node:fs/promises';
import {compile} from '../vendor/cuda-webshader/src/compiler/compiler.js';
const h=await readFile(new URL('../kernels/human.cu',import.meta.url),'utf8');const a=(await readFile(new URL('../kernels/realism.cu',import.meta.url),'utf8'))+'\n'+(await readFile(new URL('../kernels/appearance.cu',import.meta.url),'utf8'));
const entries=['generateAnatomy','generateSurface','generateFibers','surfaceNormals','shadePortrait','diffuseSkin','finishPortrait'];
await mkdir(new URL('../generated/',import.meta.url),{recursive:true});
for(const entry of entries){const now=performance.now();const result=compile((entry.startsWith('generate')||entry==='surfaceNormals')?h:h+'\n'+a,{entry,workgroupSize:[64,1,1]});console.log(entry,result.metadata?.bindings?.map(x=>x.name),Math.round(performance.now()-now)+'ms');await writeFile(new URL('../generated/'+entry+'.wgsl',import.meta.url),result.wgsl);}
