import {spawn} from 'node:child_process';import {writeFile} from 'node:fs/promises';import assert from 'node:assert/strict';import http from 'node:http';
const port=Number(process.env.TEST_PORT||8099),checks=[];const child=spawn(process.execPath,['scripts/serve.mjs'],{cwd:new URL('../',import.meta.url),env:{...process.env,PORT:String(port)},stdio:['ignore','pipe','pipe']});let log='';child.stdout.on('data',d=>log+=d);child.stderr.on('data',d=>log+=d);
const check=(name,pass)=>{checks.push({name,passed:!!pass});console.log((pass?'PASS ':'FAIL ')+name);assert.ok(pass,name);};
try{
 for(let i=0;i<100&&!log.includes('Open http');i++)await new Promise(r=>setTimeout(r,30));check('Standalone server starts on a selected local port',log.includes('Open http'));
 const root=await fetch(`http://127.0.0.1:${port}/`),html=await root.text();check('Root serves the complete DERMIS application',root.ok&&html.includes('DERMIS')&&html.includes('src/app.js'));
 for(const path of ['src/app.js','vendor/cuda-webshader/src/compiler/compiler.js','kernels/human.cu','kernels/appearance.cu','kernels/realism.cu']){const r=await fetch(`http://127.0.0.1:${port}/${path}`);check('Local dependency available: '+path,r.ok&&Number(r.headers.get('content-length'))>0);}
 const cuda=await fetch(`http://127.0.0.1:${port}/kernels/human.cu`);check('CUDA is served as readable source',cuda.headers.get('content-type').startsWith('text/plain'));
 const js=await fetch(`http://127.0.0.1:${port}/src/app.js`);check('ES modules have a JavaScript MIME type',js.headers.get('content-type').includes('javascript'));
 check('Missing assets produce HTTP 404',(await fetch(`http://127.0.0.1:${port}/does-not-exist.js`)).status===404);
 const status=await new Promise((resolve,reject)=>http.get({hostname:'127.0.0.1',port,path:'/%2e%2e%2f%2e%2e%2fetc/passwd'},r=>{r.resume();resolve(r.statusCode)}).on('error',reject));check('Encoded traversal cannot escape the project',status===403);
 check('Production entry has no remote script or stylesheet dependency',!/(?:src|href)=["']https?:\/\//.test(html));
}catch(e){console.error(e);process.exitCode=1;}finally{child.kill();await writeFile(new URL('../docs/server-validation.json',import.meta.url),JSON.stringify({passed:checks.filter(c=>c.passed).length,total:checks.length,checks},null,2));}
