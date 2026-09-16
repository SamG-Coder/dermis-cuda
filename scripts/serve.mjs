import {createServer} from 'node:http';
import {readFile,stat} from 'node:fs/promises';
import {resolve,extname,sep} from 'node:path';
import {fileURLToPath} from 'node:url';
const root=fileURLToPath(new URL('../',import.meta.url)),port=Number(process.env.PORT||8080);
const mime={'.html':'text/html','.js':'text/javascript','.mjs':'text/javascript','.css':'text/css','.json':'application/json','.cu':'text/plain','.wgsl':'text/plain','.svg':'image/svg+xml','.webp':'image/webp','.png':'image/png'};
const server=createServer(async(req,res)=>{if(!['GET','HEAD'].includes(req.method)){res.writeHead(405,{'Allow':'GET, HEAD'});return res.end('Method not allowed');}try{const url=new URL(req.url,'http://localhost');const path=resolve(root,'.'+decodeURIComponent(url.pathname==='/'?'/index.html':url.pathname));if(path!==root&&!path.startsWith(resolve(root)+sep)){res.writeHead(403);return res.end('Forbidden');}if(!(await stat(path)).isFile())throw Error('Not a file');const data=await readFile(path);res.writeHead(200,{'Content-Type':mime[extname(path)]||'application/octet-stream','Content-Length':data.byteLength,'Cache-Control':'no-cache','X-Content-Type-Options':'nosniff'});res.end(req.method==='HEAD'?undefined:data);}catch{res.writeHead(404);res.end('File not found');}});
server.on('error',e=>{console.error(e.message);process.exit(1);});
server.listen(port,'127.0.0.1',()=>console.log(`DERMIS - CUDA portrait laboratory\nOpen http://localhost:${port}\nCtrl+C stops the local server.\n`));
