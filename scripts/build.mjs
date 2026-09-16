import {cp,mkdir,rm} from 'node:fs/promises';
const root=new URL('../',import.meta.url),out=new URL('../dist/',import.meta.url);await rm(out,{recursive:true,force:true});await mkdir(out,{recursive:true});for(const name of ['index.html','showcase.html','src','kernels','assets','vendor','README.md','LICENSE','THIRD_PARTY_NOTICES.md'])await cp(new URL(name,root),new URL(name,out),{recursive:true});console.log('Static WebGPU application built in dist/. Serve it on localhost or HTTPS.');

