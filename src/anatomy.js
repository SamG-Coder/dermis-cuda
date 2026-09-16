// CC0 anatomical mesh, with locally generated eyes, brows and strand geometry.
// CPU builds immutable rest data only; CUDA deforms every displayed vertex.
import {geometryLayout} from './config.js';
const unit=a=>{const l=Math.hypot(...a)||1;return a.map(x=>x/l);};
const cross=(a,b)=>[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];
const dot=(a,b)=>a.reduce((s,x,i)=>s+x*b[i],0);
const hash=x=>{x=Math.imul(x^(x>>>16),0x7feb352d);x=Math.imul(x^(x>>>15),0x846ca68b);return ((x^(x>>>16))>>>0)/4294967296;};
export function anatomicalHairline(x,z){const c=Math.cos(Math.atan2(x,z+.50));return .54+.41*Math.max(c,0)-.58*Math.max(-c,0)-.045*Math.cos(4*Math.atan2(x,z+.50));}
export function buildAnatomy(asset,quality){
 const tier=geometryLayout(quality),hair=tier.layout.find(p=>p.kind===13),brow=tier.layout.find(p=>p.kind===14),P=asset.positions,N=asset.normals,I=asset.indices;
 const eyeCount=96*48*6*2,bodyCount=I.length+eyeCount,browCount=brow.nu*4*6*2,vertices=bodyCount+browCount+hair.count;
 const positions=new Float32Array(vertices*4),normals=new Float32Array(vertices*4),roots=new Float32Array(vertices*4);
 const closure=new Float32Array(bodyCount*4);
 let cursor=0;function emit(p,n,mat,t=0,root=p){positions.set([...p,mat],cursor*4);normals.set([...n,t],cursor*4);roots.set([...root,0],cursor*4);cursor++;}
 const point=i=>P.slice(i*3,i*3+3),normal=i=>N.slice(i*3,i*3+3);
 for(const id of I){closure.set([...asset.closedNormals.slice(id*3,id*3+3),0],cursor*4);emit(point(id),normal(id),1,0,asset.blink.slice(id*3,id*3+3));}
 for(const side of [-1,1])for(let j=0;j<48;j++)for(let i=0;i<96;i++)for(const [dx,dy]of [[0,0],[1,0],[0,1],[1,0],[1,1],[0,1]]){
  const u=(i+dx)/96*Math.PI*2,v=(j+dy)/48*Math.PI,n=[Math.sin(v)*Math.sin(u),Math.cos(v),Math.sin(v)*Math.cos(u)];
  emit([side*.325+n[0]*.174,.335+n[1]*.174,.453+n[2]*.174],n,side<0?3:4);
 }
 // Spatial index for placing eyebrows on the actual face, not an ellipsoid.
 const bins=new Map(),scalp=[];let area=0;
 for(let i=0;i<I.length;i+=3){const a=point(I[i]),b=point(I[i+1]),c=point(I[i+2]),center=a.map((v,k)=>(v+b[k]+c[k])/3);
  if(center[1]>.30&&center[1]<.75&&center[2]>.1){
   const lo=[Math.floor(Math.min(a[0],b[0],c[0])*40),Math.floor(Math.min(a[1],b[1],c[1])*40)],hi=[Math.floor(Math.max(a[0],b[0],c[0])*40),Math.floor(Math.max(a[1],b[1],c[1])*40)];
   for(let x=lo[0];x<=hi[0];x++)for(let y=lo[1];y<=hi[1];y++){const key=x+','+y;if(!bins.has(key))bins.set(key,[]);bins.get(key).push(i);}
  }
  if(Math.min(a[1],b[1],c[1])>anatomicalHairline(center[0],center[2])&&center[1]>.0){
   const ar=Math.hypot(...cross(b.map((v,k)=>v-a[k]),c.map((v,k)=>v-a[k])))*.5;area+=ar;scalp.push({i,area});
  }
 }
 function front(x,y){let z=-Infinity;for(const i of bins.get(Math.floor(x*40)+','+Math.floor(y*40))??[]){const a=point(I[i]),b=point(I[i+1]),c=point(I[i+2]);const d=(b[1]-c[1])*(a[0]-c[0])+(c[0]-b[0])*(a[1]-c[1]);if(Math.abs(d)<1e-12)continue;const u=((b[1]-c[1])*(x-c[0])+(c[0]-b[0])*(y-c[1]))/d,v=((c[1]-a[1])*(x-c[0])+(a[0]-c[0])*(y-c[1]))/d;if(u>=0&&v>=0&&u+v<=1)z=Math.max(z,u*a[2]+v*b[2]+(1-u-v)*c[2]);}return Number.isFinite(z)?z:.55;}
 const corners=[[0,-1],[0,1],[1,-1],[0,1],[1,1],[1,-1]];
 function strand(center,segments,width,mat){const root=center(0);for(let seg=0;seg<segments;seg++)for(const [next,edge]of corners){const t=(seg+next)/segments,p=center(t),q=center(t+.001),tangent=unit(q.map((v,k)=>v-p[k])),radial=unit([p[0],p[1]-.4,p[2]+.5]),across=unit(cross(tangent,radial));emit(p.map((v,k)=>v+across[k]*width*(1-.9*t)*edge),tangent,mat,t,root);}}
 for(const side of [-1,1])for(let id=0;id<brow.nu;id++){
  const s=hash(id*13+32),x=side*(.14+.43*s),y=.45+.030*Math.sin(s*Math.PI)-.025*s+(hash(id*37+55)-.5)*(.07*Math.pow(Math.sin(Math.PI*s),.5)+.003);
  strand(t=>{const xx=x+side*.035*t,yy=y+.020*t-.012*t*t;return [xx,yy,front(xx,yy)+.005+.006*Math.sin(Math.PI*t)];},4,.0009,8);
 }
 const hairFirst=cursor;
 for(let id=0;id<hair.nu;id++){
  const target=hash(id*17+67)*area;let lo=0,hi=scalp.length-1;while(lo<hi){const mid=(lo+hi)>>1;if(scalp[mid].area<target)lo=mid+1;else hi=mid;}
  const face=scalp[lo].i,rr=Math.sqrt(hash(id*29+991)),v=hash(id*43+52),weights=[1-rr,rr*(1-v),rr*v];let root=[0,0,0],n=[0,0,0];
  for(let j=0;j<3;j++){const p=point(I[face+j]),nn=normal(I[face+j]);for(let k=0;k<3;k++){root[k]+=p[k]*weights[j];n[k]+=nn[k]*weights[j];}}n=unit(n);root=root.map((v,k)=>v+n[k]*.006);
  const top=Math.max(0,Math.min(1,(root[1]-.65)/.6)),clump=hash(Math.floor(root[0]*9)*73856093^Math.floor(root[2]*9)*19349663);
  const dir=unit([.45*Math.sin(clump*6.28),-.40*(1-top),.7*top-.35*(1-top)]),tangent=unit(dir.map((v,k)=>v-n[k]*dot(dir,n))),length=(.05+top*.22)*(.70+.45*hash(id+7357)),lift=.008+top*(.06+.075*clump);
  strand(t=>root.map((v,k)=>v+tangent[k]*length*t+n[k]*(lift*Math.sin(Math.PI*t*.85)-.30*length*length*t*t)+((k===0)?top*.015*Math.sin(t*Math.PI*2)*Math.sin(Math.PI*t):0)),hair.nv,.00125+.00065*hash(id+673),10+.8*hash(id+414));
 }
 if(cursor!==vertices)throw Error('Anatomy vertex count mismatch.');
 return {positions,normals,roots,closure,layout:{vertices,triangles:vertices/3,strands:hair.nu+2*brow.nu,scalpStrands:hair.nu,skinVertices:hairFirst,hairFirst,bodyCount,layout:[]}};
}


