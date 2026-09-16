export const clamp=(x,a,b)=>Math.min(b,Math.max(a,x));
export const sub=(a,b)=>a.map((v,i)=>v-b[i]);
export const dot=(a,b)=>a.reduce((s,v,i)=>s+v*b[i],0);
export const cross=(a,b)=>[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];
export const unit=a=>{const d=Math.hypot(...a)||1;return a.map(x=>x/d);};
export function multiply(a,b){const c=new Float32Array(16);for(let r=0;r<4;r++)for(let s=0;s<4;s++)for(let k=0;k<4;k++)c[r*4+s]+=a[r*4+k]*b[k*4+s];return c;}
export function lookAt(eye,target){const z=unit(sub(eye,target)),x=unit(cross([0,1,0],z)),y=cross(z,x);return new Float32Array([...x,-dot(x,eye),...y,-dot(y,eye),...z,-dot(z,eye),0,0,0,1]);}
export function perspective(fov,aspect,near=.05,far=40){const f=1/Math.tan(fov*Math.PI/360);return new Float32Array([f/aspect,0,0,0,0,f,0,0,0,0,far/(near-far),far*near/(near-far),0,0,-1,0]);}
export function ortho(size,near=.1,far=20){return new Float32Array([2/size,0,0,0,0,2/size,0,0,0,0,1/(near-far),near/(near-far),0,0,0,1]);}
export function halton(i,b){let f=1,r=0;while(i>0){f/=b;r+=f*(i%b);i=Math.floor(i/b);}return r;}
