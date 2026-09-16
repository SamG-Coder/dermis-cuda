// DERMIS / original procedural portrait. Units: approximately 0.1 metre.
// Authored CUDA C++; compiled at runtime by CUDA WebShader. No scanned mesh.
#define PI 3.14159265358979323846f
// Keep landmarks in authoring space; apply the same proportions to every mesh
// and strand. Appearance uses the inverse map, so irises and lip edges stay aligned.
#define HEIGHT_SCALE .94f
__device__ float3 portraitPoint(float3 p){return make_float3(p.x,.335f+(p.y-.335f)*HEIGHT_SCALE,p.z);}
__device__ float3 anatomyPoint(float3 p){return make_float3(p.x,.335f+(p.y-.335f)/HEIGHT_SCALE,p.z);}
__device__ float sat(float x){return fminf(1.0f,fmaxf(0.0f,x));}
__device__ float lerp1(float a,float b,float t){return a+(b-a)*t;}
__device__ float smooth(float a,float b,float x){float t=sat((x-a)/(b-a));return t*t*(3.0f-2.0f*t);}
__device__ float sq(float x){return x*x;}
__device__ float3 v3(float4 p){return make_float3(p.x,p.y,p.z);}
__device__ float3 add(float3 a,float3 b){return make_float3(a.x+b.x,a.y+b.y,a.z+b.z);}
__device__ float3 sub(float3 a,float3 b){return make_float3(a.x-b.x,a.y-b.y,a.z-b.z);}
__device__ float3 mul(float3 a,float s){return make_float3(a.x*s,a.y*s,a.z*s);}
__device__ float3 product(float3 a,float3 b){return make_float3(a.x*b.x,a.y*b.y,a.z*b.z);}
__device__ float dot3(float3 a,float3 b){return a.x*b.x+a.y*b.y+a.z*b.z;}
__device__ float3 cross3(float3 a,float3 b){return make_float3(a.y*b.z-a.z*b.y,a.z*b.x-a.x*b.z,a.x*b.y-a.y*b.x);}
__device__ float3 unit(float3 a){float d=dot3(a,a);return d>1.0e-24f?mul(a,rsqrtf(d)):make_float3(0.0f,0.0f,1.0f);}
__device__ float3 mix3(float3 a,float3 b,float t){return add(mul(a,1.0f-t),mul(b,t));}
__device__ unsigned int hash32(unsigned int x){x^=x>>16u;x*=0x7feb352du;x^=x>>15u;x*=0x846ca68bu;x^=x>>16u;return x;}
__device__ float rnd(unsigned int x){return float(hash32(x)&0x00ffffffu)/16777216.0f;}
__device__ float gauss(float x,float y,float cx,float cy,float wx,float wy){return expf(-sq((x-cx)/wx)-sq((y-cy)/wy));}
__device__ float noise3(float3 p){
 int ix=int(floorf(p.x)),iy=int(floorf(p.y)),iz=int(floorf(p.z));
 float x=p.x-floorf(p.x),y=p.y-floorf(p.y),z=p.z-floorf(p.z);x=x*x*(3.0f-2.0f*x);y=y*y*(3.0f-2.0f*y);z=z*z*(3.0f-2.0f*z);
 float sum=0.0f;
 for(unsigned int i=0u;i<8u;i++){int dx=int(i&1u),dy=int((i>>1u)&1u),dz=int((i>>2u)&1u);unsigned int seed=(unsigned int)(ix+dx)*73856093u^(unsigned int)(iy+dy)*19349663u^(unsigned int)(iz+dz)*83492791u;sum+=rnd(seed)*(dx==1?x:1.0f-x)*(dy==1?y:1.0f-y)*(dz==1?z:1.0f-z);}
 return sum;
}
__device__ float headWidth(float y,float jaw){float t=(y-0.28f)/1.34f;float w=.835f*sqrtf(fmaxf(0.000001f,1.0f-t*t));return w*(1.0f+.015f*expf(-sq((y+.68f)/.35f)))*(1.0f+(jaw-1.0f)*.24f*expf(-sq((y+.6f)/.5f)));}
__device__ float faceZ(float x,float y,float smile,float nose){
 float z=.575f+.045f*expf(-sq((y-.87f)/.58f));
 z+=.066f*gauss(fabsf(x),y,.43f,-.07f,.23f,.29f); // zygomatic fat pad
 z-=.010f*gauss(fabsf(x),y,.53f,-.40f,.24f,.34f); // cheek hollow
 z+=.030f*gauss(x,y,0.0f,-.76f,.31f,.20f); // mental prominence
 z+=.072f*gauss(x,y,0.0f,-.44f,.31f,.22f); // muzzle / orbicularis region
 z+=nose*(.155f*gauss(x,y,0.0f,.18f,.10f,.34f)+.160f*gauss(x,y,0.0f,-.083f,.135f,.14f)+.075f*gauss(fabsf(x),y,.126f,-.164f,.063f,.080f));
 z-=.033f*gauss(fabsf(x),y,.186f,-.20f,.043f,.091f); // alar crease
 z-=.019f*gauss(x,y,0.0f,-.32f,.044f,.070f); // philtrum
 z+=.038f*gauss(fabsf(x),y,.325f,.515f,.25f,.12f); // supraorbital ridge
 z-=.012f*gauss(fabsf(x),y,.34f,.155f,.22f,.090f); // infraorbital crease
 z+=smile*.032f*gauss(fabsf(x),y,.39f,-.19f,.19f,.18f);
 // Orbital hollow is covered by independent, smoothly blended eyelid surfaces.
 z-=.065f*gauss(fabsf(x),y,.325f,.335f,.219f,.110f);
 return z;
}
__device__ float frontEnvelope(float y){if(y>.68f)return sqrtf(fmaxf(.000001f,1.0f-sq((y-.68f)/.94f)));if(y<-.64f)return sqrtf(fmaxf(.000001f,1.0f-sq((y+.64f)/.42f)));return 1.0f;}
// The lower jaw converges beneath the chin, rather than beneath the skull.
__device__ float sectionCenterZ(float y){return .045f+.20f*(1.0f-smooth(-1.06f,-.42f,y));}
__device__ float frontZ(float x,float y,float jaw,float smile,float nose){float w=headWidth(y,jaw);float c=sqrtf(fmaxf(.000001f,1.0f-sq(x/w)));return sectionCenterZ(y)+faceZ(x,y,smile,nose)*powf(c,.72f)*frontEnvelope(y);}
__device__ float3 headPoint(float u,float v,float jaw,float smile,float nose){
 float a=2.0f*PI*u,y=-1.06f+2.68f*v,w=headWidth(y,jaw),x=w*sinf(a),c=cosf(a);float z;
 if(v>=.999999f)return make_float3(0.0f,1.62f,.045f);
 if(c>=0.0f)z=sectionCenterZ(y)+faceZ(x,y,smile,nose)*powf(fmaxf(c,.000001f),.72f)*frontEnvelope(y);
 else z=sectionCenterZ(y)+.87f*(.70f+.30f*smooth(-.85f,.35f,y))*sqrtf(fmaxf(.000001f,1.0f-sq((y-.28f)/1.34f)))*c;
 return make_float3(x,y,z);
}
__device__ float eyeShape(float x){return powf(fmaxf(0.0f,1.0f-x*x),.67f);}
__device__ float eyeZ(float x,float y,float side,float gx,float gy){float dx=x-side*.325f,dy=y-.335f;float b=sqrtf(fmaxf(.00001f,.250f*.250f-dx*dx-dy*dy));float cor=expf(-sq((dx-gx*.038f)/.102f)-sq((dy-gy*.030f)/.102f));float z=.370f+b+.010f*cor;float corner=smooth(.110f,.175f,fabsf(dx));return lerp1(z,frontZ(x,y,1.0f,0.0f,1.0f)+.003f,corner);}
__device__ float3 eyePoint(float u,float v,float side,float blink,float gx,float gy){
 float s=2.0f*u-1.0f,f=eyeShape(s),x=side*(.325f+.175f*s),mid=.335f+.012f*s;
 float upper=.073f*(1.0f-blink),lower=-.047f*(1.0f-blink);mid-=.012f*blink*f;
 float y=mid+lerp1(lower,upper,v)*f;
 return make_float3(x,y,eyeZ(x,y,side,gx,gy));
}
__device__ float3 lidPoint(float u,float v,float side,float blink,float jaw,float smile,float nose,float gx,float gy){
 float a=2.0f*PI*u,s=cosf(a),h=sinf(a),f=eyeShape(s),open=1.0f-blink;
 float xi=side*(.325f+.175f*s),yi=.335f+.012f*s-.012f*blink*f+(h>=0.0f?.073f:-.047f)*open*f;
 float xo=side*(.325f+.252f*s),yo=.335f+.018f*s+(h>=0.0f?.145f:.113f)*h;
 float t=smooth(0.0f,1.0f,v),x=lerp1(xi,xo,t),y=lerp1(yi,yo,t),zi=eyeZ(xi,yi,side,gx,gy)+.005f,zo=frontZ(xo,yo,jaw,smile,nose)+.0014f;
 float z=lerp1(zi,zo,t)+.005f*sq(sinf(PI*v))*(h>=0.0f?1.0f:.65f);
 z-=.0022f*expf(-sq((v-.61f)/.09f))*f*(h>0.0f?1.0f:.4f);
 z=fmaxf(z,frontZ(x,y,jaw,smile,nose)+.0012f);return make_float3(x,y,z);
}
__device__ float mouthY(float s,float smile){return -.492f+.057f*smile*s*s+.005f*cosf(s*PI);}
__device__ float3 lipPoint(float u,float v,int part,float smile,float jaw,float nose){
 float s=2.0f*u-1.0f,x=s*(.287f+.022f*smile),f=powf(fmaxf(0.0f,1.0f-s*s),.70f),mid=mouthY(s,smile);
 float cupid=.042f+.018f*expf(-sq((fabsf(s)-.33f)/.22f))-.010f*expf(-sq(s/.16f));
 float y=mid;if(part==0)y+=f*lerp1(.003f,cupid,v);if(part==1)y-=f*lerp1(.006f,.066f,v);if(part==2)y+=f*lerp1(-.0065f,.0035f,v);
 float z=frontZ(x,y,jaw,smile,nose)+f*(part==2?.017f:.016f*(1.0f-v)+.032f*sinf(PI*v));
 return make_float3(x,y,z);
}
// Pinna lies along the side of the skull, with a rolled helix and concha.
// Its tilted local axes give it real depth in both front and profile views.
__device__ float3 earPoint(float u,float v,float side){
 float a=2.0f*PI*u,r=v,h=r*cosf(a),vertical=r*sinf(a);
 float helix=.055f*expf(-sq((r-.84f)/.13f));
 float antihelix=.025f*expf(-sq((r-.49f)/.14f));
 float bowl=-.025f*expf(-sq(r/.32f));
 float relief=helix+antihelix+bowl;
 float x=side*(.825f+.095f*h+relief),y=.15f+.29f*vertical;
 float z=-.015f-.175f*h+.35f*relief;
 return make_float3(x,y,z);
}
__device__ float3 neckPoint(float u,float v){float a=2.0f*PI*u,y=-1.70f+1.22f*v;float r=.385f+.14f*sq(1.0f-v)+.020f*cosf(a*2.0f);float x=r*sinf(a),z=-.015f+r*.80f*cosf(a);z+=.019f*cosf(a)*sinf(v*PI);return make_float3(x,y,z);}
__device__ float3 shirtPoint(float u,float v){float a=2.0f*PI*u;float r=.505f+1.35f*powf(v,.70f),y=-1.48f-.96f*v;float x=r*sinf(a),z=-.09f+r*.53f*cosf(a);y+=.15f*sinf(PI*v)*fabsf(sinf(a));z+=.009f*sinf(a*31.0f+v*5.0f)*sinf(PI*v);return make_float3(x,y,z);}
__device__ float hairline(float theta){float c=cosf(theta);return .25f+.73f*fmaxf(c,0.0f)-.49f*fmaxf(-c,0.0f)-.055f*cosf(theta*4.0f);}
__device__ float3 scalpPoint(float u,float v){float a=2.0f*PI*u;float y=lerp1(1.62f,hairline(a),v);float3 p=headPoint(u,(y+1.06f)/2.68f,1.0f,0.0f,1.0f);float3 n=unit(make_float3(p.x,(p.y-.28f)*.65f,p.z-.045f));return add(p,mul(n,.012f));}
// Trace strands directly in scalp coordinates. The root is exactly scalpPoint,
// avoiding the former approximate ellipsoid reprojection and floating roots.
__device__ float3 fiberCenter(unsigned int id,float t,float length,float time,float breeze){
 float theta=2.0f*PI*rnd(id*17u+67u),v=.00001f+.9999f*rnd(id*29u+991u);
 float rootY=lerp1(1.62f,hairline(theta),v);
 float top=smooth(.45f,1.23f,rootY);float travel=length*(.14f+.32f*rnd(id+7357u))*(.30f+.70f*top);
 float theta2=theta+t*travel*(.7f+.5f*sinf(theta*5.0f+rootY*13.0f));
 float cy=fminf(1.0f,fmaxf(-1.0f,(rootY-.28f)/1.34f));float lat=atan2f(sqrtf(fmaxf(0.0f,1.0f-cy*cy)),cy);
 float yy=.28f+1.34f*cosf(lat+t*travel*.25f);
 yy=fmaxf(hairline(theta2),yy);
 float3 p=headPoint(theta2/(2.0f*PI),(yy+1.06f)/2.68f,1.0f,0.0f,1.0f);
 float3 n=unit(make_float3(p.x,(p.y-.28f)*.65f,p.z-.045f));
 float clump=.5f+.5f*sinf(theta*19.0f+rootY*23.0f);float lift=.012f+length*(.016f+top*(.05f+.065f*clump))*sinf(PI*t)+top*length*.025f*t;
 p=add(p,mul(n,lift));
 float wave=top*length*.024f*sinf(t*PI*2.0f+clump*3.0f)*sinf(PI*t);p.x+=wave;p.z-=wave*.6f;
 float w=breeze*.020f*t*t*sinf(time*1.7f+float(id%137u)*.07f);p.x+=w;p.z+=w*.45f;return p;
}
__device__ float3 browCenter(unsigned int id,float t,float side,float jaw,float smile,float nose,float lift){float s=rnd(id*13u+32u),x=side*(.145f+.405f*s),y=.493f+.033f*sinf(s*PI*.9f)-.018f*s+lift*.047f;float scatter=(rnd(id*37u+55u)-.5f)*(.075f*powf(fmaxf(.0f,sinf(PI*s)),.45f)*(1.0f-.5f*s)+.005f);x+=side*.045f*t;y+=scatter+.023f*t-.019f*t*t;float z=frontZ(x,y,jaw,smile,nose)+.006f+.006f*sinf(t*PI);return make_float3(x,y,z);}
__device__ float3 lashCenter(unsigned int id,float t,float side,float blink,float gx,float gy){float s=.035f+.93f*rnd(id*97u+17u);bool lower=(id%4u)==0u;float3 p=eyePoint(s,lower?0.0f:1.0f,side,blink,gx,gy);float out=side*(s*2.0f-1.0f);float l=.012f+.018f*rnd(id+69u);p.x+=out*l*.35f*t;p.y+=(lower?-1.0f:1.0f)*l*(.18f*t+.82f*t*t);p.z+=.005f+l*.55f*sinf(t*PI*.65f);return p;}
__device__ float4 poseA(const float4* settings){float4 a=settings[0];if(settings[2].z>.5f){float t=settings[1].w;float phase=t-floorf(t/5.2f)*5.2f;a.w=fmaxf(a.w,.97f*expf(-sq((phase-3.7f)/.105f)));}return a;}
__device__ float4 poseB(const float4* settings){float4 b=settings[1];if(settings[2].z>.5f){b.x+=.13f*sinf(b.w*.57f);b.y+=.06f*sinf(b.w*.41f);}return b;}
// Anatomical rest-mesh deformation. Tangent derivatives below also transform
// normals, so changes to the jaw, nose and expression retain correct lighting.
__device__ float3 deformAnatomy(float3 p,float mat,const float4* settings){
 float4 a=poseA(settings);
 p.x*=1.0f+(a.x-1.0f)*.30f*expf(-sq((p.y+.55f)/.40f));
 p.z+=(a.y-1.0f)*.20f*gauss(p.x,p.y,0.0f,.01f,.14f,.30f)*smooth(.35f,.65f,p.z);
 p.y+=a.z*.055f*gauss(fabsf(p.x),p.y,.23f,-.38f,.15f,.13f)*smooth(.35f,.60f,p.z);
 p.y+=settings[2].y*.05f*gauss(fabsf(p.x),p.y,.32f,.53f,.26f,.095f)*smooth(.20f,.50f,p.z);
 return p;
}
__global__ void generateAnatomy(const float4* settings,const float4* rest,const float4* directions,const float4* roots,const float4* closure,float4* positions,float4* normals,unsigned int first,unsigned int count){
 unsigned int lane=blockIdx.x*blockDim.x+threadIdx.x;if(lane>=count)return;unsigned int i=first+lane;
 float4 src=rest[i],dir=directions[i];float3 p=v3(src),n=unit(v3(dir));float mat=src.w;
 if(mat<2.5f){float amount=poseA(settings).w;p=add(p,mul(v3(roots[i]),amount));n=unit(mix3(n,v3(closure[i]),amount));}
 if(mat>=10.0f&&mat<11.0f){float3 root=v3(roots[i]);p=add(root,mul(sub(p,root),settings[1].z));float w=settings[2].x*.014f*dir.w*dir.w*sinf(settings[1].w*1.7f+root.x*19.0f+root.z*11.0f);p.x+=w;}
 float3 q=deformAnatomy(p,mat,settings),out=n;
 if(mat>=8.0f&&mat<12.0f)out=unit(sub(deformAnatomy(add(p,mul(n,.001f)),mat,settings),q));
 else {float3 axis=fabsf(n.y)<.9f?make_float3(0.0f,1.0f,0.0f):make_float3(1.0f,0.0f,0.0f);float3 u=unit(cross3(axis,n)),v=cross3(n,u);out=unit(cross3(sub(deformAnatomy(add(p,mul(u,.001f)),mat,settings),q),sub(deformAnatomy(add(p,mul(v,.001f)),mat,settings),q)));}
 positions[i]=make_float4(q.x,q.y,q.z,mat);normals[i]=make_float4(out.x,out.y,out.z,dir.w);
}
// layouts = kind, nu, nv, first; kind 0=head, 1/2=eyes, 3/4=lids,
// 5/6=lips, 7=mouth seam, 8/9=ears, 10=neck, 11=shirt, 12=scalp,
// 13=hair ribbons, 14/15=brows, 16/17=lashes, 18/19=nostrils.
__device__ float4 surface(unsigned int kind,float u,float v,const float4* settings){
 float4 a=poseA(settings),b=poseB(settings);float jaw=a.x,nose=a.y,smile=a.z,blink=a.w;float3 p=make_float3(0.0f,0.0f,0.0f);float mat=1.0f;
 if(kind==0u)p=headPoint(u,v,jaw,smile,nose);
 if(kind==1u||kind==2u){float side=kind==1u?-1.0f:1.0f;p=eyePoint(u,v,side,blink,b.x,b.y);mat=float(kind+2u);}
 if(kind==3u||kind==4u)p=lidPoint(u,v,kind==3u?-1.0f:1.0f,blink,jaw,smile,nose,b.x,b.y);
 if(kind>=5u&&kind<=7u){p=lipPoint(u,v,int(kind)-5,smile,jaw,nose);mat=kind==7u?7.0f:2.0f;}
 if(kind==8u||kind==9u)p=earPoint(u,v,kind==8u?-1.0f:1.0f);
 if(kind==10u)p=neckPoint(u,v);
 if(kind==11u){p=shirtPoint(u,v);mat=13.0f;}
 if(kind==12u){p=scalpPoint(u,v);mat=9.0f;}
 if(kind==18u||kind==19u){float side=kind==18u?-1.0f:1.0f,angle=u*PI*2.0f;float x=side*(.117f+.041f*v*cosf(angle)),y=-.190f+.012f*v*sinf(angle);p=make_float3(x,y,frontZ(x,y,jaw,smile,nose)+.0025f);mat=7.0f;}
 p=portraitPoint(p);return make_float4(p.x,p.y,p.z,mat);
}
__global__ void generateSurface(const float4* settings,float4* positions,float4* normals,unsigned int kind,unsigned int nu,unsigned int nv,unsigned int first,unsigned int count){
 unsigned int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;unsigned int cell=i/6u,k=i%6u;
 float du=(k==1u||k==3u||k==4u)?1.0f:0.0f,dv=(k==2u||k==4u||k==5u)?1.0f:0.0f;
 float u=(float(cell%nu)+du)/float(nu),v=(float(cell/nu)+dv)/float(nv);
 float4 p=surface(kind,u,v,settings);positions[first+i]=p;normals[first+i]=make_float4(0.0f,0.0f,1.0f,0.0f);
}
// A second dispatch derives smooth normals from GPU-generated grid neighbours.
// This avoids evaluating the complete anatomical surface five times per vertex.
__device__ float3 gridPoint(const float4* positions,int x,int y,unsigned int nu,unsigned int nv,unsigned int first){
 x=min(int(nu),max(0,x));y=min(int(nv),max(0,y));unsigned int cx=min(nu-1u,(unsigned int)x),cy=min(nv-1u,(unsigned int)y),dx=(unsigned int)x-cx,dy=(unsigned int)y-cy;unsigned int k=dy==0u?(dx==0u?0u:1u):(dx==0u?2u:4u);return v3(positions[first+(cy*nu+cx)*6u+k]);
}
__global__ void surfaceNormals(const float4* settings,const float4* positions,float4* normals,unsigned int kind,unsigned int nu,unsigned int nv,unsigned int first,unsigned int count){
 unsigned int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;unsigned int cell=i/6u,k=i%6u;int x=int(cell%nu)+((k==1u||k==3u||k==4u)?1:0),y=int(cell/nu)+((k==2u||k==4u||k==5u)?1:0);
 int xm=x-1,xp=x+1;bool wrap=kind==0u||kind==3u||kind==4u||kind==8u||kind==9u||kind==10u||kind==11u||kind==12u||kind>=18u;
 if(wrap){if(xm<0)xm=int(nu)-1;if(xp>int(nu))xp=1;}
 float3 pu=sub(gridPoint(positions,xp,y,nu,nv,first),gridPoint(positions,xm,y,nu,nv,first)),pv=sub(gridPoint(positions,x,y+1,nu,nv,first),gridPoint(positions,x,y-1,nu,nv,first));float3 p=v3(positions[first+i]),n=unit(cross3(pu,pv));
 if(dot3(n,n)<.1f)n=unit(make_float3(p.x,p.y-.28f,p.z-.04f));
 if(kind==0u||kind==10u||kind==11u||kind==12u){if(dot3(n,make_float3(p.x,0.0f,p.z-((kind==0u||kind==12u)?sectionCenterZ(anatomyPoint(p).y):-.015f)))<0.0f)n=mul(n,-1.0f);}else if(kind==8u||kind==9u){float side=kind==8u?-1.0f:1.0f;if(dot3(n,make_float3(side,0.0f,.54f))<0.0f)n=mul(n,-1.0f);}else if(n.z<0.0f)n=mul(n,-1.0f);
 float v=float(y)/float(nv);
 if((kind==3u||kind==4u||kind==5u||kind==6u)&&v>.6f){float e=.0002f;float4 a=poseA(settings);float3 ap=anatomyPoint(p);float zx=(frontZ(ap.x+e,ap.y,a.x,a.z,a.y)-frontZ(ap.x-e,ap.y,a.x,a.z,a.y))/(2.0f*e),zy=(frontZ(ap.x,ap.y+e,a.x,a.z,a.y)-frontZ(ap.x,ap.y-e,a.x,a.z,a.y))/(2.0f*e*HEIGHT_SCALE);n=unit(mix3(n,unit(make_float3(-zx,-zy,1.0f)),smooth(.60f,1.0f,v)));}
 // Longitude has no unique tangent at the crown. Blend to an outward cap
 // normal instead of letting its nearly degenerate triangles flip downward.
 if((kind==0u||kind==12u)&&p.y>1.40f){float3 ap=anatomyPoint(p);float3 cap=unit(make_float3(ap.x,(ap.y-.28f)*.65f/HEIGHT_SCALE,ap.z-.045f));n=unit(mix3(n,cap,smooth(1.40f,1.50f,p.y)));}
 normals[first+i]=make_float4(n.x,n.y,n.z,0.0f);
}

__global__ void generateFibers(const float4* settings,float4* positions,float4* normals,unsigned int kind,unsigned int strands,unsigned int segments,unsigned int first,unsigned int count){
 unsigned int i=(blockIdx.y*gridDim.x+blockIdx.x)*blockDim.x+threadIdx.x;if(i>=count)return;unsigned int strand=i/(segments*6u),seg=(i/6u)%segments,k=i%6u;
 bool next=k==2u||k==4u||k==5u;float edge=(k==1u||k==3u||k==4u)?1.0f:-1.0f,t=(float(seg)+(next?1.0f:0.0f))/float(segments);
 float4 a=poseA(settings),b=poseB(settings);float3 p=make_float3(0.0f,0.0f,0.0f),p2=p;float mat=10.0f,width=.0010f;float d=.002f;
 if(kind==13u){p=fiberCenter(strand,t,b.z,b.w,settings[2].x);p2=fiberCenter(strand,t+d,b.z,b.w,settings[2].x);width=(.00070f+.00065f*rnd(strand+673u))*(1.0f-.89f*powf(t,2.0f));mat=10.0f+.8f*rnd(strand+414u);}
 if(kind==14u||kind==15u){float side=kind==14u?-1.0f:1.0f;p=browCenter(strand,t,side,a.x,a.z,a.y,settings[2].y);p2=browCenter(strand,t+d,side,a.x,a.z,a.y,settings[2].y);width=.0008f*(1.0f-.86f*t);mat=8.0f;}
 if(kind==16u||kind==17u){float side=kind==16u?-1.0f:1.0f;p=lashCenter(strand,t,side,a.w,b.x,b.y);p2=lashCenter(strand,t+d,side,a.w,b.x,b.y);width=.0009f*(1.0f-.95f*t);mat=11.0f;}
 float3 tangent=unit(sub(p2,p)),outward=kind==13u?unit(make_float3(p.x,p.y-.28f,p.z-.04f)):make_float3(0.0f,0.0f,1.0f),across=unit(cross3(tangent,outward));
 p=portraitPoint(add(p,mul(across,width*edge)));tangent=unit(make_float3(tangent.x,tangent.y*HEIGHT_SCALE,tangent.z));positions[first+i]=make_float4(p.x,p.y,p.z,mat);normals[first+i]=make_float4(tangent.x,tangent.y,tangent.z,t);
}
