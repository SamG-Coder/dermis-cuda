// DERMIS appearance / all portrait lighting and material evaluation is CUDA.
// Compile order: human.cu -> realism.cu -> appearance.cu.
// Skin/eye microstructure lives in the independently editable realism.cu.
// A thin raster bridge only projects geometry and writes position/normal buffers.
__device__ float4 transform(const float4* scene,unsigned int first,float3 p){float4 a=scene[first],b=scene[first+1u],c=scene[first+2u],d=scene[first+3u];return make_float4(a.x*p.x+a.y*p.y+a.z*p.z+a.w,b.x*p.x+b.y*p.y+b.z*p.z+b.w,c.x*p.x+c.y*p.y+c.z*p.z+c.w,d.x*p.x+d.y*p.y+d.z*p.z+d.w);}
__device__ float shadowAt(const float* shadow,const float4* scene,float3 p,float nl,unsigned int size){
 float4 clip=transform(scene,5u,p);float u=clip.x/clip.w*.5f+.5f,v=.5f-clip.y/clip.w*.5f,d=clip.z/clip.w;
 if(u<.002f||u>.998f||v<.002f||v>.998f||d<0.0f||d>1.0f)return 1.0f;
 int x=int(u*float(size)),y=int(v*float(size));float lit=0.0f,bias=.0018f+.0030f*(1.0f-sat(nl));
 for(int dy=-1;dy<=1;dy++)for(int dx=-1;dx<=1;dx++){int xx=min(int(size)-1,max(0,x+dx*5)),yy=min(int(size)-1,max(0,y+dy*5));float depth=shadow[(unsigned int)yy*size+(unsigned int)xx];lit+=d-bias<=depth?1.0f:0.0f;}
 return lit/9.0f;
}
__device__ float ggx(float3 n,float3 v,float3 l,float rough,float f0){
 float3 h=unit(add(v,l));float nv=fmaxf(.001f,dot3(n,v)),nl=fmaxf(.001f,dot3(n,l)),nh=sat(dot3(n,h)),vh=sat(dot3(v,h));
 float a=rough*rough,a2=a*a,den=nh*nh*(a2-1.0f)+1.0f,d=a2/(PI*den*den+.0000001f),k=sq(rough+1.0f)*.125f;
 float g1=nv/(nv*(1.0f-k)+k),g2=nl/(nl*(1.0f-k)+k),f=f0+(1.0f-f0)*powf(1.0f-vh,5.0f);
 return d*g1*g2*f/(4.0f*nv+.0001f);
}
__device__ float3 reflect3(float3 incident,float3 n){return sub(incident,mul(n,2.0f*dot3(incident,n)));}
__device__ float3 hairAlbedo(float melanin,float seed){float3 blonde=make_float3(.20f,.158f,.105f),brown=make_float3(.025f,.011f,.0047f);return mul(mix3(blonde,brown,melanin),.76f+.44f*seed);}
__global__ void shadePortrait(const float4* position,const float4* normal,const float* shadow,const float4* scene,const float4* settings,float4* diffuse,float4* specular,unsigned int width,unsigned int height,unsigned int shadowSize,int view){
 unsigned int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=width*height)return;
 float4 pp=position[i],nn=normal[i];float3 p=v3(pp),n=unit(v3(nn));float mat=pp.w;bool anatomical=scene[11u].y>.5f;float4 skin=scene[9u],look=scene[10u];
 if(mat<.5f){float x=(float(i%width)/float(width)-.5f),y=(float(i/width)/float(height)-.48f);float halo=expf(-(x*x*3.0f+y*y*1.8f)*4.0f);float3 bg=mix3(make_float3(.007f,.010f,.016f),make_float3(.033f,.046f,.066f),halo);diffuse[i]=make_float4(bg.x,bg.y,bg.z,0.0f);specular[i]=make_float4(0.0f,0.0f,0.0f,0.0f);return;}
 float3 cam=v3(scene[4u]);float footprint=sqrtf(dot3(sub(cam,p),sub(cam,p)))*.57349f/float(height);float3 v=unit(sub(cam,p)),key=unit(v3(scene[12u])),fill=unit(make_float3(.70f,.20f,.63f)),rim=unit(make_float3(.58f,.62f,-.53f));
 if(anatomical&&mat<1.5f){
  float theta=atan2f(p.x,p.z+.50f),c=cosf(theta),line=.54f+.41f*fmaxf(c,0.0f)-.58f*fmaxf(-c,0.0f)-.045f*cosf(4.0f*theta);
  if(p.y< -1.36f+.10f*fabsf(p.x))mat=13.0f;
  else if(scene[11u].z>.5f&&p.y>line&&p.y>0.0f)mat=9.0f;
 }
 float3 ap=anatomical?p:anatomyPoint(p);float3 albedo=skinAlbedo(ap,skin.x,skin.w),spec=make_float3(0.0f,0.0f,0.0f),diff=spec;float rough=skin.y,sss=1.0f;
 if(anatomical&&mat<1.5f){float lip=gauss(p.x,p.y,0.0f,-.40f,.235f,.060f)*smooth(.45f,.60f,p.z);albedo=mix3(albedo,make_float3(.32f,.135f,.115f),lip*.28f);}
 bool hair=mat>=8.0f&&mat<9.0f||mat>=10.0f&&mat<12.0f,eye=mat>2.5f&&mat<4.5f;
 if(mat<2.5f){n=skinDetailNormal(p,n,skin.w,footprint);rough=skinDetailRoughness(p,rough,skin.w);
  if(mat>1.5f){float s=p.x/(.287f+.022f*poseA(settings).z),f=powf(fmaxf(.0001f,1.0f-s*s),.70f),mid=mouthY(s,poseA(settings).z);float upper=.042f+.018f*expf(-sq((fabsf(s)-.33f)/.22f))-.010f*expf(-sq(s/.16f));float lipv=fabsf(ap.y-mid)/(f*(ap.y>mid?upper:.066f));albedo=mix3(albedo,make_float3(.32f,.106f,.090f),.38f*(1.0f-smooth(.72f,1.0f,lipv)));float lines=sinf(p.x*640.0f+6.0f*sinf(p.y*35.0f));n=unit(add(n,make_float3(lines*.033f*skin.w,0.0f,0.0f)));rough*=.94f;}
 }
 if(eye){float3 ep=ap;if(anatomical)ep.y=.335f+(ep.y-.335f)/HEIGHT_SCALE;albedo=eyeAlbedoFiltered(ep,mat<3.5f?-1.0f:1.0f,poseB(settings).x,poseB(settings).y,int(look.x),scene[11u].x,footprint);rough=.045f;sss=0.0f;}
 if(mat>6.5f&&mat<7.5f){albedo=make_float3(.025f,.0038f,.0044f);rough=.64f;sss=0.0f;}
 if(mat>8.5f&&mat<9.5f){albedo=mul(hairAlbedo(look.y,.35f),.49f);rough=.6f;sss=0.0f;}
 if(mat>12.5f){float weave=.92f+.08f*sinf(p.x*500.0f)*sinf(p.y*500.0f);albedo=mul(make_float3(.025f,.034f,.045f),weave);rough=.87f;sss=0.0f;}
 float sh=shadowAt(shadow,scene,p,dot3(n,key),shadowSize);
 float3 kc=make_float3(1.00f,.905f,.824f),fc=make_float3(.75f,.86f,1.0f),rc=make_float3(.59f,.77f,1.0f);float power=scene[12u].w;
 if(hair){float3 tangent=n;float3 radial=unit(make_float3(p.x,(p.y-.28f)*.6f,p.z));if(mat<9.0f||mat>=11.0f)radial=make_float3(0.0f,0.0f,1.0f);
  albedo=hairAlbedo(look.y,mat-10.0f);if(mat<9.0f)albedo=mul(hairAlbedo(fmaxf(.55f,look.y),.2f),.7f);if(mat>=11.0f)albedo=make_float3(.012f,.006f,.003f);
  float hairDiffuse=.14f+.48f*sqrtf(fmaxf(0.0f,1.0f-sq(dot3(tangent,key))))*(.28f+.72f*sh);diff=product(albedo,add(mul(kc,hairDiffuse*power),make_float3(.045f,.055f,.067f)));
  float3 h=unit(add(v,key));float primary=powf(sqrtf(fmaxf(0.0f,1.0f-sq(dot3(unit(add(tangent,mul(radial,.12f))),h)))),lerp1(130.0f,35.0f,look.z));float secondary=powf(sqrtf(fmaxf(0.0f,1.0f-sq(dot3(unit(sub(tangent,mul(radial,.18f))),h)))),25.0f);
  spec=add(mul(kc,primary*.08f*sh*power),product(albedo,mul(kc,secondary*.40f*sh*power)));// Pigment-tinted rim: the old unshadowed neutral rim made dark hair look foggy.
  spec=add(spec,product(albedo,mul(rc,.20f*powf(sat(dot3(radial,rim)),5.0f))));sss=0.0f;
 }else{
  float nl=sat(dot3(n,key)),nf=sat(dot3(n,fill)),nr=sat(dot3(n,rim));float ambient=.13f+.07f*sat(n.y);
  float3 illumination=add(add(mul(kc,nl*sh*power),mul(fc,nf*.36f)),add(mul(rc,nr*.72f),make_float3(ambient*.84f,ambient*.92f,ambient)));
  if(sss>.5f){float transmitted=powf(sat(dot3(mul(n,-1.0f),key)),3.0f)*skin.z*(.05f+.24f*smooth(.76f,.95f,fabsf(p.x)));illumination=add(illumination,make_float3(transmitted*.85f,transmitted*.20f,transmitted*.09f));}
  diff=product(albedo,illumination);
  spec=add(mul(kc,ggx(n,v,key,rough,.035f)*sh*power),mul(fc,ggx(n,v,fill,rough,.035f)*.36f));spec=add(spec,mul(rc,ggx(n,v,rim,rough,.035f)*.72f));
  if(sss>.5f)spec=add(spec,mul(kc,ggx(n,v,key,fmaxf(.19f,rough*.64f),.024f)*sh*power*.28f));
  if(eye){float3 reflected=reflect3(mul(v,-1.0f),n);float3 eyeBox=unit(make_float3(-.43f,.30f,.85f));float3 bx=unit(cross3(make_float3(0.0f,1.0f,0.0f),eyeBox)),by=cross3(eyeBox,bx);float rx=dot3(reflected,bx),ry=dot3(reflected,by);
   float box=(1.0f-smooth(.065f,.091f,fabsf(rx)))*(1.0f-smooth(.10f,.125f,fabsf(ry)))*smooth(.8f,.95f,dot3(reflected,eyeBox));spec=add(spec,mul(kc,box*1.2f));
   float small=powf(sat(dot3(reflected,fill)),280.0f);spec=add(spec,mul(fc,small*.8f));diff=mul(diff,.83f);
  }
 }
 if(view==1){diff=albedo;spec=make_float3(0.0f,0.0f,0.0f);sss=0.0f;}
 if(view==2){diff=add(mul(n,.5f),make_float3(.5f,.5f,.5f));spec=make_float3(0.0f,0.0f,0.0f);sss=0.0f;}
 if(view==3){diff=make_float3(sh,sh,sh);spec=make_float3(0.0f,0.0f,0.0f);sss=0.0f;}
 if(view==4){float edge=smooth(.015f,.040f,nn.w);diff=mix3(make_float3(.04f,.45f,.58f),make_float3(.035f,.045f,.065f),edge);spec=make_float3(0.0f,0.0f,0.0f);sss=0.0f;}
 if(view==5){diff=make_float3(0.0f,0.0f,0.0f);sss=0.0f;}
 diffuse[i]=make_float4(diff.x,diff.y,diff.z,sss);specular[i]=make_float4(spec.x,spec.y,spec.z,1.0f);
}
// Edge-aware, separable RGB diffusion. Artistic approximation, not a volumetric solve.
__global__ void diffuseSkin(const float4* input,const float4* position,float4* output,unsigned int width,unsigned int height,int vertical,float strength,float scale){
 unsigned int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=width*height)return;float4 center=input[i];if(center.w<.5f||strength<=.001f){output[i]=center;return;}
 int x=int(i%width),y=int(i/width);float3 sum=make_float3(0.0f,0.0f,0.0f),weights=sum;float3 p=v3(position[i]);
 for(int k=-6;k<=6;k++){int offset=int(float(k)*scale),xx=min(int(width)-1,max(0,x+(vertical==0?offset:0))),yy=min(int(height)-1,max(0,y+(vertical!=0?offset:0)));unsigned int j=(unsigned int)yy*width+(unsigned int)xx;float4 s=input[j];float3 delta=sub(v3(position[j]),p);float gate=s.w>.5f?expf(-dot3(delta,delta)*1300.0f):0.0f;
  float f=float(k),wr=expf(-f*f/17.0f)*gate,wg=expf(-f*f/5.4f)*gate,wb=expf(-f*f/1.45f)*gate;
  sum=add(sum,make_float3(s.x*wr,s.y*wg,s.z*wb));weights=add(weights,make_float3(wr,wg,wb));
 }
 float3 blurred=make_float3(sum.x/fmaxf(.00001f,weights.x),sum.y/fmaxf(.00001f,weights.y),sum.z/fmaxf(.00001f,weights.z));float3 c=mix3(v3(center),blurred,sat(strength));output[i]=make_float4(c.x,c.y,c.z,center.w);
}
__device__ float film(float x){return sat((x*(2.51f*x+.03f))/(x*(2.43f*x+.59f)+.14f));}
__device__ float srgb(float x){return x<=.0031308f?12.92f*x:1.055f*powf(x,1.0f/2.4f)-.055f;}
__global__ void finishPortrait(const float4* diffuse,const float4* specular,float4* history,float4* output,unsigned int width,unsigned int height,unsigned int samples,float exposure,int view){
 unsigned int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=width*height)return;float3 c=add(v3(diffuse[i]),v3(specular[i]));
 if(view==0||view==5){c=mul(c,exposure);c=make_float3(film(c.x),film(c.y),film(c.z));}
 c=make_float3(srgb(sat(c.x)),srgb(sat(c.y)),srgb(sat(c.z)));
 float3 blended=samples>0u?mix3(v3(history[i]),c,1.0f/float(samples+1u)):c;history[i]=make_float4(blended.x,blended.y,blended.z,1.0f);output[i]=history[i];
}


