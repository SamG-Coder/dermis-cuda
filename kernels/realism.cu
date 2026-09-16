// DERMIS skin and eye material detail. Original procedural CUDA source.
// Compile after human.cu and before appearance.cu. No image textures or scans.
// Coordinates are object-space, so detail stays attached while the camera moves.
// Pixel-footprint filtering suppresses subpixel pores and iris fibers in motion.
__device__ float detailFilter(float frequency,float footprint){return 1.0f-smooth(.35f,1.1f,frequency*footprint);}
__device__ float3 poreField(float x,float y){
 float sx=x*155.0f,sy=y*155.0f;int ix=int(floorf(sx)),iy=int(floorf(sy));float fx=sx-floorf(sx),fy=sy-floorf(sy),best=10.0f,gx=0.0f,gy=0.0f,size=30.0f;
 for(int j=-1;j<=1;j++)for(int i=-1;i<=1;i++){unsigned int seed=(unsigned int)(ix+i)*19349663u^(unsigned int)(iy+j)*73856093u;float dx=fx-float(i)-rnd(seed),dy=fy-float(j)-rnd(seed+23u),d=dx*dx+dy*dy;if(d<best){best=d;gx=dx;gy=dy;size=22.0f+36.0f*rnd(seed+41u);}}
 float h=expf(-best*size);return make_float3(h,gx*h,gy*h);
}
__device__ float3 skinAlbedo(float3 p,float melanin,float detail){
 float3 base=mix3(make_float3(.55f,.305f,.221f),make_float3(.106f,.048f,.026f),melanin);
 float broad=noise3(mul(p,4.0f))-.5f,medium=noise3(mul(p,17.0f))-.5f,fine=noise3(mul(p,51.0f))-.5f;
 float face=smooth(.05f,.45f,p.z),cheek=gauss(fabsf(p.x),p.y,.46f,-.12f,.25f,.27f),nose=gauss(p.x,p.y,0.0f,.015f,.15f,.22f);
 float blood=face*(.07f*cheek+.055f*nose)+.022f*smooth(.75f,.9f,fabsf(p.x));
 float variation=detail*(.09f*broad+.075f*medium+.030f*fine);
 base=product(base,make_float3(1.0f+blood+variation,1.0f-blood*.30f+variation*.85f,1.0f-blood*.43f+variation*.70f));
 // Sparse pigment clusters, not a uniform spotted layer over the whole face.
 float pigment=smooth(.76f,.91f,noise3(mul(p,88.0f)))*face*cheek*detail;
 base=mul(base,1.0f-.12f*pigment);
 float under=face*gauss(fabsf(p.x),p.y,.325f,.16f,.20f,.085f);
 base=product(base,make_float3(1.0f-.018f*under,1.0f-.027f*under,1.0f-.015f*under));
 return base;
}
__device__ float3 skinDetailNormal(float3 p,float3 n,float detail,float footprint){
 float3 xy=poreField(p.x,p.y),zy=poreField(p.z,p.y),xz=poreField(p.x,p.z);
 float3 w=make_float3(powf(fabsf(n.x),4.0f),powf(fabsf(n.y),4.0f),powf(fabsf(n.z),4.0f));float sum=w.x+w.y+w.z+.000001f;w=mul(w,1.0f/sum);
 float3 gradient=make_float3(xy.y*w.z+xz.y*w.y,xy.z*w.z+zy.z*w.x,zy.y*w.x+xz.z*w.y);
 float oil=gauss(p.x,p.y,0.0f,.05f,.20f,.30f)+.35f*gauss(fabsf(p.x),p.y,.40f,.02f,.23f,.25f);
 float amount=detail*(2.6f+1.3f*oil)*detailFilter(155.0f,footprint);
 // Project the perturbation into the local tangent plane; no side-view seams.
 gradient=sub(gradient,mul(n,dot3(gradient,n)));
 return unit(sub(n,mul(gradient,amount)));
}
__device__ float skinDetailRoughness(float3 p,float rough,float detail){
 float face=smooth(.10f,.45f,p.z),oil=face*(.8f*gauss(p.x,p.y,0.0f,.04f,.19f,.33f)+.30f*gauss(p.x,p.y,0.0f,.83f,.50f,.20f));
 float dry=face*gauss(fabsf(p.x),p.y,.5f,-.3f,.25f,.3f);
 return fminf(.78f,fmaxf(.24f,rough+detail*((noise3(mul(p,29.0f))-.5f)*.05f-.065f*oil+.025f*dry)));
}
__device__ float3 eyeAlbedoFiltered(float3 p,float side,float gx,float gy,int colour,float pupil,float footprint){
 float dx=p.x-side*.325f-gx*.038f,dy=(p.y-.335f-gy*.030f)*HEIGHT_SCALE;
 float radius=sqrtf(dx*dx+dy*dy)/.081f,angle=atan2f(dy,dx),aa=fmaxf(.008f,footprint/.081f*.65f);
 float3 polar=make_float3(cosf(angle),sinf(angle),radius);
 float scleraNoise=noise3(make_float3(dx*47.0f,dy*47.0f,side*3.0f))-.5f;
 float3 sclera=mul(make_float3(.63f,.611f,.570f),1.0f+scleraNoise*.035f);
 float edge=smooth(1.20f,2.2f,radius),path=angle+.065f*sinf(radius*11.0f+angle*3.0f);
 float veins=powf(sat(.5f+.5f*sinf(path*23.0f+radius*9.0f)),36.0f)*edge*detailFilter(95.0f,footprint);
 sclera=mix3(sclera,make_float3(.42f,.17f,.14f),veins*.14f);
 sclera=mul(sclera,1.0f-.10f*edge);
 float3 dark=make_float3(.024f,.076f,.105f),light=make_float3(.11f,.22f,.26f);
 if(colour==1){dark=make_float3(.033f,.065f,.020f);light=make_float3(.17f,.21f,.081f);}
 if(colour==2){dark=make_float3(.065f,.037f,.012f);light=make_float3(.25f,.17f,.060f);}
 if(colour==3){dark=make_float3(.016f,.007f,.003f);light=make_float3(.100f,.041f,.012f);}
 float warp=(noise3(mul(polar,4.0f))-.5f)*.14f;
 float fibers=.5f+.22f*sinf((angle+warp)*119.0f+radius*18.0f)*detailFilter(240.0f,footprint);
 fibers+=.13f*sinf(angle*227.0f-radius*31.0f+2.0f*sinf(angle*17.0f))*detailFilter(450.0f,footprint);
 fibers+=.50f*(noise3(make_float3(polar.x*18.0f,polar.y*18.0f,radius*8.0f))-.5f);
 // Vary fiber bundles independently in each eye, with continuous polar coordinates.
 fibers+=.65f*(noise3(make_float3(polar.x*37.0f+side*13.0f,polar.y*37.0f,radius*4.0f))-.5f)*detailFilter(160.0f,footprint);
 float3 iris=mix3(dark,light,sat(fibers));
 float collarette=expf(-sq((radius-(pupil+.18f+.032f*sinf(angle*13.0f)))/.075f));
 iris=mix3(iris,make_float3(.19f,.105f,.032f),collarette*(colour==3?.14f:.28f));
 float crypts=powf(sat(.5f+.5f*sinf(angle*41.0f+2.0f*sinf(angle*9.0f))),7.0f)*expf(-sq((radius-.64f)/.16f));
 iris=mul(iris,1.0f-.27f*crypts*detailFilter(95.0f,footprint));
 float furrows=.5f+.5f*sinf(radius*96.0f+.9f*sinf(angle*11.0f));
 iris=mul(iris,1.0f-.055f*furrows*smooth(.62f,.84f,radius)*detailFilter(190.0f,footprint));
 iris=mul(iris,1.0f-.53f*smooth(.84f,1.02f,radius));
 float pupilEdge=pupil+.001f*sinf(angle*23.0f);
 iris=mix3(make_float3(.001f,.0015f,.002f),iris,smooth(pupilEdge-aa,pupilEdge+aa,radius));
 return mix3(iris,sclera,smooth(1.0f-aa,1.0f+aa,radius));
}
// Unfiltered reference entry for CPU oracles and material probes.
__device__ float3 eyeAlbedo(float3 p,float side,float gx,float gy,int colour,float pupil){return eyeAlbedoFiltered(p,side,gx,gy,colour,pupil,0.0f);}
