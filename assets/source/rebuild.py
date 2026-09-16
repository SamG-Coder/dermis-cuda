import numpy as np
from pathlib import Path
p=Path(__file__).resolve().parent;verts=[];groups={};g=''
for line in (p/'base.obj').read_text().splitlines():
 if line.startswith('v '):verts.append(list(map(float,line.split()[1:4])))
 elif line.startswith('g '):g=line[2:];groups[g]=[]
 elif line.startswith('f '):groups[g].append([int(x.split('/')[0])-1 for x in line.split()[1:]])
verts=np.array(verts)
for file in ['male.target','average.target']:
 for line in (p/file).read_text().splitlines():
  if not line or line.startswith('#'):continue
  v,*delta=line.split();verts[int(v)]+=np.array(list(map(float,delta)))

import json
deltas=np.zeros_like(verts)
for file in ['eye-left-closure.target','eye-right-closure.target']:
 for line in (p/file).read_text().splitlines():
  if not line or line.startswith('#'):continue
  idx,*delta=line.split();deltas[int(idx)]+=np.array(list(map(float,delta)))
v=np.concatenate([verts,deltas],axis=1);faces=groups['body']
# Keep the connected head, neck and upper shoulders; discard helper meshes.
faces=[list(f) for f in faces if np.min(v[f,1])>5.85]
ids=sorted(set(i for f in faces for i in f));remap={old:i for i,old in enumerate(ids)};v=v[ids];faces=[[remap[i] for i in f] for f in faces]
v[:,0]*=.325/.293125;v[:,1]=(v[:,1]-8.2164)*(.325/.293125)+.335;v[:,2]=v[:,2]*(.325/.293125)-1.011
v[:,3:]*=.325/.293125
# Catmull-Clark subdivision produces smooth anatomical contours without scans.
def subdiv(v,faces):
 centers=np.array([v[f].mean(axis=0) for f in faces]);edges={};vf=[[] for _ in v]
 for fi,f in enumerate(faces):
  for j,x in enumerate(f):
   vf[x].append(fi);edge=tuple(sorted((x,f[(j+1)%len(f)])));edges.setdefault(edge,[]).append(fi)
 ve=[[] for _ in v];boundary=[[] for _ in v]
 for (x,y),adj in edges.items():
  ve[x].append((x,y));ve[y].append((x,y))
  if len(adj)==1:boundary[x].append(y);boundary[y].append(x)
 out=[]
 for i,pos in enumerate(v):
  if boundary[i]:out.append(pos*.75+v[boundary[i]].mean(axis=0)*.25)
  else:
   n=len(vf[i]);F=centers[vf[i]].mean(axis=0);R=np.array([(v[x]+v[y])*.5 for x,y in ve[i]]).mean(axis=0);out.append((F+2*R+(n-3)*pos)/n)
 edgeids={}
 for (x,y),adj in edges.items():
  edgeids[(x,y)]=len(out);out.append((v[x]+v[y]+centers[adj].sum(axis=0))/(2+len(adj)) if len(adj)==2 else (v[x]+v[y])*.5)
 ci=len(out);out.extend(centers);new=[]
 for fi,f in enumerate(faces):
  for j,x in enumerate(f):new.append([x,edgeids[tuple(sorted((x,f[(j+1)%len(f)])))],ci+fi,edgeids[tuple(sorted((f[j-1],x)))]])
 return np.array(out),new
for _ in range(2):v,faces=subdiv(v,faces)
blink=v[:,3:].copy();v=v[:,:3].copy()
tri=np.array([[f[0],f[1],f[2]] for f in faces]+[[f[0],f[2],f[3]] for f in faces]);n=np.zeros_like(v)
fn=np.cross(v[tri[:,1]]-v[tri[:,0]],v[tri[:,2]]-v[tri[:,0]])
for k in range(3):np.add.at(n,tri[:,k],fn)
n/=np.maximum(1e-12,np.linalg.norm(n,axis=1))[:,None]
closed=v+blink;cn=np.zeros_like(v);cf=np.cross(closed[tri[:,1]]-closed[tri[:,0]],closed[tri[:,2]]-closed[tri[:,0]])
for k in range(3):np.add.at(cn,tri[:,k],cf)
cn/=np.maximum(1e-12,np.linalg.norm(cn,axis=1))[:,None]
out={'blink':np.round(blink,6).reshape(-1).tolist(),'closedNormals':np.round(cn,6).reshape(-1).tolist(),'positions':np.round(v,6).reshape(-1).tolist(),'normals':np.round(n,6).reshape(-1).tolist(),'indices':tri.reshape(-1).tolist(),'source':'MakeHuman hm08 CC0 base mesh with adult male macro targets; cropped and Catmull-Clark subdivided twice.'}
dest=p.parent
(dest/'male-anatomy.json').write_text(json.dumps(out,separators=(',',':')))
print(len(v),'vertices',len(tri),'triangles',v.min(axis=0),v.max(axis=0))
