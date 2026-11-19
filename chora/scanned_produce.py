"""CC0 produce surfaces. These assets do not supply material mechanics.

Keep source geometry proportions, orient in the work frame and uniformly scale
to the trial's nominal height. A coarse FEM envelope is a separate approximation.
"""
import json
from functools import lru_cache
import numpy as np
import mujoco
from .common import ROOT

ASSETS={'apple':('food_apple_01','food_apple_01'),'banana':('bananas','bananas_b')}

@lru_cache(maxsize=2)
def source_surface(kind):
    name,node_name=ASSETS[kind];folder=ROOT/'assets/produce'/name
    g=json.loads((folder/(name+'_2k.gltf')).read_text());buffers=[(folder/b['uri']).read_bytes() for b in g['buffers']]
    def read(index):
        a=g['accessors'][index];b=g['bufferViews'][a['bufferView']]
        dtype=np.dtype({5126:'<f4',5125:'<u4',5123:'<u2',5121:'u1'}[a['componentType']]);n={'SCALAR':1,'VEC2':2,'VEC3':3,'VEC4':4}[a['type']]
        return np.ndarray((a['count'],n),dtype=dtype,buffer=buffers[b['buffer']],offset=b.get('byteOffset',0)+a.get('byteOffset',0),strides=(b.get('byteStride',n*dtype.itemsize),dtype.itemsize)).copy()
    node=next(n for n in g['nodes'] if n.get('name')==node_name);p=g['meshes'][node['mesh']]['primitives'][0]
    v=read(p['attributes']['POSITION']).astype(float);v=v[:,[0,2,1]];v[:,1]*=-1 # glTF Y-up → Z-up
    v-=.5*(v.min(0)+v.max(0))
    if kind=='banana':
        _,R=np.linalg.eigh(np.cov(v[:,:2].T));axis=R[:,-1];across=np.array([axis[1],-axis[0]])
        v[:,:2]=v[:,:2]@np.stack([across,axis],axis=1)
        v-=.5*(v.min(0)+v.max(0))
    uv=read(p['attributes']['TEXCOORD_0']).astype(float)
    f=read(p['indices']).reshape(-1,3).astype(int)
    return v,uv,f,folder/'textures'/(name+'_diff_2k.jpg')

def dimensions(kind,height):
    v,*_=source_surface(kind);extent=np.ptp(v,axis=0)*height/np.ptp(v[:,2])
    return dict(radius=float(extent[0]/2),height=height,aspect=float(extent[1]/extent[0]),visual_asset=ASSETS[kind][0])

def surface(kind,height,side=None):
    v,uv,faces,_=source_surface(kind);v=v*(height/np.ptp(v[:,2]))
    if side is None:return v,uv,faces
    # Clip triangles at the actual cut plane, interpolating both positions/UVs.
    vertices=[];coords=[];triangles=[]
    for face in faces:
        poly=[(v[i],uv[i]) for i in face];out=[]
        for i,(a,ta) in enumerate(poly):
            b,tb=poly[(i+1)%len(poly)];ina=side*a[1]>=0;inb=side*b[1]>=0
            if ina:out.append((a,ta))
            if ina!=inb:
                t=-a[1]/(b[1]-a[1]);out.append((a+t*(b-a),ta+t*(tb-ta)))
        if len(out)<3:continue
        start=len(vertices)
        for p,t in out:vertices.append(p);coords.append(t)
        triangles.extend([[start,start+i,start+i+1] for i in range(1,len(out)-1)])
    return np.array(vertices),np.array(coords),np.array(triangles)

def material(s,kind):
    from PIL import Image
    path=source_surface(kind)[3];png=path.with_suffix('.png')
    if not png.exists():Image.open(path).convert('RGB').save(png)
    name='scan_'+kind;s.add_texture(name=name,type=mujoco.mjtTexture.mjTEXTURE_2D,file=str(png))
    s.add_material(name=name,textures=['',name,'','','','','','','',''],specular=.25,roughness=.65)
    return name
