"""Visual surface detail only: CC0 appearance analogues, not product scans.

Noncontact tile bevel/relief and a continuous mesh of the mortar height field do
not alter collision geometry, mass, material volume or constitutive parameters.
"""
from functools import lru_cache
from pathlib import Path
import mujoco
import numpy as np
from PIL import Image
from scipy.ndimage import map_coordinates

from .paths import WORKSPACE_ROOT
ASSETS=WORKSPACE_ROOT/'assets/tiling_surfaces'

def materials(spec):
    for name,file in [('tile_surface','terracotta_floor_tiles_diff_4k.png'),('mortar_surface','scuffed_cement_diff_1k.png')]:
        spec.add_texture(name=name,type=mujoco.mjtTexture.mjTEXTURE_2D,file=str(ASSETS/file))
    a=spec.add_material(name='tile_photo',rgba=[1,1,1,1],specular=.08,shininess=.12)
    a.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB]='tile_surface'
    b=spec.add_material(name='mortar_photo',rgba=[.9,.92,.94,1],specular=.16,shininess=.22)
    b.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB]='mortar_surface'


@lru_cache(maxsize=1)
def displacement():return np.asarray(Image.open(ASSETS/'terracotta_floor_tiles_disp_4k.jpg').convert('L'),dtype=float)

def add_tile(spec,body,index,half):
    # Select one photographed tile interior by UV, excluding the atlas's joints.
    col,row=[(1,1),(3,2),(5,4),(7,6)][index]
    u0,u1=(col+.10)/9,(col+.90)/9
    v0,v1=1-(row+.90)/9,1-(row+.10)/9
    n=25;a=np.linspace(-1,1,n);xx,yy=np.meshgrid(a,a,indexing='ij')
    uv=np.c_[u0+(xx.ravel()+1)/2*(u1-u0),v0+(yy.ravel()+1)/2*(v1-v0)]
    disp=displacement()
    sampled=map_coordinates(disp,[(1-uv[:,1])*(disp.shape[0]-1),uv[:,0]*(disp.shape[1]-1)],order=1)
    z=half[2]+(sampled-sampled.mean())/255*.00012
    verts=np.c_[xx.ravel()*(half[0]-.0008),yy.ravel()*(half[1]-.0008),z]
    faces=[]
    for i in range(n-1):
        for j in range(n-1):
            q=i*n+j;faces.extend([[q,q+n,q+1],[q+1,q+n,q+n+1]])
    # Closed sides with a small visual bevel; the contact box remains unchanged.
    boundary=list(range(0,n))+[i*n+n-1 for i in range(1,n)]+list(range(n*n-2,(n-1)*n-1,-1))+[i*n for i in range(n-2,0,-1)]
    for level,scale in [(half[2]-.0008,1.),(-half[2],.997)]:
        start=len(verts);ring=np.array([[np.sign(verts[k,0])*half[0]*scale if abs(verts[k,0])>half[0]-.001 else verts[k,0],np.sign(verts[k,1])*half[1]*scale if abs(verts[k,1])>half[1]-.001 else verts[k,1],level] for k in boundary])
        previous=boundary if start==n*n else list(range(start-len(boundary),start))
        verts=np.r_[verts,ring];uv=np.r_[uv,uv[boundary]]
        for k in range(len(boundary)):
            b=(k+1)%len(boundary);faces.extend([[previous[k],previous[b],start+k],[previous[b],start+b,start+k]])
    last=len(verts)-len(boundary)
    for k in range(1,len(boundary)-1):faces.append([last,last+k,last+k+1])
    spec.add_mesh(name=f'tile_detail_{index}',uservert=verts.ravel().tolist(),userface=np.array(faces).ravel().tolist(),usertexcoord=uv.ravel().tolist(),userfacetexcoord=np.array(faces).ravel().tolist(),smoothnormal=True)
    body.add_geom(name=f'tile_visual_{index}',type=mujoco.mjtGeom.mjGEOM_MESH,meshname=f'tile_detail_{index}',material='tile_photo',mass=0,contype=0,conaffinity=0)


class MortarSurface:
    n=65
    @classmethod
    def add(cls,spec,targets,width,surface):
        a=np.linspace(-width/2,width/2,cls.n);xx,yy=np.meshgrid(a,a,indexing='ij')
        top=np.c_[xx.ravel(),yy.ravel(),np.zeros(xx.size)]
        corners=np.array([[-width/2,-width/2,-.001],[-width/2,width/2,-.001],[width/2,width/2,-.001],[width/2,-width/2,-.001]])
        verts=np.r_[top,corners];faces=[]
        for i in range(cls.n-1):
            for j in range(cls.n-1):
                q=i*cls.n+j;faces.extend([[q,q+cls.n,q+1],[q+1,q+cls.n,q+cls.n+1]])
        boundary=list(range(cls.n))+[i*cls.n+cls.n-1 for i in range(1,cls.n)]+list(range(cls.n**2-2,(cls.n-1)*cls.n-1,-1))+[i*cls.n for i in range(cls.n-2,0,-1)]
        # A closed thin backing is below the substrate and carries no physics.
        for k,p in enumerate(boundary):
            q=boundary[(k+1)%len(boundary)];corner=cls.n**2+(k//(cls.n-1))%4
            faces.append([p,q,corner])
        for side in range(4):faces.append([boundary[((side+1)*(cls.n-1))%len(boundary)],cls.n**2+(side+1)%4,cls.n**2+side])
        faces.extend([[cls.n**2,cls.n**2+1,cls.n**2+2],[cls.n**2,cls.n**2+2,cls.n**2+3]])
        faces=np.array(faces);normals=np.tile([0.,0.,1.],(len(verts),1))
        for i,t in enumerate(targets):
            uv=np.c_[(verts[:,0]+t[0])/1.6,(verts[:,1]+t[1])/1.6]
            spec.add_mesh(name=f'mortar_surface_{i}',uservert=verts.ravel().tolist(),userface=faces.ravel().tolist(),usernormal=normals.ravel().tolist(),userfacenormal=faces.ravel().tolist(),usertexcoord=uv.ravel().tolist(),userfacetexcoord=faces.ravel().tolist(),smoothnormal=True)
            spec.worldbody.add_geom(name=f'mortar_surface_{i}',type=mujoco.mjtGeom.mjGEOM_MESH,meshname=f'mortar_surface_{i}',material='mortar_photo',pos=[t[0],t[1],surface],mass=0,contype=0,conaffinity=0,rgba=[1,1,1,0])
    def __init__(self,model,data,layer):
        self.model=model;self.data=data;self.layer=layer
        self.mesh_ids=[model.mesh(f'mortar_surface_{i}').id for i in range(4)]
        self.geom_ids=[model.geom(f'mortar_surface_{i}').id for i in range(4)]
        a=np.linspace(-layer.width/2,layer.width/2,self.n)
        xx,yy=np.meshgrid(a,a,indexing='ij');self.xy=np.c_[xx.ravel(),yy.ravel()]
        self.coords=[(xx.ravel()+layer.width/2)/layer.dx-.5,(yy.ravel()+layer.width/2)/layer.dx-.5]
    def update(self):
        m=self.model;d=self.data
        for i,(mid,gid) in enumerate(zip(self.mesh_ids,self.geom_ids)):
            h=map_coordinates(self.layer.height[i].reshape(self.layer.cells,self.layer.cells),self.coords,order=1,mode='nearest').reshape(self.n,self.n)
            xyz=np.c_[self.xy+self.layer.targets[i,:2],self.layer.surface+np.where(h.ravel()>1e-7,h.ravel(),-.00002)]
            R=d.geom_xmat[gid].reshape(3,3);adr=m.mesh_vertadr[mid]
            m.mesh_vert[adr:adr+self.n**2]=(xyz-d.geom_xpos[gid])@R
            dx=self.layer.width/(self.n-1);gx,gy=np.gradient(h,dx,dx)
            normals=np.c_[-gx.ravel(),-gy.ravel(),np.ones(self.n**2)];normals/=np.linalg.norm(normals,axis=1)[:,None]
            na=m.mesh_normaladr[mid]
            if m.mesh_normalnum[mid]!=m.mesh_vertnum[mid]:raise RuntimeError('Mortar normal indexing changed during mesh compilation')
            m.mesh_normal[na:na+self.n**2]=normals@R
            m.geom_rgba[gid,3]=float(self.layer.volume[i].sum()>1e-12)
        return self.mesh_ids
