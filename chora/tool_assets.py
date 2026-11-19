"""Imported textured tool meshes; physical contact proxies are separate."""
from functools import lru_cache
import numpy as np
import mujoco
from scipy.spatial import ConvexHull
from .common import ROOT

# Effective collision-edge truncation, not a measured blade sharpness.
KNIFE_EDGE_HALF_THICKNESS_M=15e-6

@lru_cache(maxsize=2)
def geometry(kind):
    path=ROOT/('assets/ycb/048_hammer/google_16k/textured.obj' if kind=='hammer' else 'assets/kitchen_knife/knife.obj')
    if not path.is_file():
        raise FileNotFoundError('Imported knife asset is local-only because its license is unresolved; see docs/ASSETS.md. No substitute tool is installed automatically.' if kind=='knife' else 'Hammer asset missing: '+str(path))
    v=[];uv=[];faces=[];fuv=[]
    for line in path.read_text().splitlines():
        p=line.split()
        if not p:continue
        if p[0]=='v':v.append(list(map(float,p[1:4])))
        elif p[0]=='vt':uv.append(list(map(float,p[1:3])))
        elif p[0]=='f':
            entries=[q.split('/') for q in p[1:]]
            for i in range(1,len(entries)-1):
                tri=[entries[j] for j in [0,i,i+1]];faces.append([int(q[0])-1 for q in tri]);fuv.append([int(q[1])-1 for q in tri])
    v=np.array(v)
    if kind=='hammer':
        axis=np.array([-.39,.92,0.]);axis/=np.linalg.norm(axis)
    else:
        _,R=np.linalg.eigh(np.cov(v.T));axis=R[:,-1];axis*=np.sign(axis[0]);axis[2]=0;axis/=np.linalg.norm(axis)
    across=np.array([0.,0.,-1.]);up=np.cross(axis,across);R=np.array([axis,across,up]);v=v@R.T
    if kind=='hammer':
        # The scanned flat striking face is the lowest steel surface.
        face=v[v[:,2]<v[:,2].min()+.002];origin=face.mean(0);origin[2]=v[:,2].min()
    else:
        # Blade's cutting edge, separated from the black handle.
        x=v[:,0].max()-.12;band=v[np.abs(v[:,0]-x)<.015];origin=np.array([x,0.,band[:,2].min()])
    v-=origin
    uv=np.array(uv)
    if kind=='hammer':uv[:,1]=1-uv[:,1]
    return v,np.array(faces),uv,np.array(fuv)


def add_tool(spec,body,kind):
    v,f,uv,fu=geometry(kind)
    texture=ROOT/('assets/ycb/048_hammer/google_16k/texture_map.png' if kind=='hammer' else 'assets/kitchen_knife/basecolor.png')
    spec.add_texture(name='tool_rgb',type=mujoco.mjtTexture.mjTEXTURE_2D,file=str(texture))
    spec.add_material(name='tool_material',textures=['','tool_rgb','','','','','','','',''],specular=.3,shininess=.3)
    spec.add_mesh(name='real_tool',uservert=v.ravel(),userface=f.ravel(),usertexcoord=uv.ravel(),userfacetexcoord=fu.ravel(),smoothnormal=True)
    body.add_geom(name='tool_visual',type=mujoco.mjtGeom.mjGEOM_MESH,meshname='real_tool',material='tool_material',contype=0,conaffinity=0,mass=0)
    if kind=='hammer':
        body.add_geom(name='tool_contact',type=mujoco.mjtGeom.mjGEOM_CYLINDER,pos=[0,0,.009],size=[.014,.009,0],mass=.7,contype=2,conaffinity=5,solref=[.0005,1],solimp=[.98,.995,.001,.5,2])
    else:
        # Convex blade-profile proxy follows the imported cutting edge; it does
        # not extend a phantom box underneath the knife handle.
        blade=v[v[:,0]>=-.10];hull=ConvexHull(blade[:,[0,2]]);profile=blade[hull.vertices][:,[0,2]]
        # Lower silhouette is the cutting edge. Taper from a finite, explicit
        # 30 µm truncated edge to the 1.2 mm blade body over 15 mm of height.
        lower=[]
        for i,a in enumerate(profile):
            b=profile[(i+1)%len(profile)]
            if b[0]>a[0]:lower.append((a,b))
        def edge_height(x):
            values=[a[1]+(b[1]-a[1])*(x-a[0])/(b[0]-a[0]) for a,b in lower if a[0]-1e-8<=x<=b[0]+1e-8]
            return min(values) if values else float(profile[:,1].min())
        cv=[]
        for sign in [-1,1]:
            for x,z in profile:
                width=KNIFE_EDGE_HALF_THICKNESS_M+(.0006-KNIFE_EDGE_HALF_THICKNESS_M)*np.clip((z-edge_height(x))/.015,0,1)
                cv.append([x,sign*width,z])
        cv=np.array(cv);cf=ConvexHull(cv)
        faces=cf.simplices.copy()
        for i,face in enumerate(faces):
            if np.dot(np.cross(cv[face[1]]-cv[face[0]],cv[face[2]]-cv[face[0]]),cf.equations[i,:3])<0:faces[i]=face[::-1]
        spec.add_mesh(name='blade_contact_mesh',uservert=cv.ravel(),userface=faces.ravel())
        body.add_geom(name='tool_contact',type=mujoco.mjtGeom.mjGEOM_MESH,meshname='blade_contact_mesh',mass=.25,contype=2,conaffinity=5,friction=[.15,.001,.0001],solref=[.002,1])
    # The real handle is enclosed by a visible, bolted tool-holder. No grasp claim.
    handle=v[v[:,0]<np.quantile(v[:,0],.17)].mean(0)
    body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX,pos=handle+[0,0,.005],size=[.022,.027,.018],rgba=[.13,.17,.19,1],mass=0,contype=0,conaffinity=0)
    for x in [-.012,.012]:
        for y in [-.018,.018]:body.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER,pos=handle+[x,y,.025],size=[.003,.002,0],rgba=[.6,.65,.68,1],mass=0,contype=0,conaffinity=0)
    return handle
