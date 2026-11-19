"""Deterministic procedural appearance maps and mesh surfaces (not photo scans)."""
from functools import lru_cache
import numpy as np
import mujoco
from PIL import Image, ImageDraw
from .common import ROOT

@lru_cache(maxsize=None)
def texture_path(kind,inside=False):
    folder=ROOT/'assets/chora';folder.mkdir(parents=True,exist_ok=True)
    path=folder/(kind+('-inside' if inside else '-skin')+'.png')
    rng=np.random.default_rng(512+sum(map(ord,kind))+int(inside));n=512
    y,x=np.mgrid[-1:1:complex(n),-1:1:complex(n)];noise=rng.normal(0,1,(n,n))
    if kind=='pine':
        grain=np.sin(90*y+4*np.sin(5*x)+2*np.sin(11*x))*.5+.3*np.sin(210*y+8*x)
        a=np.array([192,143,87])+grain[...,None]*np.array([25,24,18])+noise[...,None]*2
    elif not inside:
        if kind=='apple':
            blush=(np.sin(22*x+3*np.sin(4*y))+np.sin(7*y))/2
            a=np.stack([165+35*blush,42+28*blush,24+8*blush],-1)+noise[...,None]*3
            fleck=rng.random((n,n))>.994;a[fleck]=[215,171,94]
        elif kind=='tomato':a=np.array([201,39,24])+np.sin(4*x+3*y)[...,None]*np.array([18,5,2])+noise[...,None]*2
        else:
            a=np.array([230,196,45])+np.sin(8*x)[...,None]*np.array([9,10,3])+noise[...,None]*2
            for _ in range(60):
                cx,cy=rng.uniform(-1,1,2);r=rng.uniform(.005,.025);a[(x-cx)**2+(y-cy)**2<r*r]=[112,77,30]
    else:
        r=np.sqrt(x*x+y*y);theta=np.arctan2(y,x)
        if kind=='apple':
            a=np.array([239,226,184])+noise[...,None]*2
            core=(r<.24+.035*np.cos(5*theta));a[core]=[220,208,157]
            for ang in np.linspace(0,2*np.pi,6)[:-1]:
                cx,cy=.14*np.cos(ang),.14*np.sin(ang);seed=((x-cx)/.025)**2+((y-cy)/.05)**2<1;a[seed]=[79,45,24]
        elif kind=='tomato':
            a=np.array([218,62,40])+noise[...,None]*3
            locule=(r>.28)&(r<.77)&(np.cos(3*theta)>.0);a[locule]=[170,127,54]
            for _ in range(65):
                ang=rng.uniform(0,2*np.pi);rad=rng.uniform(.35,.68);cx,cy=rad*np.cos(ang),rad*np.sin(ang)
                if np.cos(3*ang)>0:a[((x-cx)/.018)**2+((y-cy)/.03)**2<1]=[232,203,116]
            a[r<.22]=[236,178,126]
        else:
            a=np.array([241,227,171])+noise[...,None]*2
            core=(r<.25+.09*np.cos(3*theta));a[core]=[207,177,114]
            a[r>.90]=[215,182,49]
    Image.fromarray(np.uint8(np.clip(a,0,255))).save(path)
    return path


def material(spec,kind,inside=False):
    name=kind+('_inside' if inside else '_skin')
    spec.add_texture(name=name,type=mujoco.mjtTexture.mjTEXTURE_2D,file=str(texture_path(kind,inside)))
    spec.add_material(name=name,textures=['',name,'','','','','','','',''],texuniform=False,roughness=.6,specular=.18 if inside else .3,shininess=.15)
    return name


def fruit_half_mesh(spec,kind,side,radius,height):
    """Closed half mesh with skin UVs; separate face has planar interior UVs."""
    vertices=[];uv=[];faces=[];rings=25;segments=49
    for i in range(rings):
        phi=.0001+(np.pi-.0002)*i/(rings-1)
        for j in range(segments):
            theta=(0 if side==1 else np.pi)+np.pi*j/(segments-1)
            radial=radius*np.sin(phi)*(1+.035*np.cos(5*theta)*np.sin(phi) if kind=='apple' else 1)
            xx=radial*np.cos(theta);yy=radial*np.sin(theta);zz=height*.5*np.cos(phi)
            if kind=='banana':yy*=2.7;zz+=.004*(yy/(radius*2.7))**2
            vertices.append([xx,yy,zz]);uv.append([j/(segments-1),i/(rings-1)])
    for i in range(rings-1):
        for j in range(segments-1):
            a=i*segments+j;b=a+segments;faces.extend([[a,b,a+1],[a+1,b,b+1]])
    # Close the flat face using the boundary ring, with outward winding.
    boundary=[i*segments for i in range(rings)]+[i*segments+segments-1 for i in range(rings-1,-1,-1)]
    center=len(vertices);vertices.append([0,0,0]);uv.append([.5,.5])
    for i,b in enumerate(boundary):faces.append([center,b,boundary[(i+1)%len(boundary)]])
    name=f'{kind}_{side}'
    spec.add_mesh(name=name,uservert=np.array(vertices).ravel(),userface=np.array(faces).ravel(),usertexcoord=np.array(uv).ravel(),userfacetexcoord=np.array(faces).ravel(),smoothnormal=True)
    # Planar disk, slightly outside the half so internal face wins depth testing.
    v=[[0,-side*.00005,0]];uv=[[.5,.5]]
    for a in np.linspace(0,2*np.pi,97)[:-1]:
        xx=radius*np.cos(a);zz=height*.5*np.sin(a)
        v.append([xx,-side*.00005,zz]);uv.append([.5+.5*np.cos(a),.5+.5*np.sin(a)])
    f=[[0,j+1,(j+1)%96+1] for j in range(96)]
    if side<0:f=[t[::-1] for t in f]
    spec.add_mesh(name=name+'_face',uservert=np.array(v).ravel(),userface=np.array(f).ravel(),usertexcoord=np.array(uv).ravel(),userfacetexcoord=np.array(f).ravel(),smoothnormal=False,inertia=mujoco.mjtMeshInertia.mjMESH_INERTIA_SHELL)
    return name
