from pxr import Usd,UsdGeom
import numpy as np
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
s=Usd.Stage.Open(str(ROOT/'assets/kitchen_knife/source.usd'));p=s.GetPrimAtPath('/Piece/Visual');mesh=UsdGeom.Mesh(p)
v=np.array(mesh.GetPointsAttr().Get());m=np.array(UsdGeom.Xformable(p).ComputeLocalToWorldTransform(Usd.TimeCode.Default()));v=(np.c_[v,np.ones(len(v))]@m)[:,:3]
uvp=UsdGeom.PrimvarsAPI(p).GetPrimvar('st');uv=np.array(uvp.Get());uvi=uvp.GetIndices();uvi=np.array(uvi) if uvi else None
indices=np.array(mesh.GetFaceVertexIndicesAttr().Get());counts=np.array(mesh.GetFaceVertexCountsAttr().Get());print('bbox',v.min(0),v.max(0),'UV',uv.shape,uvp.GetInterpolation(), 'faces',counts.shape)
lines=['# RoboCasa knife_22 via EmbodiedSWE/robobench-assets; see SOURCE-NOTICE.txt']
lines+=['v '+' '.join(map(str,x)) for x in v];lines+=['vt '+' '.join(map(str,x)) for x in uv]
cursor=0
for n in counts:
 face=indices[cursor:cursor+n];t=(uvi[cursor:cursor+n] if uvi is not None else np.arange(cursor,cursor+n)) if uvp.GetInterpolation()=='faceVarying' else face
 for k in range(1,n-1):lines.append('f '+' '.join(f'{face[j]+1}/{t[j]+1}' for j in [0,k,k+1]))
 cursor+=n
(ROOT/'assets/kitchen_knife/knife.obj').write_text('\n'.join(lines)+'\n')
