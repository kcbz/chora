"""Separate, reproducible robot-model screening. Does not alter Chora scenes.

Pose IK/collision screening and quasi-static loads, NOT full manipulation or
impact validation. Native pinned models, bare flanges, explicit tool TCP/CoM.
"""
import argparse
from collections import Counter
import hashlib,itertools,json,time
from pathlib import Path
import mujoco
import numpy as np
from scipy.optimize import least_squares,lsq_linear
from scipy.spatial.transform import Rotation
from .common import ROOT

OUT=ROOT/'outputs/robot_evaluation'
REVISION='370c43965323c0944b433b202aac5058b5f907a9'
MODELS={
 'panda':dict(name='Franka Panda',path='franka_emika_panda/panda_nohand.xml',flange='attachment_site',velocity=[2.175]*4+[2.61]*3),
 'fr3':dict(name='Franka FR3 v2',path='franka_fr3_v2/fr3v2.xml',body='fr3v2_link8',velocity=[2.62]*4+[5.26,4.18,5.26]),
 'ur5e':dict(name='UR5e · pinned legacy model',path='universal_robots_ur5e/ur5e.xml',flange='attachment_site',velocity=[np.pi]*6),
 'ur10e':dict(name='UR10e · pinned model',path='universal_robots_ur10e/ur10e.xml',flange='attachment_site',velocity=[2*np.pi/3]*2+[np.pi]*4),
 'iiwa14':dict(name='KUKA iiwa 14',path='kuka_iiwa_14/iiwa14.xml',flange='attachment_site',velocity=np.deg2rad([85,85,100,75,130,135,135]).tolist()),
}
TASKS={
 'bowl':dict(tcp=[0,0,.15],com=[0,0,.08],mass=1.5,reaction=[0,0,0]),
 'cutting':dict(tcp=[.17,0,.12],com=[.06,0,.08],mass=1.1,reaction=[10,0,60]),
 'hammer':dict(tcp=[.23,0,.12],com=[.11,0,.10],mass=1.4,reaction=[0,0,100]),
 'pouring':dict(tcp=[0,0,.16],com=[.03,0,.10],mass=1.8,reaction=[0,0,0]),
}
DOWN=np.diag([1.,-1.,-1.])

def build_spec(key,base_x=0.):
    meta=MODELS[key];path=ROOT/'assets/menagerie'/meta['path'];s=mujoco.MjSpec.from_file(str(path))
    root={"panda":"link0","fr3":"fr3v2_link0","ur5e":"base","ur10e":"base","iiwa14":"base"}[key]
    s.body(root).pos[0]+=base_x
    if 'flange' in meta:
        site=s.site(meta['flange']);body=site.parent;pos=np.array(site.pos);quat=np.array(site.quat)
    else:body=s.body(meta['body']);pos=np.zeros(3);quat=np.array([1.,0,0,0])
    R=np.zeros(9);mujoco.mju_quat2Mat(R,quat);R=R.reshape(3,3)
    for name,t in TASKS.items():body.add_site(name='tcp_'+name,pos=pos+R@t['tcp'],quat=quat,size=[.005]*3,rgba=[.2,.8,.55,1])
    body.add_site(name='eval_flange',pos=pos,quat=quat,size=[.003]*3)
    add_bench(s)
    return s


def add_bench(s):
    s.worldbody.add_geom(name='eval_floor',type=mujoco.mjtGeom.mjGEOM_PLANE,size=[2,2,.01],rgba=[.16,.20,.22,1])
    s.worldbody.add_geom(name='eval_bench',type=mujoco.mjtGeom.mjGEOM_BOX,pos=[.50,0,.05],size=[.28,.32,.05],rgba=[.37,.42,.40,1])


def poses(task):
    if task=='bowl':return [(list(p),DOWN.copy()) for p in itertools.product([.35,.50,.65],[-.15,.15],[.16,.32])]
    if task=='cutting':return [(list(p),DOWN.copy()) for p in itertools.product([.40,.50,.60],[-.10,.10],[.16,.24,.32])]
    if task=='hammer':return [(list(p),DOWN.copy()) for p in itertools.product([.40,.50,.60],[-.10,.10],[.16,.28])]
    return [([x,y,.38],Rotation.from_euler('y',tilt,degrees=True).as_matrix()@DOWN) for x,y,tilt in itertools.product([.45,.55],[-.10,.10],[0,30,60,90,110])]


def contacts(m,d):
    return [{'bodies':[m.body(int(m.geom_bodyid[g])).name for g in (c.geom1,c.geom2)],'depth_m':round(float(-c.dist),6)} for c in d.contact if c.dist<-.0001]


def solve(m,d,site_ids,targets,rng,seed_q=None,attempts=10):
    lo=m.jnt_range[:,0]+.025;hi=m.jnt_range[:,1]-.025
    home=m.key_qpos[0].copy();best=None
    def residual(q):
        d.qpos[:]=q;mujoco.mj_kinematics(m,d)
        return np.concatenate([np.r_[d.site_xpos[sid]-pos,.15*Rotation.from_matrix(R@d.site_xmat[sid].reshape(3,3).T).as_rotvec()] for sid,(pos,R) in zip(site_ids,targets)])
    seeds=[home if seed_q is None else seed_q,home]
    for attempt in range(attempts):
        initial=seeds[attempt] if attempt<len(seeds) else (home+rng.normal(0,1.1,m.nq) if attempt<6 else rng.uniform(lo,hi))
        q=np.clip(initial,lo+.0001,hi-.0001)
        result=least_squares(residual,q,bounds=(lo,hi),max_nfev=100,ftol=1e-8,xtol=1e-8,gtol=1e-8)
        residual(result.x)
        errors=[(float(np.linalg.norm(d.site_xpos[sid]-pos)),float(np.linalg.norm(Rotation.from_matrix(R@d.site_xmat[sid].reshape(3,3).T).as_rotvec()))) for sid,(pos,R) in zip(site_ids,targets)]
        pose_ok=all(p<.002 and a<np.deg2rad(2) for p,a in errors)
        if pose_ok:
            d.qvel[:]=0;mujoco.mj_forward(m,d);coll=contacts(m,d)
            row=dict(qpos=result.x.tolist(),position_error_m=max(x[0] for x in errors),rotation_error_deg=np.rad2deg(max(x[1] for x in errors)),collisions=coll,pose_ok=True,collision_free=not coll,attempt=attempt+1)
            if not coll:return row
            if best is None or len(coll)<len(best['collisions']):best=row
    if best:return best
    return dict(qpos=result.x.tolist(),position_error_m=max(x[0] for x in errors),rotation_error_deg=float(np.rad2deg(max(x[1] for x in errors))),collisions=[],pose_ok=False,collision_free=False,attempt=attempts)


def torque_limits(m):
    limits=np.full(m.nv,np.nan)
    for a in range(m.nu):
        if m.actuator_forcelimited[a] and m.actuator_trntype[a]==mujoco.mjtTrn.mjTRN_JOINT:
            joint=int(m.actuator_trnid[a,0]);dof=int(m.jnt_dofadr[joint]);limits[dof]=max(abs(m.actuator_forcerange[a]))*abs(m.actuator_gear[a,0])
    return limits


def loads(m,d,site,task,velocity_limits):
    t=TASKS[task];flange=m.site('eval_flange').id;R=d.site_xmat[flange].reshape(3,3);body=int(m.site_bodyid[flange])
    com=d.site_xpos[flange]+R@t['com'];Jcom=np.zeros((3,m.nv));mujoco.mj_jac(m,d,Jcom,None,com,body)
    Jp=np.zeros((3,m.nv));Jr=Jp.copy();mujoco.mj_jacSite(m,d,Jp,Jr,site)
    tau=d.qfrc_bias.copy()-Jcom.T@np.array([0,0,-t['mass']*9.81])-Jp.T@t['reaction']
    limit=torque_limits(m);finite=np.all(np.isfinite(limit));ratio=float(np.max(np.abs(tau)/limit)) if finite else None
    smin=float(np.linalg.svd(np.r_[Jp,.15*Jr],compute_uv=False)[-1])
    row=dict(required_static_torque_nm=tau.tolist(),peak_model_limit_ratio=ratio,model_torque_limits_nm=[float(x) if np.isfinite(x) else None for x in limit],within_model_limits=bool(ratio<=1) if finite else None,below_70_percent_model_limits=bool(ratio<=.7) if finite else None,scaled_jacobian_sigma_min=smin)
    if task=='hammer':
        J=np.r_[Jp,Jr];v=np.asarray(velocity_limits);target=np.array([0,0,-1.,0,0,0])
        qd=lsq_linear(J*v[None,:],target,bounds=(-1,1),tol=1e-9)
        row.update(one_mps_downward_twist_error=float(np.linalg.norm(J@(v*qd.x)-target)),one_mps_velocity_feasible=bool(np.linalg.norm(J@(v*qd.x)-target)<.01))
    return row


def single(key,base_x=0.,output_key=None):
    start=time.perf_counter();meta=MODELS[key];s=build_spec(key,base_x);m=s.compile();d=mujoco.MjData(m);rng=np.random.default_rng(42);rows=[]
    for task in TASKS:
        previous=None
        for i,(pos,R) in enumerate(poses(task)):
            sid=m.site('tcp_'+task).id;row=solve(m,d,[sid],[(np.array(pos),R)],rng,previous)
            row.update(task=task,index=i,target_m=pos,target_rotation=R.tolist())
            if row['pose_ok'] and row['collision_free']:
                d.qpos[:]=row['qpos'];d.qvel[:]=0;mujoco.mj_forward(m,d);row['load_screen']=loads(m,d,sid,task,meta['velocity']);previous=d.qpos.copy()
                if task=='cutting':
                    old=TASKS[task]['reaction'];row['upward_force_sweep']={}
                    for force in [20,40,60,100]:
                        TASKS[task]['reaction']=[10,0,force];row['upward_force_sweep'][str(force)]=loads(m,d,sid,task,meta['velocity'])['peak_model_limit_ratio']
                    TASKS[task]['reaction']=old
            rows.append(row)
        subset=[r for r in rows if r['task']==task];print(key,task,sum(r['pose_ok'] and r['collision_free'] for r in subset),'/',len(subset),flush=True)
    summary={}
    for task in TASKS:
        subset=[r for r in rows if r['task']==task];valid=[r for r in subset if 'load_screen' in r];screen=[r['load_screen'] for r in valid]
        summary[task]=dict(poses=len(subset),ik_reached=sum(r['pose_ok'] for r in subset),collision_free=len(valid),within_model_static_limits=sum(r['within_model_limits'] is True for r in screen) if all(x is not None for x in torque_limits(m)) and np.all(np.isfinite(torque_limits(m))) else None,
          static_ratios=[r['peak_model_limit_ratio'] for r in screen if r['peak_model_limit_ratio'] is not None],velocity_feasible=sum(r.get('one_mps_velocity_feasible',False) for r in screen) if task=='hammer' else None)
    report=dict(id=output_key or key,base_x_m=base_x,name=meta['name'],source=meta['path'],sha256=hashlib.sha256((ROOT/'assets/menagerie'/meta['path']).read_bytes()).hexdigest(),revision=REVISION,arm_dofs=m.nv,has_enabled_torque_limits=bool(np.all(np.isfinite(torque_limits(m)))),summary=summary,poses=rows,wall_seconds=time.perf_counter()-start)
    OUT.mkdir(parents=True,exist_ok=True);(OUT/((output_key or key)+'.json')).write_text(json.dumps(report,indent=2)+'\n');return report


def dual():
    path=ROOT/'assets/menagerie/franka_fr3_duo/fr3_duo.xml';s=mujoco.MjSpec.from_file(str(path));add_bench(s)
    for side in ['left','right']:
        s.body(side+'_fr3v2_link8').add_site(name=side+'_tcp',pos=[0,0,.15],size=[.005]*3,rgba=[.2,.8,.55,1])
    m=s.compile();d=mujoco.MjData(m);rng=np.random.default_rng(42);sid=[m.site(side+'_tcp').id for side in ['left','right']];targets=[]
    for tilt in [0,30,60,90,110]:targets.append(('pouring',[(np.array([.50,.08,.42]),Rotation.from_euler('y',tilt,degrees=True).as_matrix()@DOWN),(np.array([.50,-.10,.25]),DOWN)]))
    for x in [.4,.5,.6]:
        targets.append(('cutting',[(np.array([x,.075,.17]),Rotation.from_euler('x',90,degrees=True).as_matrix()),(np.array([x,-.035,.23]),DOWN)]))
        targets.append(('handover',[(np.array([x,.065,.35]),Rotation.from_euler('x',90,degrees=True).as_matrix()),(np.array([x,-.065,.35]),Rotation.from_euler('x',-90,degrees=True).as_matrix())]))
        targets.append(('assembly',[(np.array([x,.08,.25]),DOWN),(np.array([x,-.08,.25]),DOWN)]))
    rows=[];previous=None
    for task,goals in targets:
        row=solve(m,d,sid,goals,rng,previous,attempts=16);row.update(task=task,targets=[dict(position=p.tolist(),rotation=R.tolist()) for p,R in goals])
        if row['collision_free']:previous=np.array(row['qpos'])
        rows.append(row);print('dual',task,row['collision_free'],flush=True)
    report=dict(id='fr3_duo',name='FR3 Duo · native Menagerie mounting',revision=REVISION,source='franka_fr3_duo/fr3_duo.xml',sha256=hashlib.sha256(path.read_bytes()).hexdigest(),arm_dofs=14,poses=rows,summary={t:dict(pairs=sum(r['task']==t for r in rows),collision_free=sum(r['task']==t and r['collision_free'] for r in rows)) for t in ['pouring','cutting','handover','assembly']})
    (OUT/'fr3_duo.json').write_text(json.dumps(report,indent=2)+'\n');return report


def main():
    p=argparse.ArgumentParser();p.add_argument('--robot',choices=list(MODELS)+['dual','all'],default='all');a=p.parse_args();OUT.mkdir(parents=True,exist_ok=True)
    for key in (MODELS if a.robot=='all' else [a.robot]):
        if key=='dual':dual()
        else:single(key)
    if a.robot=='all':dual()
    (OUT/'method.json').write_text(json.dumps(dict(revision=REVISION,scope='Pose IK and collision screen with quasi-static point-mass/tool-reaction loads. Not task execution, trajectory validation, impact qualification or a procurement recommendation.',tasks=TASKS,acceptance=dict(position_m=.002,rotation_deg=2,min_joint_margin_rad=.025),collision_penetration_tolerance_m=.0001,tool_geometry='Bare arms, point TCP/CoM only. No grippers, tool collisions, grasp forces, liquids or material fracture.',static_limits='Pinned simulator actuator-force maxima; not continuous-duty or hardware impact ratings. iiwa ships without enabled limits; capacity is not scored.'),indent=2)+'\n')

if __name__=='__main__':main()
