import json,time
import numpy as np,mujoco
from chora.arm_evaluation import build_spec,MODELS,TASKS,OUT

def run(key,file,short=False,base_x=0.):
    t=dict(TASKS['cutting']);t.update(tcp=[.08,0,.12],com=[.03,0,.08]) if short else None
    old=TASKS['cutting'];TASKS['cutting']=t
    s=build_spec(key,base_x);flange=s.site('eval_flange');R=np.zeros(9);mujoco.mju_quat2Mat(R,flange.quat);R=R.reshape(3,3)
    flange.parent.add_body(name='eval_payload',pos=np.array(flange.pos)+R@t['com'],mass=t['mass'],ipos=[0,0,0],inertia=[.001]*3,explicitinertial=True)
    m=s.compile();d=mujoco.MjData(m);report=json.loads((OUT/(file+'.json')).read_text());rows=[r for r in report['poses'] if r['task']=='cutting' and r['collision_free']]
    row=min(rows,key=lambda r:np.linalg.norm(np.array(r['target_m'])-[.5,.1,.24]));d.qpos[:]=row['qpos'];d.ctrl[:]=row['qpos'];mujoco.mj_forward(m,d)
    site=m.site('tcp_cutting').id;body=int(m.site_bodyid[site]);limit=np.max(np.abs(m.actuator_forcerange),axis=1);count=0;saturation=0;errors=[];loaded=[];baseline=None
    for i in range(round(1.8/m.opt.timestep)):
        d.qfrc_applied[:]=0
        if i*m.opt.timestep>=.6:mujoco.mj_applyFT(m,d,np.array(t['reaction'],float),np.zeros(3),d.site_xpos[site],body,d.qfrc_applied)
        mujoco.mj_step(m,d)
        if i*m.opt.timestep>=.5 and baseline is None:baseline=d.site_xpos[site].copy()
        if i*m.opt.timestep>1.3:
            errors.append(float(np.linalg.norm(d.site_xpos[site]-row['target_m'])));loaded.append(d.site_xpos[site].copy());count+=1;saturation+=bool(np.any(np.abs(d.actuator_force)/limit>.98))
    result={'robot':key,'configuration':file,'target_m':row['target_m'],'payload_mass_kg':t['mass'],'payload_inertia_diagonal_kg_m2':[.001]*3,'reaction_n':t['reaction'],'native_position_servos':True,'gravity_compensation_added':False,'mean_loaded_tcp_position_error_mm':1000*float(np.mean(errors)),'load_deflection_from_settled_baseline_mm':1000*float(np.linalg.norm(np.mean(loaded,axis=0)-baseline)),'steady_state_saturation_fraction':saturation/count,'warning_counts':d.warning.number.tolist(),'qpos':row['qpos'],'scope':'Fixed joint setpoints; no motion, force-control tuning or material contact. Hypothetical tool payload/inertia.'}
    print(file,result,flush=True);TASKS['cutting']=old;return result
results=[run('panda','panda'),run('fr3','fr3'),run('fr3','fr3_short_knife_mount',True),run('ur5e','ur5e'),run('ur10e','ur10e_base_back_250mm',base_x=-.25)]
(OUT/'native_hold_tests.json').write_text(json.dumps(results,indent=2)+'\n')
