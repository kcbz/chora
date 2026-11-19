"""Layered cutting surrogate with force/work-gated separation on a preset plane.

Free knife, frictional grasp, rigid food halves connected until the cohesive-work
criterion is met. This is NOT FEM fracture, arbitrary cut topology or calibrated
food mechanics. Appearance uses procedural skin and interior maps, not scans.
"""
import math
import mujoco
import numpy as np
from .common import config
from .hammer import build_hammer_model
from .tool_physics import PhysicalHammerSimulation
from .materials import FRUIT, cutting_resistance
from .scene_assets import material,fruit_half_mesh


def validate_cut_action(action):
    if not isinstance(action,dict):raise ValueError('Expected an action object')
    kind=action.get('action');fields={'move_to':{'action','x','y','z'},'slice':{'action','depth','duration'},'retract':{'action'},'done':{'action'}}
    if not isinstance(kind,str) or kind not in fields or set(action)!=fields[kind]:raise ValueError('Invalid cutting action fields')
    for k in fields[kind]-{'action'}:
        if isinstance(action[k],bool) or not isinstance(action[k],(int,float)) or not math.isfinite(action[k]):raise ValueError('Parameters must be finite numbers')
    if kind=='slice' and not (.005<=action['depth']<=.055 and .5<=action['duration']<=3.):raise ValueError('Slice depth 5..55 mm; duration 0.5..3 s')


class CuttingSimulation(PhysicalHammerSimulation):
    def __init__(self,cfg=None):
        cfg=cfg or config();self.kind=cfg['simulation']['object_kind'];self.food=FRUIT[self.kind]
        self.cut_work=0.;self.cut_complete=False;self.cut_progress=0.;self.peak_cut_force=0.;self.cut_force=0.;self.separation_steps=0;self.peak_impact=0.
        super().__init__(cfg)

    def build_model(self):
        s=build_hammer_model(self.cfg,physical=True,as_spec=True)
        s.modelname='SO-101 · Cutting '+self.food['name']
        verts=[[-.036,-.0012,-.10],[.036,-.0012,-.10],[.036,-.0001,-.14],[-.036,-.0001,-.14],[-.036,.0012,-.10],[.036,.0012,-.10],[.036,.0001,-.14],[-.036,.0001,-.14]]
        faces=[[0,1,2],[0,2,3],[4,6,5],[4,7,6],[0,4,5],[0,5,1],[3,2,6],[3,6,7],[0,3,7],[0,7,4],[1,5,6],[1,6,2]]
        s.add_mesh(name='steel_blade',uservert=np.array(verts).ravel(),userface=np.array(faces).ravel(),smoothnormal=False)
        blade=s.geom('hammer_head');blade.type=mujoco.mjtGeom.mjGEOM_MESH;blade.meshname='steel_blade';blade.pos=[0,0,0];blade.mass=.0294;blade.rgba=[.72,.76,.79,1];blade.conaffinity=2
        # Hide the legacy fixture entirely; its slider remains mechanically inert.
        s.body('nail').pos=[0,0,-1]
        for geom in s.body('nail').geoms:geom.rgba=[0,0,0,0];geom.contype=0;geom.conaffinity=0
        self.center=np.array([*self.cfg.get('table_center',[.175,.09]),self.cfg['table_height']+.03+self.food['height']/2])
        skin=material(s,self.kind);inside=material(s,self.kind,True)
        for side,label in [(1,'left'),(-1,'right')]:
            mesh=fruit_half_mesh(s,self.kind,side,self.food['radius'],self.food['height'])
            body=s.worldbody.add_body(name='fruit_'+label,pos=self.center)
            body.add_freejoint(name='fruit_'+label+'_free')
            body.add_geom(name='food_'+label,type=mujoco.mjtGeom.mjGEOM_MESH,meshname=mesh,mass=self.food['mass']/2,material=skin,contype=32,conaffinity=2,friction=[.6,.002,.0001])
            body.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH,meshname=mesh+'_face',mass=0,material=inside,contype=0,conaffinity=0)
            if side==1 and self.kind=='apple':
                body.add_geom(type=mujoco.mjtGeom.mjGEOM_CAPSULE,fromto=[0,0,self.food['height']/2-.002,.002,0,self.food['height']/2+.006],size=[.0013,0,0],mass=0,rgba=[.24,.12,.055,1],contype=0,conaffinity=0)
            if side==1 and self.kind=='tomato':
                for a in np.linspace(0,2*np.pi,6)[:-1]:
                    body.add_geom(type=mujoco.mjtGeom.mjGEOM_CAPSULE,fromto=[0,0,self.food['height']/2+.0002,.009*np.cos(a),.009*np.sin(a),self.food['height']/2-.001],size=[.0011,0,0],mass=0,rgba=[.15,.30,.07,1],contype=0,conaffinity=0)
        s.add_equality(name='food_cohesion',type=mujoco.mjtEq.mjEQ_WELD,objtype=mujoco.mjtObj.mjOBJ_BODY,name1='fruit_left',name2='fruit_right',data=[0,0,0,0,0,0,1,0,0,0,.01,0],solref=[.003,1])
        # A shallow cradle keeps round food from rolling; no attachment to board.
        tx,ty=self.center[:2]
        yaw=math.atan2(ty,tx-.0388353)
        for sign in [-1,1]:
            distance=self.food['radius']*(2.7 if self.kind=='banana' else 1)+.003
            s.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX,pos=[tx-sign*distance*math.sin(yaw),ty+sign*distance*math.cos(yaw),self.cfg['table_height']+.034],quat=[math.cos(yaw/2),0,0,math.sin(yaw/2)],size=[.03,.002,.004],rgba=[.34,.25,.16,1],contype=2,conaffinity=32|8)
        m=s.compile();m.vis.global_.offwidth=1280;m.vis.global_.offheight=960
        return m

    def reset(self,seed):
        self.cut_work=0.;self.cut_complete=False;self.cut_progress=0.;self.peak_cut_force=0.;self.cut_force=0.;self.separation_steps=0;self.release_budget=0.;self.last_edge_z=None
        self.fruit_ids=[self.model.body('fruit_'+label).id for label in ['left','right']]
        self.cohesion=self.model.equality('food_cohesion').id
        self.center=np.array([*self.cfg.get('table_center',[.175,.09]),self.cfg['table_height']+.03+self.food['height']/2])
        obs=super().reset(seed)
        # Rotate food's predefined cutting plane to the knife's approach yaw.
        q=self.solve_ik([*self.center[:2],.125],self.data.qpos[:5])
        self.ik_data.qpos[:5]=q;mujoco.mj_forward(self.model,self.ik_data)
        R=self.ik_data.xmat[self.gripper_id].reshape(3,3);yaw=math.atan2(R[1,0],R[0,0])
        self.cut_normal=np.array([-math.sin(yaw),math.cos(yaw),0.])
        rng=np.random.default_rng(seed);jitter=rng.uniform(-.003,.003,2);self.center[:2]+=jitter
        for label in ['left','right']:
            a=int(self.model.joint('fruit_'+label+'_free').qposadr[0]);self.data.qpos[a:a+3]=self.center
            self.data.qpos[a+3:a+7]=[math.cos(yaw/2),0,0,math.sin(yaw/2)]
            d=int(self.model.joint('fruit_'+label+'_free').dofadr[0]);self.data.qvel[d:d+6]=0
        mujoco.mj_forward(self.model,self.data);self._advance(.12)
        return self.observe()

    @property
    def bowl(self):return np.mean([self.data.xpos[i] for i in self.fruit_ids],axis=0)

    def _advance(self,seconds):
        dt=self.model.opt.timestep
        for _ in range(round(seconds/dt)):
            self.data.xfrc_applied[:]=0
            if hasattr(self,'reference_offset') and hasattr(self,'fruit_ids'):
                edge=self.tool_tip;center=self.bowl;top=center[2]+self.food['height']/2
                penetration=top-edge[2]
                normal=getattr(self,'cut_normal',np.array([0.,1.,0.]))
                lateral=edge-center
                aligned=abs(np.dot(lateral,normal))<.004 and np.linalg.norm(lateral[:2])<self.food['radius']*.7
                force=cutting_resistance(self.kind,penetration) if aligned and not self.cut_complete else 0.
                self.cut_force=force;self.peak_cut_force=max(self.peak_cut_force,force)
                if force:
                    direction=-1 if self.last_edge_z is not None and edge[2]-self.last_edge_z>1e-7 else 1
                    applied=np.array([0.,0.,force*direction])
                    self.data.xfrc_applied[self.tool_id,:3]=applied
                    self.data.xfrc_applied[self.tool_id,3:]=np.cross(edge-self.data.xipos[self.tool_id],applied)
                    for body in self.fruit_ids:self.data.xfrc_applied[body,:3]=-applied/2
                before=edge[2];self.last_edge_z=before
            else:force=0;before=0;penetration=0;aligned=False
            released_before=[self.data.xpos[b].copy() for b in self.fruit_ids]
            if self.separation_steps and self.release_budget>0:
                for body,sign in zip(self.fruit_ids,[1,-1]):self.data.xfrc_applied[body,:3]+=normal*sign*.06
                self.separation_steps-=1
            else:self.separation_steps=0
            super()._advance(dt)
            if self.separation_steps:
                released=sum(max(0,float(np.dot(self.data.xpos[b]-old,normal*sign*.06))) for b,old,sign in zip(self.fruit_ids,released_before,[1,-1]))
                self.release_budget=max(0.,self.release_budget-released)
            if force:self.cut_work+=force*max(0,before-self.tool_tip[2])
            if aligned:
                self.cut_progress=max(self.cut_progress,min(1.,max(0.,penetration/self.food['height'])))
            if self.cut_progress>=1-.001/self.food["height"] and self.cut_work>=self.food['fracture_work_j'] and not self.cut_complete:
                self.cut_complete=True;self.data.eq_active[self.cohesion]=0;self.separation_steps=80;self.release_budget=.02*self.cut_work

    def observe(self):
        return {'task':'cut_fruit','object_kind':self.kind,'object':[round(float(x),4) for x in self.bowl], 'object_yaw':0.,
                'tcp':[round(float(x),4) for x in self.tcp],'blade_edge':[round(float(x),4) for x in self.tool_tip],
                'target':[*self.cfg.get('table_center',[.175,.09]),self.cfg['table_height']+.03], 'target_half_size':[.05,.045],
                'cut_progress':round(self.cut_progress,3),'cut_complete':bool(self.cut_complete),'cut_work_j':round(self.cut_work,4),
                'cut_force_n':round(self.cut_force,2),'peak_cut_force_n':round(self.peak_cut_force,2),'food_height':self.food['height'],
                'grip_slip_mm':round(self.slip*1000,2),'grip_normal_n':round(float(self.grip_force),2),'tool_held':self.tool_held,
                'material_calibrated':False,'step':self.step_index,'last_result':self.result}

    def step(self,action):
        if self.finished:raise RuntimeError('Episode has ended')
        self.step_index+=1;self.action=action
        try:validate_cut_action(action)
        except ValueError:self.result='invalid_action'
        else:
            if action['action']=='move_to':self.result=self._move([action[k] for k in 'xyz'],.7);self._advance(.08)
            elif action['action']=='retract':self.result=self._move([*self.tool_tip[:2],.125],.6);self._advance(.08)
            elif action['action']=='slice':
                target=self.tool_tip.copy();target[2]=max(self.cfg['table_height']+.0295,target[2]-action['depth'])
                self.result=self._move(target,action['duration']);self._advance(.35)

            else:
                self._advance(.2)
                self.success=bool(self.cut_complete and self.tool_held and self.tool_tip[2]>self.cfg['table_height']+.03+self.food['height']+.01)
                self.result='success' if self.success else 'cut_incomplete';self.finished=True
        if self.slip>.012:self.result='tool_slipped';self.success=False;self.finished=True
        if self.step_index>=self.cfg['max_steps'] and not self.finished:self.result='step_limit';self.finished=True
        return self.observe()


class CuttingExpert:
    def __init__(self,seed):pass
    def act(self,obs,history):
        if not history:return {'action':'move_to','x':obs['object'][0],'y':obs['object'][1],'z':obs['target'][2]+obs['food_height']+.003}
        if history[-1]['action']=='move_to':return {'action':'slice','depth':obs['food_height']+.004,'duration':1.4}
        if history[-1]['action']=='slice':
            if not obs['cut_complete'] and sum(a['action']=='slice' for a in history)<3:return {'action':'move_to','x':obs['object'][0],'y':obs['object'][1],'z':obs['blade_edge'][2]+.003}
            return {'action':'retract'}
        return {'action':'done'}
