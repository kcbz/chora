"""Force-limited XYZ tool platform; an impact is motor-off, not a press."""
import math
import mujoco
import numpy as np
from .common import config
from .tool_assets import add_tool
from .scene_assets import material


def stage_spec(kind,timestep=.0002):
    s=mujoco.MjSpec();s.modelname='Chora · XYZ mounted '+kind
    s.option.timestep=timestep;s.option.integrator=mujoco.mjtIntegrator.mjINT_IMPLICITFAST;s.option.iterations=60
    s.option.cone=mujoco.mjtCone.mjCONE_ELLIPTIC
    s.visual.headlight.ambient=[.35]*3
    s.add_texture(name='studio_sky',type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,rgb1=[.24,.30,.34],rgb2=[.06,.09,.11],width=512,height=3072)
    w=s.worldbody
    w.add_light(pos=[.3,-.5,1.5],dir=[0,0,-1],diffuse=[.8]*3);w.add_light(pos=[-.5,.4,1],dir=[.3,0,-1],diffuse=[.5]*3)
    w.add_geom(name='floor',type=mujoco.mjtGeom.mjGEOM_PLANE,size=[2,2,.01],rgba=[.15,.19,.21,1],contype=1,conaffinity=6)
    w.add_geom(name='bench',type=mujoco.mjtGeom.mjGEOM_BOX,pos=[-.05,.1,.06],size=[.38,.24,.06],rgba=[.29,.34,.35,1],contype=1,conaffinity=6)
    wood=material(s,'pine');w.add_geom(name='board',type=mujoco.mjtGeom.mjGEOM_BOX,pos=[0,.1,.135],size=[.16,.13,.015],material=wood,contype=1,conaffinity=6,friction=[.7,.002,.0001])
    for x in [-.38,.28]:
        w.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX,pos=[x,.30,.29],size=[.025,.025,.29],rgba=[.27,.32,.34,1],contype=0,conaffinity=0)
    w.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX,pos=[-.05,.3,.56],size=[.35,.027,.025],rgba=[.38,.43,.46,1],contype=0,conaffinity=0)
    bx=w.add_body(name='x_slide');bx.add_joint(name='x',type=mujoco.mjtJoint.mjJNT_SLIDE,axis=[1,0,0],range=[-.12,.12],limited=True,damping=1)
    bx.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX,pos=[-.20,.3,.52],size=[.035,.04,.025],mass=.4,rgba=[.14,.22,.27,1],contype=0,conaffinity=0)
    by=bx.add_body(name='y_slide');by.add_joint(name='y',type=mujoco.mjtJoint.mjJNT_SLIDE,axis=[0,1,0],range=[-.05,.25],limited=True,damping=1)
    by.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX,pos=[-.20,.08,.50],size=[.02,.14,.015],mass=.3,rgba=[.45,.5,.52,1],contype=0,conaffinity=0)
    bz=by.add_body(name='tool',gravcomp=1);bz.add_joint(name='z',type=mujoco.mjtJoint.mjJNT_SLIDE,axis=[0,0,1],range=[.12,.46],limited=True,damping=.02)
    handle=add_tool(s,bz,kind)
    bx.geoms[0].pos[0]=handle[0];by.geoms[0].pos[0]=handle[0]
    by.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX,pos=[handle[0],0,.42],size=[.020,.018,.10],rgba=[.22,.30,.34,1],mass=0,contype=0,conaffinity=0)
    bz.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX,pos=[handle[0],handle[1],handle[2]+.10],size=[.012,.012,.11],rgba=[.46,.5,.52,1],mass=0,contype=0,conaffinity=0)
    for a in 'xyz':s.add_actuator(name=a,target=a,trntype=mujoco.mjtTrn.mjTRN_JOINT,gear=[1,0,0,0,0,0],gaintype=mujoco.mjtGain.mjGAIN_FIXED,gainprm=[1,0,0,0,0,0,0,0,0,0],forcelimited=True,forcerange=[-100,100],ctrllimited=True,ctrlrange=[-100,100])
    front=np.array([.80,-.80,.67]);z=front-[0,.1,.27];z/=np.linalg.norm(z);x=np.cross([0,0,1],z);x/=np.linalg.norm(x);y=np.cross(z,x)
    w.add_camera(name='front',pos=front,xyaxes=np.r_[x,y],fovy=45);w.add_camera(name='top',pos=[-.04,.10,1.05],xyaxes=[1,0,0,0,1,0],fovy=48)
    return s


class GantryImpact:
    def __init__(self,cfg=None):
        self.cfg=(cfg or config())['simulation'];s=self.build_stage()
        nail=s.worldbody.add_body(name='nail',pos=np.array(self.cfg['target'])+[0,0,.028],gravcomp=1)
        nail.add_joint(name='nail_slide',type=mujoco.mjtJoint.mjJNT_SLIDE,axis=[0,0,1],range=[-.024,0],limited=True,frictionloss=120,damping=1,solref_friction=[.0004,1],solimp_friction=[.999,.9999,.0001,.5,2])
        nail.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER,pos=[0,0,-.02],size=[.001,.02,0],mass=.002,rgba=[.63,.66,.7,1],contype=0,conaffinity=0)
        nail.add_geom(name='nail_head',type=mujoco.mjtGeom.mjGEOM_CYLINDER,size=[.004,.001,0],mass=.002,rgba=[.72,.75,.78,1],contype=4,conaffinity=2,solref=[.0005,1],solimp=[.98,.995,.001,.5,2])
        self.model=s.compile();self.model.vis.global_.offwidth=1280;self.model.vis.global_.offheight=960;self.data=mujoco.MjData(self.model)
        self.nail_q=int(self.model.joint('nail_slide').qposadr[0]);self.nail_dof=int(self.model.joint('nail_slide').dofadr[0]);self.nail_id=self.model.body('nail').id
        self.contact_id=self.model.geom('tool_contact').id;self.head_id=self.model.geom('nail_head').id
        self.on_frame=None;self.frame_stride=250;self.object_radius=.004;self.target_depth=.02;self.reset(0)

    @property
    def tcp(self):return self.data.qpos[:3].copy()
    def build_stage(self):return stage_spec('hammer')
    def set_home(self):self.data.qpos[:3]=[0,.1,.30]
    def control(self,z_force=None):
        error=self.goal-self.tcp;self.data.ctrl[:]=np.clip(5000*error-100*self.data.qvel[:3],-100,100)
        if z_force is not None:self.data.ctrl[2]=z_force
        elif self.tcp[2]<self.bowl[2]+.010:self.data.ctrl[2]=max(0.,self.data.ctrl[2])
    def downward_drive(self):return max(0.,-float(self.data.ctrl[2]))
    @property
    def bowl(self):return self.data.xpos[self.nail_id].copy()
    @property
    def depth(self):return max(0.,-float(self.data.qpos[self.nail_q]))

    def reset(self,seed):
        self.seed=int(seed);rng=np.random.default_rng(seed);mujoco.mj_resetData(self.model,self.data)
        self.model.body_pos[self.nail_id,:2]=np.array(self.cfg['target'][:2])+rng.uniform(-.015,.015,2)
        self.set_home();mujoco.mj_forward(self.model,self.data)
        self.goal=self.tcp;self.step_index=0;self.strikes=0;self.action={'action':'reset'};self.result='reset';self.success=False;self.finished=False;self.physics_steps=0;self.collision_count=0
        self.peak_force=0.;self.impulse=0.;self.impact_energy=0.;self.motor_at_contact=0.;self.max_lateral_error=0.;self.contact_seen=False
        self._advance(.15);return self.observe()

    def _advance(self,seconds,z_force=None):
        dt=self.model.opt.timestep
        for _ in range(round(seconds/dt)):
            self.control(z_force)
            self.model.dof_frictionloss[self.nail_dof]=120+3000*self.depth
            mujoco.mj_step(self.model,self.data);self.physics_steps+=1
            for i,c in enumerate(self.data.contact):
                if {int(c.geom1),int(c.geom2)}=={self.contact_id,self.head_id}:
                    force=np.zeros(6);mujoco.mj_contactForce(self.model,self.data,i,force)
                    self.peak_force=max(self.peak_force,float(abs(force[0])));self.impulse+=float(abs(force[0]))*dt
                    self.motor_at_contact=max(self.motor_at_contact,self.downward_drive());self.contact_seen=True
                    self.max_lateral_error=max(self.max_lateral_error,float(np.linalg.norm(self.tcp[:2]-self.bowl[:2])))
            if self.on_frame and self.physics_steps%self.frame_stride==0:self.on_frame(self)

    def move(self,target,seconds=.6):
        target=np.array(target,float)
        if not np.all(np.isfinite(target)) or not (-.10<=target[0]<=.10 and 0<=target[1]<=.20 and .155<=target[2]<=.42):return 'rejected_unreachable'
        start=self.tcp
        for f in np.linspace(0,1,round(seconds/self.model.opt.timestep)):
            a=f*f*(3-2*f);self.goal=start+(target-start)*a;self._advance(self.model.opt.timestep)
        self.goal=target;self._advance(.12);return 'ok'

    def strike(self,energy):
        if np.linalg.norm(self.tcp[:2]-self.bowl[:2])>.001:return 'rejected_misaligned'
        if self.tcp[2]<self.bowl[2]+.025:return 'rejected_no_runup'
        self.peak_force=0.;self.impulse=0.;self.motor_at_contact=0.;self.contact_seen=False;before=self.depth
        speed=math.sqrt(2*energy/.7);dt=self.model.opt.timestep
        # Accelerate in air; switch the Z motor off well before nail contact.
        for _ in range(1000):
            if -self.data.qvel[2]>=speed or self.tcp[2]<=self.bowl[2]+.008:break
            self._advance(dt,z_force=-30.)
        self.impact_energy=.5*.7*float(self.data.qvel[2])**2
        for _ in range(1800):
            self._advance(dt,z_force=0.)
            if self.contact_seen and self.data.qvel[2]>-.005:break
            if self.tcp[2]<.151:break
        self.goal=self.tcp.copy()
        # Retraction starts immediately; no position hold pushes on the nail.
        self.move([*self.goal[:2],max(.23,self.bowl[2]+.055)],.25)
        return 'ok' if self.depth-before>.00005 else 'no_insertion'

    def observe(self):
        return {'task':'impact_nail','object_kind':'nail','object':[round(float(x),5) for x in self.bowl],'object_yaw':0.,'tcp':[round(float(x),5) for x in self.tcp],
                'target':[0,.1,.15],'target_half_size':[.14,.11],'nail_depth':round(self.depth,5),'target_depth':.02,'strikes':self.strikes,'last_peak_force':round(self.peak_force,2),
                'impact_energy_j':round(self.impact_energy,4),'impulse_ns':round(self.impulse,4),'motor_force_at_contact_n':round(self.motor_at_contact,3),'lateral_error_mm':round(self.max_lateral_error*1000,3),
                'mounted_tool':True,'material_calibrated':False,'step':self.step_index,'last_result':self.result}

    def step(self,action):
        if self.finished:raise RuntimeError('Episode has ended')
        self.step_index+=1;self.action=action
        try:
            if action['action']=='move_to' and set(action)=={'action','x','y','z'}:self.result=self.move([action[k] for k in 'xyz'])
            elif action['action']=='strike' and set(action)=={'action','energy'} and not isinstance(action['energy'],bool) and .1<=action['energy']<=.6:
                self.strikes+=1;self.result=self.strike(action['energy'])
            elif action=={'action':'done'}:
                self.success=bool(self.depth>=.02 and self.tcp[2]>self.bowl[2]+.025);self.finished=True;self.result='success' if self.success else 'incomplete_insertion'
            else:self.result='invalid_action'
        except (KeyError,ValueError,TypeError):self.result='invalid_action'
        if self.step_index>=self.cfg['max_steps'] and not self.finished:self.finished=True;self.result='step_limit'
        return self.observe()


class ImpactExpert:
    def __init__(self,seed):pass
    def act(self,obs,history):
        if obs['nail_depth']>=obs['target_depth'] or obs['strikes']>=12:return {'action':'done'}
        if not history or history[-1]['action']=='strike':return {'action':'move_to','x':obs['object'][0],'y':obs['object'][1],'z':max(.23,obs['object'][2]+.055)}
        return {'action':'strike','energy':.35}
