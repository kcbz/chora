"""Mounted-hammer contact task with a guided, friction-resisted nail.

This deliberately models one constrained nailing fixture, not wood fracture or
arbitrary carpentry. Nail depth changes only through MuJoCo contact dynamics.
"""
import math

import mujoco
import numpy as np

from .common import ROOT,config
from .sim import Simulation


def validate_hammer_action(action):
    if not isinstance(action,dict):raise ValueError("Expected an action object")
    kind=action.get("action")
    fields={"move_to":{"action","x","y","z"},"strike":{"action","depth","duration"},"retract":{"action"},"done":{"action"}}
    if not isinstance(kind,str) or kind not in fields or set(action)!=fields[kind]:raise ValueError("Invalid hammer action fields")
    for key in fields[kind]-{"action"}:
        v=action[key]
        if isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v):raise ValueError("Action parameters must be finite numbers")
    if kind=="strike" and not (.002<=action["depth"]<=.012 and .06<=action["duration"]<=.18):
        raise ValueError("Strike depth must be 2–12 mm and duration 0.06–0.18 s")


def build_hammer_model(cfg,physical=False,as_spec=False):
    s=mujoco.MjSpec.from_file(str(ROOT/"assets/menagerie/robotstudio_so101/so101.xml"))
    s.modelname="SO-101 · Mounted hammer and guided nail"
    s.option.timestep=cfg["timestep"];s.option.iterations=80
    s.body("base").pos=[0,0,.12]
    gripper=s.body("gripper")
    gripper.add_site(name="hammer_tip",pos=[0,0,-.14],size=[.002]*3,rgba=[0,0,0,0])
    tool=s.worldbody.add_body(name="free_hammer") if physical else gripper.add_body(name="mounted_hammer")
    if physical:
        tool.add_freejoint(name="tool_free")
        gripper.add_geom(name="fixed_tool_pad",type=mujoco.mjtGeom.mjGEOM_BOX,pos=[-.0098,0,-.086],size=[.0012,.007,.013],mass=.001,rgba=[.13,.15,.14,1],contype=1,conaffinity=16,condim=6,friction=[.8,.003,.0001],priority=2,solref=[.004,1])
        s.body("moving_jaw_so101_v1").add_geom(name="moving_tool_pad",type=mujoco.mjtGeom.mjGEOM_BOX,pos=[-.0093,-.065,.01875],size=[.0012,.013,.007],mass=.001,rgba=[.13,.15,.14,1],contype=1,conaffinity=16,condim=6,friction=[.8,.003,.0001],priority=2,solref=[.004,1])
        s.site("hammer_tip").pos=[-.004,0,-.14]

    if physical:
        # Flat-sided hardwood handle resists torsion through distributed contact.
        tool.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX,pos=[0,0,-.085],size=[.006,.007,.03],mass=.007,rgba=[.44,.25,.12,1],contype=16,conaffinity=1,friction=[.65,.003,.0001],condim=6)
    else:
        tool.add_geom(type=mujoco.mjtGeom.mjGEOM_CAPSULE,fromto=[0,0,-.025,0,0,-.115],size=[.004,0,0],mass=.025,rgba=[.44,.25,.12,1],contype=0,conaffinity=0)
    tool.add_geom(name="hammer_head",type=mujoco.mjtGeom.mjGEOM_BOX,pos=[0,0,-.13],size=[.019,.008,.01],mass=.0955 if physical else .085,rgba=[.47,.52,.56,1],contype=8,conaffinity=6,solref=[.004,1])
    w=s.worldbody;tx,ty=cfg.get("table_center",[.175,.09]);height=cfg["table_height"]
    s.visual.headlight.ambient=[.4]*3;s.visual.headlight.diffuse=[.65]*3
    s.add_texture(name="studio_sky",type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,rgb1=[.24,.31,.35],rgb2=[.07,.10,.12],width=512,height=3072)
    w.add_light(pos=[0,-.5,1.5],dir=[0,0,-1],diffuse=[.85]*3)
    w.add_light(pos=[.7,.5,1],dir=[-.4,0,-1],diffuse=[.5]*3)
    w.add_geom(name="floor",type=mujoco.mjtGeom.mjGEOM_PLANE,size=[1,1,.01],rgba=[.23,.28,.30,1],contype=2,conaffinity=9)
    w.add_geom(name="pedestal",type=mujoco.mjtGeom.mjGEOM_BOX,pos=[0,0,.06],size=[.046,.045,.06],rgba=[.20,.25,.25,1],contype=2,conaffinity=9)
    w.add_geom(name="table",type=mujoco.mjtGeom.mjGEOM_BOX,pos=[tx,ty,height/2],size=[.083,.074,height/2],rgba=[.33,.39,.37,1],contype=2,conaffinity=9)
    w.add_geom(name="board",type=mujoco.mjtGeom.mjGEOM_BOX,pos=[tx,ty,height+(.015 if physical else .005)],size=[.052,.046,.015 if physical else .005],rgba=[.66,.45,.26,1],contype=2,conaffinity=9)
    nail=w.add_body(name="nail",pos=[tx,ty,height+(.058 if physical else .038)])
    # The fixture supports the nail's tiny weight. Its downward resistance is
    # frictional, so it stays put before an impact rather than creeping under
    # solver-softened static friction and gravity.
    nail.gravcomp=1
    nail.add_joint(name="nail_slide",type=mujoco.mjtJoint.mjJNT_SLIDE,axis=[0,0,1],range=[-.026,0],limited=True,damping=.4,frictionloss=3)
    nail.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER,pos=[0,0,-.018],size=[.001,.018,0],mass=.003,rgba=[.65,.68,.70,1],contype=0,conaffinity=0)
    nail.add_geom(name="nail_head",type=mujoco.mjtGeom.mjGEOM_CYLINDER,size=[.0035,.0015,0],mass=.002,rgba=[.80,.82,.84,1],contype=4,conaffinity=8,solref=[.004,1])
    front=np.array([.46,-.45,.34]);z=front-[.12,.015,.17];z/=np.linalg.norm(z);x=np.cross([0,0,1],z);x/=np.linalg.norm(x);y=np.cross(z,x)
    w.add_camera(name="front",pos=front,xyaxes=np.r_[x,y],fovy=42)
    w.add_camera(name="top",pos=[.14,.04,.62],xyaxes=[1,0,0,0,1,0],fovy=46)
    if physical:
        from .scene_assets import material
        s.geom("board").material=material(s,"pine")
        s.geom("board").rgba=[1,1,1,1]
    if as_spec:return s
    model=s.compile();model.vis.global_.offwidth=1280;model.vis.global_.offheight=960
    return model


class HammerSimulation(Simulation):
    def __init__(self,cfg=None):
        self.cfg=(cfg or config())["simulation"]
        self.model=build_hammer_model(self.cfg);self.data=mujoco.MjData(self.model);self.ik_data=mujoco.MjData(self.model)
        self.tcp_id=self.model.site("hammer_tip").id;self.gripper_id=self.model.body("gripper").id
        self.nail_id=self.model.body("nail").id
        self.nail_q=int(self.model.joint("nail_slide").qposadr[0]);self.nail_dof=int(self.model.joint("nail_slide").dofadr[0])
        self.head_geom=self.model.geom("hammer_head").id;self.nail_geom=self.model.geom("nail_head").id
        self.on_frame=None;self.frame_stride=10;self.object_radius=.0035;self.target_depth=.020
        self.reset(0)

    @property
    def bowl(self):
        # Generic rollout compatibility: the task object is the nail.
        return self.data.xpos[self.nail_id].copy()

    @property
    def depth(self):return max(0.,-float(self.data.qpos[self.nail_q]))

    def reset(self,seed):
        self.seed=int(seed);self.rng=np.random.default_rng(seed)
        mujoco.mj_resetData(self.model,self.data)
        centre=self.cfg.get("table_center",[.175,.09])
        self.model.body_pos[self.nail_id,:2]=[self.rng.uniform(centre[0]-.018,centre[0]+.018),self.rng.uniform(centre[1]-.016,centre[1]+.016)]
        self.model.dof_frictionloss[self.nail_dof]=self.rng.uniform(5,7)
        home=self.solve_ik([.17,-.005,.125],[0,-.4,.7,1.1])
        if home is None:raise RuntimeError("Hammer home pose is unreachable")
        self.data.qpos[:5]=home;self.data.qpos[5]=.0;self.data.ctrl[:]=self.data.qpos[:6]
        self.step_index=0;self.result="reset";self.finished=False;self.success=False;self.physics_steps=0;self.collision_count=0
        self.action={"action":"reset"};self.strikes=0;self.peak_impact=0.;self.last_impulse=0.
        self._advance(.25)
        return self.observe()

    def observe(self):
        return {"task":"hammer_nail","object_kind":"nail","object":[round(float(x),4) for x in self.bowl],"object_yaw":0.,
                "tcp":[round(float(x),4) for x in self.tcp],"nail_depth":round(self.depth,4),"target_depth":self.target_depth,
                "target":[*self.cfg.get("table_center",[.175,.09]),self.cfg["table_height"]+.01],"target_half_size":[.05,.045],
                "strikes":self.strikes,"last_peak_force":round(self.peak_impact,2),"step":self.step_index,"last_result":self.result}

    def _move(self,target,seconds,measure=False):
        start=self.tcp;target=np.asarray(target,float)
        count=max(12,round(seconds/self.model.opt.timestep))
        # Validate the entire path without moving the physical model.
        plan=[];q=self.data.qpos[:5].copy()
        for f in np.linspace(0,1,14):
            q=self.solve_ik(start+(target-start)*f,q)
            if q is None:return "rejected_unreachable"
            plan.append(q)
        if measure:self.peak_impact=0.;self.last_impulse=0.
        for i in range(count):
            fraction=(i+1)/count
            # A strike uses a fast, nearly constant downward speed. Ordinary
            # moves use a smooth interpolation; both drive the same actuators.
            f=fraction if measure else fraction*fraction*(3-2*fraction)
            index=f*(len(plan)-1);j=min(int(index),len(plan)-2);alpha=index-j
            self.data.ctrl[:5]=(1-alpha)*plan[j]+alpha*plan[j+1]
            self._advance(seconds/count)
            if measure:
                for contact_id,c in enumerate(self.data.contact):
                    if {int(c.geom1),int(c.geom2)}=={self.head_geom,self.nail_geom}:
                        force=np.zeros(6);mujoco.mj_contactForce(self.model,self.data,contact_id,force)
                        magnitude=abs(force[0]);self.peak_impact=max(self.peak_impact,magnitude);self.last_impulse+=magnitude*self.model.opt.timestep
        return "ok"

    def step(self,action):
        if self.finished:raise RuntimeError("Episode has ended")
        self.step_index+=1;self.action=action
        try:validate_hammer_action(action)
        except ValueError:self.result="invalid_action"
        else:
            kind=action["action"]
            if kind=="move_to":
                self.result=self._move([action[k] for k in "xyz"],.6)
                self._advance(.08)
            elif kind=="retract":
                self.result=self._move([*self.tcp[:2],.125],.18)
                self._advance(.06)
            elif kind=="strike":
                self.strikes+=1;before=self.depth
                target=self.tcp.copy();target[2]=max(self.cfg["table_height"]+.011,self.bowl[2]+.0015-action["depth"])
                self.result=self._move(target,action["duration"],measure=True)
                if self.result=="ok" and self.depth-before<.0003:self.result="missed_nail"
            else:
                self.success=self.depth>=self.target_depth and self.tcp[2]>self.bowl[2]+.018
                self._advance(.2)
                self.success=bool(self.success and self.depth>=self.target_depth and abs(self.data.qvel[self.nail_dof])<.005)
                self.result="success" if self.success else "nail_not_driven";self.finished=True
        if self.step_index>=self.cfg["max_steps"] and not self.finished:self.finished=True;self.result="step_limit"
        return self.observe()


class HammerExpert:
    def __init__(self,seed):self.last="reset"
    def act(self,obs,history):
        if not history:return {"action":"move_to","x":obs["object"][0],"y":obs["object"][1],"z":.125}
        if history[-1]["action"]=="strike":return {"action":"retract"}
        if obs["nail_depth"]>=obs["target_depth"] or obs["strikes"]>=5:return {"action":"done"}
        if history[-1]["action"]=="retract":return {"action":"move_to","x":obs["object"][0],"y":obs["object"][1],"z":.125}
        return {"action":"strike","depth":.006,"duration":.10}
