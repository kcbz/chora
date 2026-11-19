"""A free miniature hammer held only by frictional finger contacts.

Fixture-guided nail + depth-dependent insertion resistance. These parameters
are illustrative, not fitted pine data. Opening the jaw releases the tool.
"""
import mujoco
import numpy as np
from .common import config
from .hammer import HammerSimulation, build_hammer_model, HammerExpert
from .materials import PINE, STEEL, nail_resistance


class PhysicalHammerSimulation(HammerSimulation):
    def build_model(self):return build_hammer_model(self.cfg,physical=True)

    def __init__(self,cfg=None):
        self.cfg=(cfg or config())["simulation"]
        self.model=self.build_model()
        self.data=mujoco.MjData(self.model);self.ik_data=mujoco.MjData(self.model)
        self.tcp_id=self.model.site("hammer_tip").id;self.gripper_id=self.model.body("gripper").id
        self.nail_id=self.model.body("nail").id
        self.nail_q=int(self.model.joint("nail_slide").qposadr[0]);self.nail_dof=int(self.model.joint("nail_slide").dofadr[0])
        self.head_geom=self.model.geom("hammer_head").id;self.nail_geom=self.model.geom("nail_head").id
        self.tool_id=self.model.body("free_hammer").id
        self.tool_q=int(self.model.joint("tool_free").qposadr[0])
        self.on_frame=None;self.frame_stride=10;self.object_radius=.0035;self.target_depth=.02
        self.reset(0)

    def reset(self,seed):
        self.seed=int(seed);self.rng=np.random.default_rng(seed)
        mujoco.mj_resetData(self.model,self.data)
        centre=self.cfg.get("table_center",[.175,.09])
        self.model.body_pos[self.nail_id,:2]=[self.rng.uniform(centre[0]-.012,centre[0]+.012),self.rng.uniform(centre[1]-.01,centre[1]+.01)]
        home=self.solve_ik([.17,-.005,.125],[0,-.4,.7,1.1])
        if home is None:raise RuntimeError("Tool home pose unreachable")
        self.data.qpos[:5]=home;self.data.qpos[5]=-.10;self.data.ctrl[:]=self.data.qpos[:6]
        mujoco.mj_forward(self.model,self.data)
        R=self.data.xmat[self.gripper_id].reshape(3,3)
        self.data.qpos[self.tool_q:self.tool_q+3]=self.data.xpos[self.gripper_id]+R@[-.004,0,0]
        quat=np.zeros(4);mujoco.mju_mat2Quat(quat,R.ravel());self.data.qpos[self.tool_q+3:self.tool_q+7]=quat
        self.step_index=0;self.result="reset";self.finished=False;self.success=False;self.physics_steps=0;self.collision_count=0
        self.action={"action":"reset"};self.strikes=0;self.peak_impact=0.;self.last_impulse=0.;self.contact_work=0.;self.max_load=0.;self.grip_force=0.
        mujoco.mj_forward(self.model,self.data)
        # Start in a prepared grasp, close jaws physically; no constraint joins tool to robot.
        self.data.ctrl[5]=-.17
        self._advance(.35)
        self.reference_offset=np.array([-.004,0,0])
        return self.observe()

    def relative_tool(self):
        R=self.data.xmat[self.gripper_id].reshape(3,3)
        return R.T@(self.data.xpos[self.tool_id]-self.data.xpos[self.gripper_id])

    @property
    def slip(self):return float(np.linalg.norm(self.relative_tool()-getattr(self,"reference_offset",np.array([-.004,0,0]))))

    @property
    def tool_tip(self):return self.data.xpos[self.tool_id]+self.data.xmat[self.tool_id].reshape(3,3)@np.array([0,0,-.14])

    @property
    def tool_held(self):
        return bool(self.slip<.012 and all(force>.1 for force in getattr(self,'finger_forces',{}).values()) and len(getattr(self,'finger_forces',{}))==2)

    def _advance(self,seconds):
        for _ in range(round(seconds/self.model.opt.timestep)):
            self.model.dof_frictionloss[self.nail_dof]=nail_resistance(self.depth)
            before=self.depth
            mujoco.mj_step(self.model,self.data);self.physics_steps+=1
            self.max_load=max(self.max_load,float(np.max(np.abs(self.data.actuator_force)/2.94)))
            self.grip_force=0.;self.finger_forces={self.gripper_id:0.,self.model.body("moving_jaw_so101_v1").id:0.}
            for j,c in enumerate(self.data.contact):
                bodies={int(self.model.geom_bodyid[c.geom1]),int(self.model.geom_bodyid[c.geom2])}
                if self.tool_id in bodies:
                    f=np.zeros(6);mujoco.mj_contactForce(self.model,self.data,j,f)
                    for finger in self.finger_forces:
                        if finger in bodies:self.finger_forces[finger]+=abs(f[0]);self.grip_force+=abs(f[0])
                    if {int(c.geom1),int(c.geom2)}=={self.head_geom,self.nail_geom}:
                        self.peak_impact=max(self.peak_impact,abs(f[0]));self.last_impulse+=abs(f[0])*self.model.opt.timestep
            self.contact_work+=nail_resistance(self.depth)*max(0,self.depth-before)
            if self.on_frame is not None and self.physics_steps%self.frame_stride==0:self.on_frame(self)

    def _move(self,target,seconds,measure=False):
        start=self.tool_tip.copy();offset=start-self.tcp;target=np.asarray(target,float)
        plan=[];q=self.data.qpos[:5].copy()
        for f in np.linspace(0,1,18):
            q=self.solve_ik(start+(target-start)*f-offset,q)
            if q is None:return "rejected_unreachable"
            plan.append(q)
        if measure:self.peak_impact=0.;self.last_impulse=0.
        count=round(seconds/self.model.opt.timestep)
        for i in range(count):
            f=(i+1)/count
            if not measure:f=f*f*(3-2*f)
            index=f*(len(plan)-1);j=min(int(index),len(plan)-2);a=index-j
            self.data.ctrl[:5]=(1-a)*plan[j]+a*plan[j+1]
            self._advance(self.model.opt.timestep)
        return "ok"

    def observe(self):
        obs=super().observe()
        obs.update(task="hammer_pine",actual_tool_tip=[round(float(x),4) for x in self.tool_tip],grip_slip_mm=round(self.slip*1000,2),grip_normal_n=round(self.grip_force,2),
                   tool_held=self.tool_held,insertion_resistance_n=round(nail_resistance(self.depth),2),
                   insertion_work_j=round(self.contact_work,4),joint_load_fraction=round(self.max_load,3),
                   material_calibrated=False,board_material="pine",tool_material="rigid steel",target=[*self.cfg.get("table_center",[.175,.09]),self.cfg["table_height"]+.03])
        return obs

    def step(self,action):
        from .hammer import validate_hammer_action
        if self.finished:raise RuntimeError("Episode has ended")
        try:validate_hammer_action(action)
        except ValueError:return super().step(action)
        if action["action"] in ("strike","retract"):
            self.step_index+=1;self.action=action
            if action["action"]=="strike":
                self.strikes+=1;before=self.depth
                target=self.tool_tip.copy();target[2]=max(self.cfg["table_height"]+.031,self.bowl[2]+.0015-action["depth"])
                self.result=self._move(target,action["duration"],measure=True)
                if self.result=="ok" and self.depth-before<.0003:self.result="missed_nail"
            else:
                self.result=self._move([*self.tool_tip[:2],.125],.25);self._advance(.06)
        else:super().step(action)
        if self.slip>.012:
            self.success=False;self.result="tool_slipped";self.finished=True
        if action.get("action")=="done":
            self.success=bool(self.success and self.tool_held and self.tool_tip[2]>self.bowl[2]+.018)
            self.result="success" if self.success else "task_incomplete"
        if self.step_index>=self.cfg["max_steps"] and not self.finished:self.finished=True;self.result="step_limit"
        return self.observe()
