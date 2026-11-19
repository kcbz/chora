"""Real FR3 arm with installed chef knife, sharing the food mechanics adapter."""
import mujoco
import numpy as np
from .arm_control import robot_spec,studio,mounted_tool,ArmController
from .soft_food import SoftFoodSimulation
from .scene_assets import material

class ForceLimit(RuntimeError):pass

class ArmFoodSimulation(SoftFoodSimulation):
    def configure_food(self):
        if self.kind in {'apple','banana'}:
            from .scanned_produce import dimensions
            self.food.update(dimensions(self.kind,self.food['height']))
    def build_stage(self):
        s=robot_spec('fr3');studio(s);mounted_tool(s,'knife','fr3')
        s.worldbody.add_geom(name='board',type=mujoco.mjtGeom.mjGEOM_BOX,pos=[.5,0,.135],size=[.17,.15,.015],material=material(s,'pine'),friction=[.7,.002,.0001],contype=1,conaffinity=6)
        return s
    @property
    def tcp(self):return self.data.site('tcp').xpos.copy()
    def set_home(self):
        self.force_stop=False;self.guard_cut=False
        self.arm=ArmController(self.model,self.data,7)
        self.arm.initialize([('tcp',np.array([.5,0,.30]),np.eye(3))])
    def control(self):self.arm.apply()
    def _advance(self,seconds):
        dt=self.model.opt.timestep
        for _ in range(round(seconds/dt)):
            super()._advance(dt)
            if getattr(self,'guard_cut',False) and self.force>55:
                self.force_stop=True
                self.arm.ref=self.data.qpos[self.arm.q].copy();self.arm.vel[:]=0
                raise ForceLimit('Cutting force guard')
    def move(self,target,seconds):
        target=np.array(target,float)
        if not np.all(np.isfinite(target)) or not (.38<=target[0]<=.62 and -.12<=target[1]<=.12 and .148<=target[2]<=.4):return 'rejected_unreachable'
        self.guard_cut=target[2]<self.tcp[2]-.005
        self.arm.ref=self.data.qpos[self.arm.q].copy()
        try:self.arm.path([('tcp',target,np.eye(3))],seconds,self._advance)
        except ForceLimit:
            self.guard_cut=False;self.goal=self.tcp.copy();return 'force_limit'
        self.guard_cut=False;self.goal=target
        return 'ok'
    def observe(self):
        o=super().observe();o.update(robot='fr3',torque_limit_ratio=round(self.arm.peak_ratio,3),force_stop=self.force_stop,controller='bounded payload-aware joint torque tracking')
        return o
    def step(self,action):
        if action=={'action':'slice'}:
            self.step_index+=1;self.action=action
            self.result=self.move([*self.tcp[:2],.157],1.6)
            return self.observe()
        if action=={'action':'retract'}:
            self.step_index+=1;self.action=action
            self.result=self.move([*self.tcp[:2],.30],1.0)
            return self.observe()
        return super().step(action)
