"""UR10e installed hammer trial. Quasi-static model caps are not impact ratings."""
import math
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
from .arm_control import robot_spec,studio,mounted_tool,ArmController
from .gantry import GantryImpact
from .scene_assets import material

class ArmHammerSimulation(GantryImpact):
    def build_stage(self):
        s=robot_spec('ur10e',.0002);studio(s);mounted_tool(s,'hammer','ur10e')
        s.worldbody.add_geom(name='board',type=mujoco.mjtGeom.mjGEOM_BOX,pos=[.5,0,.135],size=[.17,.15,.015],material=material(s,'pine'),contype=1,conaffinity=6)
        return s
    @property
    def tcp(self):return self.data.site('tcp').xpos.copy()
    def set_home(self):
        self.arm=ArmController(self.model,self.data,6);self.striking=False;self.drive_z=0.
        self.arm.initialize([('tcp',np.array([.5,0,.30]),np.eye(3))])
    def jacobian(self):
        Jp=np.zeros((3,self.model.nv));Jr=Jp.copy();mujoco.mj_jacSite(self.model,self.data,Jp,Jr,self.model.site('tcp').id)
        return Jp[:,self.arm.v],Jr[:,self.arm.v]
    def control(self,z_force=None):
        if not self.striking:self.arm.apply();return
        Jp,Jr=self.jacobian();qd=self.data.qvel[self.arm.v]
        vel=Jp@qd;force=8000*(self.goal-self.tcp)-180*vel
        self.drive_z=0. if z_force is None else z_force;force[2]=self.drive_z
        R=self.data.site('tcp').xmat.reshape(3,3);rot=Rotation.from_matrix(R.T).as_rotvec()
        tau=self.data.qfrc_bias[self.arm.v]+Jp.T@force+Jr.T@(180*rot-20*(Jr@qd))-.3*qd
        limits=self.model.actuator_forcerange[:6,1];self.data.ctrl[:6]=np.clip(tau,-limits,limits)
        self.arm.peak_ratio=max(self.arm.peak_ratio,float(np.max(abs(self.data.ctrl[:6])/limits)))
    def downward_drive(self):return max(0.,-self.drive_z) if self.striking else 0.
    def move(self,target,seconds=.6):
        target=np.array(target,float)
        if not np.all(np.isfinite(target)) or not (.38<=target[0]<=.62 and -.12<=target[1]<=.12 and .155<=target[2]<=.42):return 'rejected_unreachable'
        self.striking=False;self.arm.ref=self.data.qpos[self.arm.q].copy()
        self.arm.path([('tcp',target,np.eye(3))],seconds,self._advance);self.goal=target
        return 'ok'
    def strike(self,energy):
        if np.linalg.norm(self.tcp[:2]-self.bowl[:2])>.002:return 'rejected_misaligned'
        if self.tcp[2]<self.bowl[2]+.025:return 'rejected_no_runup'
        self.peak_force=0.;self.impulse=0.;self.motor_at_contact=0.;self.contact_seen=False;before=self.depth
        self.goal=self.tcp.copy();self.striking=True;dt=self.model.opt.timestep
        Jp,Jr=self.jacobian();M=np.zeros((self.model.nv,self.model.nv));mujoco.mj_fullM(self.model,self.data,M)
        mass=1/max(1e-6,float(Jp[2]@np.linalg.solve(M[np.ix_(self.arm.v,self.arm.v)],Jp[2])))
        speed=math.sqrt(2*energy/mass)
        for _ in range(round(.35/dt)):
            vz=float(self.jacobian()[0][2]@self.data.qvel[self.arm.v])
            if -vz>=speed or self.tcp[2]<=self.bowl[2]+.010:break
            self._advance(dt,z_force=-35.)
        self.impact_energy=.5*mass*float(self.jacobian()[0][2]@self.data.qvel[self.arm.v])**2
        for _ in range(round(.45/dt)):
            self._advance(dt,z_force=0.)
            if self.contact_seen:break
            if self.tcp[2]<self.bowl[2]-.003:break
        self.striking=False
        self.move([*self.bowl[:2],max(.25,self.bowl[2]+.08)],.35)
        return 'ok' if self.depth-before>.00005 else 'no_insertion'
    def observe(self):
        o=super().observe();o.update(robot='ur10e',target=self.cfg['target'],torque_limit_ratio=round(self.arm.peak_ratio,3),drive_definition='commanded axial task force; joint gravity/orientation control remains enabled',benchmark_validated=False)
        return o
    def step(self,action):
        super().step(action)
        if action=={'action':'done'}:self.success=None;self.result='experiment_complete'
        return self.observe()
