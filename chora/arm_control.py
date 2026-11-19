"""Shared adapters for pinned articulated robots and bounded torque control.

Tools are installed at the flange, not welded to the world. IK only sets initial
conditions and reference trajectories; simulation motion uses joint actuators.
"""
import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from .common import ROOT

DOWN=np.diag([1.,-1.,-1.])
PATHS={'fr3':'franka_fr3_v2/fr3v2.xml','ur10e':'universal_robots_ur10e/ur10e.xml','fr3_duo':'franka_fr3_duo/fr3_duo.xml'}

def robot_spec(kind,dt=.0005):
    s=mujoco.MjSpec.from_file(str(ROOT/'assets/menagerie'/PATHS[kind]))
    # Preserve the source inertias, passive dynamics and torque bounds; replace
    # untuned position servos with explicit, bounded torque actuators.
    for a in s.actuators:
        limits=np.array(a.forcerange).copy();a.set_to_motor();a.inheritrange=0
        a.ctrllimited=True;a.ctrlrange=limits;a.forcelimited=True;a.forcerange=limits
    for key in list(s.keys):s.delete(key)
    if kind=='ur10e':s.body('base').pos[0]=-.25
    s.option.timestep=dt;s.option.integrator=mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    s.option.iterations=100;s.option.cone=mujoco.mjtCone.mjCONE_ELLIPTIC
    return s

def studio(s,center=(.5,0,.25),bench=True):
    w=s.worldbody;s.visual.headlight.ambient=[.35]*3
    w.add_light(pos=[.5,-.8,1.8],dir=[0,.3,-1],diffuse=[.8]*3)
    w.add_light(pos=[-.5,.8,1.5],dir=[.3,-.3,-1],diffuse=[.5]*3)
    w.add_geom(name='floor',type=mujoco.mjtGeom.mjGEOM_PLANE,size=[2,2,.01],rgba=[.14,.18,.21,1])
    if bench:w.add_geom(name='bench',type=mujoco.mjtGeom.mjGEOM_BOX,pos=[.5,0,.06],size=[.28,.30,.06],rgba=[.31,.36,.37,1])
    front=np.array([1.50,-1.45,1.20]);z=front-center;z/=np.linalg.norm(z);x=np.cross([0,0,1],z);x/=np.linalg.norm(x);y=np.cross(z,x)
    w.add_camera(name='front',pos=front,xyaxes=np.r_[x,y],fovy=45)
    w.add_camera(name='top',pos=[.4,0,1.9],xyaxes=[1,0,0,0,1,0],fovy=50)

def mounted_tool(s,kind,robot):
    from .tool_assets import add_tool
    flange=s.body('fr3v2_link8') if robot=='fr3' else s.site('attachment_site').parent
    pos=np.zeros(3);quat=np.array([1.,0,0,0])
    if robot=='ur10e':pos=np.array(s.site('attachment_site').pos);quat=np.array(s.site('attachment_site').quat)
    # Source MJCF permits non-unit quaternions and normalizes on compilation;
    # mju_quat2Mat requires a unit quaternion. Normalize before composing offsets.
    quat/=np.linalg.norm(quat)
    R=np.zeros(9);mujoco.mju_quat2Mat(R,quat);R=R.reshape(3,3)
    offset=np.array([.08 if kind=='knife' else .23,0,.12])
    # Tool mesh frame has +Z up. Flange frame points downward at the workpiece.
    q=np.empty(4);mujoco.mju_mulQuat(q,quat,[0,1,0,0])
    body=flange.add_body(name='tool',pos=pos+R@offset,quat=q)
    handle=add_tool(s,body,kind)
    anchor=np.array([-offset[0],0,offset[2]])
    body.add_geom(name='adapter',type=mujoco.mjtGeom.mjGEOM_CAPSULE,fromto=np.r_[anchor,handle+[0,0,.025]],size=[.014,0,0],mass=.35,rgba=[.28,.33,.36,1],contype=0,conaffinity=0)
    body.add_site(name='tcp',size=[.003]*3,rgba=[0,0,0,0])
    return body

class ArmController:
    def __init__(self,m,d,n):
        self.m=m;self.d=d;self.n=n
        self.a=np.arange(n);self.j=m.actuator_trnid[:n,0];self.q=m.jnt_qposadr[self.j];self.v=m.jnt_dofadr[self.j]
        self.lo=m.jnt_range[self.j,0]+.025;self.hi=m.jnt_range[self.j,1]-.025
        self.ikdata=mujoco.MjData(m);self.ref=d.qpos[self.q].copy();self.vel=np.zeros(n)
        self.M=np.zeros((m.nv,m.nv));self.mass=np.eye(n);self.ticks=0;self.peak_ratio=0.;self.max_error=0.
    def ik(self,goals,seed=None,attempts=3):
        d=self.ikdata;d.qpos[:]=self.d.qpos;seed=self.d.qpos[self.q].copy() if seed is None else seed.copy()
        ids=[self.m.site(name).id for name,pos,R in goals]
        def residual(q):
            d.qpos[self.q]=q;mujoco.mj_kinematics(self.m,d)
            return np.concatenate([np.r_[d.site_xpos[sid]-p,.15*Rotation.from_matrix(R@d.site_xmat[sid].reshape(3,3).T).as_rotvec()] for sid,(_,p,R) in zip(ids,goals)])
        def jac(q):
            residual(q);mujoco.mj_comPos(self.m,d);rows=[]
            for sid in ids:
                jp=np.zeros((3,self.m.nv));jr=jp.copy();mujoco.mj_jacSite(self.m,d,jp,jr,sid)
                rows.append(np.r_[jp[:,self.v],-.15*jr[:,self.v]])
            return np.concatenate(rows)
        rng=np.random.default_rng(5)
        best=None
        for i in range(attempts):
            initial=seed if i==0 else seed+rng.normal(0,.6,self.n)
            r=least_squares(residual,np.clip(initial,self.lo+1e-5,self.hi-1e-5),jac=jac,bounds=(self.lo,self.hi),max_nfev=160,gtol=1e-8,ftol=1e-8,xtol=1e-8)
            error=np.linalg.norm(residual(r.x));best=r.x
            if error<.001:return best
        raise ValueError(f'Articulated-arm pose unreachable (scaled residual {error:.4f})')
    def initialize(self,goals):
        # Franka needs a bent elbow seed to avoid its singular zero pose.
        if self.n in (7,14):
            seed=np.tile([0,0,0,-1.57,0,1.57,-.785],self.n//7)
        else:seed=np.array([-1.57,-1.57,1.57,-1.57,-1.57,0.])
        self.ref=self.ik(goals,seed,attempts=12);self.d.qpos[self.q]=self.ref;self.d.qvel[self.v]=0
        mujoco.mj_forward(self.m,self.d)
    def apply(self):
        if self.ticks%20==0:
            mujoco.mj_fullM(self.m,self.d,self.M);self.mass=self.M[np.ix_(self.v,self.v)].copy()
        self.ticks+=1
        error=self.ref-self.d.qpos[self.q];self.max_error=max(self.max_error,float(np.max(abs(error))))
        tau=self.d.qfrc_bias[self.v]+self.mass@(400*error+40*(self.vel-self.d.qvel[self.v]))
        limits=self.m.actuator_forcerange[:self.n,1]
        self.d.ctrl[:self.n]=np.clip(tau,-limits,limits)
        self.peak_ratio=max(self.peak_ratio,float(np.max(abs(self.d.ctrl[:self.n])/limits)))
    def path(self,goals,seconds,advance):
        # Full-pose IK knots, interpolated in joint space. Reject a discontinuity
        # before execution rather than teleporting through an alternate branch.
        start=self.ref.copy();knots=[start];N=max(8,round(seconds/.06))
        sites=[self.m.site(name).id for name,p,R in goals]
        starts=[self.d.site_xpos[i].copy() for i in sites]
        for f in np.linspace(0,1,N+1)[1:]:
            a=f*f*(3-2*f)
            goal=[(name,p0+(p-p0)*a,R) for (name,p,R),p0 in zip(goals,starts)]
            q=self.ik(goal,knots[-1],attempts=1)
            if np.max(abs(q-knots[-1]))>.35:raise ValueError('Rejected discontinuous joint trajectory')
            knots.append(q)
        dt=self.m.opt.timestep;segment=seconds/N
        for qa,qb in zip(knots,knots[1:]):
            for f in np.linspace(0,1,max(1,round(segment/dt)),endpoint=False):
                self.ref=qa+(qb-qa)*f;self.vel=(qb-qa)/segment;advance(dt)
        self.ref=knots[-1];self.vel[:]=0;advance(.10)
