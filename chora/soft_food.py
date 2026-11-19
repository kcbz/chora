"""Tetrahedral food deformation and local cohesive release on one cut plane.

MuJoCo bulk elasticity supplies forces; this is an uncalibrated research model,
not a fluid solver. Contact work advances a cohesive front, never scored as proof
of real food cutting. The intact skin has no visible pre-cut interior.
"""
import itertools,math
import mujoco
import numpy as np
from scipy.spatial import cKDTree
from .gantry import stage_spec
from .scene_assets import material
from .common import config

FOOD={
 'apple':dict(radius=.035,height=.064,mass=.13,young=3e6,poisson=.25,skin_young=5e6,skin_thickness=.0003,toughness=140.),
 'tomato':dict(radius=.035,height=.056,mass=.14,young=1.5e4,poisson=.42,skin_young=8e5,skin_thickness=.00015,toughness=25.),
 'banana':dict(radius=.021,height=.040,mass=.10,young=1.5e5,poisson=.35,skin_young=1e6,skin_thickness=.001,toughness=65.),
}

def surface(kind,p,side=None,inside=False):
    if not inside and p.get('visual_asset'):
        from .scanned_produce import surface as scan_surface
        return scan_surface(kind,p['height'],side)
    v=[];uv=[];f=[]
    if inside:
        v=[[0,0,0]];uv=[[.5,.5]]
        for a in np.linspace(0,2*np.pi,65)[:-1]:v.append([p['radius']*np.cos(a),0,p['height']/2*np.sin(a)]);uv.append([.5+.5*np.cos(a),.5+.5*np.sin(a)])
        f=[[0,i+1,(i+1)%64+1] for i in range(64)]
        if side==-1:f=[x[::-1] for x in f]
    else:
        nlat,nlon=25,49
        for i in range(nlat):
            phi=.00001+(np.pi-.00002)*i/(nlat-1)
            for j in range(nlon):
                theta=(0 if side in (None,1) else np.pi)+(2*np.pi if side is None else np.pi)*j/(nlon-1)
                x=p['radius']*np.sin(phi)*np.cos(theta);y=p['radius']*np.sin(phi)*np.sin(theta);z=p['height']/2*np.cos(phi)
                if kind=='banana':y*=2.0;z+=.006*(y/(p['radius']*2))**2
                if kind=='apple':x*=1+.035*np.cos(5*theta)*np.sin(phi);y*=1+.035*np.cos(5*theta)*np.sin(phi)
                v.append([x,y,z]);uv.append([(theta%(2*np.pi))/(2*np.pi),i/(nlat-1)])
        for i in range(nlat-1):
            for j in range(nlon-1):
                a=i*nlon+j;b=a+nlon;f.extend([[a,b,a+1],[a+1,b,b+1]])
    return np.array(v),np.array(uv),np.array(f)


def add_bound_skin(s,name,v,uv,faces,nodepos,names,mat,alpha=1):
    d,idx=cKDTree(nodepos).query(v,k=4);weight=1/np.maximum(d,.0001)**3;weight/=weight.sum(1)[:,None]
    ids=[[] for _ in names];weights=[[] for _ in names]
    for vertex,(nn,ww) in enumerate(zip(idx,weight)):
        for node,w in zip(nn,ww):ids[node].append(vertex);weights[node].append(float(w))
    used=[i for i,vv in enumerate(ids) if vv]
    names=[names[i] for i in used];nodepos=nodepos[used];ids=[ids[i] for i in used];weights=[weights[i] for i in used]
    s.add_skin(name=name,material=mat,rgba=[1,1,1,alpha],vert=v.ravel(),face=faces.ravel(),texcoord=uv.ravel(),bodyname=names,bindpos=nodepos.ravel(),bindquat=np.tile([1,0,0,0],len(names)),vertid=ids,vertweight=weights)


def add_food(s,kind,center,properties=None):
    p=properties or FOOD[kind];parts=[];bonds=[]
    if p.get('visual_asset'):
        from .scanned_produce import material as scan_material
        outer=scan_material(s,kind)
    else:outer=material(s,kind)
    inner=material(s,kind,True)
    for side in [1,-1]:
        points=[];names=[];lookup={};coords=list(itertools.product(range(5),range(3),range(5)))
        for i,j,k in coords:
            u=np.array([i/2-1,j/2*side,k/2-1.])
            x,y,z=u;xyz=np.array([x*np.sqrt(1-y*y/2-z*z/2+y*y*z*z/3),y*np.sqrt(1-z*z/2-x*x/2+z*z*x*x/3),z*np.sqrt(1-x*x/2-y*y/2+x*x*y*y/3)])
            xyz*=np.array([p['radius'],p['radius']*p.get('aspect',2 if kind=='banana' else 1),p['height']/2])
            if kind=='banana':xyz[2]+=.006*(xyz[1]/(p['radius']*2))**2
            pos=xyz+center;name=f'food_{side}_{i}_{j}_{k}';lookup[(i,j,k)]=len(points);points.append(pos);names.append(name)
            b=s.worldbody.add_body(name=name,pos=pos,mass=p['mass']/150,ipos=[0,0,0],inertia=[1e-8]*3,explicitinertial=True)
            for axis in np.eye(3):b.add_joint(type=mujoco.mjtJoint.mjJNT_SLIDE,axis=axis)
        points=np.array(points);tets=[]
        for i,j,k in itertools.product(range(4),range(2),range(4)):
            for perm in itertools.permutations(range(3)):
                corners=[np.array([i,j,k])];q=corners[0].copy()
                for axis in perm:q=q.copy();q[axis]+=1;corners.append(q)
                ids=[lookup[tuple(q)] for q in corners];a=points[ids]
                if np.linalg.det((a[1:]-a[0]).T)<0:ids[1],ids[2]=ids[2],ids[1]
                tets.append(ids)
        # True volumetric Saint-Venant–Kirchhoff elasticity; visual skin is bound to these nodes.
        s.add_flex(name=f'bulk_{side}',dim=3,vertbody=names,vert=np.zeros_like(points).ravel(),elem=np.array(tets).ravel(),young=p['young'],poisson=p['poisson'],damping=.005,radius=.0004,selfcollide=mujoco.mjtFlexSelf.mjFLEXSELF_NONE,contype=4,conaffinity=3,friction=[.2,.001,.0001],solref=[.003,1],rgba=[0,0,0,0],group=5)
        # A thin elastic membrane supplies separate skin mechanics.
        counts={}
        for tet in tets:
            for face in itertools.combinations(tet,3):key=tuple(sorted(face));counts[key]=counts.get(key,0)+1
        faces=[face for face,n in counts.items() if n==1 and not all(abs(points[q,1]-center[1])<1e-8 for q in face)]
        s.add_flex(name=f'skin_mechanics_{side}',dim=2,vertbody=names,vert=np.zeros_like(points).ravel(),elem=np.array(faces).ravel(),young=p['skin_young'],poisson=.25,thickness=p['skin_thickness'],damping=.005,radius=.0001,selfcollide=mujoco.mjtFlexSelf.mjFLEXSELF_NONE,contype=0,conaffinity=0,rgba=[0,0,0,0],group=5)
        v,uv,f=surface(kind,p,side);add_bound_skin(s,f'outer_{side}',v+center,uv,f,points,names,outer,0)
        v,uv,f=surface(kind,p,side,True);add_bound_skin(s,f'inside_{side}',v+center,uv,f,points,names,inner,0)
        parts.append((points,names,lookup))
    for i,k in itertools.product(range(5),range(5)):
        a=parts[0][2][(i,0,k)];b=parts[1][2][(i,0,k)];name=f'cohesion_{i}_{k}'
        s.add_equality(name=name,type=mujoco.mjtEq.mjEQ_CONNECT,objtype=mujoco.mjtObj.mjOBJ_BODY,name1=parts[0][1][a],name2=parts[1][1][b],data=[0]*12,solref=[.001,1],solimp=[.99,.999,.0001,.5,2])
        bonds.append((name,parts[0][0][a]))
    # One continuous skin before damage; no pre-cut interior faces are exposed.
    allpoints=np.r_[parts[0][0],parts[1][0]];allnames=parts[0][1]+parts[1][1]
    v,uv,f=surface(kind,p);add_bound_skin(s,'intact_skin',v+center,uv,f,allpoints,allnames,outer,1)
    return allpoints,allnames,bonds


class SoftFoodSimulation:
    def __init__(self,cfg=None):
        cfg=cfg or config();self.cfg=cfg['simulation'];self.kind=self.cfg['object_kind'];self.food=FOOD[self.kind].copy();self.configure_food()
        s=self.build_stage();s.option.integrator=mujoco.mjtIntegrator.mjINT_DISCRETE;s.option.iterations=150;s.option.tolerance=1e-6;self.center=np.array(self.cfg['target'],float)+[0,0,.001+self.food['height']/2]
        self.initial,self.node_names,self.bonds=add_food(s,self.kind,self.center,self.food)
        self.model=s.compile();self.model.vis.global_.offwidth=1280;self.model.vis.global_.offheight=960;self.data=mujoco.MjData(self.model)
        self.node_ids=np.array([self.model.body(n).id for n in self.node_names]);self.bond_ids=[self.model.equality(n).id for n,p in self.bonds]
        self.tool_id=self.model.body('tool').id;self.contact_id=self.model.geom('tool_contact').id
        self.on_frame=None;self.frame_stride=100;self.object_radius=self.food['radius'];self.reset(0)

    @property
    def tcp(self):return self.data.qpos[:3].copy()
    def build_stage(self):return stage_spec('knife',.0005)
    def configure_food(self):pass
    def set_home(self):self.data.qpos[:3]=[0,.1,.30]
    def control(self):self.data.ctrl[:]=np.clip(6000*(self.goal-self.tcp)-100*self.data.qvel[:3],-80,80)
    @property
    def bowl(self):return np.mean(self.data.xpos[self.node_ids],axis=0)

    def reset(self,seed):
        self.seed=int(seed);mujoco.mj_resetData(self.model,self.data);self.set_home();mujoco.mj_forward(self.model,self.data)
        self.goal=self.tcp;self.step_index=0;self.action={'action':'reset'};self.result='reset';self.success=None;self.finished=False;self.physics_steps=0;self.collision_count=0
        self.work=0.;self.peak_force=0.;self.force=0.;self.fractured=False;self.released=set();self.damage=np.zeros(len(self.bonds));self.max_deformation=0.
        self.model.skin_rgba[:,3]=0;self.model.skin_rgba[self.model.skin('intact_skin').id,3]=1
        self._advance(.1);self.rest_positions=self.data.xpos[self.node_ids].copy();return self.observe()

    def _advance(self,seconds):
        dt=self.model.opt.timestep
        for _ in range(round(seconds/dt)):
            prev=self.tcp.copy();self.control()
            mujoco.mj_step(self.model,self.data);self.physics_steps+=1;self.force=0.
            for i,c in enumerate(self.data.contact):
                if self.contact_id in (int(c.geom1),int(c.geom2)):
                    f=np.zeros(6);mujoco.mj_contactForce(self.model,self.data,i,f);self.force+=abs(float(f[0]))
            self.peak_force=max(self.peak_force,self.force);increment=self.force*max(0.,prev[2]-self.tcp[2]);self.work+=increment
            # Local cohesive front, driven by actual contact work and blade position.
            active=[j for j,(n,p) in enumerate(self.bonds) if j not in self.released and p[2]>=self.tcp[2]-.001 and abs(p[0]-self.tcp[0])<.065]
            if active and self.force>.5:
                area=math.pi*self.food['radius']*self.food['height']/len(self.bonds)
                for j in active:
                    self.damage[j]+=increment/len(active)
                    if self.damage[j]>=self.food['toughness']*area:
                        self.data.eq_active[self.bond_ids[j]]=0;self.released.add(j)
            if self.released and not self.fractured:
                self.fractured=True;self.model.skin_rgba[:,3]=1;self.model.skin_rgba[self.model.skin('intact_skin').id,3]=0
            if hasattr(self,'rest_positions'):self.max_deformation=max(self.max_deformation,float(np.max(np.linalg.norm(self.data.xpos[self.node_ids]-self.rest_positions,axis=1))))
            if self.on_frame and self.physics_steps%self.frame_stride==0:self.on_frame(self)

    def move(self,target,seconds):
        target=np.array(target,float)
        if not np.all(np.isfinite(target)) or not (-.1<=target[0]<=.1 and 0<=target[1]<=.2 and .148<=target[2]<=.4):return 'rejected_unreachable'
        start=self.tcp
        for f in np.linspace(0,1,round(seconds/self.model.opt.timestep)):
            self.goal=start+(target-start)*f*f*(3-2*f);self._advance(self.model.opt.timestep)
        self.goal=target;self._advance(.05);return 'ok'

    def observe(self):
        return {'task':'soft_cut','object_kind':self.kind,'object':[round(float(x),5) for x in self.bowl],'object_yaw':0.,'tcp':[round(float(x),5) for x in self.tcp],
                'target':self.cfg['target'],'target_half_size':[.14,.11],'food_height':self.food['height'],'peak_cut_force_n':round(self.peak_force,3),'cut_force_n':round(self.force,3),'cut_work_j':round(self.work,5),
                'released_bonds':len(self.released),'total_bonds':len(self.bonds),'deformation_mm':round(self.max_deformation*1000,3),'fracture_started':self.fractured,'mounted_tool':True,'material_calibrated':False,'benchmark_validated':False,'step':self.step_index,'last_result':self.result}

    def step(self,action):
        if self.finished:raise RuntimeError('Episode has ended')
        self.step_index+=1;self.action=action
        try:
            if action['action']=='move_to':self.result=self.move([action[k] for k in 'xyz'],.5)
            elif action['action']=='slice':self.result=self.move([*self.tcp[:2],.1505],1.2)
            elif action=={'action':'retract'}:self.result=self.move([*self.tcp[:2],.30],.6)
            elif action=={'action':'done'}:self.finished=True;self.result='experiment_complete';self.success=None
            else:self.result='invalid_action'
        except (KeyError,ValueError,TypeError):self.result='invalid_action'
        if self.step_index>=self.cfg['max_steps']:self.finished=True;self.result='step_limit'
        return self.observe()


class SoftFoodExpert:
    def __init__(self,seed):pass
    def act(self,obs,history):
        if not history:return {'action':'move_to','x':obs['target'][0],'y':obs['target'][1],'z':obs['target'][2]+obs['food_height']+.01}
        if history[-1]['action']=='move_to':return {'action':'slice'}
        if history[-1]['action']=='slice':return {'action':'retract'}
        return {'action':'done'}
