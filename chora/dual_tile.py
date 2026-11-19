"""Two native FR3 arms: finite suction workholding and contact-based wiping.

Coverage is a geometric contact diagnostic, NOT a grout/fluid removal model.
No tile weld, kinematic object tracking, or pose writes after reset.
"""
import mujoco
import numpy as np
import yaml
from .arm_control import robot_spec,studio,ArmController,DOWN
from .common import config,ROOT
from .tiling_materials import tile_properties,SurfaceGrout

MATERIAL=yaml.safe_load((ROOT/'configs/materials/tile_wipe.yaml').read_text())
PARAM=MATERIAL['parameters']
TILE_PRODUCT=tile_properties()
PARAM.update(tile_dimensions_m=TILE_PRODUCT['dimensions_m'],tile_mass_kg=TILE_PRODUCT['mass_kg'],tile_sliding_friction=TILE_PRODUCT['friction'])
TILE_CENTER=np.array([.5,0,.388])
TILE_HALF=np.array(PARAM['tile_dimensions_m'])/2

class DualTileSimulation:
    def __init__(self,cfg=None):
        cfg=cfg or config();self.cfg=cfg['simulation'];s=robot_spec('fr3_duo')
        studio(s,center=[.45,0,.40],bench=False)
        left=s.body('left_fr3v2_link8');right=s.body('right_fr3v2_link8')
        cup=left.add_body(name='vacuum_cup',pos=[0,0,.15])
        cup.add_geom(name='cup_stem',type=mujoco.mjtGeom.mjGEOM_CYLINDER,pos=[0,0,-.075],size=[.014,.075,0],mass=.12,rgba=[.45,.50,.55,1],contype=0,conaffinity=0)
        cup.add_geom(name='cup_contact',type=mujoco.mjtGeom.mjGEOM_CYLINDER,pos=[0,0,-.003],size=[PARAM['cup_radius_m'],.003,0],mass=.08,rgba=[.13,.21,.24,1],friction=[PARAM['cup_sliding_friction'],.01,.001],solref=[.005,1],contype=2,conaffinity=4)
        cup.add_site(name='hold_tcp',size=[.003]*3,rgba=[0,0,0,0])
        a=s.add_actuator(name='vacuum',target='vacuum_cup',trntype=mujoco.mjtTrn.mjTRN_BODY);a.set_to_adhesion(PARAM['vacuum_force_limit_n']);a.ctrllimited=True;a.ctrlrange=[0,1]
        wipe=right.add_body(name='sponge_holder',pos=[0,0,.15])
        wipe.add_geom(name='sponge_stem',type=mujoco.mjtGeom.mjGEOM_CYLINDER,pos=[0,0,-.0825],size=[.012,.0675,0],mass=.12,rgba=[.45,.5,.55,1],contype=0,conaffinity=0)
        wipe.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX,pos=[0,0,-.014],size=[.027,.020,.004],mass=.06,rgba=[.2,.27,.30,1],contype=0,conaffinity=0)
        wipe.add_geom(name='sponge',type=mujoco.mjtGeom.mjGEOM_BOX,pos=[0,0,-.005],size=np.array(PARAM['sponge_dimensions_m'])/2,mass=.025,rgba=[.92,.66,.22,1],friction=[PARAM['sponge_sliding_friction'],.005,.0001],solref=[PARAM['sponge_contact_time_constant_s'],1],solimp=[.85,.95,.005,.5,2],contype=8,conaffinity=4)
        wipe.add_site(name='wipe_tcp',size=[.003]*3,rgba=[0,0,0,0])
        tile=s.worldbody.add_body(name='tile',pos=TILE_CENTER);tile.add_freejoint(name='tile_free')
        tile.add_geom(name='tile_contact',type=mujoco.mjtGeom.mjGEOM_BOX,size=TILE_HALF,mass=PARAM['tile_mass_kg'],rgba=[.52,.24,.15,1],friction=[PARAM['tile_sliding_friction'],.005,.0001],contype=4,conaffinity=11,solref=[.004,1])
        self.surface=SurfaceGrout(PARAM['tile_dimensions_m'][0])
        for i,xy in enumerate(self.surface.centers):
            tile.add_geom(name=f'grout_cell_{i}',type=mujoco.mjtGeom.mjGEOM_BOX,pos=[*xy,TILE_HALF[2]+.0001],size=[self.surface.dx/2,self.surface.dx/2,.0001],mass=0,rgba=[.64,.62,.57,1],contype=0,conaffinity=0)
        # Surface markers show sampled contact coverage, not simulated residue.
        self.samples=np.array([[x,y,TILE_HALF[2]+.0003] for x in np.linspace(-.06,.06,13) for y in np.linspace(-.06,.06,13)])
        for i,p in enumerate(self.samples):tile.add_site(name=f'coverage_{i}',type=mujoco.mjtGeom.mjGEOM_SPHERE,pos=p,size=[.0016]*3,rgba=[0,0,0,0],group=1)
        self.model=s.compile();self.model.vis.global_.offwidth=1280;self.model.vis.global_.offheight=960
        self.data=mujoco.MjData(self.model);self.arm=ArmController(self.model,self.data,14)
        self.tile_id=self.model.body('tile').id;self.tile_q=int(self.model.joint('tile_free').qposadr[0]);self.vacuum_id=self.model.actuator('vacuum').id
        self.sponge_id=self.model.geom('sponge').id;self.tile_geom=self.model.geom('tile_contact').id
        self.sample_ids=np.array([self.model.site(f'coverage_{i}').id for i in range(len(self.samples))])
        self.grout_ids=np.array([self.model.geom(f'grout_cell_{i}').id for i in range(len(self.surface.centers))])
        self.on_frame=None;self.frame_stride=100;self.object_radius=float(max(TILE_HALF[:2]));self.reset(0)
    @property
    def tcp(self):return self.data.site('wipe_tcp').xpos.copy()
    @property
    def holder(self):return self.data.site('hold_tcp').xpos.copy()
    @property
    def bowl(self):return self.data.xpos[self.tile_id].copy()
    def reset(self,seed):
        self.seed=int(seed);mujoco.mj_resetData(self.model,self.data)
        self.data.qpos[self.tile_q:self.tile_q+3]=TILE_CENTER;self.data.qpos[self.tile_q+3:self.tile_q+7]=[1,0,0,0]
        self.hold_goal=TILE_CENTER-np.array([0,0,TILE_HALF[2]]);self.wipe_goal=np.array([.5,-.065,.455])
        self.arm.initialize(self.goals());self.arm.peak_ratio=0;self.arm.max_error=0;self.arm.ticks=0
        self.data.ctrl[self.vacuum_id]=1
        self.step_index=0;self.physics_steps=0;self.collision_count=0;self.action={'action':'reset'};self.result='reset';self.success=None;self.finished=False
        self.covered=np.zeros(len(self.samples),bool);self.model.site_rgba[self.sample_ids,3]=0
        self.surface=SurfaceGrout(PARAM['tile_dimensions_m'][0]);self.paste_force=np.zeros(3);self.update_surface()
        self.force=0.;self.peak_force=0.;self.contact_seconds=0.;self.max_tile_error=0.;self.unsafe_contact=False
        self._advance(.25);return self.observe()
    def goals(self):return [('hold_tcp',self.hold_goal,np.eye(3)),('wipe_tcp',self.wipe_goal,DOWN)]
    def update_surface(self):
        thickness=self.surface.thickness
        self.model.geom_size[self.grout_ids,2]=np.maximum(thickness/2,1e-7)
        self.model.geom_pos[self.grout_ids,2]=TILE_HALF[2]+thickness/2+1e-5
        self.model.geom_rgba[self.grout_ids,3]=np.clip(thickness/.00005,0,1)
    def _advance(self,seconds):
        dt=self.model.opt.timestep
        for _ in range(round(seconds/dt)):
            prev_local=self.data.xmat[self.tile_id].reshape(3,3).T@(self.tcp-self.bowl)
            self.data.qfrc_applied[:]=0
            point=self.tcp.copy()
            mujoco.mj_applyFT(self.model,self.data,self.paste_force,np.zeros(3),point,self.model.body('sponge_holder').id,self.data.qfrc_applied)
            mujoco.mj_applyFT(self.model,self.data,-self.paste_force,np.zeros(3),point,self.tile_id,self.data.qfrc_applied)
            self.arm.apply();mujoco.mj_step(self.model,self.data);self.physics_steps+=1;self.force=0.
            for i,c in enumerate(self.data.contact):
                if {int(c.geom1),int(c.geom2)}=={self.sponge_id,self.tile_geom}:
                    f=np.zeros(6);mujoco.mj_contactForce(self.model,self.data,i,f);self.force+=max(0.,float(f[0]))
                elif c.dist<-.001 and 'fr3v2' in (self.model.body(int(self.model.geom_bodyid[c.geom1])).name+self.model.body(int(self.model.geom_bodyid[c.geom2])).name):
                    self.collision_count+=1
                    raise RuntimeError('Robot collision guard stopped the trial')
            self.peak_force=max(self.peak_force,self.force)
            expected=self.holder+np.array([0,0,TILE_HALF[2]]);self.max_tile_error=max(self.max_tile_error,float(np.linalg.norm(self.bowl-expected)))
            R=self.data.xmat[self.tile_id].reshape(3,3);local=R.T@(self.tcp-self.bowl);velocity=(local[:2]-prev_local[:2])/dt;speed=float(np.linalg.norm(velocity))
            transfer=self.surface.contact(local[:2],[.025,.018],self.force,speed if self.force>.15 else 0.,dt)
            self.paste_force=R@np.r_[-transfer['drag_n']*velocity/speed,0.] if speed>1e-8 else np.zeros(3)
            if self.force>.15:
                self.contact_seconds+=dt
                # Require actual contact before marking geometrically covered cells.
                self.covered|=(abs(self.samples[:,0]-local[0])<.025)&(abs(self.samples[:,1]-local[1])<.018)
            if self.force>25 or np.linalg.norm(self.bowl-expected)>.03:
                self.unsafe_contact=True
                raise RuntimeError('Workholding/force guard stopped the trial')
            if self.physics_steps%self.frame_stride==0:
                self.update_surface()
                if self.on_frame:self.on_frame(self)
    def move(self,holder=None,wiper=None,seconds=.8):
        if holder is not None:self.hold_goal=np.array(holder,float)
        if wiper is not None:self.wipe_goal=np.array(wiper,float)
        self.arm.path(self.goals(),seconds,self._advance)
    def step(self,action):
        if self.finished:raise RuntimeError('Episode has ended')
        self.step_index+=1;self.action=action
        if action=={'action':'position_tile'}:
            self.move(holder=[.5,.025,TILE_CENTER[2]-TILE_HALF[2]],wiper=[.5,-.065,.455])
        elif action=={'action':'wipe_tile'}:
            # Raster on the exposed face. The holder remains actively controlled.
            for i,x in enumerate([.455,.50,.545]):
                y0,y1=(-.035,.085) if i%2==0 else (.085,-.035)
                self.move(wiper=[x,y0,.410],seconds=.45)
                working_z=float(TILE_CENTER[2]+TILE_HALF[2]-.004)
                self.move(wiper=[x,y0,working_z],seconds=.45)
                self.move(wiper=[x,y1,working_z],seconds=1.4)
                self.move(wiper=[x,y1,.425],seconds=.4)
        elif action=={'action':'retract'}:self.move(wiper=[.5,-.065,.48],seconds=.7)
        elif action=={'action':'done'}:self.finished=True;self.result='experiment_complete';return self.observe()
        else:self.result='invalid_action';return self.observe()
        self.result='ok';return self.observe()
    def observe(self):
        return dict(task='dual_tile_wipe',object_kind='ceramic tile',object=self.bowl.tolist(),object_yaw=0.,tcp=self.tcp.tolist(),holder_tcp=self.holder.tolist(),target=[.5,.025,.388],target_half_size=[.075,.075],step=self.step_index,last_result=self.result,
          coverage_percent=round(float(self.covered.mean()*100),1),contact_force_n=round(self.force,3),peak_wipe_force_n=round(self.peak_force,3),contact_seconds=round(self.contact_seconds,3),max_tile_error_mm=round(self.max_tile_error*1000,3),torque_limit_ratio=round(self.arm.peak_ratio,3),vacuum_force_limit_n=40.,material_calibrated=False,benchmark_validated=False,grout_model='reduced_surface_transport',
          tile_product=TILE_PRODUCT['name'],tile_mass_kg=PARAM['tile_mass_kg'],tile_mass_status=TILE_PRODUCT['mass_status'],
          grout_remaining_ml=round(self.surface.remaining_m3*1e6,5),grout_collected_ml=round(self.surface.collected_m3*1e6,5),grout_initial_ml=round(self.surface.initial_m3*1e6,5),
          grout_mass_balance_error_ml=round((self.surface.remaining_m3+self.surface.collected_m3-self.surface.initial_m3)*1e6,10),grout_age_s=round(self.surface.paste.age_s,2),paste_work_j=round(self.surface.work_j,6),
          fracture_model=False,installation_model=False)

class DualTileExpert:
    def __init__(self,seed):pass
    def act(self,obs,history):
        return {'action':['position_tile','wipe_tile','retract','done'][min(len(history),3)]}
