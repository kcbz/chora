"""Independent FR3 tile workcell: optional mortar application and bedding.

Native articulated dynamics and free rigid tiles; custom installed face-vacuum
and metering/comb fixtures. Mortar support comes from a declared conservative
spatial reduced model. No object-pose writes after reset, welds or fracture.
"""
import mujoco
import numpy as np
from .arm_control import ArmController, DOWN, robot_spec, studio
from .tiling_materials import tile_properties
from .tiling_task import validate_action

TILE = tile_properties()
HALF = np.array(TILE['dimensions_m']) / 2
SURFACE = .24
PITCH = TILE['dimensions_m'][0] + .009525
TARGETS = np.array([[.43 + col*PITCH, .02 + row*PITCH, SURFACE+HALF[2]]
                    for row in range(2) for col in range(2)])
RACK = np.array([.43, -.22])


def build_model(wet=False):
    s = mujoco.MjSpec()
    s.modelname = 'Chora independent FR3 tiling workcell — adhesive bedding' if wet else 'Chora independent FR3 tiling workcell — dry transfer'
    s.option.timestep = .0005
    s.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    s.option.iterations = 100
    s.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    # Separate base mounts leave the work surface open; no Duo torso/pole holder.
    for label, y in [('A', -.22), ('B', .55)]:
        child = robot_spec('fr3')
        frame = s.worldbody.add_frame(pos=[(1.0 if label=='B' else .08) if wet else 0, (.15 if label=='B' else -.15) if wet else y, 0],quat=[0,0,0,1] if label=='B' and wet else [1,0,0,0])
        s.attach(child, prefix=label+'_', frame=frame)
    from .tiling_visuals import materials,add_tile,MortarSurface
    materials(s)
    studio(s, center=[.43, 0, .3], bench=False)
    w = s.worldbody
    w.add_geom(name='substrate', type=mujoco.mjtGeom.mjGEOM_BOX,
               pos=[.51, .10, SURFACE/2], size=[.19, .19, SURFACE/2],
               rgba=[.47,.49,.46,1], contype=8, conaffinity=5)
    w.add_geom(name='rack', type=mujoco.mjtGeom.mjGEOM_BOX,
               pos=[*RACK, SURFACE/2], size=[.09,.09,SURFACE/2],
               rgba=[.27,.33,.36,1], contype=8, conaffinity=5)
    for i, p in enumerate(TARGETS):
        for axis in range(2):
            for sign in [-1, 1]:
                pos=p.copy(); pos[2]=SURFACE+.0002
                pos[axis]+=sign*HALF[axis]
                size=[HALF[0],.0006,.0002] if axis else [.0006,HALF[1],.0002]
                w.add_geom(name=f'pattern_{i}_{axis}_{sign}',type=mujoco.mjtGeom.mjGEOM_BOX,
                           pos=pos,size=size,contype=0,conaffinity=0,rgba=[.40,.70,.65,1])
    flange=s.body('A_fr3v2_link8')
    cup=flange.add_body(name='vacuum_fixture',pos=[0,0,.08])
    cup.add_geom(name='vacuum_stem',type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                 pos=[0,0,-.04],size=[.012,.04,0],mass=.15,
                 rgba=[.45,.5,.54,1],contype=1,conaffinity=9)
    cup.add_geom(name='vacuum_face',type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                 size=[.025,.004,0],mass=.12,rgba=[.12,.16,.18,1],
                 friction=[.7,.005,.001],solref=[.005,1],contype=2,conaffinity=4)
    cup.add_site(name='tile_tcp',pos=[0,0,.004],size=[.003]*3,rgba=[0,0,0,0])
    if wet:
        # Installed custom metering/comb head, not a claimed commercial trowel.
        head=s.body('B_fr3v2_link8').add_body(name='mortar_tool',pos=[0,0,.12])
        head.add_geom(name='comb_stem',type=mujoco.mjtGeom.mjGEOM_CYLINDER,pos=[0,0,-.0725],size=[.012,.0475,0],mass=.16,rgba=[.45,.5,.54,1])
        head.add_geom(name='blade_backplate',type=mujoco.mjtGeom.mjGEOM_BOX,pos=[0,0,-.0125],size=[.002,.0762,.0125],mass=.10,rgba=[.65,.69,.72,1])
        head.add_geom(name='comb_spine',type=mujoco.mjtGeom.mjGEOM_BOX,pos=[0,0,-.004],size=[.002,.0762,.004],mass=.15,rgba=[.65,.69,.72,1])
        for j in range(16):
            if j%2:
                head.add_geom(name=f'comb_tooth_{j}',type=mujoco.mjtGeom.mjGEOM_BOX,pos=[0,(j+.5)*.009525-.0762,.004],size=[.002,.009525/2,.004],mass=.005,rgba=[.65,.69,.72,1])
        head.add_geom(name='feed_manifold',type=mujoco.mjtGeom.mjGEOM_BOX,pos=[0,0,-.018],size=[.005,.0762,.004],mass=.12,rgba=[.28,.34,.37,1])
        head.add_geom(name='hopper',type=mujoco.mjtGeom.mjGEOM_CYLINDER,pos=[.045,0,-.065],size=[.037,.045,0],mass=1.2,rgba=[.38,.43,.45,1])
        head.add_geom(name='feed_neck',type=mujoco.mjtGeom.mjGEOM_CAPSULE,fromto=[.045,0,-.034,0,0,-.034],size=[.012,0,0],mass=.05,rgba=[.38,.43,.45,1])
        head.add_geom(name='feed_downpipe',type=mujoco.mjtGeom.mjGEOM_CYLINDER,pos=[0,0,-.026],size=[.005,.008,0],mass=.02,rgba=[.38,.43,.45,1])
        head.add_geom(name='feed_channel',type=mujoco.mjtGeom.mjGEOM_BOX,pos=[0,0,-.013],size=[.003,.0762,.005],mass=.04,rgba=[.28,.34,.37,1])
        head.add_site(name='park_tcp',pos=[0,0,.008],size=[.003]*3,rgba=[0,0,0,0])
        from .mortar_layer import MortarLayer
        layer=MortarLayer(TARGETS)
        MortarSurface.add(s,TARGETS,TILE['dimensions_m'][0],SURFACE)
        for i,patch in enumerate(layer.xy):
            for j,xy in enumerate(patch):
                w.add_geom(name=f'mortar_{i}_{j}',type=mujoco.mjtGeom.mjGEOM_BOX,pos=[*xy,SURFACE],size=[layer.dx/2,layer.dx/2,1e-7],rgba=[.64,.63,.58,0],group=5,contype=0,conaffinity=0)
            for j in range(4):
                axis=j//2;sign=(-1,1)[j%2];p=TARGETS[i].copy();p[axis]+=sign*(HALF[axis]+.0015);p[2]=SURFACE
                size=[HALF[0],.0015,1e-7] if axis else [.0015,HALF[1],1e-7]
                w.add_geom(name=f'squeeze_{i}_{j}',type=mujoco.mjtGeom.mjGEOM_BOX,pos=p,size=size,rgba=[.64,.63,.58,0],contype=0,conaffinity=0)
    else:
        s.body('B_fr3v2_link8').add_site(name='park_tcp',pos=[0,0,.084],size=[.003]*3,rgba=[0,0,0,0])
    a=s.add_actuator(name='vacuum',target='vacuum_fixture',trntype=mujoco.mjtTrn.mjTRN_BODY)
    a.set_to_adhesion(40);a.ctrllimited=True;a.ctrlrange=[0,1]
    for i in range(4):
        tile=w.add_body(name=f'tile_{i}',pos=[*RACK,SURFACE+HALF[2]+(3-i)*2*HALF[2]+.0001])
        tile.add_freejoint(name=f'tile_{i}_free')
        tile.add_geom(name=f'tile_{i}_contact',type=mujoco.mjtGeom.mjGEOM_BOX,
                      size=HALF,mass=TILE['mass_kg'],rgba=[.57,.26,.16,1],group=3,
                      friction=[TILE['friction'],.005,.0001],solref=[.005,1],
                      contype=4,conaffinity=15)
        add_tile(s,tile,i,HALF)
    m=s.compile();m.vis.global_.offwidth=1280;m.vis.global_.offheight=960
    return m


class TileInstallationSimulation:
    def __init__(self,cfg=None):
        self.wet=(cfg or {}).get('experiment',{}).get('task')=='tile_adhesive_bedding'
        self.model=build_model(self.wet);self.data=mujoco.MjData(self.model)
        self.targets=TARGETS.copy()
        if self.wet:self.targets[:,2]+=.003
        self.arm=ArmController(self.model,self.data,14)
        self.tiles=[self.model.body(f'tile_{i}').id for i in range(4)]
        self.tile_geoms=[self.model.geom(f'tile_{i}_contact').id for i in range(4)]
        self.vacuum_id=self.model.actuator('vacuum').id
        self.face_id=self.model.geom('vacuum_face').id
        self.on_frame=None;self.frame_stride=100;self.object_radius=float(HALF[0])
        self.cfg=(cfg or {}).get('simulation',{})
        self.reset(0)
    @property
    def tcp(self):return self.data.site('tile_tcp').xpos.copy()
    @property
    def bowl(self):return self.data.xpos[self.tiles[min(self.active_tile,3)]].copy()
    def goals(self):return [('tile_tcp',self.goal,DOWN),('park_tcp',self.park,np.diag([-1.,1.,-1.]) if self.wet else DOWN)]
    def reset(self,seed):
        self.seed=int(seed);mujoco.mj_resetData(self.model,self.data)
        self.goal=np.array([.43,-.22,.46]);self.park=np.array([.85,.50,.47] if self.wet else [.30,.65,.47])
        self.arm.initialize(self.goals());self.arm.peak_ratio=0;self.arm.max_error=0
        self.physics_steps=0;self.step_index=0;self.collision_count=0
        self.action={'action':'reset'};self.result='ready';self.finished=False;self.success=None
        self.tool_force=np.zeros(3);self.tool_peak_force=0.
        self.dirty_meshes=set()
        self.parallel_active=False;self.concurrent_motion_s=0.;self.overlap_mortaring_s=0.
        if self.wet:
            from .mortar_layer import MortarLayer
            self.mortar=MortarLayer(TARGETS)
            self.mortar_ids=np.array([[self.model.geom(f'mortar_{i}_{j}').id for j in range(256)] for i in range(4)])
            self.squeeze_ids=np.array([[self.model.geom(f'squeeze_{i}_{j}').id for j in range(4)] for i in range(4)])
            from .tiling_visuals import MortarSurface
            self.mortar_visual=MortarSurface(self.model,self.data,self.mortar)
            self.update_mortar_visuals()
        self.tile_load_n=np.zeros(4)
        self.active_tile=0;self.force_limit=60.;self.peak_force=0.;self.contact_force=0.
        self.data.ctrl[self.vacuum_id]=0;self._advance(.4)
        return self.observe()
    def _advance(self,seconds):
        for _ in range(round(seconds/self.model.opt.timestep)):
            previous=self.data.site('park_tcp').xpos.copy();previous_a=self.tcp
            deposited_before=self.mortar.applied_by_tool_m3 if self.wet else 0.
            if self.wet:
                self.data.qfrc_applied[:]=0
                self.mortar.advance(self.model.opt.timestep)
                for index,bid in enumerate(self.tiles):
                    velocity=np.zeros(6)
                    mujoco.mj_objectVelocity(self.model,self.data,mujoco.mjtObj.mjOBJ_BODY,bid,velocity,0)
                    force,torque=self.mortar.couple_tile(self.data.xpos[bid],self.data.xmat[bid].reshape(3,3),velocity[3:],velocity[:3],self.model.opt.timestep)
                    self.tile_load_n[index]=force[2]
                    mujoco.mj_applyFT(self.model,self.data,force,torque,self.data.xpos[bid],bid,self.data.qfrc_applied)
                mujoco.mj_applyFT(self.model,self.data,self.tool_force,np.zeros(3),previous,self.model.body('mortar_tool').id,self.data.qfrc_applied)
            self.arm.apply();mujoco.mj_step(self.model,self.data);self.physics_steps+=1
            if self.wet:
                self.tool_force=self.mortar.apply_tool(previous,self.data.site('park_tcp').xpos,self.model.opt.timestep)
                both_moving=np.linalg.norm(self.tcp-previous_a)/self.model.opt.timestep>.005 and np.linalg.norm(self.data.site('park_tcp').xpos-previous)/self.model.opt.timestep>.005
                if self.parallel_active and both_moving:self.concurrent_motion_s+=self.model.opt.timestep
                if self.parallel_active and self.mortar.applied_by_tool_m3>deposited_before and np.linalg.norm(self.tcp-previous_a)/self.model.opt.timestep>.005:self.overlap_mortaring_s+=self.model.opt.timestep
                self.tool_peak_force=max(self.tool_peak_force,float(np.linalg.norm(self.tool_force)))
                if np.linalg.norm(self.tool_force)>30:raise RuntimeError('Mortar tool force guard exceeded')
            self.contact_force=0.
            for i,c in enumerate(self.data.contact):
                if self.face_id in (c.geom1,c.geom2):
                    f=np.zeros(6);mujoco.mj_contactForce(self.model,self.data,i,f)
                    self.contact_force+=float(np.linalg.norm(f[:3]))
                if self.wet and any(g in self.tile_geoms for g in (c.geom1,c.geom2)) and any(self.model.body(int(self.model.geom_bodyid[g])).name=='mortar_tool' for g in (c.geom1,c.geom2)):
                    self.collision_count+=1
                    raise RuntimeError('Applicator touched a tile: '+str([(self.model.geom(g).name,self.data.geom_xpos[g].round(4).tolist()) for g in (c.geom1,c.geom2)]))
                names=[self.model.body(int(self.model.geom_bodyid[g])).name or '' for g in (c.geom1,c.geom2)]
                if c.dist<=0 and any(n.startswith('A_') or n=='vacuum_fixture' for n in names) and any(n.startswith('B_') or n=='mortar_tool' for n in names):
                    self.collision_count+=1;raise RuntimeError('Inter-arm contact stopped the coordinated task')
                # Ignore stationary native base/floor contacts, guard moving links.
                for g1,g2 in [(c.geom1,c.geom2),(c.geom2,c.geom1)]:
                    b=int(self.model.geom_bodyid[g1]);other=int(self.model.geom_bodyid[g2])
                    name=self.model.body(b).name or ''
                    moving=('_link' in name and not name.endswith('link0')) or name in {'vacuum_fixture','mortar_tool'}
                    if moving and other!=b and c.dist<-.002 and g1!=self.face_id:
                        # Native adjacent-link exclusions are retained by attachment.
                        self.collision_count+=1
                        raise RuntimeError('Moving robot collision stopped tile task: '+name)
            self.peak_force=max(self.peak_force,self.contact_force)
            if self.contact_force>self.force_limit:raise RuntimeError('Vacuum contact force limit exceeded')
            if np.any(self.data.warning.number):raise RuntimeError('MuJoCo solver warning stopped tile task')
            if self.physics_steps%self.frame_stride==0:
                if self.wet:self.update_mortar_visuals()
                if self.on_frame:self.on_frame(self)
    def update_mortar_visuals(self):
        self.dirty_meshes.update(self.mortar_visual.update())
        h=self.mortar.height
        self.model.geom_size[self.mortar_ids,2]=np.maximum(h/2,1e-7)
        self.model.geom_pos[self.mortar_ids,2]=SURFACE+h/2
        self.model.geom_rgba[self.mortar_ids,3]=np.where(h>1e-6,1.,0.)
        # Outside-volume compartment drawn uniformly at footprint edges; not a flow map.
        height=self.mortar.squeezed_m3/(8*HALF[0]*.003)
        self.model.geom_size[self.squeeze_ids,2]=np.maximum(height[:,None]/2,1e-7)
        self.model.geom_pos[self.squeeze_ids,2]=SURFACE+height[:,None]/2
        self.model.geom_rgba[self.squeeze_ids,3]=np.where(height[:,None]>1e-6,1.,0.)
        # MuJoCo caches world-fixed geom transforms; forward() does not refresh
        # them after model.geom_pos edits. Update only these noncontact visuals.
        self.data.geom_xpos[self.mortar_ids]=self.model.geom_pos[self.mortar_ids]
        self.data.geom_xpos[self.squeeze_ids]=self.model.geom_pos[self.squeeze_ids]
    def bedding_ok(self,i):
        age=self.mortar.paste.age_s-self.mortar.spread_at[i]
        from .mortar_layer import PARAM
        return bool(self.mortar.contact_mask[i].mean()>PARAM['bedding_contact_fraction'] and self.mortar.paste.workable and age<2400 and abs(self.mortar.balance_error_m3)<1e-12)
    def grip(self):
        contacts=[]
        for c in self.data.contact:
            if self.face_id in (c.geom1,c.geom2):
                other=c.geom2 if c.geom1==self.face_id else c.geom1
                if other in self.tile_geoms:contacts.append(self.tile_geoms.index(other))
        return sorted(set(contacts)) if self.data.ctrl[self.vacuum_id]>0 else []
    def placement(self,i):
        bid=self.tiles[i];p=self.data.xpos[bid];R=self.data.xmat[bid].reshape(3,3)
        j=self.model.joint(f'tile_{i}_free');v=self.data.qvel[int(j.dofadr[0]):int(j.dofadr[0])+6]
        yaw=abs(float(np.arctan2(R[1,0],R[0,0])))
        support=any(self.tile_geoms[i] in (c.geom1,c.geom2) and self.model.geom('substrate').id in (c.geom1,c.geom2) for c in self.data.contact)
        if self.wet:support=self.tile_load_n[i]>TILE['mass_kg']*9.81*.7
        return dict(position_error_mm=round(float(np.linalg.norm(p[:2]-self.targets[i,:2]))*1000,3),
                    height_error_mm=round(abs(float(p[2]-self.targets[i,2]))*1000,3),
                    yaw_error_deg=round(float(np.degrees(yaw)),3),
                    stable=bool(np.linalg.norm(v)<.01),supported=bool(support),
                    placed=bool(np.linalg.norm(p[:2]-self.targets[i,:2])<.003 and abs(p[2]-self.targets[i,2])<.002 and R[2,2]>.999 and yaw<.02 and np.linalg.norm(v)<.01 and support and i not in self.grip()))
    def observe(self):
        return dict(task='tile_adhesive_bedding' if self.wet else 'tile_dry_transfer',protocol='chora-tile-adhesive-bedding-v2' if self.wet else 'chora-tile-dry-transfer-v1',
          mortar=self.mortar.snapshot() if self.wet else None,tool_tcp_mm=(self.data.site('park_tcp').xpos*1000).round(3).tolist(),tool_peak_force_n=round(self.tool_peak_force,4),
          concurrent_motion_s=round(self.concurrent_motion_s,3),overlap_mortaring_s=round(self.overlap_mortaring_s,3),
          simulation_elapsed_s=round(float(self.data.time),3),
          milestone='apply_comb_pick_place_bed' if self.wet else 'pickup_transport_dry_placement',installation_complete=False,
          object=self.bowl.tolist(),object_kind='ceramic tile',object_yaw=0.,tcp=self.tcp.tolist(),
          target=self.targets[min(self.active_tile,3)].tolist(),target_half_size=HALF[:2].tolist(),
          tiles=[dict(index=i,position_mm=(self.data.xpos[b]*1000).round(3).tolist(),**self.placement(i)) for i,b in enumerate(self.tiles)],
          targets_mm=(self.targets*1000).tolist(),tile_thickness_mm=float(HALF[2]*2000),
          vacuum_on=bool(self.data.ctrl[self.vacuum_id]),grip_tiles=self.grip(),
          contact_force_n=round(self.contact_force,3),peak_contact_force_n=round(self.peak_force,3),
          torque_limit_ratio=round(self.arm.peak_ratio,4),tile_mass_kg=TILE['mass_kg'],tile_mass_status=TILE['mass_status'],
          vacuum_force_limit_n=40,fixture_status='simplified_custom_unverified',adhesive_model='reduced_spatial_layer' if self.wet else False,fracture_model=False,
          initial_fixture='four clean rigid tiles on rack; premixed finite adhesive hopper' if self.wet else 'four dry clean rigid tiles stacked on rack; no adhesive',
          step=self.step_index,last_result=self.result)
    def validate_pose(self,action):
        p=np.array(action['position_mm'])/1000;arm=action['arm']
        if arm=='B' and not self.wet:raise ValueError('Arm B is not equipped in dry mode')
        if action['rpy_deg']!=([180,0,180] if arm=='B' and self.wet else [180,0,0]):raise ValueError('Unsupported tool orientation')
        if not (.29<=p[0]<=(.90 if arm=='B' and self.wet else .72) and -.32<=p[1]<=.70 and (.240 if arm=='B' else .245)<=p[2]<=.56):raise ValueError('Outside workspace')
        return p
    def execute_parallel(self,commands):
        if not self.wet:return 'parallel_requires_two_equipped_arms'
        poses=[(c,self.validate_pose(c)) for c in commands if c['action']=='move_to']
        for c in commands:
            if c['action']=='dispense' and (c['arm']!='B' or c['material']!='adhesive'):raise ValueError('Unsupported material or arm')
            if c['action']=='vacuum' and c['arm']!='A':raise ValueError('Unsupported vacuum arm')
        for c in commands:
            if c['action']=='dispense':
                result=self.mortar.arm_dose(c['volume_ml'])
                if result not in {'dose_armed_for_tool_motion','valve_closed'}:raise RuntimeError(result)
            elif c['action']=='vacuum':self.data.ctrl[self.vacuum_id]=int(c['state']=='on')
        seconds=.2
        for c,p in poses:
            start=self.tcp if c['arm']=='A' else self.data.site('park_tcp').xpos.copy()
            seconds=max(seconds,float(np.linalg.norm(p-start))/(c['speed_mm_s']/1000))
            if c['arm']=='A':self.goal=p;self.force_limit=c['force_limit_n']
            else:self.park=p
        self.parallel_active=True
        try:
            if poses:self.arm.path(self.goals(),max(.4,seconds),self._advance)
            else:self._advance(seconds)
        finally:self.parallel_active=False
        if self.grip():self.active_tile=self.grip()[0]
        return 'ok'
    def execute(self,action):
        validate_action(action);name=action['action']
        if name=='parallel':return self.execute_parallel(action['actions'])
        if name=='move_to':
            if action['arm']=='B' and not self.wet:return 'unsupported_parked_arm'
            p=np.array(action['position_mm'])/1000
            if action['rpy_deg']!=([180,0,180] if action['arm']=='B' and self.wet else [180,0,0]):return 'unsupported_orientation'
            if not (.29<=p[0]<=(.90 if action['arm']=='B' and self.wet else .72) and -.32<=p[1]<=.70 and (.240 if action['arm']=='B' else .245)<=p[2]<=.56):return 'outside_workspace'
            self.force_limit=action['force_limit_n']
            if action['arm']=='A':self.goal=p;start=self.tcp
            else:self.park=p;start=self.data.site('park_tcp').xpos.copy()
            seconds=max(.4,float(np.linalg.norm(p-start))/(action['speed_mm_s']/1000))
            self.arm.path(self.goals(),seconds,self._advance)
        elif name=='vacuum':
            if action['arm']!='A':return 'unsupported_parked_arm'
            self.data.ctrl[self.vacuum_id]=int(action['state']=='on');self._advance(.2)
            if self.grip():self.active_tile=self.grip()[0]
        elif name=='dispense' and self.wet:
            if action['arm']!='B' or action['material']!='adhesive':return 'unsupported_material_or_arm'
            return self.mortar.arm_dose(action['volume_ml'])
        elif name=='wait':
            if action['seconds']>5:return 'dry_transfer_wait_limit'
            self._advance(action['seconds'])
        elif name=='inspect':self._advance(.3)
        else:return 'unsupported_in_dry_transfer'
        return 'ok'
    def step(self,action):
        if self.finished:raise RuntimeError('Episode has ended')
        self.action=action;self.step_index+=1
        if action=={'action':'finish_stage'}:
            self._advance(.6);self.success=all(self.placement(i)['placed'] for i in range(4))
            if self.wet:self.success=self.success and all(self.bedding_ok(i) for i in range(4))
            self.finished=True;self.result=('adhesive_bedding_complete' if self.success else 'adhesive_bedding_incomplete') if self.wet else ('dry_transfer_complete' if self.success else 'dry_transfer_incomplete')
        else:self.result=self.execute(action)
        return self.observe()


class TileTransferExpert:
    """Deterministic dry-transfer reference; not a full installation expert."""
    def __init__(self,seed):pass
    def act(self,obs,history):
        i=len(history)//8;phase=len(history)%8
        if i>=4:return {'action':'finish_stage'}
        tile=np.array(obs['tiles'][i]['position_mm']);target=np.array(obs['targets_mm'][i]);half=obs['tile_thickness_mm']/2
        if phase==3 and obs['grip_tiles']!=[i]:raise RuntimeError('Dry-transfer reference stopped: expected tile grip was not established')
        if phase in (2,6):return dict(action='vacuum',arm='A',state='on' if phase==2 else 'off')
        if phase==7:return dict(action='move_to',arm='A',position_mm=[*target[:2],420],rpy_deg=[180,0,0],speed_mm_s=80,force_limit_n=60)
        poses={0:[*tile[:2],420],1:[*tile[:2],tile[2]+half-.3],
               3:[*tile[:2],420],4:[*target[:2],420],5:[*target[:2],target[2]+half-.3]}
        return dict(action='move_to',arm='A',position_mm=np.round(poses[phase],3).tolist(),
                    rpy_deg=[180,0,0],speed_mm_s=35 if phase in (1,5) else 80,force_limit_n=60)


class TileBeddingExpert:
    """Shared-timeline pickup/application; local tool clearance before bedding."""
    cycle_steps=13
    def __init__(self,seed,tile_count=4):
        if isinstance(tile_count,bool) or not isinstance(tile_count,int) or not 1<=tile_count<=4:raise ValueError('Invalid reference tile count')
        self.tile_count=tile_count
    def act(self,obs,history):
        i=len(history)//self.cycle_steps;phase=len(history)%self.cycle_steps
        if i>=self.tile_count:return {'action':'finish_stage'}
        tile=np.array(obs['tiles'][i]['position_mm']);target=np.array(obs['targets_mm'][i]);half=obs['tile_thickness_mm']/2
        start=target[0]-HALF[0]*1000-5;end=target[0]+HALF[0]*1000+5
        def move(arm,pose,speed=70):return dict(action='move_to',arm=arm,position_mm=np.round(pose,3).tolist(),rpy_deg=[180,0,180] if arm=='B' else [180,0,0],speed_mm_s=speed,force_limit_n=80)
        def parallel(a,b):return dict(action='parallel',actions=[a,b])
        if phase==0:return parallel(move('A',[*tile[:2],380]),move('B',[start,target[1],320]))
        if phase==1:return parallel(move('A',[*tile[:2],tile[2]+half-.3],35),move('B',[start,target[1],240.8],35))
        if phase==2:return parallel(dict(action='vacuum',arm='A',state='on'),dict(action='dispense',arm='B',material='adhesive',volume_ml=94.))
        if phase==3:
            if obs['grip_tiles']!=[i]:raise RuntimeError('Coordinated reference stopped: tile grip missing')
            return parallel(move('A',[*tile[:2],380]),move('B',[end,target[1],240.8],35))
        if phase==4:return dict(action='dispense',arm='B',material='adhesive',volume_ml=0.)
        if phase==5:return move('B',[800,target[1],320],50)
        if phase==6:return move('A',[*target[:2],380])
        if phase==7:
            standby=[850,500,470] if i==self.tile_count-1 else [800,obs['targets_mm'][i+1][1],320]
            return parallel(move('A',[*target[:2],262.7],35),move('B',standby,50))
        if phase==8:return move('A',[*target[:2],target[2]+half-.3],2)
        if phase==9:return dict(action='wait',seconds=1.)
        if phase==10:return dict(action='vacuum',arm='A',state='off')
        if phase==11:return move('A',[*target[:2],380])
        return dict(action='inspect',target='placement')
