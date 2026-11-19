"""Physical regressions for the articulated adapters, not benchmark scores."""
import mujoco
import numpy as np
import pytest
from chora.experiments import load_config,catalog
from chora.sim import make_simulation
from chora.policy import make_expert

@pytest.mark.parametrize('name',['ur10e-hammer','fr3-cut-apple'])
def test_tool_adapter_meets_actual_flange_after_quaternion_normalization(name):
    s=make_simulation(load_config(name));m=s.model;d=s.data;g=m.geom('adapter').id
    for delta in [0.,.1,-.15]:
        d.qpos[s.arm.q]+=delta;mujoco.mj_forward(m,d)
        flange=d.site('attachment_site').xpos if name.startswith('ur10e') else d.xpos[m.body('fr3v2_link8').id]
        ends=[d.geom_xpos[g]+a*m.geom_size[g,1]*d.geom_xmat[g].reshape(3,3)[:,2] for a in [-1,1]]
        assert min(np.linalg.norm(p-flange) for p in ends)<1e-9

def test_dual_stems_meet_flange_without_floating_gap():
    s=make_simulation(load_config('fr3-duo-tile-wipe'));m=s.model;d=s.data
    for geom,side in [('cup_stem','left'),('sponge_stem','right')]:
        g=m.geom(geom).id;flange=d.xpos[m.body(side+'_fr3v2_link8').id]
        ends=[d.geom_xpos[g]+a*m.geom_size[g,1]*d.geom_xmat[g].reshape(3,3)[:,2] for a in [-1,1]]
        assert min(np.linalg.norm(p-flange) for p in ends)<1e-9

def test_tile_wipe_needs_contact_and_retains_free_tile_under_bounded_torques():
    cfg=load_config('fr3-duo-tile-wipe');s=make_simulation(cfg);p=make_expert(0,cfg);h=[]
    assert s.model.neq==0 and s.model.joint('tile_free').type==mujoco.mjtJoint.mjJNT_FREE
    assert s.covered.sum()==0
    initial_holder=s.holder.copy()
    while not s.finished:
        a=p.act(s.observe(),h);s.step(a);h.append(a)
        if a['action']=='position_tile':
            assert s.holder[1]-initial_holder[1]>.02
            assert s.covered.sum()==0
    # Geometry/thickness and the estimated tile mass now follow the selected
    # product. This is a contact/regression check, not a cleanup-quality score.
    assert s.covered.mean()>.5
    assert .5<s.peak_force<25 and s.contact_seconds>1
    assert s.max_tile_error<.001 and s.collision_count==0
    assert s.arm.peak_ratio<1
    assert not np.any(s.data.warning.number)
    assert s.success is None and s.observe()['grout_model']=='reduced_surface_transport'
    assert not s.observe()['fracture_model'] and not s.observe()['installation_model']
    assert s.surface.collected_m3>0
    assert s.surface.remaining_m3+s.surface.collected_m3==pytest.approx(s.surface.initial_m3,abs=1e-14)

def test_suction_is_finite_and_switching_it_off_removes_workholding():
    cfg=load_config('fr3-duo-tile-wipe')
    held=make_simulation(cfg);held.data.xfrc_applied[held.tile_id,0]=15;held._advance(.08)
    assert held.max_tile_error<.001
    released=make_simulation(cfg);released.data.ctrl[released.vacuum_id]=0
    released.data.xfrc_applied[released.tile_id,0]=15
    with pytest.raises(RuntimeError,match='Workholding'):released._advance(.08)
    assert released.max_tile_error>.03
    assert released.covered.sum()==0

@pytest.mark.parametrize('kind',['apple','tomato','banana'])
def test_fr3_executes_real_joints_and_retracts_after_food_contact(kind):
    cfg=load_config('fr3-cut-'+kind);s=make_simulation(cfg);p=make_expert(0,cfg);h=[]
    before=s.data.qpos[s.arm.q].copy()
    while not s.finished:
        a=p.act(s.observe(),h);s.step(a);h.append(a)
        assert s.result in {'ok','force_limit','experiment_complete'}
    assert np.linalg.norm(s.data.qpos[s.arm.q]-before)>.005
    assert s.tcp[2]>.29 and s.peak_force>1
    assert s.arm.n==7 and s.model.nflex==4
    assert s.success is None and not np.any(s.data.warning.number)
    assert s.arm.peak_ratio<=1

def test_ur10e_strikes_repeatedly_without_commanded_axial_drive_at_contact():
    cfg=load_config('ur10e-hammer');s=make_simulation(cfg);p=make_expert(0,cfg);h=[];depth=[]
    while s.strikes<3:
        a=p.act(s.observe(),h);s.step(a);h.append(a)
        assert s.result=='ok'
        if a['action']=='strike':
            depth.append(s.depth)
            assert s.motor_at_contact==0 and s.peak_force>50
            assert s.max_lateral_error<.005
            assert s.tcp[2]>s.bowl[2]+.05
    assert .0001<depth[0]<.004 and np.all(np.diff(depth)>0)
    assert s.arm.n==6 and s.arm.peak_ratio<=1
    assert not np.any(s.data.warning.number)
    active=catalog()
    assert not any(x['robot']=='xyz_tool_rig' for x in active)
    assert not cfg['physics']['learning_enabled']
