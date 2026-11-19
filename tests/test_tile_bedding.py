"""Two-arm contact/ledger regressions, not real-world material validation."""
import mujoco
import numpy as np
import pytest
from chora.tile_installation import TileInstallationSimulation,TileBeddingExpert,TILE
from chora.experiments import load_config


def test_missing_adhesive_cannot_pass_bedding_or_full_installation():
    s=TileInstallationSimulation(load_config('fr3-tile-installation'))
    assert s.wet and s.model.neq==0
    assert not any(s.bedding_ok(i) for i in range(4))
    s.mortar.volume[0,0]=s.mortar.area*.008
    s.mortar.reservoir_m3-=s.mortar.volume[0,0]
    s.update_mortar_visuals()
    assert s.data.geom_xpos[s.mortar_ids[0,0],2]==pytest.approx(.244)
    mid=s.mortar_visual.mesh_ids[0];gid=s.mortar_visual.geom_ids[0]
    verts=s.model.mesh_vert[s.model.mesh_vertadr[mid]:s.model.mesh_vertadr[mid]+s.mortar_visual.n**2]
    world=verts@s.data.geom_xmat[gid].reshape(3,3).T+s.data.geom_xpos[gid]
    assert world[:,2].max()==pytest.approx(.248,abs=1e-7)
    s.mortar.volume[0,0]=0;s.mortar.reservoir_m3=s.mortar.initial_m3
    s.update_mortar_visuals()
    assert s.execute(dict(action='dispense',arm='A',material='adhesive',volume_ml=50))=='unsupported_material_or_arm'
    s.step(dict(action='finish_stage'))
    assert s.success is False and s.result=='adhesive_bedding_incomplete'
    assert s.observe()['installation_complete'] is False


def test_both_arms_apply_then_bed_a_free_tile_with_conserved_material():
    s=TileInstallationSimulation(load_config('fr3-tile-installation'))
    p=TileBeddingExpert(0,tile_count=1);history=[]
    initial_q=s.data.qpos[s.arm.q].copy();initial_time=s.data.time
    park=s.data.site('park_tcp').xpos.copy();peak_b_motion=[0.]
    def sample(scene):peak_b_motion[0]=max(peak_b_motion[0],float(np.linalg.norm(scene.data.qpos[scene.arm.q[7:]]-initial_q[7:])))
    s.on_frame=sample
    # One declared subtask in the same four-tile scene; stop before the full-job
    # finish request, which correctly still requires all four placements.
    for _ in range(p.cycle_steps):
        action=p.act(s.observe(),history);s.step(action);history.append(action)
    assert np.linalg.norm(s.data.qpos[s.arm.q[:7]]-initial_q[:7])>.01
    assert peak_b_motion[0]>.1
    np.testing.assert_allclose(s.data.site('park_tcp').xpos,park,atol=.002)
    assert any(a['action']=='parallel' for a in history)
    assert s.concurrent_motion_s>1 and s.overlap_mortaring_s>.5
    assert s.data.time>initial_time and s.mortar.paste.age_s>0
    assert s.model.joint('tile_0_free').type==mujoco.mjtJoint.mjJNT_FREE
    assert s.placement(0)['placed'] and s.bedding_ok(0)
    assert s.mortar.loaded_n[0]==pytest.approx(TILE['mass_kg']*9.81,rel=.02)
    assert s.mortar.squeezed_m3[0]>0 and s.mortar.redistributed_m3>0
    assert abs(s.mortar.balance_error_m3)<1e-12 and s.mortar.pending_m3==0
    assert s.data.ctrl[s.vacuum_id]==0 and s.grip()==[]
    assert s.collision_count==0 and not np.any(s.data.warning.number)
    assert s.arm.peak_ratio<1 and s.observe()['installation_complete'] is False


def test_farthest_tile_approach_and_retract_have_continuous_reachable_poses():
    s=TileInstallationSimulation(load_config('fr3-tile-installation'))
    q=s.arm.ref.copy();previous=s.goal.copy()
    for goal in [np.array([.43,-.22,.42]),np.array([.591925,.181925,.42]),
                 np.array([.591925,.181925,.2554]),np.array([.591925,.181925,.42])]:
        for f in np.linspace(0,1,25)[1:]:
            position=previous+(goal-previous)*f
            candidate=s.arm.ik([('tile_tcp',position,np.diag([1.,-1.,-1.])),
                                ('park_tcp',s.park,np.diag([-1.,1.,-1.]))],q,attempts=1)
            assert np.max(abs(candidate-q))<.35
            q=candidate
        previous=goal
