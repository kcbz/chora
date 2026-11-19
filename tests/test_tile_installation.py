"""Dry-transfer physics regressions; these do not validate adhesive or hardware."""
import mujoco
import numpy as np
from chora.experiments import load_config
from chora.sim import make_simulation
from chora.policy import make_expert

def dry_config():
    cfg=load_config('fr3-tile-installation');cfg['experiment']['task']='tile_dry_transfer'
    return cfg


def test_empty_space_suction_does_not_attach_tiles_or_claim_placement():
    s=make_simulation(dry_config())
    assert s.model.neq==0
    for i in range(4):assert s.model.joint(f'tile_{i}_free').type==mujoco.mjtJoint.mjJNT_FREE
    before=np.array([s.data.xpos[b].copy() for b in s.tiles])
    s.execute(dict(action='vacuum',arm='A',state='on'))
    assert s.grip()==[]
    np.testing.assert_allclose([s.data.xpos[b] for b in s.tiles],before,atol=.0001)
    assert s.execute(dict(action='dispense',arm='B',material='adhesive',volume_ml=50))=='unsupported_in_dry_transfer'
    s.step({'action':'finish_stage'})
    assert s.success is False and s.result=='dry_transfer_incomplete'
    assert s.observe()['installation_complete'] is False


def test_four_tiles_transfer_by_contact_then_rest_released_on_substrate():
    cfg=dry_config();s=make_simulation(cfg);p=make_expert(0,cfg);h=[]
    model_identity=id(s.model);data_identity=id(s.data)
    held=set()
    while not s.finished:
        a=p.act(s.observe(),h);s.step(a);h.append(a)
        if len(h)%8==4:
            i=(len(h)-1)//8
            assert s.grip()==[i]
            assert s.data.xpos[s.tiles[i],2]>.39
            held.add(i)
    assert held=={0,1,2,3} and s.success is True
    assert id(s.model)==model_identity and id(s.data)==data_identity
    assert all(s.placement(i)['placed'] for i in range(4))
    assert s.grip()==[] and s.data.ctrl[s.vacuum_id]==0
    assert s.collision_count==0 and s.arm.peak_ratio<1
    assert not np.any(s.data.warning.number)
    assert s.observe()['installation_complete'] is False
