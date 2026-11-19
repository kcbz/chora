import numpy as np
import pytest
from chora.episodes import rollout
from chora.experiments import load_config
from chora.gallery import verify_trace
from chora.materials import nail_resistance,cutting_resistance,FRUIT
from chora.policy import make_expert,messages
from chora.sim import make_simulation
from chora.cutting import validate_cut_action


def test_physical_hammer_is_free_and_opening_jaw_releases_it():
    cfg=load_config('so101-hammer-pine');sim=make_simulation(cfg)
    assert sim.model.neq==0
    assert sim.model.body('free_hammer').jntnum==1
    assert sim.tool_held and all(f>.1 for f in sim.finger_forces.values())
    sim._advance(.5)
    assert sim.depth==pytest.approx(0,abs=1e-8)
    before=sim.tool_tip.copy();sim.data.ctrl[5]=.6;sim._advance(.5)
    assert not sim.tool_held
    assert np.linalg.norm(sim.tool_tip-before)>.05
    assert not sim.success


def test_material_law_and_physics_provenance_are_explicit():
    assert nail_resistance(.02)>nail_resistance(0)>0
    for kind in FRUIT:
        assert cutting_resistance(kind,.00001)>cutting_resistance(kind,FRUIT[kind]['height']/2)>0
        assert not load_config('so101-cut-'+kind)['physics']['calibrated']
    assert not load_config('so101-hammer-pine')['physics']['calibrated']
    assert 'physics' not in load_config()


@pytest.mark.parametrize('kind',['apple','tomato','banana'])
def test_cut_requires_alignment_and_work_then_releases_physical_halves(kind):
    cfg=load_config('so101-cut-'+kind);sim=make_simulation(cfg)
    sim._advance(.5)
    assert not sim.cut_complete and sim.cut_work==0
    assert sim.data.eq_active[sim.cohesion]
    # A slice performed away from the food cannot accumulate work or sever it.
    sim.step({'action':'slice','depth':.02,'duration':.8})
    assert not sim.cut_complete and sim.cut_work==0
    episode=rollout(sim,make_expert(0,cfg),0)
    assert episode['success']
    assert episode['final_observation']['cut_work_j']>0
    if sim.cut_complete:
        assert not sim.data.eq_active[sim.cohesion]
        assert sim.cut_work>=sim.food['fracture_work_j']
    assert 'placement_margin' not in episode
    verify_trace(sim,episode)
    prompt=messages(sim.observe(),[],{'action':'slice','depth':.02,'duration':1.4})
    assert '"depth":20' in prompt[-1]['content']
    assert 'cut the observed fruit' in prompt[0]['content']
    # Both skin and interior maps must occupy the renderer's RGB texture role.
    for name in [kind+'_skin',kind+'_inside']:
        assert sim.model.mat_texid[sim.model.material(name).id,1]>=0


@pytest.mark.parametrize('action',[
 {'action':'slice','depth':30,'duration':1.4},
 {'action':'slice','depth':.03,'duration':float('nan')},
 {'action':'slice','depth':True,'duration':1.4},
 {'action':[]},
])
def test_cut_action_contract(action):
    with pytest.raises(ValueError):validate_cut_action(action)
