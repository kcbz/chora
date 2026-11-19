import numpy as np
import pytest
from chora.experiments import load_config
from chora.sim import make_simulation
from chora.gantry import ImpactExpert
from chora.tool_assets import geometry


def test_imported_tools_have_full_size_meshes_and_texture_coordinates():
    for kind in ['hammer','knife']:
        v,f,uv,fu=geometry(kind)
        assert len(v)>1000 and len(f)>1000
        assert np.ptp(v[:,0])>.25
        assert fu.shape==f.shape
        assert np.all((uv>=0)&(uv<=1))


def test_impact_is_aligned_motor_off_and_energy_limited():
    sim=make_simulation(load_config('gantry-hammer-impact'));policy=ImpactExpert(0);history=[]
    # A misaligned strike must not be executed.
    sim.move([.06,.1,.25]);sim.step({'action':'strike','energy':.35})
    assert sim.result=='rejected_misaligned' and sim.depth==pytest.approx(0,abs=1e-8)
    sim.reset(0);energies=0.;depths=[]
    while not sim.finished:
        action=policy.act(sim.observe(),history);sim.step(action);history.append(action)
        if action['action']=='strike':
            energies+=sim.impact_energy;depths.append(sim.depth)
            assert sim.motor_at_contact==0
            assert sim.max_lateral_error<.0001
            assert 120*sim.depth+1500*sim.depth**2 <= energies*1.05
    assert sim.success and len(depths)>3
    assert depths[0]<.004
    assert np.all(np.diff(depths)>=0)
    assert not np.any(sim.data.warning.number)


@pytest.mark.parametrize('kind',['apple','tomato','banana'])
def test_food_has_live_volume_and_skin_elasticity_no_initial_cut(kind):
    sim=make_simulation(load_config('gantry-cut-'+kind))
    assert sim.model.nflex==4 and sim.model.nv>400
    assert np.any(sim.model.flex_stiffness>0)
    assert sim.model.skin_rgba[sim.model.skin('intact_skin').id,3]==1
    assert len(sim.released)==0 and sim.work==0
    assert not np.any(sim.data.warning.number)
    rest=sim.data.xpos[sim.node_ids].copy()
    sim.move([0,.1,.15+sim.food['height']-.005],.4)
    assert np.linalg.norm(sim.data.xpos[sim.node_ids]-rest)>.0001
    assert sim.peak_force>0
    assert not np.any(sim.data.warning.number)
    assert not sim.observe()['benchmark_validated']
