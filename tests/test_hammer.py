import pytest

from chora.episodes import rollout
from chora.experiments import load_config, output_dir
from chora.gallery import verify_trace
from chora.hammer import validate_hammer_action
from chora.policy import make_expert, messages
from chora.sim import make_simulation
from chora.ui import WEB, page, render_shell


def test_hammer_contact_drives_nail_and_trace_replays():
    cfg=load_config('so101-hammer-nail');sim=make_simulation(cfg)
    initial=sim.observe()
    assert initial['task']=='hammer_nail' and 'bowl' not in initial
    sim._advance(1)
    assert sim.depth==pytest.approx(0,abs=1e-9)
    sim.step({'action':'strike','depth':.006,'duration':.10})
    assert sim.depth==pytest.approx(0,abs=1e-9)
    assert sim.peak_impact==0
    assert not sim.success
    for seed in (0,3,17):
        episode=rollout(sim,make_expert(seed,cfg),seed)
        assert episode['success'] and episode['nail_depth']>=.020
        assert 'placement_margin' not in episode
        assert episode['final_observation']['last_peak_force']>0
        verify_trace(sim,episode)
    prompt=messages(initial,[],{'action':'strike','depth':.006,'duration':.1})
    assert 'guided nail' in prompt[0]['content']
    assert '"depth":6' in prompt[-1]['content']
    assert output_dir('so101-hammer-nail')!=output_dir()


@pytest.mark.parametrize('action',[
    {'action':'strike','depth':6,'duration':.1},
    {'action':'strike','depth':.006,'duration':float('nan')},
    {'action':'strike','depth':True,'duration':.1},
    {'action':'retract','x':0},
    {'action':[]},
    [],
])
def test_hammer_action_contract(action):
    with pytest.raises(ValueError):validate_hammer_action(action)


def test_all_pages_refresh_from_one_logo_and_navbar():
    html=page('Test','<main>Saved content</main>','gallery')
    stale=html.replace('by coherence','obsolete brand')
    refreshed=render_shell(stale,'gallery','so101-hammer-nail')
    assert 'obsolete brand' not in refreshed
    assert (WEB/'logo-octahedron.svg').read_text() in refreshed
    assert refreshed.count('class="app-header"')==1
    assert refreshed.count('id="appearance-settings"')==1
    assert 'Saved content' in refreshed
    assert '/outputs/experiments/so101-hammer-nail/eval_gallery.html' in refreshed
    assert 'src="/static/appearance-ui.js"' in refreshed
    assert '{{' not in refreshed
