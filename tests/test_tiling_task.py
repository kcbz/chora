"""Task-interface tests only. This fixture is not a robotics simulator."""
import pytest
from chora.tiling_task import Episode,Stage,STAGES,plan,validate_action,messages

class SceneFixture:
    def __init__(self,fixture=None):self.state={'material_age_s':0,'tile_error_mm':3,'initial_fixture':fixture};self.pass_checks=False;self.commands=[]
    def observe(self):return self.state.copy()
    def execute(self,action):
        self.commands.append(action)
        if action['action']=='wait':self.state['material_age_s']+=action['seconds']
        return 'ok'
    def checks(self,stage):return {name:self.pass_checks for name in STAGES[stage.name]}

def test_full_job_repeats_tile_operations_and_keeps_global_stages_once():
    p=plan();assert len(p)==20
    assert p[:2]==(Stage('inspect_plan'),Stage('prepare_materials'))
    assert [s.tile for s in p if s.name=='place_bed_tile']==[0,1,2,3]
    assert p[-2:]==(Stage('wait_for_set'),Stage('grout_finish'))

def test_stage_completion_requires_environment_evidence_and_keeps_same_scene():
    scene=SceneFixture();episode=Episode(scene);identity=id(episode.scene)
    episode.step({'action':'wait','seconds':12})
    episode.step({'action':'finish_stage'});assert episode.index==0 and 'incomplete' in episode.result
    scene.pass_checks=True
    episode.step({'action':'finish_stage'})
    assert episode.index==1 and id(episode.scene)==identity
    assert episode.observe()['material_age_s']==12 and episode.observe()['tile_error_mm']==3
    while not episode.finished:episode.step({'action':'finish_stage'})
    assert episode.result=='sequence_complete' and episode.mode=='full'
    with pytest.raises(RuntimeError):episode.step({'action':'finish_stage'})

def test_curricula_need_explicit_initial_fixture_and_record_their_mode():
    with pytest.raises(ValueError,match='fixture'):Episode(SceneFixture(),mode='stage',stages=['place_bed_tile'])
    e=Episode(SceneFixture('prepared_bed_v1'),mode='sequence',stages=['pick_prepare_tile','place_bed_tile'],tile=2)
    assert e.observe()['initial_fixture']=='prepared_bed_v1'
    assert e.observe()['run_mode']=='sequence' and e.observe()['tile_index']==2
    with pytest.raises(ValueError):plan(mode='stage',stages=['place_bed_tile','wait_for_set'])

@pytest.mark.parametrize('action',[
 {'action':'declare_success'}, {'action':'finish_stage','success':True},
 {'action':'wait','seconds':float('nan')}, {'action':'wait','seconds':True},
 {'action':'vacuum','arm':'C','state':'on'},
 {'action':'dispense','arm':'B','material':'grout','volume_ml':-1},
 {'action':'request_tool','arm':'B','tool':'magic_tool'},
])
def test_policy_cannot_bypass_checks_or_send_unbounded_invalid_actions(action):
    with pytest.raises(ValueError):validate_action(action)

def test_teacher_and_student_receive_same_task_protocol_without_bowl_instructions():
    e=Episode(SceneFixture());a={'action':'inspect','target':'substrate'}
    m=messages(e.observe(),[],a)
    assert 'two-arm tile-installation' in m[0]['content']
    assert 'SO-101' not in m[0]['content'] and 'tile_installation' in m[1]['content']
    assert m[-1]['role']=='assistant'


def test_parallel_schema_rejects_duplicate_arms_and_nesting():
    a={'action':'vacuum','arm':'A','state':'on'}
    b={'action':'dispense','arm':'B','material':'adhesive','volume_ml':94}
    assert validate_action({'action':'parallel','actions':[a,b]})['action']=='parallel'
    for commands in [[a,a],[a], [{'action':'parallel','actions':[a,b]},b]]:
        with pytest.raises(ValueError):validate_action({'action':'parallel','actions':commands})


def test_parallel_prompt_describes_shared_physics_timeline():
    assert 'parallel' in messages({'task':'tile_installation'},[])[0]['content']
