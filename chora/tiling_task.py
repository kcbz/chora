"""One tiling job, stage curricula and a shared expert/student task interface.

This is scheduling/interface infrastructure, not a tile physics implementation.
Checks are supplied by the physics adapter; a policy cannot declare success.
The same scene instance survives every transition in a chained episode.
"""
from dataclasses import dataclass
from typing import Protocol
import math

PROTOCOL='chora-tile-installation-v1'
STAGES={
 'inspect_plan':('substrate_inspected','pattern_registered'),
 'prepare_materials':('adhesive_batch_ready','required_tools_ready'),
 'apply_adhesive':('adhesive_in_target_region','adhesive_workable'),
 'pick_prepare_tile':('correct_tile_acquired','grip_verified','tile_back_ready'),
 'place_bed_tile':('tile_at_target','bedding_acceptable','load_limits_respected'),
 'inspect_clean_placement':('alignment_acceptable','joint_spacing_acceptable','placement_stable','joints_clear'),
 'wait_for_set':('required_setting_interval_elapsed',),
 'grout_finish':('joint_fill_acceptable','residue_acceptable','tools_clear'),
}
PER_TILE=('apply_adhesive','pick_prepare_tile','place_bed_tile','inspect_clean_placement')

@dataclass(frozen=True)
class Stage:
    name:str
    tile:int|None=None

def plan(mode='full',stages=None,tile=0,tile_count=4):
    if isinstance(tile_count,bool) or not isinstance(tile_count,int) or not 1<=tile_count<=100:raise ValueError('Invalid tile count')
    if isinstance(tile,bool) or not isinstance(tile,int) or not 0<=tile<tile_count:raise ValueError('Invalid tile index')
    if mode=='full':
        if stages is not None:raise ValueError('Full mode uses the complete job plan')
        return (Stage('inspect_plan'),Stage('prepare_materials'),
          *(Stage(name,i) for i in range(tile_count) for name in PER_TILE),
          Stage('wait_for_set'),Stage('grout_finish'))
    if mode not in {'stage','sequence'}:raise ValueError('Unknown run mode')
    if not stages or any(name not in STAGES for name in stages):raise ValueError('Choose known stages')
    if mode=='stage' and len(stages)!=1:raise ValueError('Stage mode requires one stage')
    return tuple(Stage(name,tile if name in PER_TILE else None) for name in stages)

def validate_action(action):
    """Bounded primitives shared by scripted expert and Qwen.

    Numeric limits are software action bounds, not material damage guarantees.
    A tool change must be executed/verified by a future physical tool adapter.
    """
    if not isinstance(action,dict):raise ValueError('Action must be an object')
    if action.get('action')=='parallel':
        if set(action)!={'action','actions'} or not isinstance(action['actions'],list) or len(action['actions'])!=2:raise ValueError('Parallel requires exactly two arm actions')
        commands=action['actions']
        if any(not isinstance(c,dict) or c.get('action') not in {'move_to','vacuum','dispense'} for c in commands):raise ValueError('Unsupported parallel primitive')
        for command in commands:validate_action(command)
        if {c['arm'] for c in commands}!={'A','B'}:raise ValueError('Parallel commands require different arms')
        return action
    name=action.get('action')
    shapes={
      'move_to':{'action','arm','position_mm','rpy_deg','speed_mm_s','force_limit_n'},
      'vacuum':{'action','arm','state'},
      'dispense':{'action','arm','material','volume_ml'},
      'request_tool':{'action','arm','tool'},
      'wait':{'action','seconds'},
      'inspect':{'action','target'},
      'finish_stage':{'action'},
    }
    if name not in shapes or set(action)!=shapes[name]:raise ValueError('Unknown action or fields')
    def number(v,lo,hi):return not isinstance(v,bool) and isinstance(v,(int,float)) and math.isfinite(v) and lo<=v<=hi
    if 'arm' in action and action['arm'] not in {'A','B'}:raise ValueError('Unknown arm')
    if name=='move_to':
        for key,bounds in [('position_mm',(-3000,3000)),('rpy_deg',(-180,180))]:
            v=action[key]
            if not isinstance(v,list) or len(v)!=3 or not all(number(x,*bounds) for x in v):raise ValueError('Invalid pose')
        if not number(action['speed_mm_s'],1,250) or not number(action['force_limit_n'],1,80):raise ValueError('Invalid motion bounds')
    if name=='vacuum' and action['state'] not in {'on','off'}:raise ValueError('Invalid vacuum state')
    if name=='dispense' and (action['material'] not in {'adhesive','grout'} or not number(action['volume_ml'],0,250)):raise ValueError('Invalid material dose')
    if name=='request_tool' and action['tool'] not in {'vacuum_gripper','notched_trowel','grout_float','sponge'}:raise ValueError('Unknown tool')
    if name=='wait' and not number(action['seconds'],0,172800):raise ValueError('Invalid wait')
    if name=='inspect' and action['target'] not in {'substrate','tile','placement','adhesive','joints','surface'}:raise ValueError('Unknown inspection target')
    return action

class Scene(Protocol):
    def observe(self)->dict: ...
    def execute(self,action:dict)->str: ...
    def checks(self,stage:Stage)->dict[str,bool]: ...

class Episode:
    def __init__(self,scene:Scene,mode='full',stages=None,tile=0,tile_count=4):
        self.scene=scene;self.mode=mode;self.stages=plan(mode,stages,tile,tile_count);self.index=0
        self.history=[];self.result='ready';self.finished=False
        # Scene initialization happens outside this scheduler, exactly once.
        # Stage-only mode requires a physics-backed fixture with provenance.
        initial=scene.observe()
        if mode!='full' and not initial.get('initial_fixture'):
            raise ValueError('Stage/sequence mode requires an identified initial-state fixture')
    @property
    def stage(self):return None if self.finished else self.stages[self.index]
    def observe(self):
        state=dict(self.scene.observe())
        state.update(task='tile_installation',protocol=PROTOCOL,run_mode=self.mode,
          stage=self.stage.name if self.stage else None,tile_index=self.stage.tile if self.stage else None,
          stage_index=self.index,total_stages=len(self.stages),finished=self.finished,last_result=self.result)
        return state
    def step(self,action):
        if self.finished:raise RuntimeError('Episode has ended')
        validate_action(action);before=self.observe();stage=self.stage
        if action['action']=='finish_stage':
            evidence=self.scene.checks(stage)
            missing=[name for name in STAGES[stage.name] if evidence.get(name) is not True]
            if missing:self.result='stage_incomplete:'+','.join(missing)
            else:
                self.index+=1;self.finished=self.index==len(self.stages)
                self.result='sequence_complete' if self.finished else 'stage_complete'
        else:self.result=self.scene.execute(action)
        self.history.append(dict(observation=before,action=dict(action),result=self.result))
        return self.observe()

SYSTEM='''Control the two-arm tile-installation scene using the supplied observations.
Coordinates are integer millimetres in one persistent world frame, orientations
are roll/pitch/yaw degrees, speeds mm/s, force limits N, material volumes mL and
time seconds. A operates the tile gripper; B operates application/finishing tools.
Return one JSON object. Available primitives:
{"action":"parallel","actions":[one A move_to/vacuum/dispense primitive,one B move_to/vacuum/dispense primitive]}
Parallel executes both commands on one physics timeline, with workspace checks.
{"action":"move_to","arm":"A|B","position_mm":[x,y,z],"rpy_deg":[r,p,y],"speed_mm_s":number,"force_limit_n":number}
{"action":"vacuum","arm":"A|B","state":"on|off"}
{"action":"dispense","arm":"A|B","material":"adhesive|grout","volume_ml":number}
{"action":"request_tool","arm":"A|B","tool":"vacuum_gripper|notched_trowel|grout_float|sponge"}
{"action":"wait","seconds":number}
{"action":"inspect","target":"substrate|tile|placement|adhesive|joints|surface"}
{"action":"finish_stage"}
The environment enforces pose, workspace, tool, material and load constraints.
A tool request must run a supported tool-change process; it is not a free swap.
finish_stage requests environment checks and cannot declare success. Respond to
failed grips, unstable placement and other failures using current observations.
Never assume hidden coverage, fracture or material state has been measured.'''

def messages(observation,history,action=None):
    import json
    result=[dict(role='system',content=SYSTEM),dict(role='user',content=json.dumps(dict(observation=observation,history=history[-4:]),allow_nan=False))]
    if action is not None:validate_action(action);result.append(dict(role='assistant',content=json.dumps(action,allow_nan=False)))
    return result
