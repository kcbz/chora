"""Experiment presets, validated scene settings, and isolated output folders.

SO-101 bowl placement and solid-object staging are implemented today. New robots
need an explicit adapter rather than reusing this arm's IK and grasp geometry.
"""
import copy
import math
import re
from pathlib import Path

import yaml

from .paths import WORKSPACE_ROOT as ROOT
DEFAULT="so101-bowl"
FIXED_SCENES=(DEFAULT,"so101-bowl-raised-shelf","so101-bowl-moving-platform","ur10e-hammer","fr3-cut-apple","fr3-cut-tomato","fr3-cut-banana","fr3-tile-installation")
PRESETS=ROOT/"configs/experiments"
TASKS={
    "tile_adhesive_bedding":{"name":"Tile · Adhesive and bedding","object":"tile","instructions":"Arm B meters and combs each adhesive patch while arm A acquires and lifts its tile on the same physics timeline. Local clearance permits placement; the applicator parks after its final patch. A spatial reduced mortar layer supplies contact loads, redistributes ridge volume and tracks squeeze-out. This is an assumed simulation material law, not a resolved fluid or validated hardware model. Setting, grout/cleanup and fracture remain unimplemented."},
    "tile_dry_transfer":{"name":"Tile · Installation foundation","object":"tile","instructions":"Arm A picks four dry tiles from the rack by their exposed faces and releases them into a 2 × 2 pattern. Arm B is parked. Independent FR3 mounts, native contact and finite 40 N adhesion; simplified custom vacuum fixture. This is dry transfer only: no adhesive, bedding, grout application, fracture, or complete installation yet."},
    "dual_tile_wipe":{"name":"Tile · Grout cleanup material trial","object":"tile","instructions":"Two FR3 arms hold and wipe a QuarryBasics 31IS ceramic tile. Geometry is manufacturer-reported; mass is estimated from assumed density. Native contact drives a finite-volume residue model with paste-shear reactions and finite sponge capacity. This is a simulation-only cleanup subtask, not full tile installation or a resolved grout-fluid/fracture solver. Open the material workbench for the 2×2 panel workflow and source/assumption records."},
    "impact_nail":{"name":"Hammer · Impact rig","object":"nail","instructions":"A scanned YCB claw hammer is installed in an XYZ holder. Accelerate in air, coast into the nail with the Z motor off, and retract before realigning. Watch per-hit energy, off-axis error and drive force at contact. Wood law remains uncalibrated."},
    "soft_cut":{"name":"Knife · Deformable food lab","object":"apple","instructions":"A textured chef’s knife is mounted on a force-limited XYZ platform. This material trial uses deformable tetrahedral flesh, a separate elastic skin and local cohesive release along one plane. Observe forces and shape changes; this is not a validated cutting benchmark or juice simulation."},
    "hammer_pine":{"name":"Hammer · Pine board","object":"nail","instructions":"Drive a 2 mm nail 20 mm into a 30 mm pine-model board, using a physically gripped miniature steel hammer. Watch contact force, slip and arm limits. Material law is uncalibrated; incomplete insertion is a valid outcome."},
    "cut_fruit":{"name":"Cut fruit","object":"apple","instructions":"Align the physically gripped steel knife, slice through the fruit on its board, and retract. Skin/flesh resistance and work gate separation on a predefined plane. This reduced-order model is uncalibrated; textures are procedural."},
    "hammer_nail":{"name":"Hammer a nail","object":"nail","instructions":"Align the mounted hammer over the guided nail, strike and retract until driven at least 20 mm, then retreat. This fixture models contact and sliding resistance, not wood fracture."},
    "bowl_pick_place":{"name":"Bowl pick and place","object":"bowl","instructions":"Pick up the bowl, place it inside the green target, release, and retreat."},
    "part_staging":{"name":"Construction part staging","object":"part","instructions":"Move the small construction part onto the marked assembly area, release it flat, and retreat."},
    "ingredient_staging":{"name":"Kitchen ingredient staging","object":"ingredient","instructions":"Transfer the ingredient portion onto the marked prep area, release it, and retreat."},
}
ROBOT_ADAPTERS={"fr3_pair":{"name":"Two independently mounted Franka FR3 arms","tasks":["tile_dry_transfer","tile_adhesive_bedding"]},"so101":{"name":"SO-101","tasks":[t for t in TASKS if t not in {"impact_nail","soft_cut","dual_tile_wipe"}]},"xyz_tool_rig":{"name":"XYZ mounted-tool rig","tasks":["impact_nail","soft_cut"]},"fr3":{"name":"Franka FR3 · 7 axes","tasks":["soft_cut"]},"ur10e":{"name":"UR10e · 6 axes","tasks":["impact_nail"]},"fr3_duo":{"name":"Franka FR3 Duo · 14 axes","tasks":["dual_tile_wipe"]}}


def valid_id(value):
    if not isinstance(value,str) or not re.fullmatch(r"[a-z][a-z0-9-]{1,47}",value):
        raise ValueError("Experiment IDs use 2–48 lowercase letters, numbers, and hyphens")
    return value


def output_dir(experiment=DEFAULT):
    valid_id(experiment)
    return ROOT/"outputs" if experiment==DEFAULT else ROOT/"outputs/experiments"/experiment


def output_url(experiment=DEFAULT):
    valid_id(experiment)
    return "/outputs" if experiment==DEFAULT else "/outputs/experiments/"+experiment


def definition(experiment=DEFAULT):
    valid_id(experiment)
    if experiment==DEFAULT:
        return {"id":DEFAULT,"name":"Bowl · Standard platform","description":"The original trained experiment and its saved benchmark.","robot":"so101","task":"bowl_pick_place","simulation":{}}
    file=PRESETS/(experiment+".yaml")
    if not file.is_file():
        raise ValueError("Unknown experiment")
    value=yaml.safe_load(file.read_text())
    if not isinstance(value,dict) or value.get("id")!=experiment:
        raise ValueError("Invalid experiment definition")
    if value.get("robot") not in ROBOT_ADAPTERS:
        raise ValueError("This robot needs a robot adapter before it can run")
    if value.get("task") not in ROBOT_ADAPTERS[value["robot"]]["tasks"]:
        raise ValueError("This task needs a scenario adapter before it can run")
    return value


def validate_scene(scene):
    def number(value,lo,hi,label):
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or not lo<=value<=hi:
            raise ValueError(f"{label} must be between {lo*1000:g} and {hi*1000:g} mm")
    for key,lo,hi in (("spawn_x",.13,.22),("spawn_y",-.12,-.035)):
        values=scene[key]
        if not isinstance(values,list) or len(values)!=2:
            raise ValueError("Spawn bounds require two coordinates")
        for value in values:number(value,lo,hi,key)
        if values[0]>=values[1]:raise ValueError("Minimum spawn coordinate must be below maximum")
    number(scene["table_height"],.03,.10,"Table height")
    if len(scene["target"])!=3 or len(scene["target_half_size"])!=2:
        raise ValueError("Invalid target dimensions")
    number(scene["target"][0],.13,.21,"Target x")
    number(scene["target"][1],.065,.135,"Target y")
    if scene["target"][2]!=scene["table_height"]:
        raise ValueError("Target height must match the table height")
    for value in scene["target_half_size"]:number(value,.032,.055,"Target half-size")
    centre=scene.get("table_center",[.175,.09])
    if not isinstance(centre,list) or len(centre)!=2:raise ValueError("Table placement requires x and y coordinates")
    number(centre[0],.14,.205,"Table x")
    number(centre[1],.075,.135,"Table y")
    if any(abs(scene["target"][i]-centre[i])+scene["target_half_size"][i]>[.083,.074][i] for i in range(2)):
        raise ValueError("The target region must fit on the table")
    if scene.get("table_style") not in (None, "shelf"):
        raise ValueError("Unknown table style")
    if "platform_motion" in scene:
        motion = scene["platform_motion"]
        if not isinstance(motion, dict) or set(motion) != {"travel", "peak_speed"}:
            raise ValueError("Platform motion requires travel and peak speed limits")
        number(motion["travel"], .01, .04, "Platform half-travel")
        speeds = motion["peak_speed"]
        if not isinstance(speeds, list) or len(speeds) != 2:
            raise ValueError("Platform peak speed requires two values")
        for speed in speeds:
            number(speed, .005, .04, "Platform peak speed")
        if speeds[0] > speeds[1]:
            raise ValueError("Platform speed range is reversed")
        if centre[1] - motion["travel"] - .074 <= scene["spawn_y"][1] + .029:
            raise ValueError("Platform travel overlaps the pickup area")


def load_config(experiment=DEFAULT):
    value=definition(experiment)
    cfg=yaml.safe_load((ROOT/"configs/default.yaml").read_text())
    if value['robot']=='fr3_pair':
        from .tiling_materials import tile_properties
        cfg['simulation'].update(timestep=.0005,max_steps=80,object_kind='tile',table_height=.24,target=[.43,.02,.24935],target_half_size=[.0762,.0762])
        cfg['physics']=dict(calibrated=False,benchmark_validated=False,learning_enabled=False,
          learning_readiness='Adhesive/bedding milestone only. Expert generalization checks, stage integration and the task-specific Qwen data/student pipeline remain to be implemented. Physical calibration is not a gate.',
          model='Native free rigid tiles, torque-bounded independently mounted FR3s, finite face adhesion and a conservative spatial mortar foundation with assumed HB shear and plastic squeeze response. No grout application, cured bonding, fracture or full installation.',
          controller='Native articulated dynamics; bounded joint torques and Cartesian trajectory references',
          robot_source='MuJoCo Menagerie revision 370c43965323c0944b433b202aac5058b5f907a9',
          tool_grip='Simplified installed face vacuum and raised-feed metering/comb head; no vendor CAD or measured hardware model',
          appearance='CC0 Poly Haven terracotta appearance analogue, visual bevel/microrelief; cement surface reference on a continuously updated mortar height mesh. Neither surface is a scan of the selected product or validated wet mortar.',
          mount_status='custom_unverified',tile=tile_properties(),simulation_only=True,
          material_workbench_url='/materials/tiling',material_parameters=yaml.safe_load((ROOT/'configs/materials/mortar_layer.yaml').read_text()))
        cfg['experiment']={k:value[k] for k in ('id','name','description','robot','task')}
        return cfg
    if value['robot'] in {'fr3','ur10e','fr3_duo'}:
        robot=value['robot'];kind=value.get('fruit','tile' if robot=='fr3_duo' else 'nail')
        cfg['simulation'].update(timestep=.0002 if robot=='ur10e' else .0005,max_steps=30,table_height=.12,table_center=[.5,0],target=[.5,0,.15],target_half_size=[.14,.11],spawn_x=[.485,.515],spawn_y=[-.015,.015],object_kind=kind)
        cfg['physics']={'calibrated':False,'benchmark_validated':False,'learning_enabled':False,'robot_source':'MuJoCo Menagerie revision 370c43965323c0944b433b202aac5058b5f907a9','controller':'Native articulated dynamics; payload-aware bounded joint torques, Cartesian IK trajectory references','tool_grip':'CUSTOM simulated fixture, not manufacturer hardware; unverified dimensions/stiffness; no tool pickup or frictional grasp claim','mount_status':'custom_unverified'}
        if robot=='fr3_duo':
            from .tiling_materials import tile_properties,catalog as material_catalog
            tile=tile_properties()
            cfg['simulation'].update(target=[.5,.025,.388],table_height=0.,target_half_size=[x/2 for x in tile['dimensions_m'][:2]])
            cfg['physics'].update(model='Finite 40 N contact adhesion; native sponge contact plus equal/opposite assumed paste shear; finite-volume grout residue and sponge collection. No full installation, resolved free-surface flow or tile fracture.',tile=tile,vacuum={'force_limit_n':40,'provenance':'assumed effective pressure-area force, not a calibrated cup'},sponge={'friction':.45,'contact_time_constant_s':.018,'provenance':'assumed contact response, not calibrated foam'},model_parameters='configs/materials/tiling_baseline.yaml',material_workbench_url='/materials/tiling',simulation_only=True)
            cfg['physics']['material_record']=yaml.safe_load((ROOT/'configs/materials/tile_wipe.yaml').read_text())
            cfg['physics']['product_records']=material_catalog()
            cfg['physics']['learning_readiness']='Simulation learning is allowed in principle. This cleanup subtask currently has no scored expert benchmark or task-specific student action protocol; full installation is still unimplemented. Physical calibration is not the gate.'
        elif robot=='fr3':
            from .soft_food import FOOD
            food=FOOD[kind].copy()
            if kind in {'apple','banana'}:
                from .scanned_produce import dimensions
                food.update(dimensions(kind,food['height']))
            cfg['physics'].update(model='Coarse tetrahedral FEM envelope + skin + predefined cohesive plane; illustrative food parameters, no juice solver',food=food,tool_asset='Imported RoboCasa knife_22 chef knife; short CUSTOM unverified adapter',force_guard_n=55,appearance='CC0 Poly Haven textured surface for apple/banana, uniformly scaled; coarse FEM envelope and procedural interiors. Tomato appearance remains procedural.',knife_edge={'effective_full_edge_thickness_m':.00003,'blade_body_thickness_m':.0012,'bevel_height_m':.015,'status':'assumed collision wedge, not measured sharpness'})
        else:cfg['physics'].update(model='Guided 2 mm nail with insertion resistance 120 + 3000*depth N; no wood fracture; nominal 0.35 J task command. Commanded downward task force removed before contact; joint gravity/orientation control stays enabled.',tool_asset='YCB scanned claw hammer',base_offset_m=-.25,impact_qualification=False)
        cfg['experiment']={k:value[k] for k in ('id','name','description','robot','task')}
        return cfg
    if value["robot"]=="xyz_tool_rig":
        kind=value.get("fruit","nail");impact=value["task"]=="impact_nail"
        cfg["simulation"].update(timestep=.0002 if impact else .0005,max_steps=30 if impact else 8,table_height=.12,table_center=[0,.1],target=[0,.1,.15],target_half_size=[.14,.11],spawn_x=[-.015,.015],spawn_y=[.085,.115],object_kind=kind)
        cfg["physics"]={"calibrated":False,"benchmark_validated":False,"learning_enabled":False,"platform":"Custom XYZ rig, force-limited 100 N axes; illustrative hardware, not a certified industrial platform","tool_grip":"Installed tool in visible bolted holder; no grasp or pickup claim"}
        if impact:cfg["physics"].update(model="Motor-off kinetic impact; stiff dry-friction nail insertion, 120 + 3000*depth N; guided nail, no wood splitting",tool_asset="YCB scanned full claw hammer",moving_z_mass_kg=.7,impact_energy_j=.35,friction_time_constant_s=.0004)
        else:
            from .soft_food import FOOD
            cfg["physics"].update(model="Tetrahedral bulk FEM, separate elastic skin, contact-work cohesive front on a predefined plane; no liquid/juice solver",food=FOOD[kind],tool_asset="RoboCasa knife_22 chef knife via Robobench",appearance="Continuous intact skin before damage; procedural food textures bound to deformable nodes",learning_enabled=False)
        cfg["experiment"]={k:value[k] for k in ("id","name","description","robot","task")}
        return cfg
    overrides=value.get("simulation",{})
    allowed={"spawn_x","spawn_y","target","target_half_size","table_height","table_center","table_style","platform_motion"}
    if set(overrides)-allowed:
        raise ValueError("Preset contains unsupported scene settings")
    cfg["simulation"].update(copy.deepcopy(overrides))
    if value["task"]!="bowl_pick_place":
        cfg["simulation"]["object_kind"]=TASKS[value["task"]]["object"]
    if experiment!=DEFAULT:validate_scene(cfg["simulation"])
    if value["task"] in {"hammer_nail","hammer_pine","cut_fruit"}:
        centre=cfg["simulation"].get("table_center",[.175,.09])
        cfg["simulation"].update(timestep=.002,max_steps=20,spawn_x=[centre[0]-.018,centre[0]+.018],spawn_y=[centre[1]-.016,centre[1]+.016])
    if value["task"] in {"hammer_pine","cut_fruit"}:
        cfg["simulation"].update(timestep=.001,max_steps=20 if value["task"]=="hammer_pine" else 12)
        if value["task"]=="cut_fruit":
            cfg["simulation"].update(object_kind=value["fruit"],spawn_x=[centre[0]-.003,centre[0]+.003],spawn_y=[centre[1]-.003,centre[1]+.003])
            cfg["data"].update(min_episodes=80,max_per_bin=15)
        from .materials import PINE,STEEL,FRUIT
        cfg["physics"]={"calibrated":False,"tool_grip":"free body, frictional finger contact; prepared grasp, no pickup", "steel":STEEL,"grip_pads":{"material":"rubber-like compliant pads","friction_coefficient":.8,"calibrated":False},"handle":"flat-sided hardwood, 7 g"}
        if value["task"]=="hammer_pine":cfg["physics"].update(wood=PINE,nail_diameter_m=.002,model="tip bearing + embedded shank shear; guided nail, no wood splitting or steel yield")
        else:cfg["physics"].update(food=FRUIT[value["fruit"]],model="layered resistance + work-gated predefined-plane separation; rigid halves, no FEM deformation",appearance="procedural skin and interior maps, not photo scans")
    cfg["experiment"]={k:value[k] for k in ("id","name","description","robot","task")}
    return cfg


def public_definition(experiment=DEFAULT):
    cfg=load_config(experiment)
    task=TASKS[cfg["experiment"]["task"]]
    meta=definition(experiment)
    instructions=task['instructions']
    if cfg["simulation"].get("platform_motion"):
        instructions = "Pick up the bowl, intercept the moving platform, release it inside the green outline, and retreat. The bowl must remain upright and stable relative to the platform."
    if cfg['experiment']['robot']=='fr3':instructions='A native 7-axis FR3 carries the imported chef knife in a short installed adapter. Inspect real articulated motion, deformable food contact and the 55 N cutting-force stop. Food/skin parameters are uncalibrated; there is no juice solver or validated cutting score.'
    elif cfg['experiment']['robot']=='ur10e':instructions='A native 6-axis UR10e carries the scanned claw hammer. Align, accelerate in air, remove commanded downward task force before contact, and retract. Joint gravity/orientation control remains enabled. Guided-nail resistance is uncalibrated; this is not hardware impact qualification.'
    missing=[]
    if cfg['experiment']['task']=='soft_cut':
        for file in ['knife.obj','basecolor.png']:
            if not (ROOT/'assets/kitchen_knife'/file).is_file():missing.append('local-only knife asset (see docs/ASSETS.md)')
    availability={'available':not missing,'unavailable_reason':'; '.join(sorted(set(missing)))}
    return {**availability,**cfg["experiment"],"simulation":cfg["simulation"],"physics":cfg.get("physics"),"output_url":output_url(experiment),"instructions":instructions,"task_name":task["name"],"robot_name":ROBOT_ADAPTERS[cfg["experiment"]["robot"]]["name"],"category":meta.get("category","experiment"),"hidden":bool(meta.get("hidden",False))}


def catalog():
    names=[name for name in FIXED_SCENES if name==DEFAULT or (PRESETS/(name+".yaml")).exists()]
    return [public_definition(name) for name in names]


def create_experiment(payload):
    raise ValueError("Choose one of the fixed scenes. Creating scene copies is no longer supported.")
