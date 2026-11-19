"""Reproduce the separate arm study; does not mutate application scenes."""
import argparse,json,os,subprocess,sys
from copy import deepcopy
from chora.arm_evaluation import MODELS,TASKS,OUT,ROOT,single,dual

p=argparse.ArgumentParser();p.add_argument('--render',action='store_true');args=p.parse_args()
for key in MODELS:single(key)
dual()
single('ur10e',-.25,'ur10e_base_back_250mm')
single('iiwa14',-.15,'iiwa14_base_back_150mm')
original=deepcopy(TASKS['cutting'])
try:
    TASKS['cutting'].update(tcp=[.08,0,.12],com=[.03,0,.08])
    result=single('fr3',output_key='fr3_short_knife_mount');result['task_assumptions']=deepcopy(TASKS)
    (OUT/'fr3_short_knife_mount.json').write_text(json.dumps(result,indent=2)+'\n')
finally:TASKS['cutting']=original
# Main module writes the same scope/assumptions when run independently.
method=dict(revision='370c43965323c0944b433b202aac5058b5f907a9',tasks=TASKS,scope='IK endpoints, collision and quasi-static load screen; not full tasks or hardware validation.',acceptance=dict(position_m=.002,rotation_deg=2,min_joint_margin_rad=.025),tool_geometry='Bare flanges, point TCP/CoM; no grippers/tool collisions or contact materials.',static_limits='Simulator actuator maxima, not continuous or impact ratings; unbounded iiwa capacity unscored.')
(OUT/'method.json').write_text(json.dumps(method,indent=2)+'\n')
env={**os.environ,'PYTHONPATH':str(ROOT)}
subprocess.run([sys.executable,str(ROOT/'scripts/robot_arm_hold_screen.py')],cwd=ROOT,env=env,check=True)
subprocess.run([sys.executable,str(ROOT/'scripts/build_arm_evaluation_report.py'),*(['--render'] if args.render else [])],cwd=ROOT,env=env,check=True)
