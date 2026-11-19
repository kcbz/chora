"""Build an evidence-backed local report; optional native model renders."""
import argparse,json
from html import escape
from pathlib import Path
import numpy as np,mujoco
from PIL import Image
from chora.arm_evaluation import OUT,ROOT,MODELS,TASKS,build_spec,add_bench
from chora.ui import page

SOURCES=[
 ('Menagerie pinned revision','https://github.com/google-deepmind/mujoco_menagerie/tree/370c43965323c0944b433b202aac5058b5f907a9'),
 ('Franka FR3 datasheet','https://franka.de/hubfs/Datasheet_FrankaResearch3_R02212_2.4.pdf?hsLang=en'),
 ('Franka joint/controller limits','https://frankarobotics.github.io/docs/robot_specifications.html'),
 ('Franka models and simulation','https://franka.de/models-and-simulation'),
 ('UR5e 2024 datasheet','https://www.universal-robots.com/media/1807465/ur5e_e-series_datasheets_web.pdf'),
 ('UR10e 2024 datasheet','https://www.universal-robots.com/media/1807466/ur10e_e-series_datasheets_web.pdf'),
 ('Current UR model-name changes','https://www.universal-robots.com/insights/faq/'),
 ('KUKA iiwa 14 datasheet','https://www.kuka.com/-/media/kuka-downloads/imported/8350ff3ca11642998dbdc81dcc2ed44c/0000246833_en.pdf?hash=DE684100A66E9D16E4777318889B67E4&rev=59bb52b77ea24187a00e205042b0114a'),
 ('Franka impact experiment paper','https://arxiv.org/html/2402.05502v2'),
 ('ABB YuMi','https://one.robotics.abb.com/en/p/IRB-14000-YuMi-Dual-Arm'),
 ('ALOHA 2 paper','https://aloha-2.github.io/assets/aloha2.pdf'),
 ('ViperX working-load guidance','https://docs.trossenrobotics.com/interbotix_xsarms_docs/specifications/vx300s.html'),
]
RECOMMEND=[
 ('Existing miniature bowl benchmark','Keep SO-101','One','Preserve the trained experiment and its original scores. Full-size household objects can move to FR3/Panda.'),
 ('General cooking / measuring and pouring','Franka FR3','One initially; two for active workholding','Seven-axis redundancy, joint torque sensing and a native dual model make this the primary family to develop. Start with moderate loads and explicit gripper/tool models.'),
 ('Cutting with a short rigid knife adapter','FR3 for the first force-controlled prototype','Second FR3 to hold/reposition food','The 80 mm knife offset passed the 60 N static screen. The 170 mm offset did not consistently pass. Controller tuning, fixture geometry and calibrated material response still required.'),
 ('Long-offset / higher-force cutting','UR5e or UR10e model as alternatives','Optional second arm or passive fixture','UR models had more static wrist margin. UR10e needed a larger stand-off from the table. This is not a demonstrated cutting task.'),
 ('Steel hammer / construction tools','UR10e next simulation candidate; iiwa 14 later','One tool arm + board clamp initially','UR10e cleared the 100 N quasi-static probe after moving its base back. This is not impact qualification. iiwa needs its missing torque limits audited first.'),
 ('Bimanual assembly, lids, handover, bowl stabilization','FR3 Duo / paired FR3','Two','Native dual-FR3 model reached 14/14 requested paired stationary TCP poses without arm collisions. Tools, grippers and coordinated paths were not tested.'),
]


def table(head,rows):
 return '<div class="scroll"><table><thead><tr>'+''.join('<th>'+escape(str(x))+'</th>' for x in head)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+escape(str(x))+'</td>' for x in row)+'</tr>' for row in rows)+'</tbody></table></div>'


def render(key,data):
 if key=='fr3_duo':
  s=mujoco.MjSpec.from_file(str(ROOT/'assets/menagerie/franka_fr3_duo/fr3_duo.xml'));add_bench(s)
  rows=[r for r in data['poses'] if r['collision_free']];q=rows[1]['qpos']
 else:
  base=data.get('base_x_m',0);s=build_spec(key,base)
  rows=[r for r in data['poses'] if r['task']=='bowl' and r['collision_free']];row=min(rows,key=lambda r:np.linalg.norm(np.array(r['target_m'])-[.5,.15,.32]));q=row['qpos']
 s.add_texture(name='eval_sky',type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,rgb1=[.23,.29,.34],rgb2=[.07,.10,.12],width=512,height=3072)
 s.worldbody.add_light(pos=[.2,-.5,1.8],dir=[0,0,-1],diffuse=[.8]*3);s.visual.headlight.ambient=[.4]*3
 front=np.array([1.4,-1.4,1.1]);z=front-[.25,0,.35];z/=np.linalg.norm(z);x=np.cross([0,0,1],z);x/=np.linalg.norm(x);y=np.cross(z,x)
 s.worldbody.add_camera(name='comparison',pos=front,xyaxes=np.r_[x,y],fovy=43)
 m=s.compile();d=mujoco.MjData(m);d.qpos[:]=q;mujoco.mj_forward(m,d);m.vis.global_.offwidth=960;m.vis.global_.offheight=720
 r=mujoco.Renderer(m,480,640);r.update_scene(d,camera='comparison');Image.fromarray(r.render()).save(OUT/(key+'.png'));r.close()


def main():
 p=argparse.ArgumentParser();p.add_argument('--render',action='store_true');args=p.parse_args()
 names=['panda','fr3','ur5e','ur10e','iiwa14'];data={k:json.loads((OUT/(k+'.json')).read_text()) for k in names}
 dual=json.loads((OUT/'fr3_duo.json').read_text());variants={k:json.loads((OUT/(k+'.json')).read_text()) for k in ['fr3_short_knife_mount','ur10e_base_back_250mm','iiwa14_base_back_150mm']}
 if args.render:
  for key in names:render(key,variants['ur10e_base_back_250mm'] if key=='ur10e' else data[key])
  render('fr3_duo',dual)
 content='<main><div class="page-intro"><div class="eyebrow">Robot selection / 18 November 2026</div><h1>Choose the arm for <em>the task.</em></h1><p>Recommendation for the next Chora implementations: FR3 as the main cooking and bimanual research family; UR10e as the stronger tool/construction simulation candidate. These are provisional engineering choices, not validated task or hardware performance.</p><p>The current Chora scenes and trained bowl experiment were not changed by this evaluation.</p></div>'
 content+='<section class="card"><h2>Task recommendations</h2>'+table(['Task','Candidate','Arms','Reason / next validation'],RECOMMEND)+'</section>'
 content+='<section class="cards">'
 for key,label in [(k,data[k]['name']) for k in names]+[('fr3_duo','FR3 Duo · two actual arms')]:
  content+='<article class="card"><h2>'+escape(label)+'</h2><img src="'+key+'.png" alt="'+escape(label)+' native model in the screening scene"><p>Native model geometry. Bare flanges; no gripper or tool installed in this screen.</p></article>'
 content+='</section><section class="card"><h2>What was actually tested</h2><p>Five single-arm models at one shared tabletop layout: 62 target TCP poses per arm (12 bowl, 18 knife, 12 hammer, 20 pouring). Position tolerance 2 mm, orientation tolerance 2°, joint-limit margin 0.025 rad. Multistart bounded inverse kinematics; native collision geometry plus a floor and work surface.</p><p>Geometric IK was solved first; poses were not optimized for torque margin. Reached poses are evidence of feasibility. A failed numerical search is not proof that a pose is unreachable. No motion paths, gripper/tool collisions, grasp stability, material interaction or actual task completion were evaluated.</p>'
 content+=table(['Model','Bowl poses','Knife poses','Hammer poses','Pour poses'],[(data[k]['name'],*[f"{data[k]['summary'][t]['collision_free']}/{data[k]['summary'][t]['poses']}" for t in ['bowl','cutting','hammer','pouring']]) for k in names])+'</section>'
 rows=[]
 for k,r in data.items():
  for task in TASKS:
   v=r['summary'][task];ratios=v['static_ratios'];rows.append((r['name'],task,f"{v['within_model_static_limits']}/{v['collision_free']}" if v['within_model_static_limits'] is not None else 'Not scored: unbounded model',f'{max(ratios):.2f}×' if ratios else '—'))
 content+='<section class="card"><h2>Static load screen</h2><p>Quasi-static inverse loads include native arm gravity, assumed gripper/tool mass and an external reaction at the tool point. Force limits are simulator actuator maxima, not continuous-duty or certified impact ratings. The 100 N hammer probe is a hypothetical steady wrench, not a measured wood impact.</p>'+table(['Model','Case','Below model limit / reached poses','Worst ratio'],rows)+table(['Case','TCP from flange (m)','Combined mass (kg)','Reaction force (N)'],[(t,v['tcp'],v['mass'],v['reaction']) for t,v in TASKS.items()])+'</section>'
 sweep=[]
 for k,r in data.items():
  if not r['has_enabled_torque_limits']:continue
  rows=[x for x in r['poses'] if x['task']=='cutting' and x.get('upward_force_sweep')]
  sweep.append((r['name'],*[f"{sum(x['upward_force_sweep'][str(f)]<=1 for x in rows)}/{len(rows)}" for f in [20,40,60,100]]))
 content+='<section class="card"><h2>Knife force sensitivity</h2><p>Same selected joint poses, 170 mm horizontal tool offset; upward reaction varied while 10 N lateral draw force remained. These are design probes, not calibrated fruit forces.</p>'+table(['Model','20 N','40 N','60 N','100 N'],sweep)+'</section>'
 vr=[]
 for name in variants:
  r=variants[name]
  for task in ['cutting','hammer','pouring']:
   v=r['summary'][task];vr.append((name,task,f"{v['collision_free']}/{v['poses']}",f"{v['within_model_static_limits']}/{v['collision_free']}" if v['within_model_static_limits'] is not None else 'Not scored'))
 content+='<section class="card"><h2>Tool mount and base placement change the answer</h2><p>FR3 short knife: horizontal TCP offset reduced from 170 to 80 mm, with a correspondingly shorter CoM offset. UR10e base moved 250 mm away from the table. iiwa base moved 150 mm away. These are screening variants, not additional app scenes or validated physical fixtures.</p>'+table(['Variant','Case','Pose + collision pass','Below static model limits'],vr)+'</section>'
 content+='<section class="card"><h2>Two-arm assessment</h2><p>The existing Menagerie FR3 Duo model contains two seven-axis arms on its native mount. All 14 requested paired stationary poses were collision-free in this search: 5 pour/hold, 3 cut/hold, 3 handover and 3 assembly pairs. This checks arm clearance at the endpoints only. End effectors, objects, tools, trajectories and simultaneous contact control are still untested. Franka’s vendor page does not currently list a released MuJoCo reference-kit setup; this Menagerie model is a URDF-derived community conversion.</p><p>A second arm earns its complexity when it must actively orient or reposition the workpiece: holding a bowl while pouring/mixing, stabilizing/repositioning food while cutting, opening a jar, handover, or holding parts during assembly. A bowl that stays on the bench, or a board held in a clamp, usually needs one manipulation arm. For nailing, start with a board clamp and one tool arm.</p><p>ALOHA is worth considering later for light bimanual demonstrations; its ViperX-class working-load limits are a poor default for our assumed full tool/liquid loads. YuMi is designed around light assembly (0.5 kg per arm), so it is not the primary choice here.</p></section>'
 hold=json.loads((OUT/'native_hold_tests.json').read_text())
 content+='<details class="card"><summary>Native position-controller audit</summary><p>A 1.1 kg rigid hypothetical tool payload was added with 0.001 kg·m² diagonal inertia. Native joint setpoints held a single pose. A constant world wrench (10, 0, 60) N was applied for 1.2 s after settling. No gravity feedforward or force-control tuning was added. The constant wrench persists after the arm moves; this is a controller stress test, not material contact. These errors do not measure vendor hardware accuracy.</p>'+table(['Configuration','Loaded TCP error (mm)','Saturated steps','Solver warnings'],[(r['configuration'],f"{r['mean_loaded_tcp_position_error_mm']:.1f}",f"{r['steady_state_saturation_fraction']:.0%}",sum(r['warning_counts'])) for r in hold])+'<p>The untuned native position controllers are not task-ready. The next implementation needs an explicitly designed controller, payload model and task-specific tool frame before any cooking or construction result is claimed.</p></details>'
 content+='<section class="card"><h2>Important limits</h2><p>Heavier payload capacity does not certify a robot for hammer impacts. The published Franka hammer study used a printed ABS hammer, a foam base and pilot holes; it is not evidence for a steel claw hammer driving into pine. No arm is declared hardware-ready for our hammer task here.</p><p>The pinned iiwa model has no actuator-force limits enabled; its static capacity was deliberately not scored. UR5e/UR10e names refer to the pinned asset versions and 2024 datasheets; current UR naming/payload revisions must not be silently mixed into those models. A new arm cannot validate our current wood or food material models.</p></section>'
 content+='<section class="card"><h2>Reproduce and inspect</h2><p>Pinned Menagerie revision: <code>370c43965323c0944b433b202aac5058b5f907a9</code>. Full joint solutions, errors, contacts, limits and assumptions are retained in the downloadable JSON files.</p><p>'+ ' · '.join('<a href="'+k+'.json">'+escape(k)+'</a>' for k in names+['fr3_duo','fr3_short_knife_mount','ur10e_base_back_250mm','iiwa14_base_back_150mm','native_hold_tests','method'])+'</p><ul>'+''.join('<li><a href="'+url+'">'+escape(label)+'</a></li>' for label,url in SOURCES)+'</ul></section></main>'
 css='main{max-width:1440px;margin:auto;padding:32px}.page-intro{max-width:1000px;margin:20px 0 32px}.page-intro h1{max-width:none}.card{margin-bottom:22px}.cards{display:grid;grid-template-columns:repeat(3,1fr);gap:18px}.cards img{width:100%;border-radius:8px}.scroll{overflow-x:auto}table{width:100%;border-collapse:collapse;font-size:13px}th,td{text-align:left;padding:12px;border-bottom:1px solid var(--line);vertical-align:top}p,li{line-height:1.7;color:var(--muted)}summary{cursor:pointer}code{overflow-wrap:anywhere}@media(max-width:900px){.cards{grid-template-columns:1fr 1fr}}@media(max-width:600px){.cards{grid-template-columns:1fr}}'
 (OUT/'index.html').write_text(page('Robot arm evaluation',content,'reports',css))
 (OUT/'recommendations.json').write_text(json.dumps(dict(date='2026-11-18',recommendations=RECOMMEND,sources=SOURCES,scope='Provisional simulation-platform selection; no app scene or hardware purchase committed.'),indent=2)+'\n')
 print(OUT/'index.html')

if __name__=='__main__':main()
