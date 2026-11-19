"""Record isolated, unscored articulated-arm trials; never touch bowl results."""
import argparse,json,os,subprocess,sys
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('experiment',choices=['fr3-tile-installation','fr3-duo-tile-wipe','ur10e-hammer','fr3-cut-apple','fr3-cut-tomato','fr3-cut-banana']);p.add_argument('--fps',type=int,choices=range(1,61),default=20);a=p.parse_args()
    os.environ['CHORA_EXPERIMENT']=a.experiment
    from chora.common import ROOT,OUTPUT,config
    from chora.sim import make_simulation
    from chora.policy import make_expert
    from chora.visualization import Recorder
    from PIL import Image
    s=make_simulation();policy=make_expert(0);history=[];trace=[]
    r=Recorder(s,OUTPUT/'material_trial.mp4',fps=a.fps)
    try:
        Image.fromarray(r.camera.render(s,overlay=False)).save(OUTPUT/'initial.png')
        while not s.finished:
            action=policy.act(s.observe(),history);before=s.observe();s.step(action);history.append(action)
            trace.append(dict(observation=before,action=action,result=s.result))
            if a.experiment=='fr3-tile-installation' and len(history) in (4,13):
                name='application_overlap.png' if len(history)==4 else 'first_tile_bedded.png'
                Image.fromarray(r.camera.render(s,overlay=False)).save(OUTPUT/name)
            print(a.experiment,action,s.result,flush=True)
        Image.fromarray(r.camera.render(s,overlay=False)).save(OUTPUT/'final.png')
        result=dict(experiment=a.experiment,seed=0,success=None,benchmark_validated=False,physics=config()['physics'],steps=trace,final_observation=s.observe(),solver_warnings=s.data.warning.number.tolist(),collisions=s.collision_count)
        (OUTPUT/'material_trial.json').write_text(json.dumps(result,indent=2)+'\n')
    finally:r.close()
    subprocess.run([sys.executable,'-m','chora','report'],cwd=ROOT,check=True,capture_output=True,text=True)
    print(json.dumps(s.observe()),flush=True)

if __name__=='__main__':main()
