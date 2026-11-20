"""Compact, evidence-backed episode wall and verified final-state thumbnails."""
import json
from html import escape

from .common import OUTPUT, exclusive_job, job_progress, read_json, save_json
from .ui import page


def verify_trace(sim,episode):
    sim.reset(episode["seed"])
    for step in episode["steps"]:
        if sim.observe()!=step["observation"]:
            raise RuntimeError(f"Saved state mismatch for seed {episode['seed']}")
        if sim.step(step["action"])["last_result"]!=step["result"]:
            raise RuntimeError(f"Saved action result mismatch for seed {episode['seed']}")
    if sim.observe()!=episode["final_observation"] or sim.success!=episode["success"]:
        raise RuntimeError(f"Saved outcome mismatch for seed {episode['seed']}")


@exclusive_job("thumbnails")
def thumbnails(policy="all"):
    from .sim import make_simulation
    from .visualization import Camera
    names=("student","expert","base") if policy=="all" else (policy,)
    groups=[(name,read_json(OUTPUT/"evaluation"/name/"episodes.json",[])) for name in names]
    total=sum(len(rows) for _,rows in groups)
    if not total:
        raise ValueError("No scored episodes are available. Evaluate a policy first.")
    sim=make_simulation(); camera=Camera(sim,640,360); completed=0
    try:
        for name,episodes in groups:
            folder=OUTPUT/"evaluation"/name/"thumbs";folder.mkdir(parents=True,exist_ok=True)
            for e in episodes:
                verify_trace(sim,e)
                relative=f"evaluation/{name}/thumbs/seed_{e['seed']}.jpg"
                file=OUTPUT/relative;temporary=file.with_suffix(".jpg.tmp")
                temporary.write_bytes(camera.jpeg(sim,overlay=False));temporary.replace(file)
                e.update(thumbnail=relative,thumbnail_verified=True)
                completed+=1
                job_progress(completed,total,f"Preparing verified episode thumbnails · {completed}/{total}")
            save_json(OUTPUT/"evaluation"/name/"episodes.json",episodes)
    finally:
        camera.close()
    from .report import build_report
    build_report()
    job_progress(total,total,f"Prepared {completed} verified episode thumbnails")
    print(f"Created {completed} verified final-state thumbnails. Scores and policy weights are unchanged.",flush=True)
    return completed


CSS="""
.gallery-intro{margin-bottom:28px}.gallery-intro p{color:var(--muted);font-size:13px;line-height:1.8;max-width:75ch}.episode-wall-heading{display:flex;align-items:center;justify-content:space-between;gap:24px;margin:30px 0 20px}.episode-wall-heading h2{font-size:24px;font-weight:500;letter-spacing:-.8px;margin:0}.wall-controls{display:flex;gap:10px;flex-wrap:wrap}.pill-group{display:flex;align-items:center;background:var(--panel);border:1px solid var(--line);border-radius:999px;padding:4px;gap:3px}.pill-group button{background:transparent;color:var(--muted);border:0;border-radius:999px;padding:9px 15px;font-size:12px;cursor:pointer;white-space:nowrap}.pill-group button[aria-pressed=true]{background:var(--ink);color:var(--bg)}.pill-group button:disabled{opacity:.35;cursor:not-allowed}
.episode-wall{display:grid;grid-template-columns:repeat(10,minmax(0,1fr));gap:8px}.episode-tile{display:block;position:relative;padding:0;border:1px solid var(--line);border-radius:8px;aspect-ratio:16/9;overflow:hidden;background:var(--viewer-bg);cursor:pointer;transition:transform .15s,border-color .15s;min-width:0}.episode-tile.failure{border-color:color-mix(in srgb,#d34c4c 80%,var(--line))}.episode-tile:hover{transform:scale(1.045);z-index:2;border-color:var(--accent)}.episode-tile:focus-visible{outline:2px solid var(--accent);outline-offset:3px;z-index:3}.episode-tile img,.episode-tile video{position:absolute;inset:0;width:100%;height:100%;object-fit:cover}.episode-tile video{opacity:0;pointer-events:none;object-fit:contain;background:var(--viewer-bg)}.episode-tile.previewing video{opacity:1}.outcome-badge{position:absolute;right:5px;top:5px;width:19px;height:19px;display:grid;place-items:center;border-radius:50%;color:#fff;font:600 13px system-ui;z-index:2}.outcome-badge.ok{background:#159b37}.outcome-badge.fail{background:#d34343}.tile-seed{position:absolute;bottom:4px;left:5px;color:#fff;background:#0a111aa3;border-radius:3px;padding:2px 4px;font:9px ui-monospace,monospace;opacity:0;transition:opacity .15s}.episode-tile:hover .tile-seed,.episode-tile:focus-visible .tile-seed{opacity:1}.clip-indicator{position:absolute;right:5px;bottom:4px;font:9px ui-monospace;color:#fff;background:#0a111a88;border-radius:3px;padding:2px 4px}.tile-placeholder{position:absolute;inset:0;display:grid;place-items:center;color:var(--muted);font:10px ui-monospace}.episode-tile[hidden]{display:none}.wall-caption{font-size:12px;color:var(--muted);line-height:1.8;margin-top:20px}.wall-empty{color:var(--muted);padding:40px;text-align:center;border:1px dashed var(--line);border-radius:8px}
#episode-dialog{width:min(940px,calc(100vw - 32px));max-height:92vh;overflow:auto;padding:0;border:1px solid var(--line);border-radius:12px;background:var(--panel);color:var(--ink)}#episode-dialog::backdrop{background:#050a10aa;backdrop-filter:blur(5px)}.episode-modal-header{padding:18px 22px;display:flex;align-items:center;justify-content:space-between;gap:15px}.episode-modal-header h2{font-size:17px;font-weight:500;margin:0}.episode-modal-header button{padding:5px 12px;font-size:20px}.episode-media{position:relative;background:var(--viewer-bg);aspect-ratio:16/9;display:grid;place-items:center}.episode-media video,.episode-media img{width:100%;height:100%;max-height:60vh;object-fit:contain}.episode-media [hidden]{display:none}.episode-info{padding:18px 22px}.episode-info p{color:var(--muted);font-size:12px;line-height:1.8;margin:0 0 15px}.episode-info details{border-top:1px solid var(--line);padding-top:14px}.episode-info summary{cursor:pointer;font-size:12px;color:var(--muted)}.episode-info pre{white-space:pre-wrap;word-break:break-word;font:11px/1.9 ui-monospace;color:var(--muted)}.episode-loading{position:absolute;display:flex;align-items:center;gap:9px;color:var(--muted);background:var(--panel);padding:10px;border-radius:5px;pointer-events:none}.episode-loading[hidden]{display:none}
@media(max-width:1250px){.episode-wall{grid-template-columns:repeat(8,minmax(0,1fr))}.episode-wall-heading{align-items:flex-start;flex-direction:column;gap:16px}}@media(max-width:900px){.episode-wall{grid-template-columns:repeat(6,minmax(0,1fr))}}@media(max-width:600px){.episode-wall{grid-template-columns:repeat(4,minmax(0,1fr));gap:6px}.episode-wall-heading h2{font-size:20px}.pill-group button{padding:8px 12px;font-size:11px}.outcome-badge{width:17px;height:17px;right:4px;top:4px}.episode-modal-header,.episode-info{padding:16px}}@media(max-width:430px){.wall-controls{width:100%}.wall-controls>.pill-group:first-child{width:100%;display:grid;grid-template-columns:repeat(3,minmax(0,1fr));border-radius:12px}.wall-controls>.pill-group:first-child button{white-space:normal;padding:8px 6px;min-height:40px;line-height:1.3;font-size:11px}}@media(max-width:380px){.episode-wall{grid-template-columns:repeat(3,minmax(0,1fr))}}@media(prefers-reduced-motion:reduce){.episode-tile{transition:none}.episode-tile:hover{transform:none}}
"""

SCRIPT="""
const gallery=JSON.parse(document.getElementById('episode-data').textContent);
let policy=gallery.initial || (gallery.policies.student?'student':Object.keys(gallery.policies)[0]),outcome='all';
const titles={student:'trained student',base:'untrained baseline',expert:'scripted expert'};
function filterWall(nextPolicy=policy,nextOutcome=outcome){
 policy=nextPolicy;outcome=nextOutcome;let shown=0,clips=0;
 document.querySelectorAll('.episode-tile').forEach(tile=>{tile.hidden=tile.dataset.policy!==policy||(outcome!=='all'&&tile.dataset.success!==outcome);const video=tile.querySelector('video');if(tile.hidden){video?.pause();tile.classList.remove('previewing')}else{shown++;if(tile.dataset.video==='true')clips++}});
 document.querySelectorAll('[data-wall-policy]').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.wallPolicy===policy)));
 document.querySelectorAll('[data-wall-outcome]').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.wallOutcome===outcome)));
 const summary=gallery.policies[policy];document.getElementById('wall-title').textContent=summary?`${summary.episodes} episodes, ${titles[policy]} · ${summary.successes}/${summary.episodes}`:'No evaluation yet';
 document.getElementById('all-outcomes').textContent=summary?'All '+summary.episodes:'All';
 document.getElementById('wall-caption').textContent=`${shown} episodes shown · ${clips} recordings available. Hover a movie tile to preview; click any tile for details. Thumbnails show verified final states. Episodes without a saved movie show a still image.`;
 document.getElementById('wall-empty').hidden=shown!==0;
}
function openEpisode(key){
 const e=gallery.episodes[key],dialog=document.getElementById('episode-dialog'),video=document.getElementById('episode-video'),image=document.getElementById('episode-image'),loading=document.getElementById('episode-loading');
 document.querySelectorAll('.episode-tile video').forEach(v=>v.pause());
 document.getElementById('episode-title').textContent=`${titles[e.policy]} / seed ${e.seed} / ${e.success?'Success':'Failure'}`;
 video.pause();video.removeAttribute('src');video.hidden=!e.video;image.hidden=Boolean(e.video);loading.hidden=!e.video;
 if(e.video){video.src=e.video;video.load();video.play().catch(()=>{})}else if(e.thumbnail){image.src=e.thumbnail;image.alt='Verified final state for seed '+e.seed}else{image.removeAttribute('src');image.alt='No thumbnail available'}
 document.getElementById('episode-description').textContent=(e.video?'Recorded playback of the scored action trace.':'No movie was saved. This still image shows the verified final state.')+' '+e.steps.length+' decisions · '+e.result.replaceAll('_',' ')+'.';
 document.getElementById('episode-trace').textContent=e.steps.map((s,i)=>String(i+1).padStart(2,'0')+'  '+JSON.stringify(s.action)+' → '+s.result).join('\\n');
 dialog.showModal();
}
document.querySelectorAll('.episode-tile').forEach(tile=>{
 tile.addEventListener('click',()=>openEpisode(tile.dataset.key));const video=tile.querySelector('video');if(!video)return;
 tile.addEventListener('pointerenter',()=>{if(!matchMedia('(hover:hover)').matches)return;if(!video.src)video.src=video.dataset.src;video.play().then(()=>tile.classList.add('previewing')).catch(()=>{})});
 tile.addEventListener('pointerleave',()=>{video.pause();tile.classList.remove('previewing')});
});
const episodeVideo=document.getElementById('episode-video'),loading=document.getElementById('episode-loading');
for(const event of ['loadstart','waiting'])episodeVideo.addEventListener(event,()=>loading.hidden=false);
for(const event of ['playing','canplay','error'])episodeVideo.addEventListener(event,()=>loading.hidden=true);
document.getElementById('episode-dialog').addEventListener('close',()=>{episodeVideo.pause();episodeVideo.removeAttribute('src');episodeVideo.load()});
filterWall();
"""


def render_gallery(results, output, experiment=None, method=None):
    modes = ("expert",) if method == "expert" else ("student", "base", "expert")
    initial = {"expert": "expert", "qwen": "student"}.get(method)
    tiles=[];data={"policies":{key: results[key] for key in modes if key in results},
                   "episodes":{}, "initial":initial}
    for policy in modes:
        for e in read_json(output/"evaluation"/policy/"episodes.json",[]):
            key=f"{policy}-{e['seed']}";passed=e["success"];thumbnail=e.get("thumbnail");video=e.get("video")
            data["episodes"][key]={"policy":policy,"seed":e["seed"],"success":passed,"thumbnail":thumbnail,"video":video,"result":e["final_observation"]["last_result"],"steps":[{"action":s["action"],"result":s["result"]} for s in e["steps"]]}
            image=f'<img loading="lazy" src="{escape(thumbnail)}" alt="Seed {e["seed"]}, {"success" if passed else "failure"}">' if thumbnail else f'<span class="tile-placeholder">Seed {e["seed"]}</span>'
            movie=f'<video muted loop playsinline preload="none" data-src="{escape(video)}" aria-hidden="true"></video><span class="clip-indicator" aria-hidden="true">▶</span>' if video else ''
            tiles.append(f'<button class="episode-tile {"success" if passed else "failure"}" data-policy="{policy}" data-success="{str(passed).lower()}" data-key="{key}" data-video="{str(bool(video)).lower()}" aria-label="{policy}, seed {e["seed"]}, {"success" if passed else "failure"}. {"Watch recording" if video else "View final state"}">{image}{movie}<span class="outcome-badge {"ok" if passed else "fail"}" aria-hidden="true">{"✓" if passed else "×"}</span><span class="tile-seed">{e["seed"]}</span></button>')
    controls=[]
    for policy,label in (("student","Trained student"),("base","Untrained baseline"),("expert","Scripted expert")):
        if policy not in modes:
            continue
        disabled='' if policy in results else ' disabled title="Not evaluated yet."'
        controls.append(f'<button data-wall-policy="{policy}" aria-pressed="false" onclick="filterWall(\'{policy}\',outcome)"{disabled}>{label}</button>')
    payload=json.dumps(data,separators=(',',':'),ensure_ascii=False).replace('&','\\u0026').replace('<','\\u003c').replace('>','\\u003e')
    baseline_note='' if method == 'expert' or 'base' in results else '<p class="baseline-note">The untrained baseline has not been evaluated for this experiment yet.</p>'
    description = ('Recorded outcomes from the scripted expert.' if method == 'expert' else
                   'Compare Qwen with the scripted expert. The untrained baseline measures Qwen before fine-tuning.')
    body='<main><div class="gallery-intro"><div class="eyebrow">Evaluation / episode wall</div><h1>Every scene. Every <em>outcome</em>.</h1><p>'+description+'</p>'+baseline_note+'</div><section aria-labelledby="wall-title"><div class="episode-wall-heading"><h2 id="wall-title">Evaluation episodes</h2><div class="wall-controls"><div class="pill-group" role="group" aria-label="Policy">'+''.join(controls)+'</div><div class="pill-group" role="group" aria-label="Outcome"><button id="all-outcomes" data-wall-outcome="all" aria-pressed="true" onclick="filterWall(policy,\'all\')">All</button><button data-wall-outcome="true" aria-pressed="false" onclick="filterWall(policy,\'true\')">Successes</button><button data-wall-outcome="false" aria-pressed="false" onclick="filterWall(policy,\'false\')">Failures</button></div></div></div><div class="episode-wall">'+''.join(tiles)+'</div><p class="wall-empty" id="wall-empty" hidden>No episodes match this filter.</p><p class="wall-caption" id="wall-caption" aria-live="polite"></p></section></main>'
    body+='<dialog id="episode-dialog" aria-labelledby="episode-title"><div class="episode-modal-header"><h2 id="episode-title">Episode</h2><button class="secondary" onclick="document.getElementById(\'episode-dialog\').close()" aria-label="Close episode">×</button></div><div class="episode-media"><video id="episode-video" controls playsinline hidden></video><img id="episode-image" alt="Episode final state" hidden><div class="episode-loading" id="episode-loading" hidden><span class="spinner" aria-hidden="true"></span>Loading recording…</div></div><div class="episode-info"><p id="episode-description"></p><details><summary>Action trace</summary><pre id="episode-trace"></pre></details></div></dialog><script id="episode-data" type="application/json">'+payload+'</script>'
    return page("Evaluation gallery", body, "gallery", CSS, SCRIPT, experiment=experiment, policy=method)


def build_gallery(results, output=None):
    output = output or OUTPUT
    (output / "eval_gallery.html").write_text(render_gallery(results, output))
