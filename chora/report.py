"""Reports are assembled only from measured files in this project's outputs."""
from __future__ import annotations

from html import escape

from .common import OUTPUT, read_json, read_jsonl
from .ui import page

REPORT_CSS = """
.page-intro{margin-bottom:28px}.page-intro h1{max-width:none}.page-intro p{color:var(--muted);line-height:1.7;max-width:72ch}.page-actions{display:flex;gap:18px;flex-wrap:wrap;font-size:12px;margin-top:18px}
.report-metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:24px}.report-metric{padding:22px;border:1px solid var(--line);border-radius:9px;background:var(--panel)}.report-metric strong{display:block;font-size:30px;font-weight:450;margin:10px 0 5px}.report-metric small{color:var(--muted)}
.report-grid{display:grid;grid-template-columns:1fr 1fr;gap:18px}.report-section{margin-bottom:20px}.report-section h2{font-size:18px;font-weight:500}.report-section p{color:var(--muted);line-height:1.8;font-size:12px}.report-section img{width:100%;border-radius:6px;margin-top:15px}.report-section table{width:100%;border-collapse:collapse;font-size:12px}.report-section th,.report-section td{text-align:left;padding:12px 7px;border-bottom:1px solid var(--line)}.report-section th{font-weight:500;color:var(--muted)}
.gallery-filters{display:flex;gap:18px;align-items:center;flex-wrap:wrap;margin:24px 0}.gallery-filters label{font-size:12px;color:var(--muted);display:flex;gap:9px;align-items:center}.gallery-filters select{width:150px}.gallery-filters input[type=checkbox]{width:auto;accent-color:var(--accent)}.gallery-count{font:11px ui-monospace,monospace;color:var(--muted)}
.gallery-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,310px),1fr));gap:18px}.gallery-card{background:var(--panel);border:1px solid var(--line);border-radius:10px;overflow:hidden}.gallery-card .meta{display:flex;justify-content:space-between;gap:12px;padding:15px;font-size:12px}.gallery-card .meta b{font-weight:500}.gallery-card .ok{color:var(--accent)}.gallery-card .fail{color:var(--error)}.gallery-card video{width:100%;display:block;background:var(--viewer-bg)}.no-video{height:90px;display:grid;place-items:center;font-size:11px;color:var(--muted)}.gallery-card details{padding:14px;font-size:11px;color:var(--muted)}.gallery-card summary{cursor:pointer}.gallery-card pre{white-space:pre-wrap;line-height:1.8;word-break:break-word}.gallery-card[hidden]{display:none}.gallery-empty{color:var(--muted);padding:40px;text-align:center;border:1px dashed var(--line);border-radius:9px}
.video-wrap{position:relative}.video-loading{position:absolute;inset:0;display:flex;align-items:center;justify-content:center;gap:10px;background:color-mix(in srgb,var(--panel) 80%,transparent);font-size:12px}.video-loading[hidden]{display:none}
.spinner{display:inline-block;width:14px;height:14px;flex-shrink:0;border:2px solid color-mix(in srgb,var(--accent) 22%,transparent);border-top-color:var(--accent);border-radius:50%;animation:spin .8s linear infinite}@keyframes spin{to{transform:rotate(360deg)}}@media(prefers-reduced-motion:reduce){.spinner{animation:none}}
@media(max-width:760px){.report-metrics{grid-template-columns:1fr 1fr}.report-grid{grid-template-columns:1fr}}
"""


def metric(label,value,note=""):
    return '<div class="report-metric"><span class="label">'+escape(label)+'</span><strong>'+escape(str(value))+'</strong><small>'+escape(note)+'</small></div>'


def table(headers,rows):
    return '<table><thead><tr>'+''.join('<th>'+escape(str(h))+'</th>' for h in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+escape(str(c))+'</td>' for c in row)+'</tr>' for row in rows)+'</tbody></table>'


def build_html_pages(results,oracle,curation,training,selection,total_wall):
    from .gallery import build_gallery
    build_gallery(results,output=OUTPUT)
    rows = [(name,r["successes"],r["episodes"],f'{r["mean_steps_to_success"]:.2f}') for name,r in results.items()]
    student = results.get("student",{})
    metrics = metric("Expert check",f'{oracle.get("successes","—")}/{oracle.get("episodes","—")}')+metric("Student test",f'{student.get("successes","—")}/{student.get("episodes","—")}')+metric("Training examples",f'{curation.get("training_pairs",0):,}')+metric("Training time",f'{total_wall/60:.1f} min',"Across all saved attempts")
    failure_rows = [(reason.replace('_',' ').capitalize(),n) for reason,n in student.get("failure_breakdown",{}).items()]
    content = '<main><div class="page-intro"><div class="eyebrow">Experiment / measured results</div><h1>Learning, <em>measured</em>.</h1><p>The expert follows a programmed plan. Qwen learns from that expert’s demonstrations. Scores below are separate, measured results.</p><div class="page-actions"><a href="eval_gallery.html">Watch evaluation recordings ↗</a><a href="curation_report.html">Explore the training data ↗</a><a href="REPORT.md" download>Download Markdown report ↓</a></div></div><div class="report-metrics">'+metrics+'</div><div class="report-grid"><section class="card report-section"><h2>Held-out evaluation</h2>'+table(["Policy","Successes","Scenes","Decisions to success"],rows)+'<p>Final evaluation uses reserved seeds 100000–100099. A smaller evaluation is a partial benchmark.</p></section><section class="card report-section"><h2>Where the student struggled</h2>'+table(["Failure","Episodes"],failure_rows)+f'<p>Malformed commands: {student.get("invalid_actions","not measured")}. These trials use exact scene coordinates from the simulator.</p></section></div>'
    if (OUTPUT/"training_curve.png").exists():
        content += '<section class="card report-section"><h2>Learning progress</h2><p>Training loss measures prediction error on demonstrations. Validation checks examples kept out of training. The selected adapter was chosen before final evaluation.</p><img src="training_curve.png" alt="Training and validation loss, with the selected checkpoint marked"></section>'
    content += '</main>'
    (OUTPUT/"experiment_report.html").write_text(page("Experiment report",content,"reports",REPORT_CSS))

    metrics = metric("Collected",f'{curation.get("raw_episodes",0):,}',"Raw demonstration episodes")+metric("Passed quality",f'{curation.get("passed_quality",0):,}')+metric("Selected",f'{curation.get("selected_episodes",0):,}',"Clean, diverse episodes")+metric("Training pairs",f'{curation.get("training_pairs",0):,}',"One observation and next action")
    reasons = {"failed_episode":"Task did not finish successfully","unclean_action":"Invalid action, collision, or failed grasp","edge_placement":"Placed too close to the target boundary","near_duplicate_start":"Nearly duplicate initial scene","reserved_seed":"Reserved evaluation scene"}
    drops = [(reasons.get(k,k.replace('_',' ')),v) for k,v in curation.get("drop_reasons",{}).items()]
    cap = curation.get("passed_quality",0)-curation.get("selected_episodes",0)
    if cap>0: drops.append(("Diversity selection: per-bin limit",cap))
    content = '<main><div class="page-intro"><div class="eyebrow">Training data / curation</div><h1>Good examples. <em>Broad coverage.</em></h1><p>Collection saves what the expert did, including failures. Curation chooses clean, varied successes for the student to learn from.</p><div class="page-actions"><a href="experiment_report.html">Back to experiment report ↗</a><a href="curation_report.md" download>Download curation details ↓</a></div></div><div class="report-metrics">'+metrics+'</div><div class="report-grid"><section class="card report-section"><h2>Why examples were removed</h2>'+table(["Reason","Episodes"],drops)+'</section><section class="card report-section"><h2>Keep validation independent</h2>'+table(["Split","Episodes","Observation–action pairs"],[("Training",curation.get("train_episodes",0),curation.get("training_pairs",0)),("Validation",curation.get("validation_episodes",0),curation.get("validation_pairs",0))])+f'<p>Whole episodes stay together. Scene selection covers {curation.get("occupied_bins",0)} of {curation.get("total_bins",64)} position-and-orientation bins.</p></section></div>'
    if (OUTPUT/"coverage.png").exists():
        content += '<section class="card report-section"><h2>Scene coverage</h2><p>Compare where objects started before and after selection.</p><img src="coverage.png" alt="Raw and curated object-start coverage"></section>'
    content += '</main>'
    (OUTPUT/"curation_report.html").write_text(page("Curation report",content,"reports",REPORT_CSS))


def plot_training():
    path = OUTPUT/"training_metrics.jsonl"
    if not path.exists():
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = read_jsonl(path)
    train = [r for r in rows if r["type"]=="train"]
    valid = [r for r in rows if r["type"]=="validation"]
    selection = read_json(OUTPUT/"selection.json",{})
    with plt.style.context("dark_background"):
        fig,ax = plt.subplots(figsize=(9,4.2))
        fig.patch.set_facecolor("#101819"); ax.set_facecolor("#101819")
        ax.plot([r["iteration"] for r in train],[r["train_loss"] for r in train],color="#99a8a0",lw=1,alpha=.7,label="Training")
        ax.plot([r["iteration"] for r in valid],[r["val_loss"] for r in valid],color="#c4e795",marker="o",lw=2,label="Validation")
        if selection:
            ax.axvline(selection["checkpoint"],color="#df9d73",ls="--",lw=1.5,label="Selected adapter")
        ax.set(title="Chora · latest training run",xlabel="Optimizer step in this run",ylabel="Assistant-token cross entropy")
        ax.grid(alpha=.12); ax.legend(frameon=False); fig.tight_layout()
        fig.savefig(OUTPUT/"training_curve.png",dpi=150)
        plt.close(fig)


def build_report():
    from .common import config
    if config()["experiment"]["task"] in {"soft_cut","impact_nail","dual_tile_wipe","tile_dry_transfer","tile_adhesive_bedding"}:return build_material_trial_report()
    results = read_json(OUTPUT/"results.json",{})
    oracle = read_json(OUTPUT/"oracle.json",{})
    curation = read_json(OUTPUT/"curation.json",{})
    training = read_json(OUTPUT/"training.json",{})
    selection = read_json(OUTPUT/"selection.json",{})
    archived = [read_json(p,{}) for p in sorted((OUTPUT/"training_runs").glob("*/training.json"))]
    total_wall = sum(r.get("wall_seconds",0) for r in archived+[training])
    lines = ["# Chora by Coherence — experiment report","",
             "All figures below come from local run outputs. Expert and student results are separate.","",
             f"- Expert qualification: {oracle.get('successes','pending')} / {oracle.get('episodes','pending')}.",
             f"- Raw demonstrations: {curation.get('raw_episodes','pending')}.",
             f"- Curated episodes: {curation.get('selected_episodes','pending')}.",
             f"- Training pairs: {curation.get('training_pairs','pending')}.",
             f"- Validation pairs: {curation.get('validation_pairs','pending')}.",
             f"- Current training status: {training.get('status','not started')}.",
             f"- Latest run wall time: {training.get('wall_seconds',0):.1f} seconds.",
             f"- Total training wall time across attempts: {total_wall:.1f} seconds ({total_wall/60:.1f} minutes).",
             f"- Selected checkpoint in latest run: {selection.get('checkpoint','not selected')}.",
             "","| Policy | Successes | Episodes | Mean decisions to success |","| --- | ---: | ---: | ---: |"]
    for name,r in results.items():
        lines.append(f"| {name} | {r['successes']} | {r['episodes']} | {r['mean_steps_to_success']:.2f} |")
    lines.extend(["","Test seeds: 100000–100099. A smaller completed run is a partial benchmark.","",
        "Training uses structured simulator observations, a scripted expert, and high-level motion primitives. This is not a camera-based or real-hardware benchmark.","",
        "See curation_report.md, results.json, training.json, training_metrics.jsonl, and eval_gallery.html for details."])
    if (OUTPUT/"training_metrics.jsonl").exists():
        plot_training()
        lines.extend(["","![Latest training run and selected adapter](training_curve.png)"])
    OUTPUT.mkdir(parents=True,exist_ok=True)
    (OUTPUT/"REPORT.md").write_text("\n".join(lines)+"\n")
    build_html_pages(results,oracle,curation,training,selection,total_wall)
    return OUTPUT/"REPORT.md"


def build_material_trial_report():
    from .common import config
    cfg=config();trial=read_json(OUTPUT/"material_trial.json",{});obs=trial.get("final_observation",{})
    content='<main><div class="page-intro"><div class="eyebrow">Material development / '+escape(cfg["experiment"]["name"])+'</div><h1>Material response, <em>under test.</em></h1><p>This is a deformable-material trial, not a validated cooking benchmark. Bulk flesh and a separate skin deform under real solver contact forces. Cohesive release follows one predefined cut plane. There is no juice-flow model; food textures are procedural.</p></div>'
    impact=cfg["experiment"]["task"]=="impact_nail"
    if impact:
        content='<main><div class="page-intro"><div class="eyebrow">Impact experiment</div><h1>Impact response, <em>under test.</em></h1><p>A scanned claw hammer is installed on an XYZ rig. The Z drive accelerates in air and turns off before the hammer contacts the nail. The material model is uncalibrated; this trial does not establish real pine-driving performance.</p></div>'
    articulated=cfg['experiment']['robot'] in {'fr3','ur10e','fr3_duo','fr3_pair'}
    dual=cfg['experiment']['task']=='dual_tile_wipe'
    if articulated:
        from .experiments import public_definition
        content='<main><div class="page-intro"><div class="eyebrow">Articulated robot / '+escape(cfg['experiment']['name'])+'</div><h1>Contact, <em>under test.</em></h1><p>'+escape(public_definition(cfg['experiment']['id'])['instructions'])+'</p></div>'
    if (OUTPUT/"material_trial.mp4").exists():content+='<section class="card report-section"><h2>Recorded engineering trial</h2><video controls preload="metadata" style="width:100%" src="material_trial.mp4"></video></section>'
    rows=[("Peak contact force",str(obs.get("peak_cut_force_n","—"))+" N"),("Contact work",str(obs.get("cut_work_j","—"))+" J"),("Maximum node displacement",str(obs.get("deformation_mm","—"))+" mm"),("Released cohesive bonds",str(obs.get("released_bonds","—"))+" / "+str(obs.get("total_bonds","—")))]
    if impact:rows=[("Strikes",obs.get("strikes","—")),("Nail depth",str(round(obs.get("nail_depth",0)*1000,2))+" mm"),("Per-hit incoming energy",str(obs.get("impact_energy_j","—"))+" J"),("Downward drive at contact",str(obs.get("motor_force_at_contact_n","—"))+" N"),("Off-axis error",str(obs.get("lateral_error_mm","—"))+" mm")]
    if dual:rows=[('Sampled contact coverage',str(obs.get('coverage_percent','—'))+'%'),('Peak wipe force',str(obs.get('peak_wipe_force_n','—'))+' N'),('Tile displacement relative to holder',str(obs.get('max_tile_error_mm','—'))+' mm'),('Contact time',str(obs.get('contact_seconds','—'))+' s')]
    if dual:
        content+='<section class="card report-section"><h2>Product data and material model</h2><p><a href="/materials/tiling">Open the tile, adhesive and grout workbench ↗</a></p><p>Manufacturer facts are separated from simulation assumptions. This recording is a prepared cleanup subtask, not a full floor/wall installation. Grout residue uses a finite-volume transport approximation; tile fracture is not modeled.</p></section>'
        rows.extend([('Tile mass (estimated)',str(obs.get('tile_mass_kg','—'))+' kg'),('Grout remaining',str(obs.get('grout_remaining_ml','—'))+' mL'),('Grout collected',str(obs.get('grout_collected_ml','—'))+' mL'),('Volume balance error',str(obs.get('grout_mass_balance_error_ml','—'))+' mL')])
    if cfg['experiment']['task']=='tile_dry_transfer':
        rows=[('Dry tiles placed',str(sum(t['placed'] for t in obs.get('tiles',[])))+' / 4'),('Peak vacuum-face contact force',str(obs.get('peak_contact_force_n','—'))+' N'),('Full installation','Not implemented'),('Arm B','Parked; application tools pending')]
    if cfg['experiment']['task']=='tile_adhesive_bedding':
        m=obs.get('mortar',{})
        rows=[('Tiles placed',str(sum(t['placed'] for t in obs.get('tiles',[])))+' / 4'),('Both arms moving',str(obs.get('concurrent_motion_s','—'))+' s'),('Pickup/application overlap',str(obs.get('overlap_mortaring_s','—'))+' s'),('Simulated duration',str(obs.get('simulation_elapsed_s','—'))+' s'),('Adhesive deposited',str(m.get('deposited_ml','—'))+' mL'),('Squeeze-out',str(m.get('squeezeout_ml','—'))+' mL'),('Volume balance error',str(m.get('volume_balance_error_ml','—'))+' mL'),('Modeled bed contact',str(m.get('contact_area_percent','—'))+' %'),('Full installation','Not implemented; no grout finish, cured bond or fracture')]
    if cfg['experiment']['task']=='tile_adhesive_bedding' and obs.get('torque_limit_ratio',0)>=.999:
        rows.append(('Actuator margin','Torque cap reached on an applicator retract; reduce/replan that reference before a varied expert benchmark'))
    if articulated:rows.append(('Peak actuator limit fraction',obs.get('torque_limit_ratio','—')))
    mount='Native pinned articulated robot dynamics with bounded torque actuators and installed flange tools. Materials and tool response are uncalibrated.' if articulated else 'The imported tool is installed on a force-limited XYZ stage.'
    content+='<section class="card report-section"><h2>Measured quantities</h2>'+table(["Quantity","Recorded trial"],rows)+'</section><section class="card report-section"><h2>Model and tool</h2><p>'+mount+'</p><pre style="white-space:pre-wrap">'+escape(__import__("json").dumps(cfg["physics"],indent=2))+'</pre></section></main>'
    OUTPUT.mkdir(parents=True,exist_ok=True)
    for name,active in [("eval_gallery.html","gallery"),("experiment_report.html","reports"),("curation_report.html","reports")]:
        (OUTPUT/name).write_text(page("Engineering trial",content,active,REPORT_CSS))
    (OUTPUT/"REPORT.md").write_text("# "+cfg["experiment"]["name"]+"\n\nUncalibrated tool/material-development trial. No validated success rate. No student training.\n\n"+"\n".join(f"- {a}: {b}" for a,b in rows)+"\n")
    return trial
