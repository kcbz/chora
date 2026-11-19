from __future__ import annotations

import hashlib
import json
import math
import time
from collections import Counter, defaultdict

import numpy as np

from .common import OUTPUT, append_jsonl, config, read_jsonl, save_json, exclusive_job, verify_dataset, job_progress
from .policy import make_expert, messages
from .sim import make_simulation


def rollout(sim, policy, seed, timing=False):
    obs = sim.reset(seed)
    history, steps = [], []
    while not sim.finished:
        started = time.perf_counter()
        action = policy.act(obs,history)
        decision_seconds = time.perf_counter()-started
        before = obs
        obs = sim.step(action)
        steps.append({"observation":before,"action":action,"result":obs["last_result"]})
        if timing:
            steps[-1]["policy_seconds"] = decision_seconds
            if hasattr(policy,"last_text"):
                steps[-1]["raw_model_output"] = policy.last_text
        history.append(action)
    path = sum(np.linalg.norm(np.array(b["observation"]["tcp"])-a["observation"]["tcp"]) for a,b in zip(steps,steps[1:]))
    margin = min(np.array(sim.cfg["target_half_size"])-np.abs(sim.bowl[:2]-sim.cfg["target"][:2])-sim.object_radius)
    episode={"seed":seed,"success":sim.success,"steps":steps,"final_observation":obs,
            "path_length":float(path),"placement_margin":float(margin),"collisions":sim.collision_count}
    if obs.get("task") in {"hammer_nail","hammer_pine","impact_nail"}:
        episode.update(task=obs["task"],nail_depth=sim.depth,strikes=sim.strikes)
        episode.pop("placement_margin")
    if obs.get("task")=="cut_fruit":
        episode.update(task="cut_fruit",cut_work_j=sim.cut_work,cut_progress=sim.cut_progress)
        episode.pop("placement_margin")
    if obs.get('task') in {'soft_cut','dual_tile_wipe','tile_dry_transfer','tile_adhesive_bedding'}:
        episode.update(task=obs['task'],benchmark_validated=False)
        episode.pop('placement_margin')
    return episode


@exclusive_job("check")
def oracle_check(n=50, seed_start=10000):
    if not 1 <= n <= 1000:
        raise ValueError("Expert check requires 1..1000 episodes")
    sim = make_simulation()
    episodes = []
    job_progress(0,n,"Preparing the expert check…")
    for seed in range(seed_start,seed_start+n):
        episode = rollout(sim,make_expert(seed),seed)
        episodes.append({"seed":seed,"success":episode["success"],"result":sim.result})
        completed = len(episodes)
        passed = sum(e["success"] for e in episodes)
        job_progress(completed,n,f"Checked {completed} of {n} scenes · {passed} successful")
        if completed%5==0:
            print(f"Expert check: {completed}/{n}; successful {passed}",flush=True)
    successes = sum(e["success"] for e in episodes)
    report = {"policy":"expert","episodes":n,"successes":successes,"success_rate":successes/n,"trials":episodes}
    save_json(OUTPUT/"oracle.json",report)
    print(f"Expert: {successes}/{n} ({successes/n:.0%})",flush=True)
    if successes/n < .9:
        raise RuntimeError("Expert below 90%; fix simulation before generating training data")
    job_progress(n,n,f"Expert check passed · {successes} of {n} scenes successful")
    return report


@exclusive_job("generate")
def generate(n=None):
    verify_dataset()
    cfg = config()
    n = n or cfg["data"]["attempts"]
    if not 1 <= n <= 10000:
        raise ValueError("Training generation is restricted to seeds 0..9999")
    path = OUTPUT/"data/raw.jsonl"
    existing = read_jsonl(path) if path.exists() else []
    seen = {e["seed"] for e in existing}
    successes = sum(e["success"] for e in existing if e["seed"]<n)
    already = sum(seed in seen for seed in range(n))
    completed = already
    job_progress(already,n,f"Resuming from {already} saved demonstrations")
    if already==n:
        save_json(OUTPUT/"generation.json",{"attempts":n,"successes":successes,"requested":n})
        message = f"All {n:,} demonstrations are already saved. No additional collection was needed."
        job_progress(n,n,message)
        print(message,flush=True)
        return path
    sim = make_simulation(cfg)
    for seed in range(n):
        if seed in seen:
            continue
        episode = rollout(sim,make_expert(seed),seed)
        append_jsonl(path,episode)
        successes += episode["success"]
        completed += 1
        job_progress(completed,n,f"Collected {completed:,} of {n:,} demonstrations · {successes:,} successful")
        if completed%50 == 0 or completed==n:
            progress = {"attempts":completed,"successes":successes,"requested":n}
            save_json(OUTPUT/"generation.json",progress)
            print(f"Demonstrations: {completed}/{n}; successes {successes}",flush=True)
    return path


def object_observation(episode):
    obs=episode["steps"][0]["observation"]
    return obs.get("object",obs.get("bowl")),obs.get("object_yaw",obs.get("bowl_yaw"))


def bin_key(episode,cfg):
    o = episode["steps"][0]["observation"]
    cells = []
    for i,axis in enumerate(("spawn_x","spawn_y")):
        lo,hi = cfg["simulation"][axis]
        cells.append(int(np.clip(4*(o.get("object",o.get("bowl"))[i]-lo)/(hi-lo),0,3)))
    cells.append(int(np.clip(4*(o.get("object_yaw",o.get("bowl_yaw"))+math.pi)/(2*math.pi),0,3)))
    return tuple(cells)


@exclusive_job("curate")
def curate():
    job_progress(0,4,"Reading collected demonstrations…")
    manifest = verify_dataset()
    cfg = config()
    raw_path = OUTPUT/"data/raw.jsonl"
    raw = read_jsonl(raw_path)
    dropped = Counter()
    groups = defaultdict(list)
    signatures = set()
    for e in raw:
        reason = None
        if not e["success"]:
            reason = "failed_episode"
        elif e["collisions"] or any(s["result"] not in ("ok","success") for s in e["steps"]):
            reason = "unclean_action"
        elif e.get("task") not in {"hammer_nail","hammer_pine","impact_nail","cut_fruit"} and e["placement_margin"] < .002:
            reason = "edge_placement"
        elif not 0 <= e["seed"] < 10000:
            reason = "reserved_seed"
        duplicate_resolution=.0005 if e.get("task")=="cut_fruit" else .002
        signature = tuple(round(x/duplicate_resolution) for x in object_observation(e)[0][:2]) + (round(object_observation(e)[1]/.1),)
        if reason is None and signature in signatures:
            reason = "near_duplicate_start"
        if reason:
            dropped[reason] += 1
            continue
        signatures.add(signature)
        e["quality"] = float(100*e.get("nail_depth",e.get("placement_margin",0)) + e.get("cut_progress",0)-.1*len(e["steps"])-e["path_length"])
        groups[bin_key(e,cfg)].append(e)
    passed = sum(len(v) for v in groups.values())
    job_progress(1,4,f"Quality filters kept {passed:,} episodes")
    selected, train, valid = [], [], []
    rng = np.random.default_rng(cfg["seed"])
    for key in sorted(groups):
        values = sorted(groups[key],key=lambda e:-e["quality"])[:cfg["data"]["max_per_bin"]]
        selected.extend(values)
        order = rng.permutation(len(values))
        nval = max(1,round(len(values)*cfg["data"]["validation_fraction"])) if len(values)>1 else 0
        valid.extend(values[i] for i in order[:nval])
        train.extend(values[i] for i in order[nval:])
    if len(selected)<cfg["data"]["min_episodes"]:
        raise RuntimeError(f"Only {len(selected)} clean diverse episodes; need {cfg['data']['min_episodes']}. Generate more first.")
    job_progress(2,4,"Balancing scene coverage and splitting whole episodes…")
    directory = OUTPUT/"data/curated"
    directory.mkdir(parents=True,exist_ok=True)
    for name,eps in (("train",train),("valid",valid)):
        with (directory/f"{name}.jsonl").open("w") as f:
            for e in eps:
                history = []
                for step in e["steps"]:
                    f.write(json.dumps({"messages":messages(step["observation"],history,step["action"]),"episode_seed":e["seed"]})+"\n")
                    history.append(step["action"])
    report = {"dataset_fingerprint":manifest["fingerprint"],"raw_episodes":len(raw),"passed_quality":passed,"selected_episodes":len(selected),
              "train_episodes":len(train),"validation_episodes":len(valid),
              "training_pairs":sum(len(e["steps"]) for e in train),
              "validation_pairs":sum(len(e["steps"]) for e in valid),
              "drop_reasons":dict(dropped),"occupied_bins":len(groups),"total_bins":16 if cfg["experiment"]["task"] in {"hammer_nail","hammer_pine","impact_nail","cut_fruit"} else 64,
              "train_seeds":[e["seed"] for e in train],"validation_seeds":[e["seed"] for e in valid],
              "raw_sha256":hashlib.sha256(raw_path.read_bytes()).hexdigest(),
              "quality_score":("100 * nail_depth_m (hammer) or cut_progress (cutting) - 0.1 * decisions - tcp_path_m" if cfg["experiment"]["task"] in {"hammer_nail","hammer_pine","impact_nail","cut_fruit"} else "100 * placement_margin_m - 0.1 * decisions - tcp_path_m"),
              "action_counts":dict(Counter(s["action"]["action"] for e in selected for s in e["steps"]))}
    save_json(OUTPUT/"curation.json",report)
    job_progress(3,4,"Writing the curated dataset and coverage report…")
    save_json(directory/"episodes.json",selected)
    (OUTPUT/"curation_report.md").write_text(
        "# Chora curation\n\n"+"\n".join(f"- {k}: {v}" for k,v in report.items() if not k.endswith("seeds"))+"\n\nEpisode-level, stratified split; held-out test seeds begin at 100000.\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axes = plt.subplots(1,2,figsize=(10,4))
    for ax,eps,title in zip(axes,[raw,selected],["Raw demonstrations","Curated demonstrations"]):
        xy = np.array([object_observation(e)[0][:2] for e in eps])
        ax.hist2d(xy[:,0],xy[:,1],bins=4,cmap="Greens")
        ax.set(title=title,xlabel="x (m)",ylabel="y (m)")
    fig.tight_layout(); fig.savefig(OUTPUT/"coverage.png",dpi=140); plt.close(fig)
    print(f"Curated {len(selected)} episodes; {report['training_pairs']} train pairs, {report['validation_pairs']} validation pairs",flush=True)
    from .report import build_report
    build_report()
    job_progress(4,4,f"Selected {len(selected):,} episodes · {report['training_pairs']:,} training examples")
    return report
