"""MLX QLoRA training and inference, sized for a 16 GB Apple Silicon Mac."""
from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

from .common import OUTPUT, ROOT, append_jsonl, config, read_json, read_jsonl, save_json, exclusive_job, verify_dataset, job_progress
from .policy import messages
from .sim import validate_action

MODEL_DIR = ROOT/"assets/qwen"


def assistant_loss(model,batch,lengths,completion_length=96):
    """Project only assistant positions through Qwen's large vocabulary head.

    The full prompt still flows through every transformer layer. Gathering
    hidden states before the output projection avoids a vocabulary-sized logits
    tensor for tokens whose loss is masked out anyway. Padding is never a label.
    """
    import mlx.core as mx
    import mlx.nn as nn
    inputs,targets = batch[:,:-1],batch[:,1:]
    hidden = model.model(inputs)
    indices = lengths[:,0:1]-1+mx.arange(completion_length)[None,:]
    safe = mx.clip(indices,0,inputs.shape[1]-1)
    selected = mx.take_along_axis(hidden,safe[:,:,None],axis=1)
    logits = model.model.embed_tokens.as_linear(selected) if model.args.tie_word_embeddings else model.lm_head(selected)
    labels = mx.take_along_axis(targets,safe,axis=1)
    mask = (indices>=0)&(indices<lengths[:,1:2]-1)
    count = mask.sum()
    loss = (nn.losses.cross_entropy(logits,labels)*mask).astype(mx.float32).sum()/count
    return loss,count


def download_model():
    from huggingface_hub import snapshot_download
    snapshot_download(config()["model"]["id"],local_dir=MODEL_DIR,
                      allow_patterns=["*.json","*.safetensors","*.txt","*.jinja","*.model","LICENSE*"])
    print(f"Model downloaded to {MODEL_DIR}",flush=True)


def prefix(tokenizer,msgs):
    return tokenizer.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True,enable_thinking=False)


class TokenSamples:
    def __init__(self,samples):
        self.samples = samples
    def __len__(self):
        return len(self.samples)
    def __getitem__(self,index):
        return self.samples[index]
    def process(self,item):
        return item


def token_dataset(path,tokenizer,max_length):
    samples = []
    for row in read_jsonl(path):
        msgs = row["messages"]
        prompt = prefix(tokenizer,msgs[:-1])
        start = tokenizer.encode(prompt,add_special_tokens=False)
        full = tokenizer.encode(prompt+msgs[-1]["content"]+tokenizer.eos_token,add_special_tokens=False)
        if len(full)>max_length:
            raise ValueError(f"Example length {len(full)} exceeds {max_length}; increase max_seq_length rather than silently truncate the action")
        samples.append((full,len(start)))
    return TokenSamples(samples)


def publish_training_run(run, output):
    """Promote a completed adapter and archive the former run and student scores."""
    if not (run / "adapter/adapters.safetensors").is_file():
        raise RuntimeError("Training produced no adapter")
    prior_paths = ["adapter", "training.json", "training_metrics.jsonl",
                   "selection.json", "training_curve.png", "evaluation/student"]
    results = read_json(output / "results.json", {})
    if "student" in results:
        prior_paths.append("results.json")
        save_json(run / "results.json", {k: v for k, v in results.items() if k != "student"})
    prior_paths = [name for name in prior_paths if (output / name).exists()]
    archive = None
    if prior_paths:
        archives = output / "training_runs"
        archives.mkdir(exist_ok=True)
        archive = Path(tempfile.mkdtemp(prefix=time.strftime("%Y%m%d_%H%M%S-"), dir=archives))

    moved = []
    def move(source, destination):
        destination.parent.mkdir(parents=True, exist_ok=True)
        source.rename(destination)
        moved.append((source, destination))

    try:
        for name in prior_paths:
            move(output / name, archive / name)
        for path in run.iterdir():
            move(path, output / path.name)
    except BaseException:
        for source, destination in reversed(moved):
            destination.rename(source)
        raise
    return archive


@exclusive_job("train")
def train(iterations=None, resume=False, batch_size=None, *, replace=False):
    verify_dataset()
    active = OUTPUT / "adapter/adapters.safetensors"
    if active.is_file() and not (resume or replace):
        raise RuntimeError("A trained adapter exists. Use --replace for a fresh run or --resume to continue.")
    if resume and not active.is_file():
        raise RuntimeError("No saved adapter to resume")
    with tempfile.TemporaryDirectory(prefix=".training-", dir=OUTPUT) as directory:
        run = Path(directory)
        summary = fit_training(run, iterations, active if resume else None, batch_size)
        publish_training_run(run, OUTPUT)
    from .report import build_report
    build_report()
    job_progress(summary["iteration"], summary["requested_iterations"], "Model adapter saved · ready to run")
    return summary


def fit_training(run, iterations, previous, batch_size):
    job_progress(message="Loading the local Qwen model…")
    import mlx.core as mx
    from mlx.utils import tree_flatten
    from mlx_lm import load
    from mlx_lm.lora import CONFIG_DEFAULTS
    from mlx_lm.tuner.trainer import train as fit, TrainingArgs
    from mlx_lm.tuner.utils import linear_to_lora_layers
    from mlx_lm.utils import save_config
    import mlx.optimizers as optim
    from mlx_lm.tuner.callbacks import TrainingCallback
    cfg = config()
    import numpy as np
    np.random.seed(cfg["seed"])
    curated = read_json(OUTPUT/"curation.json")
    if not curated or curated["selected_episodes"] < cfg["data"]["min_episodes"]:
        raise RuntimeError("Generate and curate at least 200 episodes before training")
    if not (MODEL_DIR/"config.json").exists():
        raise RuntimeError("Run `python -m chora download` first")
    adapter = run / "adapter"
    metrics = run / "training_metrics.jsonl"
    adapter.mkdir()
    model,tokenizer = load(str(MODEL_DIR))
    t = cfg["training"]
    train_set = token_dataset(OUTPUT/"data/curated/train.jsonl",tokenizer,t["max_seq_length"])
    val_set = token_dataset(OUTPUT/"data/curated/valid.jsonl",tokenizer,t["max_seq_length"])
    n = iterations or t["iterations"]
    if not 1 <= n <= 100000 or not 1 <= (batch_size or t["batch_size"]) <= 16:
        raise ValueError("Iterations must be 1..100000 and batch size 1..16")
    args = SimpleNamespace(**{**CONFIG_DEFAULTS,"model":str(MODEL_DIR),"data":str(OUTPUT/"data/curated"),"seed":cfg["seed"],
        "iters":n,"batch_size":batch_size or t["batch_size"],"learning_rate":t["learning_rate"] if previous is None else t["learning_rate"]*.5,
        "num_layers":t["num_layers"],"max_seq_length":t["max_seq_length"],
        "adapter_path":str(OUTPUT/"adapter"),"save_every":50,"steps_per_report":10,
        "steps_per_eval":100,"val_batches":20,"grad_checkpoint":True,"mask_prompt":True,
        "resume_adapter_file":str(previous) if previous else None,
        "lora_parameters":{"rank":t["rank"],"scale":32.0,"dropout":0.0}})
    model.freeze()
    mx.random.seed(cfg["seed"])
    linear_to_lora_layers(model,args.num_layers,args.lora_parameters)
    if previous:
        model.load_weights(str(previous),strict=False)
    save_config(vars(args),adapter/"adapter_config.json")
    fit_args = TrainingArgs(batch_size=args.batch_size,iters=n,val_batches=20,
        steps_per_report=10,steps_per_eval=100,steps_per_save=50,
        adapter_file=adapter/"adapters.safetensors",max_seq_length=t["max_seq_length"],grad_checkpoint=True)
    max_completion = max(len(tokens)-offset for dataset in (train_set,val_set) for tokens,offset in dataset.samples)
    from functools import partial
    objective = partial(assistant_loss,completion_length=max_completion)
    started = time.monotonic()
    summary = {"status":"running","model":cfg["model"]["id"],"backend":"MLX QLoRA","requested_iterations":n,"iteration":0,"batch_size":args.batch_size,"resumed_from":str(previous) if previous else None}
    save_json(run/"training.json",summary)
    job_progress(0,n,"Preparing training examples…")
    class BudgetReached(Exception):
        pass
    class Progress(TrainingCallback):
        def on_train_loss_report(self,info):
            info = {k:float(v) if not isinstance(v,int) else v for k,v in info.items()}
            info["elapsed_seconds"] = time.monotonic()-started
            append_jsonl(metrics,{"type":"train",**info})
            summary.update(info)
            summary["estimated_total_seconds"] = info["elapsed_seconds"]*n/info["iteration"]
            save_json(run/"training.json",summary)
            job_progress(info["iteration"],n,f"Training step {info['iteration']:,} of {n:,}")
            print(f"Training {info['iteration']}/{n}: loss {info['train_loss']:.4f}, {info['peak_memory']:.2f} GB peak",flush=True)
            if info["elapsed_seconds"] >= t["max_minutes"]*60:
                raise BudgetReached
        def on_val_loss_report(self,info):
            append_jsonl(metrics,{"type":"validation",**info})
            summary["validation_loss"] = info["val_loss"]
            save_json(run/"training.json",summary)
            job_progress(summary["iteration"],n,"Checking the model on validation examples…")
    try:
        fit(model,optim.Adam(learning_rate=args.learning_rate),train_set,val_dataset=val_set,
            args=fit_args,loss=objective,training_callback=Progress())
        summary["status"] = "complete"
    except BudgetReached:
        mx.save_safetensors(str(adapter/"adapters.safetensors"),dict(tree_flatten(model.trainable_parameters())))
        summary["status"] = "stopped_with_adapter"
    except Exception as e:
        summary.update(status="failed",error=str(e))
        raise
    finally:
        summary["wall_seconds"] = time.monotonic()-started
        save_json(run/"training.json",summary)
    return summary


class Student:
    def __init__(self,base=False,adapter_path=None):
        from mlx_lm import load
        from mlx_lm.sample_utils import make_sampler
        adapter_path = Path(adapter_path or OUTPUT/"adapter")
        if not base and not (adapter_path/"adapters.safetensors").exists():
            raise RuntimeError("No trained adapter found. Train first or evaluate with --base")
        self.model,self.tokenizer = load(str(MODEL_DIR),adapter_path=None if base else str(adapter_path))
        self.sampler = make_sampler(temp=0)
        self.last_text = ""

    def act(self,obs,history):
        from mlx_lm import generate
        self.last_text = generate(self.model,self.tokenizer,
            prompt=prefix(self.tokenizer,messages(obs,history)),max_tokens=config()["model"]["max_tokens"],
            sampler=self.sampler,verbose=False)
        try:
            action = json.loads(self.last_text)
            if not isinstance(action,dict):raise ValueError("Expected an action object")
            if obs.get("task") in {"hammer_nail","hammer_pine","cut_fruit"}:
                from .hammer import validate_hammer_action
                if action.get("action") in {"strike","slice"}:action["depth"]/=1000
                if obs.get("task")=="cut_fruit":
                    from .cutting import validate_cut_action
                    validate_cut_action(action)
                else:validate_hammer_action(action)
            else:validate_action(action)
            if action["action"]=="move_to":
                for key in "xyz":
                    action[key] /= 1000
            return action
        except (json.JSONDecodeError,ValueError,TypeError):
            return {"action":"invalid_model_output"}


@exclusive_job("select")
def select_checkpoint(checkpoint):
    """Freeze a checkpoint before final evaluation, preserving the latest weights."""
    import hashlib
    import shutil
    from datetime import datetime, timezone
    source = OUTPUT/"adapter"/f"{checkpoint:07d}_adapters.safetensors"
    if not source.exists():
        raise ValueError("Checkpoint does not exist in this run")
    adapter = OUTPUT/"adapter/adapters.safetensors"
    shutil.copy2(adapter,OUTPUT/"adapter/last_training_adapters.safetensors")
    temporary = adapter.with_suffix(".selected.tmp")
    shutil.copy2(source,temporary)
    temporary.replace(adapter)
    validation = [r for r in read_jsonl(OUTPUT/"training_metrics.jsonl") if r["type"]=="validation" and r["iteration"]<checkpoint]
    selected = {"checkpoint":checkpoint,"sha256":hashlib.sha256(adapter.read_bytes()).hexdigest(),
                "selected_at":datetime.now(timezone.utc).isoformat(),
                "criterion":"Development rollouts and validation loss, before final student evaluation",
                "preceding_validation":validation[-1] if validation else None}
    save_json(OUTPUT/"selection.json",selected)
    summary = read_json(OUTPUT/"training.json",{})
    summary.update(status="complete_selected",selected_checkpoint=checkpoint)
    summary["last_validation_loss"] = summary.get("validation_loss")
    if selected["preceding_validation"]:
        summary["validation_loss"] = selected["preceding_validation"]["val_loss"]
    save_json(OUTPUT/"training.json",summary)
    return selected


def preview(checkpoint=None,n=5):
    """Development scenes only. Never touch final test seeds during training."""
    import shutil
    from .episodes import rollout
    from .sim import make_simulation
    from .visualization import Recorder
    checkpoints = sorted((OUTPUT/"adapter").glob("[0-9]*_adapters.safetensors"))
    source = OUTPUT/"adapter"/f"{checkpoint:07d}_adapters.safetensors" if checkpoint else checkpoints[-1]
    if not source.exists():
        raise RuntimeError("Requested checkpoint has not been saved")
    step = int(source.name.split("_")[0])
    training = read_json(OUTPUT/"training.json",{})
    run = Path(training["resumed_from"]).parent.parent.name if training.get("resumed_from") else "base_run"
    out = OUTPUT/"previews"/run/f"step_{step}"
    (out/"adapter").mkdir(parents=True,exist_ok=True)
    shutil.copy2(source,out/"adapter/adapters.safetensors")
    shutil.copy2(OUTPUT/"adapter/adapter_config.json",out/"adapter/adapter_config.json")
    policy = Student(adapter_path=out/"adapter")
    sim = make_simulation()
    episodes = []
    for i in range(n):
        seed = 20000+i
        sim.reset(seed)
        r = Recorder(sim,out/f"seed_{seed}.mp4") if i==0 else None
        try:
            e = rollout(sim,policy,seed)
        finally:
            if r: r.close()
        episodes.append(e)
        print(f"Development preview step {step}: {i+1}/{n}, successes {sum(e['success'] for e in episodes)}",flush=True)
    summary = {"checkpoint":step,"episodes":n,"successes":sum(e["success"] for e in episodes),"seed_start":20000,"is_final_test":False}
    save_json(out/"episodes.json",episodes)
    save_json(out/"summary.json",summary)
    return summary


@exclusive_job("replay")
def replay(policy_name="student",failures=2):
    """Re-render scored action traces; verify every observation and outcome.

    Model inference and metrics are never rerun or altered by this operation.
    Extra failure recordings make it possible to inspect unsuccessful episodes.
    """
    from .sim import make_simulation
    from .visualization import Recorder
    path = OUTPUT/"evaluation"/policy_name/"episodes.json"
    episodes = read_json(path)
    if not episodes:
        raise ValueError("Evaluate this policy first")
    sim = make_simulation()
    remaining = failures
    recorded = 0
    for e in episodes:
        video = e.get("video")
        if video is None:
            if e["success"] or remaining<=0:
                continue
            video = f"evaluation/{policy_name}/seed_{e['seed']}.mp4"
            remaining -= 1
        target = OUTPUT/video
        temporary = target.with_name(target.stem+".rerender.mp4")
        sim.reset(e["seed"])
        recorder = Recorder(sim,temporary)
        try:
            for step in e["steps"]:
                if sim.observe()!=step["observation"]:
                    raise RuntimeError(f"Replay state mismatch on seed {e['seed']}; original clip retained")
                obs = sim.step(step["action"])
                if obs["last_result"]!=step["result"]:
                    raise RuntimeError(f"Replay action mismatch on seed {e['seed']}; original clip retained")
            if sim.observe()!=e["final_observation"] or sim.success!=e["success"]:
                raise RuntimeError(f"Replay outcome mismatch on seed {e['seed']}; original clip retained")
        finally:
            recorder.close()
        temporary.replace(target)
        e["video"] = video
        e["replay_verified"] = True
        recorded += 1
        print(f"{policy_name}: verified high-resolution replay {e['seed']} ({recorded})",flush=True)
    save_json(path,episodes)
    from .report import build_report
    build_report()
    return recorded


@exclusive_job("evaluate")
def evaluate(n=None,policy_name="student",record=3):
    from .episodes import rollout
    from .policy import make_expert
    from .sim import make_simulation
    from .gallery import save_thumbnail
    from .visualization import Camera, Recorder
    cfg = config(); n = n or cfg["evaluation"]["episodes"]
    if not 1<=n<=100:
        raise ValueError("Evaluation has at most 100 held-out seeds")
    job_progress(0,n,"Loading the policy for evaluation…")
    sim = make_simulation(cfg)
    policy = None if policy_name=="expert" else Student(base=policy_name=="base")
    results = []
    out = OUTPUT/"evaluation"/policy_name
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".evaluation-", dir=OUTPUT) as directory:
        stage = Path(directory)
        trial_dir = stage / "evaluation" / policy_name
        trial_dir.mkdir(parents=True)
        camera = Camera(sim,640,480)
        try:
            for i in range(n):
                seed = cfg["evaluation"]["seed_start"]+i
                sim.reset(seed)
                recorder = Recorder(sim,trial_dir/f"seed_{seed}.mp4") if i<record else None
                try:
                    e = rollout(sim,make_expert(seed,cfg) if policy is None else policy,seed,timing=True)
                finally:
                    if recorder: recorder.close()
                e["video"] = f"evaluation/{policy_name}/seed_{seed}.mp4" if recorder else None
                save_thumbnail(camera,sim,e,policy_name,stage)
                results.append(e)
                job_progress(i+1,n,f"Evaluated {i+1} of {n} scenes · {sum(x['success'] for x in results)} successful")
                print(f"{policy_name}: {i+1}/{n}; successes {sum(x['success'] for x in results)}",flush=True)
        finally:
            camera.close()
        save_json(trial_dir/"episodes.json",results)
        previous = stage / "previous"
        if out.exists():
            out.rename(previous)
        try:
            trial_dir.rename(out)
        except BaseException:
            if previous.exists():
                previous.rename(out)
            raise
    failures = {}
    for e in results:
        if not e["success"]:
            reason = next((s["result"] for s in e["steps"] if s["result"] not in ("ok","success")),"missed_target")
            failures[reason] = failures.get(reason,0)+1
    summary = {"policy":policy_name,"episodes":n,"successes":sum(e["success"] for e in results),
               "failure_breakdown":failures,"seed_start":100000,"complete_benchmark":n==100,
               "mean_steps_to_success":sum(len(e["steps"]) for e in results if e["success"])/max(1,sum(e["success"] for e in results))}
    decisions = [s for e in results for s in e["steps"]]
    summary["decisions"] = len(decisions)
    summary["mean_policy_seconds"] = sum(s["policy_seconds"] for s in decisions)/len(decisions)
    summary["invalid_actions"] = sum(s["result"]=="invalid_action" for s in decisions)
    summary["rejected_moves"] = sum(s["result"]=="rejected_unreachable" for s in decisions)
    from .episodes import bin_key
    bins = {}
    for e in results:
        key = ",".join(map(str,bin_key(e,cfg)))
        row = bins.setdefault(key,{"episodes":0,"successes":0})
        row["episodes"] += 1
        row["successes"] += e["success"]
    summary["workspace_bins"] = bins
    save_json(out/"summary.json",summary)
    all_results = read_json(OUTPUT/"results.json",{})
    all_results[policy_name] = summary
    save_json(OUTPUT/"results.json",all_results)
    from .report import build_report
    build_report()
    job_progress(n,n,f"Evaluation complete · {summary['successes']} of {n} scenes successful")
    return summary
