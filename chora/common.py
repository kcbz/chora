from __future__ import annotations

import json
import os
import functools
import time
from pathlib import Path

import yaml
from .experiments import DEFAULT, load_config, output_dir

from .paths import WORKSPACE_ROOT as ROOT
OUTPUT = output_dir(os.environ.get("CHORA_EXPERIMENT",DEFAULT))
# Keep model downloads in the project, rather than the user's global cache.
os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / ".cache"))


def dataset_manifest():
    """Bump the protocol version if task physics or expert semantics change."""
    import hashlib
    import importlib.metadata
    staging=config()["experiment"]["task"]!="bowl_pick_place"
    value = {"protocol":"so101-staging-v1" if staging else "so101-rim-v1","expert":"staging-planner-v1" if staging else "observation-planner-v1",
             "simulation":config()["simulation"],"mujoco":importlib.metadata.version("mujoco"),
             "robot_xml_sha256":hashlib.sha256((ROOT/"assets/menagerie/robotstudio_so101/so101.xml").read_bytes()).hexdigest()}
    if config()["experiment"]["task"]=="hammer_nail":
        value.update(protocol="so101-guided-nail-v1",expert="mounted-hammer-planner-v1")
    if config()["experiment"]["task"] in {"hammer_pine","cut_fruit"}:
        value.update(protocol="so101-material-contact-v1",expert="physical-tool-planner-v1",physics=config()["physics"])
    if config()["experiment"]["task"] in {"impact_nail","soft_cut"}:
        value.update(protocol="xyz-tool-contact-v1",expert="xyz-tool-trial-v1",physics=config()["physics"])
    robot=config()['experiment']['robot']
    if robot in {'fr3','ur10e','fr3_duo'}:
        from .arm_control import PATHS
        value.update(protocol='articulated-contact-trial-v1',expert='scripted-articulated-trial-v1',physics=config()['physics'],robot_xml_sha256=hashlib.sha256((ROOT/'assets/menagerie'/PATHS[robot]).read_bytes()).hexdigest())
    if robot=='fr3_pair':
        value.update(protocol='chora-tile-adhesive-bedding-v2',expert='coordinated-bedding-reference-v2',physics=config()['physics'],
          material_catalog_sha256=hashlib.sha256((ROOT/'configs/materials/tiling_baseline.yaml').read_bytes()).hexdigest(),
          visuals_sha256=hashlib.sha256(Path(__file__).with_name('tiling_visuals.py').read_bytes()).hexdigest(),
          surface_manifest_sha256=hashlib.sha256((ROOT/'assets/tiling_surfaces/sources.json').read_bytes()).hexdigest(),
          mortar_sha256=hashlib.sha256(Path(__file__).with_name('mortar_layer.py').read_bytes()).hexdigest(),
          robot_xml_sha256=hashlib.sha256((ROOT/'assets/menagerie/franka_fr3_v2/fr3v2.xml').read_bytes()).hexdigest(),
          scene_sha256=hashlib.sha256(Path(__file__).with_name('tile_installation.py').read_bytes()).hexdigest())
    value["fingerprint"] = hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()
    return value


def verify_dataset():
    expected = dataset_manifest()
    path = OUTPUT/"data/manifest.json"
    existing = read_json(path)
    if existing and existing["fingerprint"] != expected["fingerprint"]:
        raise RuntimeError("Simulation settings or versions changed. Archive outputs/ and collect a fresh dataset.")
    if not existing:
        save_json(path,expected)
    return expected


def exclusive_job(name):
    """One pipeline writer at a time, including jobs launched outside the UI."""
    def decorate(fn):
        @functools.wraps(fn)
        def wrapped(*args,**kwargs):
            import fcntl
            OUTPUT.mkdir(parents=True,exist_ok=True)
            with (OUTPUT/"pipeline.lock").open("a+") as lock:
                try:
                    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError:
                    raise RuntimeError("Another pipeline stage is already running") from None
                state = {"name":name,"pid":os.getpid(),"id":str(os.getpid()),"running":True,
                         "status":"running","started_at":time.time(),"progress":{"message":"Starting…"}}
                save_json(OUTPUT/"active_job.json",state)
                save_json(OUTPUT/"jobs"/f"{name}.json",state)
                try:
                    result = fn(*args,**kwargs)
                except BaseException as error:
                    state = read_json(OUTPUT/"active_job.json",state)
                    state.update(status="cancelled" if isinstance(error,KeyboardInterrupt) else "failed",error=str(error) or "Stopped")
                    raise
                else:
                    state = read_json(OUTPUT/"active_job.json",state)
                    state["status"] = "complete"
                    return result
                finally:
                    state.update(running=False,finished_at=time.time())
                    save_json(OUTPUT/"active_job.json",state)
                    save_json(OUTPUT/"jobs"/f"{name}.json",state)
        return wrapped
    return decorate


def job_progress(current=None,total=None,message="",**details):
    """Publish progress only from the process holding the pipeline lock."""
    state = read_json(OUTPUT/"active_job.json",{})
    if state.get("pid")!=os.getpid() or not state.get("running"):
        return
    state["progress"] = {"current":current,"total":total,"message":message,**details}
    state["updated_at"] = time.time()
    save_json(OUTPUT/"active_job.json",state)


def config():
    return load_config(os.environ.get("CHORA_EXPERIMENT",DEFAULT))


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.replace(path)


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except FileNotFoundError:
        return default


def append_jsonl(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(value, allow_nan=False) + "\n")


def read_jsonl(path):
    with Path(path).open() as f:
        return [json.loads(line) for line in f if line.strip()]
