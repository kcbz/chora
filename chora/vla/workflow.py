"""Versioned lab workflow for the Standard bowl visual action pilot."""

from argparse import ArgumentParser
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from ..common import OUTPUT, ROOT, config, exclusive_job, job_progress, read_json, save_json
from ..experiments import DEFAULT

STAGES = ("vla-check", "vla-record", "vla-prepare", "vla-train", "vla-evaluate")
INITIAL = {
    "recording": "runs/bowl-tick-pilot-v1",
    "dataset": "datasets/bowl-pilot-v1",
    "training": "training/bowl-direct-familiar-v1",
    "familiar": "rollouts/bowl-direct-familiar-v1",
}


def artifact(directory, value, parent):
    if not isinstance(value, str):
        return None
    base = (directory / "vla" / parent).resolve()
    path = (directory / "vla" / value).resolve()
    return path if path.is_relative_to(base) and path.is_dir() else None


def state(directory):
    return read_json(directory / "vla/workflow.json", INITIAL.copy())


def details(directory):
    saved = state(directory)

    def load(key, parent, file):
        path = artifact(directory, saved.get(key), parent)
        return read_json(path / file, {}) if path else {}

    return {
        "expert": read_json(directory / "vla/expert-check.json", {}),
        "recording": load("recording", "runs", "manifest.json"),
        "dataset": load("dataset", "datasets", "dataset.json"),
        "training": load("training", "training", "report.json"),
        "familiar": load("familiar", "rollouts", "report.json"),
        "unseen": load("unseen", "rollouts", "report.json"),
    }


def summary(report):
    if report.get("status") != "complete":
        return None
    trials = report.get("trials", [])
    return dict(successes=sum(t["success"] is True for t in trials), episodes=len(trials),
                collisions=sum(t.get("collisions", 0) for t in trials))


def familiar_passed(score):
    return score["successes"] == score["episodes"] == 2 and score["collisions"] == 0


def evaluation_outcome(familiar, unseen):
    if unseen:
        passed = (unseen["episodes"] >= 10 and unseen["successes"] / unseen["episodes"] >= .8
                  and unseen["collisions"] == 0)
        return "passed" if passed else "did-not-pass"
    if familiar and not familiar_passed(familiar):
        return "did-not-pass"
    return None


def status(directory):
    d = details(directory)
    recording = d["recording"]
    episodes = d["dataset"].get("episodes", [])
    training = d["training"]
    familiar, unseen = summary(d["familiar"]), summary(d["unseen"])
    return {
        "ready": (ROOT / ".venv-vla/bin/python").is_file(),
        "expert": d["expert"] or None,
        "recording": {"successes": recording["successes"], "episodes": len(recording["episodes"])}
        if recording.get("status") == "complete" else None,
        "dataset": {"train": sum(e["split"] == "train" for e in episodes),
                    "validation": sum(e["split"] == "validation" for e in episodes)} if episodes else None,
        "trained": training.get("final", {}).get("step") if training.get("status") == "complete" else None,
        "familiar": familiar,
        "unseen": unseen,
        "evaluation_outcome": evaluation_outcome(familiar, unseen),
    }


def directory(parent, label):
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return OUTPUT / "vla" / parent / f"lab-{label}-{stamp}-{uuid4().hex[:6]}"


def verify_expert():
    from .env import BowlTickEnv, TickExpert

    env = BowlTickEnv(images=False)
    successes = 0
    try:
        for i, seed in enumerate(range(24000, 24010), 1):
            env.reset(seed)
            expert = TickExpert(env, seed)
            while not env.done:
                action = expert.act()
                env.step_tick(action)
                if expert.exhausted and not expert.pending:
                    break
            successes += env.success
            job_progress(i,10,f"Checked {i}/10 visual expert episodes")
    finally:
        env.close()
    result = {"successes": successes, "episodes": 10}
    save_json(OUTPUT / "vla/expert-check.json", result)
    return result


def checkpoint_path(training_dir):
    report = read_json(training_dir / "report.json", {})
    value = report.get("final", {}).get("checkpoint")
    if not isinstance(value, str):
        raise ValueError("Train a camera policy first.")
    checkpoint = (ROOT / value).resolve()
    if not checkpoint.is_relative_to(training_dir) or not (checkpoint / "model.safetensors").is_file():
        raise ValueError("This run's trained checkpoint could not be found.")
    return checkpoint


def run(stage):
    if config()["experiment"]["id"] != DEFAULT:
        raise ValueError("The camera action pipeline supports Bowl · Standard platform.")
    if stage == "vla-check":
        return verify_expert()
    saved = state(OUTPUT)
    current = status(OUTPUT)
    if stage == "vla-record":
        from .record import collect

        expert = current["expert"]
        if not expert or expert["successes"] < .9 * expert["episodes"]:
            raise ValueError("Pass the fixed-tick expert check before recording.")
        job_progress(message="Recording two camera views and joint targets for 12 expert episodes…")
        result = collect(episodes=12, seed_start=23000, output=directory("runs", "record"))
        saved = {"recording": str(result.relative_to(OUTPUT / "vla"))}
    elif stage == "vla-prepare":
        from .dataset import prepare

        source = artifact(OUTPUT, saved.get("recording"), "runs")
        if not source or not current["recording"]:
            raise ValueError("Record visual demonstrations first.")
        job_progress(message="Splitting complete episodes and normalizing joint targets…")
        result = prepare([source], directory("datasets", "prepare"))
        saved = {"recording": saved["recording"], "dataset": str(result.relative_to(OUTPUT / "vla"))}
    elif stage == "vla-train":
        from .train import run as train

        dataset = artifact(OUTPUT, saved.get("dataset"), "datasets")
        if not dataset or not current["dataset"]:
            raise ValueError("Prepare a visual dataset first.")
        result = train.__wrapped__(str(dataset), steps=1200, batch_size=8, overfit_windows=0,
                                   direct_pilot=True, device="mps", eval_every=1200,
                                   output=str(directory("training", "train")))
        saved = {key: saved[key] for key in ("recording", "dataset") if key in saved}
        saved["training"] = str(result.relative_to(OUTPUT / "vla"))
    else:
        from .runner import run as rollout

        dataset = artifact(OUTPUT, saved.get("dataset"), "datasets")
        training = artifact(OUTPUT, saved.get("training"), "training")
        if not dataset or not training:
            raise ValueError("Train a camera policy first.")
        model = checkpoint_path(training)
        rows = read_json(dataset / "dataset.json")["episodes"]
        familiar = [row["seed"] for row in rows if row["split"] == "train"][:2]
        if len(familiar) != 2:
            raise ValueError("Two training scenes are needed for the familiar-scene gate.")
        result = rollout(model, dataset, output=directory("rollouts", "familiar"),
                         seeds=familiar, execute=4, job_tracking=True)
        make_posters(result)
        saved["familiar"] = str(result.relative_to(OUTPUT / "vla"))
        saved.pop("unseen", None)
        score = summary(read_json(result / "report.json"))
        if familiar_passed(score):
            known = {row["seed"] for row in rows}
            heldout = [seed for seed in range(25000, 26000) if seed not in known][:10]
            result = rollout(model, dataset, output=directory("rollouts", "unseen"),
                             seeds=heldout, execute=4, job_tracking=True)
            make_posters(result)
            saved["unseen"] = str(result.relative_to(OUTPUT / "vla"))
        else:
            job_progress(2,2,"Familiar scene check complete. Held-out evaluation requires 2/2 without collisions.")
    save_json(OUTPUT / "vla/workflow.json", saved)
    return status(OUTPUT)


def make_posters(path):
    import numpy as np
    from PIL import Image

    report = read_json(path / "report.json", {})
    for row in report.get("trials", []):
        seed = row["seed"]
        with np.load(path / f"episode-{seed}.npz", allow_pickle=False) as recording:
            Image.fromarray(recording["frames"][-1]).save(path / f"episode-{seed}.jpg", quality=92)
            raw = recording["raw_actions"]
            for item in row.get("inferences", []):
                if item["tick"] < len(raw):
                    item["first_target_rad"] = raw[item["tick"]].tolist()
    save_json(path / "report.json", report)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=STAGES)
    stage = parser.parse_args().stage
    print(exclusive_job(stage)(run)(stage))


if __name__ == "__main__":
    main()
