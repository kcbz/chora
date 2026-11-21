"""Inspect scored final frames with a local, pinned vision-language model."""

from argparse import ArgumentParser
from collections import Counter
import math
import re
import time

from .common import OUTPUT, ROOT, config, exclusive_job, job_progress, read_json, save_json
from .experiments import output_url
from .gallery import media_url

MODEL = "mlx-community/Qwen3-VL-2B-Instruct-4bit"
REVISION = "9c4f5209e57b31f4b9dfba735de3fb983739c9cc"
SUPPORTED = {"so101-bowl", "so101-part-raised-shelf"}


def model_directory():
    cache = ROOT / ".cache/huggingface/hub"
    return cache / ("models--" + MODEL.replace("/", "--")) / "snapshots" / REVISION


def ready():
    path = model_directory()
    return (ROOT / ".venv-vlm/bin/python").is_file() and (path / "config.json").is_file() and any(
        p.is_file() for p in path.glob("*.safetensors")
    )


def status(scene, output):
    supported = scene in SUPPORTED
    benchmark = read_json(output / "vlm/benchmark.json", {}) if supported else {}
    score = {key: benchmark[key] for key in ("correct", "examples")} if benchmark.get("view") == "overhead" else None
    return dict(supported=supported, ready=supported and ready(), benchmark=score,
                latest=read_json(output / "vlm/latest.json") if supported else None)


def cases(scene, output):
    if scene not in SUPPORTED:
        return []
    root = output.resolve()
    prefix = output_url(scene) + "/"
    rows = []
    for policy in ("student", "expert"):
        for trial in read_json(root / "evaluation" / policy / "episodes.json", []):
            thumbnail = trial.get("inspection_frame")
            url = media_url(root, thumbnail, prefix) if trial.get("thumbnail_verified") else None
            if url:
                rows.append(dict(id=f"{policy}-{trial['seed']}", seed=trial["seed"],
                                 policy=policy, image=url, thumbnail=thumbnail,
                                 observation=trial["final_observation"]))
    return rows


def public_cases(scene, output):
    return [{key: row[key] for key in ("id", "seed", "policy", "image")}
            for row in cases(scene, output)]


def footprint_inside(observation):
    """Match the placement scorer's horizontal fit criterion."""
    point = observation.get("bowl", observation.get("object"))
    radius = .027 if "bowl" in observation else math.sqrt(2) * .008
    target = observation["target"]
    return all(abs(point[i] - target[i]) + radius < observation["target_half_size"][i]
               for i in (0, 1))


def parse_answer(text):
    match = re.fullmatch(r"(inside|outside)[.!]?", text.strip().lower())
    return {"inside": "yes", "outside": "no"}[match[1]] if match else "uncertain"


def inspection_camera(sim):
    from .visualization import Camera

    camera = Camera(sim, 896, 672)
    camera.set_orbit(dict(azimuth=90, elevation=-89, distance=.49, lookat=[.16, 0, .06]))
    return camera


def save_inspection_frame(camera, sim, episode, policy, output):
    relative = f"evaluation/{policy}/inspection/seed_{episode['seed']}.jpg"
    path = output / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".jpg.tmp")
    temporary.write_bytes(camera.jpeg(sim, camera="orbit", overlay=False))
    temporary.replace(path)
    episode["inspection_frame"] = relative


class Inspector:
    def __init__(self):
        from mlx_vlm import load, generate
        from mlx_vlm.prompt_utils import apply_chat_template
        from mlx_vlm.utils import load_config

        path = model_directory()
        self.model, self.processor = load(str(path), trust_remote_code=False)
        self.model_config = load_config(str(path))
        self.format_prompt = apply_chat_template
        self.generate = generate

    def ask(self, image, object_name):
        from PIL import Image

        subject = "orange bowl" if object_name == "bowl" else "small orange block"
        question = f"Is the entire {object_name} inside the target outline?"
        prompt = (
            f"This is an overhead image of a robot workcell. Where is the {subject} relative to the pale rectangular "
            "outline painted on the blue platform? If the whole object is in the rectangle, answer inside. "
            "If it is on the gray floor or any part is outside the rectangle, answer outside. "
            "If you cannot see the object, answer uncertain. Reply in one word: inside, outside, or uncertain."
        )
        formatted = self.format_prompt(self.processor, self.model_config, prompt, num_images=1)
        started = time.perf_counter()
        with Image.open(image) as frame:
            rgb = frame.convert("RGB")
            result = self.generate(self.model, self.processor, formatted, [rgb],
                                   max_tokens=12, temperature=0, verbose=False)
        return question, result.text.strip(), round((time.perf_counter() - started) * 1000)


def inspect_case(inspector, row, output):
    observation = row["observation"]
    question, text, latency = inspector.ask(output / row["thumbnail"], "bowl" if "bowl" in observation else "block")
    answer = parse_answer(text)
    expected = "yes" if footprint_inside(observation) else "no"
    return {k: row[k] for k in ("id", "seed", "policy", "image")} | dict(
        question=question, answer=answer, expected=expected, correct=answer==expected,
        latency_ms=latency, raw_answer=text, model=MODEL, revision=REVISION,
    )


@exclusive_job("vlm-inspect")
def inspect(episode):
    scene = config()["experiment"]["id"]
    row = next((row for row in cases(scene, OUTPUT) if row["id"] == episode), None)
    if row is None:
        raise ValueError("Choose a saved episode with a verified frame.")
    job_progress(0,1,"Loading the local vision model…")
    result = inspect_case(Inspector(), row, OUTPUT)
    save_json(OUTPUT / "vlm/latest.json", result)
    job_progress(1,1,"Image inspection complete")
    return result


@exclusive_job("vlm-benchmark")
def benchmark(per_class):
    if not 1 <= per_class <= 25:
        raise ValueError("Select 1 to 25 examples per visible placement class.")
    scene = config()["experiment"]["id"]
    rows = cases(scene, OUTPUT)
    # Use a fixed ordering of student episodes, whose failures exercise the visual check.
    selected = []
    for label in (False, True):
        group = [row for row in rows if row["policy"] == "student" and footprint_inside(row["observation"]) == label]
        if len(group) < per_class:
            raise ValueError("The student evaluation has too few verified examples in one class.")
        selected.extend(group[round(i * (len(group)-1) / max(1, per_class-1))] for i in range(per_class))
    inspector = Inspector()
    evidence = []
    for row in selected:
        evidence.append(inspect_case(inspector, row, OUTPUT))
        job_progress(len(evidence), len(selected), "Checking visual answers against simulator geometry")
    report = dict(scene=scene, model=MODEL, revision=REVISION, view="overhead", examples=len(evidence),
                  correct=sum(row["correct"] for row in evidence),
                  answers=dict(Counter(row["answer"] for row in evidence)), trials=evidence)
    save_json(OUTPUT / "vlm/benchmark.json", report)
    return {key: value for key, value in report.items() if key != "trials"}


def main():
    parser = ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("download")
    job = commands.add_parser("inspect")
    job.add_argument("--episode", required=True)
    check = commands.add_parser("benchmark")
    check.add_argument("--per-class", type=int, default=8)
    args = parser.parse_args()
    if args.command == "download":
        from huggingface_hub import snapshot_download

        print(snapshot_download(MODEL, revision=REVISION,
                                cache_dir=ROOT / ".cache/huggingface/hub",
                                allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja", "*.model"]))
    elif args.command == "inspect":
        print(inspect(args.episode))
    else:
        print(benchmark(args.per_class))


if __name__ == "__main__":
    main()
