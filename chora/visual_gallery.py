"""Episode and inspection evidence for the shared visual-method gallery."""

from html import escape
import json

from .common import read_json
from .experiments import DEFAULT, definition, output_dir, output_url
from .gallery import CSS, media_url
from .ui import page


def inspection_rows(experiment, output):
    from .visual_inspection import cases, saved_answers, status

    statistics = status(experiment, output)
    frames = cases(experiment, output)
    answers = saved_answers(output, frames)
    rows = []
    for frame in frames:
        rows.append({key: frame[key] for key in ("id", "seed", "policy", "image")} |
                    dict(answer=answers.get(frame["id"]), label="Qwen" if frame["policy"] == "student" else "Expert"))
    rows.sort(key=lambda row: (not bool(row["answer"]), row["policy"] != "student", row["seed"]))
    return rows, statistics["ready"]


def action_rows(experiment, output):
    from .vla.workflow import artifact, state

    rows = []
    if experiment != DEFAULT:
        return rows
    workflow = state(output)
    groups = []
    for group, label in (("unseen", "Held-out"), ("familiar", "Familiar")):
        directory = artifact(output, workflow.get(group), "rollouts")
        if directory:
            groups.append((group, label, directory, None))
    for directory in sorted((output / "vla/ui-runs").glob("*"), reverse=True):
        if directory.is_dir():
            groups.append((f"live-{directory.name}", "Live run", directory, "latest.jpg"))

    url = output_url(experiment) + "/"
    for group, label, directory, image_name in groups:
        report_path = directory / "report.json"
        report = read_json(report_path, {})
        if report.get("status") != "complete":
            continue
        for trial in report.get("trials", []):
            seed = trial["seed"]
            image_path = directory / (image_name or f"episode-{seed}.jpg")
            image = media_url(output.resolve(), str(image_path.relative_to(output)), url)
            rows.append(dict(id=f"{group}-{seed}", seed=seed, label=label, image=image,
                             report=media_url(output.resolve(), str(report_path.relative_to(output)), url),
                             **{key: trial.get(key) for key in ("success", "collisions", "reason", "ticks", "inferences")}))
    return rows


def render(experiment, method):
    output = output_dir(experiment)
    if method == "vlm":
        rows, ready = inspection_rows(experiment, output)
        label = "Visual inspection"
        detail = "Open an overhead frame to see its model answer and simulator check, or inspect it again."
    else:
        rows, ready = action_rows(experiment, output), False
        label = "Visual policy rollouts"
        detail = "Review the recorded front and wrist camera frames, outcomes, and replan traces."
    tiles = []
    for row in rows:
        badge = row.get("answer")
        success = row.get("success") if method == "vla" else badge and badge.get("correct")
        mark = "" if success is None else f'<span class="outcome-badge {"ok" if success else "fail"}" aria-hidden="true">{"✓" if success else "×"}</span>'
        image = f'<img loading="lazy" src="{escape(row["image"])}" alt="Seed {row["seed"]}">' if row["image"] else f'<span class="tile-placeholder">Seed {row["seed"]}</span>'
        tiles.append(f'<button type="button" class="episode-tile" data-visual-entry="{escape(row["id"])}" aria-label="{escape(row["label"])} seed {row["seed"]}">{image}{mark}<span class="tile-seed">{row["seed"]}</span></button>')
    payload = json.dumps(dict(experiment=experiment, method=method, ready=ready, rows=rows), separators=(",",":"))
    payload = payload.replace("&","\\u0026").replace("<","\\u003c").replace(">","\\u003e")
    empty = '<p class="wall-empty">No saved visual results for this scene yet.</p>' if not rows else ''
    scene = escape(definition(experiment)["name"])
    content = f'''<main><div class="gallery-intro"><h1>{label}</h1><p>{scene} · {detail}</p></div>
<div class="episode-wall">{''.join(tiles)}</div>{empty}</main>
<dialog id="episode-dialog" data-method="{method}" aria-labelledby="episode-title"><div class="episode-modal-header">
<h2 id="episode-title">Episode</h2><button class="secondary" id="close-visual-episode" aria-label="Close episode">×</button>
</div><div class="episode-media"><img id="visual-frame" alt="Recorded final frame"></div>
<div class="episode-info"><p id="visual-description" role="status"></p>
<button id="inspect-visual-frame" class="secondary" type="button" hidden>Inspect frame</button>
<a id="visual-raw-report" hidden>Open recorded report ↗</a>
<details id="visual-trace-details"><summary id="visual-trace-label">Recorded trace</summary><pre id="visual-trace"></pre></details></div></dialog>
<script type="application/json" id="visual-gallery-data">{payload}</script>
<script src="/static/visual-gallery.js"></script>'''
    css = CSS + '''
#episode-dialog[data-method="vlm"] .episode-media{aspect-ratio:4/3}
#episode-dialog[data-method="vla"] .episode-media{aspect-ratio:2/1}
#inspect-visual-frame,#visual-raw-report{margin-bottom:12px}
'''
    return page(label, content, "gallery", css, experiment=experiment, policy=method)
