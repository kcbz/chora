"""Show the visual policy's saved training and evaluation evidence."""
from html import escape
from urllib.parse import urlencode

from ..experiments import DEFAULT, definition, output_dir
from ..jobs import stage_history
from ..ui import page
from .workflow import STAGES, details, evaluation_outcome, summary


def render(experiment, service):
    query = escape(urlencode(dict(experiment=experiment, policy="vla")), quote=True)
    scene = escape(definition(experiment)["name"])
    content = f'<main class="vla-report"><div class="intro"><div><h1>VLA report</h1><p>{scene}</p></div></div>'
    links = f'<div class="report-links"><a href="/?{query}">Back to lab ↗</a>'
    if experiment != DEFAULT:
        content += f'<section class="card"><p>Visual policy evaluation currently supports the standard bowl scene.</p>{links}</div></section>'
        return page("VLA report", content + "</main>", experiment=experiment, policy="vla")

    directory = output_dir(experiment)
    data = details(directory)
    training = data["training"]
    step = training.get("final", {}).get("step") if training.get("status") == "complete" else None
    familiar, held_out = summary(data["familiar"]), summary(data["unseen"])
    outcome = evaluation_outcome(familiar, held_out)
    ready = bool(service and service["ready"])
    live = "Available" if ready else "Locked"
    if held_out:
        verdict = "Held-out evaluation passed" if outcome == "passed" else "Held-out evaluation did not pass"
        explanation = "Live runs require at least 8 of 10 held-out placements with zero collisions and a matching checkpoint."
    elif outcome == "did-not-pass":
        verdict = "Familiar evaluation did not pass"
        explanation = "Both familiar trials must pass without collisions. Held-out testing has yet to run."
    elif step:
        verdict = "Training complete"
        explanation = "Evaluate the saved checkpoint to qualify it for live runs."
    else:
        verdict = "Training pending"
        explanation = "Train the camera policy and evaluate its rollouts."

    training_text = f"{step:,} updates saved" if step else "Awaiting a completed run"
    if familiar or held_out:
        links += f'<a href="/gallery?{query}">View rollout frames and replans ↗</a>'
    content += (f'<section class="card"><h2>{verdict}</h2><p>{explanation}</p>'
                f'<div class="metrics"><div class="metric"><span>TRAINING</span><strong>{training_text}</strong></div>'
                f'<div class="metric"><span>RUN EPISODE</span><strong>{live}</strong></div></div>{links}</div></section>')

    history = stage_history(output=directory)
    latest = max((history[name] for name in STAGES if name in history),
                 key=lambda job: job.get("started_at", 0), default={})
    if latest.get("status") in {"failed", "cancelled"}:
        label = dict(zip(STAGES, ("Check the visual expert", "Record demonstrations", "Prepare the visual dataset",
                                  "Train the camera policy", "Evaluate the VLA"))).get(latest["name"], "Pipeline stage")
        result = "Job failed" if latest["status"] == "failed" else "Stopped"
        error = escape(latest.get("error") or "See Run details in the lab.")
        content += f'<section class="card"><h2>{label} · {result}</h2><p>{error}</p></section>'

    for title, score, key in (("Familiar trials", familiar, "familiar"), ("Held-out trials", held_out, "unseen")):
        if not score:
            continue
        collisions = score["collisions"]
        count = f'{score["successes"]}/{score["episodes"]} placements · {collisions} collision{"s" if collisions != 1 else ""}'
        rows = []
        for trial in data[key]["trials"]:
            reason = "Placed" if trial["success"] else str(trial.get("reason", "failed")).replace("_", " ").capitalize()
            if trial.get("stop_error"):
                reason += f' · {trial["stop_error"]}'
            style = '' if trial["success"] else ' class="trial-failure"'
            rows.append(f'<tr{style}><th scope="row">{int(trial["seed"])}</th><td>{escape(reason)}</td>'
                        f'<td>{int(trial["ticks"])}</td><td>{int(trial.get("collisions", 0))}</td></tr>')
        content += (f'<section class="card"><h2>{title}</h2><p>{count}</p><div class="trial-table"><table>'
                    '<thead><tr><th scope="col">Seed</th><th scope="col">Outcome</th>'
                    '<th scope="col">Ticks</th><th scope="col">Collisions</th></tr></thead>'
                    f'<tbody>{"".join(rows)}</tbody></table></div></section>')
    return page("VLA report", content + "</main>", experiment=experiment, policy="vla")
