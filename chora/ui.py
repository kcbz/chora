"""Shared navigation, palettes, and appearance controls for every HTML page."""
from html import escape
from pathlib import Path
import os
import re
import json
from urllib.parse import urlencode

from .common import ROOT
from .experiments import DEFAULT, output_url

from .paths import PACKAGE_ROOT
WEB = PACKAGE_ROOT / "web"


CONTROL_METHODS = {
    "expert": ("Scripted expert", "Run the programmed controller for this scene."),
    "qwen": ("Qwen · State-based imitation", "Learn high-level actions from structured simulator state."),
    "vla": ("VLA · Camera-based control", "Predict joint targets from camera frames, an instruction, and joint state."),
    "vlm": ("VLM · Visual inspection", "Use images and questions to inspect the scene and describe the result."),
    "planner": ("Planner + action model", "Choose the next step and call a trained action model. A vision model can help verify progress."),
}


def control_method(value):
    return value if value in CONTROL_METHODS else "expert"


def render_shell(template, active="lab", experiment=None, policy=None):
    experiment=experiment or os.environ.get("CHORA_EXPERIMENT",DEFAULT)
    navbar = (WEB/"navbar.html").read_text().replace(
        f'data-nav="{active}"', f'data-nav="{active}" aria-current="page"')
    navbar=navbar.replace("{{LOGO}}",(WEB/"logo-octahedron.svg").read_text())
    if policy is None:
        navbar = navbar.replace('href="/"', f'href="/?experiment={experiment}"')
        navbar = navbar.replace('/outputs/', output_url(experiment) + '/')
    else:
        policy = control_method(policy)
        query = escape(urlencode(dict(experiment=experiment, policy=policy)), quote=True)
        navbar = navbar.replace('href="/"', f'href="/?{query}"')
        navbar = navbar.replace('href="/outputs/eval_gallery.html"', f'href="/gallery?{query}"')
        navbar = navbar.replace('href="/outputs/experiment_report.html"', f'href="/reports?{query}"')
    options = []
    if "{{POLICY_OPTIONS}}" in template:
        for key, (label, description) in CONTROL_METHODS.items():
            selected = ' selected' if key == (policy or 'expert') else ''
            options.append(f'<option value="{key}" data-description="{escape(description, quote=True)}"{selected}>{escape(label)}</option>')
    parts = {
        "NAVBAR": navbar,
        "APPEARANCE_DIALOG": (WEB/"appearance-dialog.html").read_text(),
        "STYLES": '<link rel="stylesheet" href="/static/app.css">',
        "APPEARANCE_CORE": '<script src="/static/appearance-core.js"></script>',
        "APPEARANCE_UI": '<script src="/static/appearance-ui.js"></script>',
        "LAB_SCRIPT": '<script src="/static/lab.js"></script>',
        "POLICY_OPTIONS": ''.join(options),
    }
    # Saved reports keep their content, but their chrome is refreshed from the
    # same components used by the live lab on each HTTP request.
    if "{{NAVBAR}}" not in template:
        template=re.sub(r'<header class="app-header">.*?</header>',lambda _:navbar,template,count=1,flags=re.S)
        template=re.sub(r'<dialog id="appearance-settings".*?</dialog>',lambda _:parts["APPEARANCE_DIALOG"],template,count=1,flags=re.S)
    for key,value in parts.items():
        template = template.replace("{{"+key+"}}",value)
    return template


def page(title, content, active="reports", css="", script="", experiment=None, policy=None):
    experiment=experiment or os.environ.get("CHORA_EXPERIMENT",DEFAULT)
    if experiment!=DEFAULT:
        from .experiments import definition
        context='<aside class="experiment-context"><span class="label">Experiment</span> '+escape(definition(experiment)["name"])+'</aside>'
        from .experiments import load_config
        physics=load_config(experiment).get("physics")
        if physics:context+='<aside class="experiment-context"><strong>Uncalibrated material prototype.</strong> '+escape(physics["model"])+'. '+escape(physics.get('tool_grip',''))+'. Results are simulation-only.</aside>'
        content=context+content
    template = ('<!doctype html><html lang="en" data-theme="porcelain"><head>'
                '<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
                '<title>'+escape(title)+' — Chora</title>{{APPEARANCE_CORE}}{{STYLES}}'
                '<style>'+css+'</style></head><body>{{NAVBAR}}{{PAGE_CONTENT}}'+
                '{{APPEARANCE_DIALOG}}{{APPEARANCE_UI}}<script>'+script+'</script></body></html>')
    return render_shell(template,active,experiment=experiment,policy=policy).replace("{{PAGE_CONTENT}}",content)


def policy_results(experiment, policy, active, vla_status=None):
    """Use the shared gallery and reports for the current scene and control method."""
    from .common import read_json
    from .experiments import definition, output_dir

    directory = output_dir(experiment)
    label = CONTROL_METHODS[policy][0]
    if policy in {"expert", "qwen"}:
        if active == "gallery":
            from .gallery import render_gallery

            return render_gallery(read_json(directory / "results.json", {}) or {},
                                  directory, experiment=experiment, method=policy)
        saved = directory / "experiment_report.html"
        if saved.is_file():
            return render_shell(saved.read_text(), active, experiment, policy)
        detail = "Run the scene's evaluations to see its report here."
    elif policy == "vla":
        visual = vla_status or {}
        chosen = next(iter(visual.get("candidates", [])), None)
        evidence = chosen.get("evaluation", {}) if chosen else {}
        if experiment != DEFAULT:
            detail = "Visual policy evaluation currently supports the standard bowl scene."
        elif not evidence:
            detail = "Results will appear here after a visual policy passes development evaluation."
        elif active == "gallery":
            detail = "No video recordings have been published for the evaluated visual policy."
        else:
            detail = (f'{evidence["unseen_successes"]}/{evidence["unseen_episodes"]} '
                      "unseen development scenes completed.")
    else:
        detail = "This method is planned. Results will appear once it can run in the lab."
    title = "Evaluation gallery" if active == "gallery" else "Experiment report"
    scene = escape(definition(experiment)["name"])
    query = escape(urlencode(dict(experiment=experiment, policy=policy)), quote=True)
    content = (f'<main><div class="intro"><div><h1>{title}</h1><p>{scene} · {escape(label)}</p></div></div>'
               f'<section class="card"><p>{escape(detail)}</p><a href="/?{query}">Back to lab ↗</a></section></main>')
    return page(title, content, active, experiment=experiment, policy=policy)
