"""Shared navigation, palettes, and appearance controls for every HTML page."""
from html import escape
from pathlib import Path
import os
import re
import json

from .common import ROOT
from .experiments import DEFAULT, output_url

from .paths import PACKAGE_ROOT
WEB = PACKAGE_ROOT / "web"


def render_shell(template, active="lab", experiment=None):
    experiment=experiment or os.environ.get("CHORA_EXPERIMENT",DEFAULT)
    navbar = (WEB/"navbar.html").read_text().replace(
        f'data-nav="{active}"', f'data-nav="{active}" aria-current="page"')
    navbar=navbar.replace("{{LOGO}}",(WEB/"logo-octahedron.svg").read_text())
    navbar=navbar.replace('href="/"',f'href="/?experiment={experiment}"')
    navbar=navbar.replace('/outputs/',output_url(experiment)+'/')
    parts = {
        "NAVBAR": navbar,
        "APPEARANCE_DIALOG": (WEB/"appearance-dialog.html").read_text(),
        "STYLES": '<link rel="stylesheet" href="/static/app.css">',
        "APPEARANCE_CORE": '<script src="/static/appearance-core.js"></script>',
        "APPEARANCE_UI": '<script src="/static/appearance-ui.js"></script>',
        "LAB_SCRIPT": '<script src="/static/lab.js"></script><script src="/static/vla.js"></script>',
        "VLA_PANEL": (WEB/"vla-panel.html").read_text(),
    }
    # Saved reports keep their content, but their chrome is refreshed from the
    # same components used by the live lab on each HTTP request.
    if "{{NAVBAR}}" not in template:
        template=re.sub(r'<header class="app-header">.*?</header>',lambda _:navbar,template,count=1,flags=re.S)
        template=re.sub(r'<dialog id="appearance-settings".*?</dialog>',lambda _:parts["APPEARANCE_DIALOG"],template,count=1,flags=re.S)
    for key,value in parts.items():
        template = template.replace("{{"+key+"}}",value)
    return template


def page(title, content, active="reports", css="", script="", experiment=None):
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
    return render_shell(template,active,experiment=experiment).replace("{{PAGE_CONTENT}}",content)
