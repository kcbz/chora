"""Shared-style, source/assumption material workbench for the tiling project."""
from html import escape
import json
from .tiling_materials import catalog,tile_properties,reference_panel
from .ui import page

CSS='''
.material-intro{max-width:78ch;margin-bottom:30px}.material-intro p{color:var(--muted);line-height:1.7}
.material-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,330px),1fr));gap:20px;margin:20px 0}
.material-grid .card{padding:24px}.material-grid h2{font-size:21px;margin-top:0}.material-grid p,.material-note{color:var(--muted);line-height:1.65;font-size:14px}
.material-facts{width:100%;border-collapse:collapse;font-size:13px}.material-facts td,.material-facts th{text-align:left;border-bottom:1px solid var(--line);padding:12px 8px;vertical-align:top}.material-facts th{color:var(--muted);font-weight:500}
.material-controls{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:18px}.material-controls label{display:grid;gap:8px;font-size:13px}.material-controls input{width:100%}
.material-badge{display:inline-block;border:1px solid var(--line);padding:5px 8px;border-radius:5px;color:var(--muted);font-size:11px}
.material-results{padding:20px 0;display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:20px}.material-results strong{font-size:25px;font-weight:500;display:block;margin-top:8px}
.material-error{color:var(--error);min-height:1.5em}details pre{white-space:pre-wrap;word-break:break-word;font-size:12px}
'''
SCRIPT='''
const form=document.getElementById('material-controls'),results=document.getElementById('material-results'),err=document.getElementById('material-error');
let timer;form.addEventListener('input',()=>{clearTimeout(timer);timer=setTimeout(refresh,200)});
async function refresh(){
 results.setAttribute('aria-busy','true');err.textContent='';
 try{const params=new URLSearchParams(new FormData(form));const r=await fetch('/api/tiling-materials?'+params);const d=await r.json();if(!r.ok)throw Error(d.error);
 const b=d.prediction;const rows=[['Tile mass · estimated',b.tile.mass_kg.toFixed(3)+' kg'],['Adhesive batch',b.adhesive.workable?'Within pot-life anchor':'Expired'],['Spread adhesive',b.adhesive.placement_window_open?'Within placement window':'Outside placement window'],['Grout batch',b.grout.workable?'Within pot-life anchor':'Expired'],['Adhesive apparent viscosity · assumed',b.adhesive.apparent_viscosity_pa_s.toFixed(1)+' Pa·s'],['Squeeze-film force · assumed',b.squeeze_force_n===null?'Not applicable':b.squeeze_force_n.toFixed(1)+' N'],['Layer volume',b.layer_volume_ml.toFixed(1)+' mL']];
 results.replaceChildren();for(const [label,value] of rows){const box=document.createElement('div'),name=document.createElement('span'),v=document.createElement('strong');name.textContent=label;v.textContent=value;box.append(name,v);results.append(box)}
 }catch(e){err.textContent=e.message}finally{results.setAttribute('aria-busy','false')}
}refresh();
'''

def render():
    d=catalog();tile=tile_properties();cards=[]
    for key,p in d['products'].items():
        facts=[]
        for name,r in p['published'].items():
            src=d['sources'][r['source']]
            facts.append('<tr><th>'+escape(name.replace('_',' '))+'</th><td>'+escape(str(r['value']))+' '+escape(r['unit'])+'<br><small>'+escape(r['context'])+'</small><br><a href="'+escape(src['url'],quote=True)+'">Manufacturer source</a></td></tr>')
        assumptions=''.join('<tr><th>'+escape(k.replace('_',' '))+'</th><td>'+escape(str(v['value']))+' '+escape(v['unit'])+'</td></tr>' for k,v in p['assumptions'].items())
        cards.append('<section class="card"><span class="material-badge">'+escape(p['manufacturer'])+'</span><h2>'+escape(p['name'])+'</h2><table class="material-facts">'+''.join(facts)+'</table><details><summary>Simulation assumptions and missing properties</summary><table class="material-facts">'+assumptions+'</table><ul>'+''.join('<li>'+escape(x)+'</li>' for x in p['missing'])+'</ul></details></section>')
    seq=reference_panel()
    content='''<main><div class="material-intro"><div class="eyebrow">Coherence / Tiling</div><h1>Materials, with provenance.</h1>
<p>A simulation-only baseline for a 2 × 2 ceramic-tile panel. Manufacturer facts and model choices are shown separately. Physical calibration is not required to explore this version.</p>
<p>The linked robot scene is a prepared tile-holding and grout-cleanup subtask. Full floor/wall installation and tile fracture are not implemented.</p>
<div class="page-actions"><a href="/?experiment=fr3-duo-tile-wipe">Open two-arm cleanup scene ↗</a> · <a href="/outputs/experiments/fr3-duo-tile-wipe/experiment_report.html">Recorded trial ↗</a></div></div>
<section class="card" style="padding:24px"><h2>Explore the material model</h2><p class="material-note">At a fixed 21°C baseline. Each paste has its own clock; this comparison sets both to the selected batch age. Viscosity and squeeze force are assumed constitutive-model predictions, not manufacturer measurements.</p>
<form id="material-controls" class="material-controls">
<label>Batch age (seconds)<input name="age_s" type="number" min="0" max="172800" value="600" step="60"></label>
<label>Time since spreading (seconds)<input name="exposure_s" type="number" min="0" max="172800" value="300" step="30"></label>
<label>Approach speed (m/s)<input name="speed_m_s" type="number" min="0" max="0.005" value="0.0001" step="0.0001"></label>
<label>Adhesive layer gap (m)<input name="gap_m" type="number" min="0.001" max="0.012" value="0.003" step="0.001"></label></form>
<div id="material-results" class="material-results" aria-live="polite"></div><p id="material-error" class="material-error" role="status"></p>
<p class="material-note">The published 800 lbf bending result is not a universal breaking-force threshold. Tile mass is estimated from an assumed 2,200 kg/m³ density. Foot-traffic DCOF is not used as robot contact friction.</p></section>
<div class="material-grid">'''+''.join(cards)+'''</div>
<section class="card" style="padding:24px"><h2>2 × 2 panel material sequence</h2><p class="material-note">Numerical accounting reference: spread and bed four tiles, advance the adhesive clock, mix fresh grout, and fill the joints. This is not a robot rollout. Coverage is a volume-based upper bound; spatial wetting, voids and wall slip remain unevaluated.</p><pre style="white-space:pre-wrap">'''+escape(json.dumps(seq,indent=2))+'''</pre></section></main>'''
    return page('Tiling material workbench',content,'reports',CSS,SCRIPT,experiment='fr3-duo-tile-wipe')
