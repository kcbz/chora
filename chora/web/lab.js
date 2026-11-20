function actionLabel(a){
 if(a.action==='parallel')return a.actions.map(actionLabel).join(' ∥ ');
 return a.action==='move_to'?(a.position_mm?`Arm ${a.arm} → (${a.position_mm.join(', ')}) mm`:`move_to (${Math.round(a.x*1000)}, ${Math.round(a.y*1000)}, ${Math.round(a.z*1000)}) mm`):a.action==='dispense'?`Arm ${a.arm} · ${a.volume_ml===0?'close adhesive valve':`set ${a.material} dose · ${a.volume_ml} mL`}`:a.action==='vacuum'?`Arm ${a.arm} · vacuum ${a.state}`:a.action==='wait'?`wait · ${a.seconds} s`:a.action==='inspect'?`inspect · ${a.target}`:a.energy!==undefined?`impact · ${a.energy} J`:a.depth!==undefined?`${a.action} · ${Math.round(a.depth*1000)} mm / ${a.duration} s`:a.action;
}
let speedRequestPending=false;
async function setPlaybackSpeed(value){
 speedRequestPending=true;
 try{await post('/api/control',{operation:'speed',speed:Number(value)})}catch{}
 finally{speedRequestPending=false}
}
function renderPlayback(s){
 const supported=typeof s.playback_speed==='number';
 $('playback-control').hidden=!supported;if(!supported)return;
 if(!speedRequestPending)document.querySelectorAll('[data-playback-speed]').forEach(button=>{
  const selected=Number(button.dataset.playbackSpeed)===s.playback_speed;
  button.classList.toggle('active',selected);button.setAttribute('aria-pressed',String(selected));
 });
 const actual=s.achieved_speed;
 $('playback-timing').textContent=typeof actual==='number'?`${s.running?'Actual':'Last run'} ${actual.toFixed(2)}×`:'';
}
const $=id=>document.getElementById(id);let state={},busy=false,loadingFrame=false,pendingJob=null,pendingRun=false,requestError=null;
async function post(url,data){$('error').textContent='';try{let r=await fetch(url,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(data)});let d=await r.json();if(!r.ok)throw Error(d.error);return d}catch(e){$('error').textContent=e.message;throw e}}
async function control(operation){
 if(operation==='run'){pendingRun=true;renderViewer(state)}
 try{await post('/api/control',{operation,seed:Number($('seed').value),policy:$('policy').value})}
 catch{pendingRun=false;renderViewer(state)}
}
async function camera(name){try{await post('/api/control',{operation:'camera',camera:name});['front','top','orbit'].forEach(n=>$(n).classList.toggle('active',n===name))}catch{}}

let orbit={azimuth:135,elevation:-25,distance:.65,lookat:[.12,0,.16]},drag=null,orbitTimer=null,orbitDirty=false;
function scheduleOrbit(){
 orbitDirty=true;
 if(orbitTimer)return;
 orbitTimer=setTimeout(async()=>{
  orbitDirty=false;
  try{await post('/api/control',{operation:'orbit',orbit:structuredClone(orbit)})}catch{}
  orbitTimer=null;
  if(orbitDirty)scheduleOrbit();
 },75);
 ['front','top','orbit'].forEach(n=>$(n).classList.toggle('active',n==='orbit'));
}
const viewport=$('viewport');
viewport.addEventListener('contextmenu',e=>e.preventDefault());
viewport.addEventListener('pointerdown',e=>{
 if(!state.ready||!([0,2].includes(e.button)))return;
 e.preventDefault();drag={x:e.clientX,y:e.clientY,button:e.button};
 viewport.setPointerCapture(e.pointerId);viewport.classList.add('dragging');
});
viewport.addEventListener('pointermove',e=>{
 if(!drag)return;
 const dx=e.clientX-drag.x,dy=e.clientY-drag.y;
 drag.x=e.clientX;drag.y=e.clientY;
 if(e.shiftKey||drag.button===2){
  const a=orbit.azimuth*Math.PI/180,el=orbit.elevation*Math.PI/180,k=orbit.distance*.0016;
  const right=[Math.sin(a),-Math.cos(a),0],up=[-Math.cos(a)*Math.sin(el),-Math.sin(a)*Math.sin(el),Math.cos(el)];
  orbit.lookat=orbit.lookat.map((v,i)=>Math.max(-1.8,Math.min(1.8,v+k*(-dx*right[i]+dy*up[i]))));
 }else{orbit.azimuth=(orbit.azimuth-dx*.4+360)%360;orbit.elevation=Math.max(-89,Math.min(-5,orbit.elevation-dy*.3));}
 scheduleOrbit();
});
function endDrag(){drag=null;viewport.classList.remove('dragging')}
viewport.addEventListener('pointerup',endDrag);viewport.addEventListener('pointercancel',endDrag);
viewport.addEventListener('wheel',e=>{if(!state.ready)return;e.preventDefault();orbit.distance=Math.max(.22,Math.min(1.8,orbit.distance*Math.exp(e.deltaY*.0012)));scheduleOrbit()},{passive:false});

const JOBS={
 check:{title:'Expert check',button:'Run expert check',working:'Checking'},
 generate:{title:'Demonstration collection',button:'Collect demonstrations',working:'Collecting'},
 curate:{title:'Dataset curation',button:'Curate dataset',working:'Curating'},
 train:{title:'Student training',button:'Train student',working:'Training'},
 evaluate:{title:'Student evaluation',button:'Evaluate student',working:'Evaluating'},
 thumbnails:{title:'Gallery thumbnails',working:'Preparing thumbnails'},replay:{title:'Recording update',working:'Updating recordings'},select:{title:'Adapter selection',working:'Selecting'}
};
function uiJob(s){
 const live=s.job||{};
 if(pendingJob&&pendingJob.id&&live.id===pendingJob.id)pendingJob=null;
 if(pendingJob)return {name:pendingJob.name,running:true,status:'running',progress:{message:'Starting…'},pending:true};
 if(live.running)return live;
 return requestError||live;
}
function buttonLabel(button,text,loading){
 let label=button.querySelector('.button-label');
 if(!label){button.replaceChildren();const spinner=document.createElement('span');spinner.className='spinner';spinner.setAttribute('aria-hidden','true');button.append(spinner);label=document.createElement('span');label.className='button-label';button.append(label)}
 button.querySelector('.spinner').hidden=!loading;label.textContent=text;button.setAttribute('aria-busy',String(loading));
}
function renderViewer(s){
 if(s.running)pendingRun=false;
 const running=Boolean(s.running||pendingRun||s.switching),preparing=s.switching||pendingRun||['preparing','loading_policy'].includes(s.phase);
 const pipeline=Boolean(uiJob(s).running);
 buttonLabel($('run'),running?(preparing?'Preparing…':'Running…'):'Run episode',running);
 $('run').disabled=!s.ready||running||pipeline;$('reset').disabled=!s.ready||running||pipeline;$('stop').disabled=!s.running;
 const activity=$('viewer-activity');activity.hidden=!running&&Boolean(s.ready);
  const activityLabel=!s.ready?'Preparing scene…':s.switching?'Loading experiment…':preparing?'Loading policy…':s.policy==='student'?'Student running':'Expert running';if(!activity.hidden){if(activity.dataset.label!==activityLabel||!activity.firstElementChild){activity.replaceChildren();const spinner=document.createElement('span');spinner.className='spinner';spinner.setAttribute('aria-hidden','true');const text=document.createElement('span');text.textContent=activityLabel;activity.append(spinner,text);activity.dataset.label=activityLabel}}
 document.querySelector('.viewer').setAttribute('aria-busy',String(running||!s.ready));
}
function savedSummary(name,s){
 if(name==='check'&&s.oracle)return `Last completed check: ${s.oracle.successes}/${s.oracle.episodes} scenes successful.`;
 if(name==='generate'&&s.generation)return `${s.generation.attempts.toLocaleString()} demonstrations saved · ${s.generation.successes.toLocaleString()} successful.`;
 if(name==='curate'&&s.curation?.selected_episodes)return `${s.curation.selected_episodes.toLocaleString()} episodes selected · ${s.curation.training_pairs.toLocaleString()} training examples.`;
 if(name==='train'&&s.adapter_ready)return 'A trained Qwen policy is ready. Select it above to run an episode.';
 if(name==='evaluate'&&s.results?.student){const r=s.results.student;return `Last completed evaluation: ${r.successes}/${r.episodes} scenes successful.`}
 return '';
}
function renderPipeline(s){
 const material=s.experiment?.physics?.learning_enabled===false;document.querySelector(".pipeline").hidden=material;$("material-trial-note").hidden=!material;
 const current=uiJob(s),jobBusy=Boolean(current.running),busy=jobBusy||s.switching,history=s.jobs||{};
 const resultIds={check:'verifyText',generate:'collectText',curate:'curateText',train:'trainText',evaluate:'evalText'};
 for(const [name,id] of Object.entries(resultIds)){
  const meta=JOBS[name],card=document.querySelector(`[data-stage="${name}"]`),button=card.querySelector('button[data-job]');
  const record=current.name===name?current:(history[name]?.name?history[name]:{});
  const running=current.name===name&&jobBusy,status=running?'running':record.status||'idle';
  card.dataset.status=status;card.setAttribute('aria-busy',String(running));
  const statusText=running?(current.pending?'Starting':'Running'):status==='complete'?'Complete':status==='failed'?'Failed':status==='cancelled'?'Stopped':savedSummary(name,s)?'Saved result':'Ready';
  $('stage-status-'+name).textContent=statusText;
  $(id).textContent=savedSummary(name,s)||'No saved result yet.';
  const progress=record.progress||{},total=Number(progress.total),done=Number(progress.current),determinate=total>0&&Number.isFinite(done);
  const bar=$('stage-progress-'+name);bar.hidden=!running;bar.classList.toggle('indeterminate',!determinate);
  if(determinate){const percentage=Math.max(0,Math.min(100,100*done/total));bar.firstElementChild.style.width=percentage+'%';bar.setAttribute('aria-valuemin','0');bar.setAttribute('aria-valuemax',String(total));bar.setAttribute('aria-valuenow',String(done))}
  else{bar.removeAttribute('aria-valuenow');bar.firstElementChild.style.width='35%'}
  bar.setAttribute('aria-valuetext',progress.message||statusText);
  const ready=name==='generate'?s.oracle?.success_rate>=.9:name==='curate'?Boolean(s.generation):name==='train'?Boolean(s.curation?.selected_episodes&&s.model_ready&&!s.adapter_ready):name==='evaluate'?s.adapter_ready:true;
  button.disabled=busy||s.running||pendingRun||!ready;
  let label=running?(current.pending?'Starting…':meta.working+(determinate?` ${done.toLocaleString()} / ${total.toLocaleString()}`:'…')):meta.button+' ↗';
  if(name==='train'&&s.adapter_ready&&!running)label='Student ready';
  buttonLabel(button,label,running);
  button.title=running?(progress.message||meta.working):busy?'Wait for the current stage to finish.':s.running?'Stop the live episode first.':name==='train'&&s.adapter_ready?'A saved student is already available in the Policy selector.':!ready?'Finish the preceding stages first.':meta.button;
 }
 if(!current.name){$('job').style.display='none';return}
 const meta=JOBS[current.name]||{title:'Pipeline stage'},status=current.status||(current.running?'running':current.exit_code===0?'complete':'idle');
 $('job').style.display='block';$('job').dataset.status=status;$('job').setAttribute('aria-busy',String(jobBusy));$('job-spinner').hidden=!jobBusy;
 $('jobTitle').textContent=meta.title+(jobBusy?(current.pending?' is starting':' is running'):status==='complete'?' complete':status==='failed'?' failed':status==='cancelled'?' stopped':'');
 const message=current.progress?.message;
 $('job-summary').textContent=status==='failed'?(current.error||'This stage could not finish. Open Run details for the error.'):message&&message!=='Starting…'?message:jobBusy?'Preparing this stage. Progress will appear here.':savedSummary(current.name,s)||'This stage has finished.';
 const log=(current.log||'').trim();$('job-details').hidden=!log;$('jobLog').textContent=log;
 if($('job-details').open)$('jobLog').scrollTop=$('jobLog').scrollHeight;
}
async function job(name){
 if(pendingJob||uiJob(state).running)return;
 requestError=null;pendingJob={name,id:null};renderPipeline(state);renderViewer(state);
 try{const result=await post('/api/job',{name});pendingJob.id=result.job_id}
 catch(error){pendingJob=null;requestError={name,status:'failed',running:false,error:error.message};renderPipeline(state);renderViewer(state)}
}
let experimentCatalog=[],pendingExperiment=null;
const selectMeasure=document.createElement('canvas').getContext('2d');
function fitScenePicker(){
 const picker=$('experiment-picker'),option=picker.selectedOptions[0];if(!option)return;
 selectMeasure.font=getComputedStyle(picker).font;
 const available=picker.parentElement.clientWidth-($('experiment-loading').hidden?0:24);
 picker.style.width=Math.min(Math.ceil(selectMeasure.measureText(option.textContent).width)+52,Math.max(120,available))+'px';
}
addEventListener('resize',fitScenePicker);
document.fonts?.ready.then(fitScenePicker);
let requestedExperiment=new URLSearchParams(location.search).get('experiment');
async function loadExperimentCatalog(){
 const r=await fetch('/api/experiments');if(!r.ok)throw Error('Could not load experiments');
 const data=await r.json();experimentCatalog=data.experiments;
 const picker=$('experiment-picker');picker.replaceChildren();
 for(const item of experimentCatalog){const option=document.createElement('option');option.value=item.id;option.textContent=item.name+(item.available===false?' · asset unavailable':'');option.disabled=item.available===false;option.title=item.unavailable_reason||'';picker.append(option)}
 picker.value=state.experiment?.id||data.active;
 fitScenePicker();
}
function syncExperiment(s){
 if(!s.experiment)return;
 $("policy").querySelector('option[value="expert"]').textContent=s.experiment.physics?.learning_enabled===false?"Scripted test controller":"Scripted expert";
 $("robot-label").textContent=(s.experiment.robot_name||"SO-101")+" / MuJoCo";
 if(pendingExperiment===s.experiment.id)pendingExperiment=null;
 if(s.error&&!s.switching)pendingExperiment=null;
 const switching=Boolean(s.switching||pendingExperiment),pipeline=Boolean(uiJob(s).running);
 $('experiment-picker').value=pendingExperiment||s.experiment.id;
 fitScenePicker();
 $('experiment-picker').disabled=switching||pipeline||s.running;
 $('experiment-loading').hidden=!switching;document.querySelector('.experiment-bar').setAttribute('aria-busy',String(switching));
 $('experiment-note').textContent=switching?'Loading experiment…':s.experiment.description;$('task-instructions').textContent=s.experiment.instructions;
 const prefix=s.output_url||'/outputs';
 const labUrl='/?experiment='+encodeURIComponent(s.experiment.id);document.querySelector('[data-nav="lab"]').href=labUrl;document.querySelector('.brand').href=labUrl;if(!pendingExperiment&&(!requestedExperiment||requestedExperiment===s.experiment.id))history.replaceState(null,'',labUrl);
 for(const [nav,file] of [['gallery','eval_gallery.html'],['reports','experiment_report.html']])document.querySelector(`[data-nav="${nav}"]`).href=prefix+'/'+file;
 document.querySelectorAll('footer a').forEach(a=>{const file=a.getAttribute('href').split('/').at(-1);a.href=prefix+'/'+file});
 if(!s.adapter_ready&&$('policy').value==='student')$('policy').value='expert';
 if(!s.oracle){$('expertMetric').textContent='—';$('expertCaption').textContent='Not checked for this experiment'}
 if(!s.results?.student){$('studentMetric').textContent='—';$('studentCaption').textContent='Not evaluated for this experiment'}
 if(requestedExperiment&&experimentCatalog.length){if(!experimentCatalog.some(item=>item.id===requestedExperiment))requestedExperiment="so101-bowl";const requested=requestedExperiment;requestedExperiment=null;if(requested!==s.experiment.id)selectExperiment(requested)}
}
async function selectExperiment(id){
 if(id===state.experiment?.id)return;
 pendingExperiment=id;syncExperiment(state);renderViewer({...state,switching:true});renderPipeline({...state,switching:true});
 try{await post('/api/experiment',{id})}catch{pendingExperiment=null;syncExperiment(state)}
}
loadExperimentCatalog().catch(error=>$('error').textContent=error.message);

async function frame(){if(loadingFrame||!state.ready)return;loadingFrame=true;try{const r=await fetch('/frame.jpg?t='+Date.now());if(!r.ok)return;let blob=await r.blob(),url=URL.createObjectURL(blob),old=$('frame').src;$('frame').src=url;$('frame').style.visibility='visible';$('loading').style.display='none';if(old.startsWith('blob:'))URL.revokeObjectURL(old)}catch{}finally{loadingFrame=false}}
function updateConnection(s){
 const indicator=document.querySelector('.connection-indicator');
 const job=uiJob(s),working=s.running||s.switching||job.running;
 let label=s.running?'Episode running':s.switching?'Loading experiment…':job.running?(JOBS[job.name]?.working||'Working')+' · on device':s.ready?'Lab online · on device':'Preparing simulation…';
 if(s.error&&!s.ready)label='Lab unavailable';
 if($('connection').textContent!==label)$('connection').textContent=label;
 indicator.dataset.state=s.error&&!s.ready?'offline':working||!s.ready?'busy':'online';
 indicator.title=label;$('dot').classList.toggle('pulse',Boolean(working));
}
async function poll(){try{let r=await fetch('/api/status');if(!r.ok)throw Error('Lab unavailable');state=await r.json();syncExperiment(state);renderPlayback(state);if(!drag&&!orbitTimer&&state.orbit)orbit=state.orbit;updateConnection(state);renderViewer(state);$('student').disabled=!state.adapter_ready;$('student').textContent=state.adapter_ready?'Qwen · trained student':'Qwen · train first';let o=state.observation||{};renderPhysics(state,o);$('scene').textContent=`SEED ${state.seed}   ·   DECISION ${o.step||0}   ·   ${state.camera.toUpperCase()}${['hammer_nail','hammer_pine','impact_nail'].includes(o.task)?`   ·   NAIL ${Math.round(o.nail_depth*1000)} / ${Math.round(o.target_depth*1000)} mm`:''}`;$('step').textContent=`${o.step||0} / ${state.experiment?.simulation?.max_steps||15}`;$('result').textContent=state.result==='experiment_complete'?'Engineering trial complete · materials uncalibrated':state.success===true?(['hammer_nail','hammer_pine','impact_nail'].includes(o.task)?`✓ Nail driven ${Math.round(o.nail_depth*1000)} mm · hammer clear`:o.task==='cut_fruit'?'✓ Cut complete · knife clear':o.task==='tile_adhesive_bedding'?'✓ Adhesive and bedding complete · reduced model':o.task==='tile_dry_transfer'?'✓ Four dry tiles placed · installation materials pending':'✓ Placement completed successfully'):state.success===false?'Episode ended · '+state.result:state.running?'Executing '+(state.history.at(-1)?.action||'first action'):'Ready to explore';if(state.error){$('error').textContent=state.error;pendingRun=false;renderViewer(state)}
let trace=state.history||[];$('actions').replaceChildren();if(!trace.length){let li=document.createElement('li');li.textContent='Run an episode to see each decision.';$('actions').append(li)}trace.forEach((a,i)=>{let li=document.createElement('li'),n=document.createElement('span'),v=document.createElement('span');n.className='num';n.textContent=String(i+1).padStart(2,'0');v.textContent=actionLabel(a);li.append(n,v);$('actions').append(li)});$('actions').scrollTop=$('actions').scrollHeight;
if(state.oracle){$('expertMetric').textContent=`${state.oracle.successes}/${state.oracle.episodes}`;$('expertCaption').textContent='Randomized scene check'}
if(state.results?.student){const r=state.results.student;$('studentMetric').textContent=`${r.successes}/${r.episodes}`;$('studentCaption').textContent=r.complete_benchmark?'100 held-out scenes':'Partial evaluation'}
renderPipeline(state);await frame()}catch(e){$('connection').textContent='Reconnecting to lab…';document.querySelector('.connection-indicator').dataset.state='offline';$('dot').classList.remove('pulse')}finally{setTimeout(poll,700)}}poll();setInterval(frame,110);

function renderPhysics(s,o){
 const card=$('physics-card');card.hidden=!s.experiment?.physics;if(card.hidden)return;
 $('material-workbench-link').hidden=!s.experiment.physics.material_workbench_url;
 const cut=['cut_fruit','soft_cut'].includes(o.task);
 let rows=[];
 if(o.task==='tile_adhesive_bedding'){
  const m=o.mortar||{};
  $('physics-description').textContent='Arm B applies each adhesive patch while arm A acquires its tile. A nearby clearance move lets arm A place and bed it; the applicator parks after its last patch. Spatial reduced mortar support, ridge redistribution and tracked squeeze-out use declared assumed laws. No resolved fluid, cured bonding, grout finish or fracture.';
  rows=[['Tiles placed',`${(o.tiles||[]).filter(t=>t.placed).length} / 4`],['Both arms moving',`${o.concurrent_motion_s??0} s`],['Pickup/application overlap',`${o.overlap_mortaring_s??0} s`],['Simulation elapsed',`${o.simulation_elapsed_s??0} s`],['Mortar reservoir',`${m.reservoir_ml??0} mL`],['Mortar deposited',`${m.deposited_ml??0} mL`],['Mortar age',`${Math.round(m.age_s??0)} s`],['Volume balance error',`${m.volume_balance_error_ml??0} mL`],['Peak applicator drag',`${o.tool_peak_force_n??0} N`],...(o.tiles||[]).flatMap(t=>[[`Tile ${t.index+1} · position error`,`${t.position_error_mm} mm`],[`Tile ${t.index+1} · modeled bed contact`,`${m.contact_area_percent?.[t.index]??0}% · ${m.current_load_n?.[t.index]??0} N`],[`Tile ${t.index+1} · squeeze-out`,`${m.squeezeout_ml?.[t.index]??0} mL`]])];
 }else if(o.task==='tile_dry_transfer'){
  $('physics-description').textContent='Dry transfer foundation: four free-body ceramic tiles, face pickup with a finite 40 N vacuum fixture, and a 2 × 2 target pattern. Arm B is parked. Adhesive, bedding, grout application and fracture are not implemented in this scene yet.';
  rows=[['Tiles placed',`${(o.tiles||[]).filter(t=>t.placed).length} / 4`],['Vacuum',o.vacuum_on?'On':'Off'],['Tile contact',o.grip_tiles?.length?o.grip_tiles.map(i=>i+1).join(', '):'None'],['Tile mass · estimated',`${(o.tile_mass_kg??0).toFixed(3)} kg`],['Vacuum-face contact force',`${o.contact_force_n??0} N`],['Peak face contact force',`${o.peak_contact_force_n??0} N`],...(o.tiles||[]).map(t=>[`Tile ${t.index+1} · position error`,`${t.position_error_mm} mm${t.placed?' · placed':''}`])];
 }else if(o.task==='dual_tile_wipe'){
  $('physics-description').textContent='QuarryBasics 31IS ceramic tile: published geometry, estimated mass. Native contact drives a reduced grout-residue model with paste shear and finite collection. Simulation-only cleanup subtask; full installation and tile fracture are not modeled.';
  rows=[['Tile mass · estimated',`${(o.tile_mass_kg??0).toFixed(3)} kg`],['Contact coverage',`${o.coverage_percent??0}%`],['Grout remaining',`${(o.grout_remaining_ml??0).toFixed(3)} mL`],['Collected by sponge',`${(o.grout_collected_ml??0).toFixed(3)} mL`],['Volume balance error',`${o.grout_mass_balance_error_ml??0} mL`],['Grout age',`${Math.round(o.grout_age_s??0)} s`],['Sponge force',`${o.contact_force_n??0} N`],['Peak wipe force',`${o.peak_wipe_force_n??0} N`],['Tile / holder displacement',`${o.max_tile_error_mm??0} mm`]];
 }else if(o.task==='impact_nail'){
  $('physics-description').textContent=o.robot==='ur10e'?'Native UR10e with scanned hammer. Commanded downward task force is removed before contact; joint gravity and orientation control remain active. Nail resistance is uncalibrated.':'Scanned claw hammer in an XYZ holder. Drive switches off before impact. Nail insertion remains uncalibrated.';
  rows=[['Mount','Installed tool · bolted holder'],['Impact energy',`${o.impact_energy_j??0} J`],['Downward drive at contact',`${o.motor_force_at_contact_n??0} N`],['Off-axis error',`${o.lateral_error_mm??0} mm`],['Last strike peak',`${o.last_peak_force??0} N`],['Contact impulse',`${o.impulse_ns??0} N·s`]];
 }else if(o.task==='soft_cut'){
  $('physics-description').textContent='Volumetric FEM flesh and elastic skin. Local cohesive release on one plane; uncalibrated. Soft-solid tomato interior, without juice flow. No cutting success score.';
  rows=[['Mount','Installed chef’s knife'],['Peak contact force',`${o.peak_cut_force_n??0} N`],['Maximum node movement',`${o.deformation_mm??0} mm`],['Contact work',`${o.cut_work_j??0} J`],['Released material bonds',`${o.released_bonds??0} / ${o.total_bonds??0}`]];
 }else{
  $('physics-description').textContent=cut?'Archived rigid-half cutting model.':'Archived SO-101 contact model.';
  rows=[['Finger normal force',`${o.grip_normal_n??0} N`],['Handle slip',`${o.grip_slip_mm??0} mm`]];
 }
 if(s.experiment?.physics?.mount_status==='custom_unverified')rows.push(['Tool mounting','Custom simulation fixture · unverified']);
 if(o.torque_limit_ratio!==undefined)rows.push(['Peak actuator limit',`${Math.round(o.torque_limit_ratio*100)}%`]);
 if(o.force_stop!==undefined)rows.push(['Cutting force guard',o.force_stop?'Triggered · stopped cut':'55 N threshold']);
 const list=$('physics-readouts');list.replaceChildren();for(const [label,value] of rows){const dt=document.createElement('dt'),dd=document.createElement('dd');dt.textContent=label;dd.textContent=value;list.append(dt,dd)}
 const details=JSON.stringify(s.experiment.physics,null,2);if($('physics-assumptions').textContent!==details)$('physics-assumptions').textContent=details;
}
