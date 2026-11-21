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
 const supported=usesSceneControls()&&typeof s.playback_speed==='number';
 $('playback-control').hidden=!supported;if(!supported)return;
 if(!speedRequestPending)document.querySelectorAll('[data-playback-speed]').forEach(button=>{
  const selected=Number(button.dataset.playbackSpeed)===s.playback_speed;
  button.classList.toggle('active',selected);button.setAttribute('aria-pressed',String(selected));
 });
 const actual=s.achieved_speed;
 $('playback-timing').textContent=typeof actual==='number'?`${s.running?'Actual':'Last run'} ${actual.toFixed(2)}×`:'';
}
const $ = id => document.getElementById(id);
const methodPicker = $('control-method');
let state = {}, pendingJob = null, pendingRun = false, pendingVla = false, requestError = null;
let method = methodPicker.value || 'expert';
const seeds = {expert: '0', qwen: '0', vla: '20003'};
const usesSceneControls = () => method === 'expert' || method === 'qwen';
const currentPolicy = () => method === 'qwen' ? 'student' : 'expert';
const visualEpisode = s => method === 'vla' && Boolean(pendingVla || s.vla?.running ||
 (s.vla?.ready && s.vla?.id));
const visualRun = s => visualEpisode(s) && Boolean(pendingVla || s.vla?.running || s.vla?.has_frame);

async function post(url,data){$('error').textContent='';try{let r=await fetch(url,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(data)});let d=await r.json();if(!r.ok)throw Error(d.error);return d}catch(e){$('error').textContent=e.message;throw e}}

async function control(operation) {
 if (operation !== 'stop' && !$('seed').reportValidity()) return;
 if (method === 'vla') {
  try {
   if (operation === 'reset') {
    await post('/api/vla', {operation, seed: Number($('seed').value)});
    state.vla = {...state.vla, id:null, phase:'idle', has_frame:false, report_url:null, trace:[]};
   } else if (operation === 'stop') {
    await post('/api/vla', {operation});
    state.vla = {...state.vla, phase: 'stopping'};
   } else if (operation === 'run' && state.vla?.supported_scene && state.vla.ready) {
    pendingVla = true;
    renderWorkspace(state);
    const result = await post('/api/vla', {operation, checkpoint: state.vla.recommended_id, seed: Number($('seed').value)});
    state.vla = {...state.vla, id: result.id, running: true, phase: 'loading', has_frame: false,
      frame_revision: null, report_url: null, tick: undefined};
   }
  } catch {} finally {
   pendingVla = false;
   renderWorkspace(state);
  }
  return;
 }
 if (!usesSceneControls()) return;
 if (operation === 'run') {
  pendingRun = true;
  renderWorkspace(state);
 }
 try {
  await post('/api/control', {operation, seed: Number($('seed').value), policy: currentPolicy()});
 } catch {
  pendingRun = false;
  renderWorkspace(state);
 }
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
 if(!state.ready||visualRun(state)||!([0,2].includes(e.button)))return;
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
viewport.addEventListener('wheel',e=>{if(!state.ready||visualRun(state))return;e.preventDefault();orbit.distance=Math.max(.22,Math.min(1.8,orbit.distance*Math.exp(e.deltaY*.0012)));scheduleOrbit()},{passive:false});

const JOBS={
 'vlm-inspect':{title:'Image inspection',working:'Inspecting frame'},
 'vlm-benchmark':{title:'Visual inspection benchmark',working:'Scoring visual inspection'},
 'vlm-model':{title:'Local model check',working:'Checking the vision model'},
 'vlm-frames':{title:'Inspection frame preparation',working:'Preparing inspection frames'},
 'vla-check':{title:'Fixed-tick expert check',working:'Checking the visual expert'},
 'vla-record':{title:'Visual demonstration recording',working:'Recording camera demonstrations'},
 'vla-prepare':{title:'Visual dataset preparation',working:'Preparing visual data'},
 'vla-train':{title:'Visual policy training',working:'Training visual policy'},
 'vla-test':{title:'Visual policy test',working:'Testing visual policy'},
 'vla-evaluate':{title:'Visual policy evaluation',working:'Evaluating visual policy'},
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
function supportsQwenTraining(s) {
 return Boolean(s.experiment && s.experiment.available !== false && s.experiment.physics?.learning_enabled !== false);
}
function qwenStatus(s) {
 if (!s.experiment) return 'Checking Qwen availability…';
 if (!supportsQwenTraining(s)) return 'This is a scripted trial. Qwen training is unavailable for this scene.';
 if (!s.adapter_ready) return 'Qwen has not been trained for this scene.';
 if (!s.model_ready) return 'The Qwen base model is missing. Run the Chora setup to restore it.';
 return '';
}
function trainingStatus(s, active) {
 if (active.running) return ['running', JOBS[active.name]?.working || 'Working'];
 if (s.switching || pendingExperiment) return ['running','Loading scene'];
 if (s.vla?.running) return ['running','Running visual policy'];
 if (s.running) return ['running','Running episode'];

 const stages = Object.values(methodStages(s));
 const byName = new Map(stages.map(([name, title]) => [name, title]));
 const latest = byName.has(active.name) ? active : Object.values(s.jobs || {})
  .filter(job => byName.has(job.name))
  .sort((a,b) => (b.started_at || 0) - (a.started_at || 0))[0];
 if (latest && ['failed','cancelled'].includes(latest.status)) {
  const outcome = latest.status === 'cancelled' ? 'Stopped' : 'Job failed';
  return ['failed', `${byName.get(latest.name)} · ${outcome}`];
 }

 if (method === 'vla' && s.vla?.supported_scene) {
  const workflow = s.vla.workflow || {};
  if (workflow.evaluation_outcome === 'did-not-pass') return ['failed','Evaluate the VLA · Did not pass'];
  if (s.vla.ready) return ['complete','VLA ready to run'];
  if (workflow.evaluation_outcome === 'passed') return ['pending','Evaluation passed · Checkpoint unavailable'];
  if (workflow.trained) return ['pending','Training complete · Evaluate the VLA'];
  if (!workflow.ready) return ['pending','VLA setup required'];
 }
 if (method === 'qwen' && s.adapter_ready) {
  return s.results?.student ? ['complete','Student evaluation complete'] : ['pending','Training complete · Evaluate student'];
 }
 if (method === 'vlm' && s.vlm?.benchmark) return ['complete','VLM benchmark complete'];
 const next = stages.find(([, , , , saved]) => !saved);
 return next ? ['pending', `Next · ${next[1]}`] : ['', ''];
}

function renderTrainingShortcut(s) {
 const visualTraining = method === 'vla' && s.vla?.supported_scene;
 const inspectionPipeline = method === 'vlm' && s.vlm?.supported;
 const qwenTraining = usesSceneControls() && supportsQwenTraining(s);
 const shortcut = $('training-shortcut');
 shortcut.hidden = !visualTraining && !qwenTraining && !inspectionPipeline;
 const active = uiJob(s);
 const [status, label] = trainingStatus(s, active);
 const progress = active.progress || {};
 const counts = active.running && Number.isFinite(progress.current) && progress.total > 0 ?
  ` · ${progress.current.toLocaleString()}/${progress.total.toLocaleString()}` : '';
 $('training-tools').hidden = shortcut.hidden && !label;
 const activity = $('training-activity');
 activity.hidden = !label;
 activity.dataset.status = status;
 activity.querySelector('.spinner').hidden = status !== 'running';
 if ($('training-activity-label').textContent !== label + counts) $('training-activity-label').textContent = label + counts;
 shortcut.disabled = Boolean((method === 'expert' && methodPicker.disabled) || s.switching || pendingExperiment);
 shortcut.textContent = inspectionPipeline ? 'Go to inspection pipeline ↓' : 'Go to training ↓';
}
$('training-shortcut').addEventListener('click', () => {
 if (method === 'expert') selectMethod('qwen');
 const heading = $('pipeline-title');
 heading.focus({preventScroll: true});
 heading.scrollIntoView({behavior: matchMedia('(prefers-reduced-motion: reduce)').matches ? 'instant' : 'smooth'});
});

function renderViewer(s) {
 const v = s.vla || {};
 const isVisual = method === 'vla';
 const visual = visualRun(s);
 const running = Boolean(isVisual ? v.running || pendingVla : s.running || pendingRun);
 const preparing = Boolean(s.switching || (isVisual ? pendingVla || v.phase === 'loading' : pendingRun || ['preparing','loading_policy'].includes(s.phase)));
 const blocked = Boolean(uiJob(s).running || s.switching || pendingExperiment || (isVisual ? s.running : v.running));
 const canRun = isVisual ? Boolean(v.supported_scene && v.ready && v.recommended_id) :
  method === 'expert' || (method === 'qwen' && !qwenStatus(s));
 const availability =  method === 'planner' ? 'Planned pilot: a Qwen planner calls checked robot skills and uses simulation feedback to recover.' :
  isVisual && !v.supported_scene ? 'VLA currently supports Bowl · Standard platform.' :
  isVisual && !canRun ? vlaRunRequirement(v) :
  method === 'qwen' ? (qwenStatus(s) ||
   (!s.results?.student ? 'Trained and ready to run. Evaluate student to publish scores and gallery previews.' : '')) : '';
 $('method-status').hidden = !availability;
 $('method-status').textContent = availability;
 $('experiment-controls').hidden = method === 'vlm';
 $('seed-field').hidden = !usesSceneControls() && !isVisual;
 $('episode-buttons').hidden = !usesSceneControls() && !isVisual;
 $('seed').disabled = running || blocked;
 $('run').disabled = !s.ready || running || blocked || !canRun;
 $('reset').hidden = !usesSceneControls() && !isVisual;
 $('reset').disabled = !s.ready || running || blocked || (isVisual && !canRun);
 $('stop').disabled = isVisual ? !v.running || v.phase === 'stopping' : !s.running;
 buttonLabel($('run'), running ? (isVisual && v.phase === 'stopping' ? 'Stopping…' : preparing ? 'Preparing…' : 'Running…') : 'Run episode', running);

 $('camera-tabs').hidden = visual;
 $('orbit-hint').hidden = visual;
 const activity = $('viewer-activity');
 const activityLabel = !s.ready ? 'Preparing scene…' : s.switching ? 'Loading scene…' :
  running ? (preparing ? 'Loading policy…' : isVisual ? 'VLA running' : method === 'qwen' ? 'Qwen running' : 'Expert running') : '';
 activity.hidden = !activityLabel;
 if (activityLabel && activity.dataset.label !== activityLabel) {
  const spinner = document.createElement('span');
  spinner.className = 'spinner';
  spinner.setAttribute('aria-hidden', 'true');
  activity.replaceChildren(spinner, document.createTextNode(activityLabel));
  activity.dataset.label = activityLabel;
 }
 document.querySelector('.viewer').setAttribute('aria-busy', String(running || !s.ready || Boolean(s.switching)));
 $('mode').textContent = visual ? 'VLA · FRONT + WRIST' : usesSceneControls() && canRun ? 'PHYSICS SIMULATION' : 'SCENE PREVIEW';
}
function savedSummary(name,s){
 if(name==='check'&&s.oracle)return `Last completed check: ${s.oracle.successes}/${s.oracle.episodes} scenes successful.`;
 if(name==='generate'&&s.generation)return `${s.generation.attempts.toLocaleString()} demonstrations saved · ${s.generation.successes.toLocaleString()} successful.`;
 if(name==='curate'&&s.curation?.selected_episodes)return `${s.curation.selected_episodes.toLocaleString()} episodes selected · ${s.curation.training_pairs.toLocaleString()} training examples.`;
 if(name==='train'&&s.adapter_ready)return 'A trained Qwen policy is ready. Select it above to run an episode.';
 if(name==='evaluate'&&s.results?.student){const r=s.results.student;return `Last completed evaluation: ${r.successes}/${r.episodes} scenes successful.`}
 return '';
}
const pipelineCards = [...document.querySelectorAll('[data-stage]')].map(card => ({
 card, key:card.dataset.stage, heading:card.querySelector('h3'), description:card.querySelector('.stage-description'),
 title:card.querySelector('h3').textContent, detail:card.querySelector('.stage-description').textContent,
 order:card.querySelector('.n').textContent
}));
const resultIds = {check:'verifyText',generate:'collectText',curate:'curateText',train:'trainText',evaluate:'evalText'};
const ratio = score => score ? `${score.successes}/${score.episodes}` : '';

function vlaRunRequirement(visual) {
 const {familiar, unseen, trained} = visual.workflow || {};
 if (visual.workflow?.evaluation_outcome === 'did-not-pass') {
  if (unseen) return 'Held-out evaluation fell short. Run episode requires at least 8 of 10 placements with zero collisions.';
  const collisions = familiar.collisions ? ` with ${familiar.collisions} collision${familiar.collisions === 1 ? '' : 's'}` : '';
  return `Familiar evaluation passed ${ratio(familiar)} trials${collisions}. Both must pass without collisions before held-out evaluation can start.`;
 }
 if (trained && !unseen) return 'Training finished. Evaluate the checkpoint to qualify it for live runs.';
 return visual.unavailable_reason || 'A visual policy is not available for this scene.';
}

function methodStages(s) {
 if (method === 'vla') {
  if (!s.vla?.supported_scene) return {};
  const v = s.vla.workflow || {}, score = v.unseen || v.familiar;
  return {
   check:['vla-check','Check the visual expert','Verify 10 scenes with the fixed-tick teacher.',v.ready,ratio(v.expert),'Run expert check'],
   generate:['vla-record','Record demonstrations','Capture front and wrist images plus joint targets in 12 expert episodes.',
    v.ready&&v.expert?.successes>=9,v.recording&&`${ratio(v.recording)} successful recordings`,'Record demonstrations'],
   curate:['vla-prepare','Prepare the visual dataset','Split whole episodes and normalize joint targets.',
    v.ready&&v.recording?.successes>=2,v.dataset&&`${v.dataset.train} train · ${v.dataset.validation} validation episodes`,'Prepare dataset'],
   train:['vla-train','Train the camera policy','Fit the compact direct-action policy to expert sequences.',
    v.ready&&v.dataset?.train&&v.dataset?.validation,v.trained&&`${v.trained.toLocaleString()} training updates saved`,v.trained?'Retrain VLA':'Train VLA'],
   evaluate:['vla-evaluate','Evaluate the VLA','Check two familiar scenes before ten held-out scenes. Run episode requires qualification.',
    v.ready&&v.trained,score&&`${ratio(score)} ${v.unseen?'held-out':'familiar'} placements · ${score.collisions} collision${score.collisions === 1 ? '' : 's'}`,'Evaluate VLA']
  };
 }
 if (method === 'vlm') {
  if (!s.vlm?.supported) return {};
  const v = s.vlm, score = v.benchmark;
  return {
   check:['vlm-model','Check the local model','Load the pretrained Qwen3-VL inspector.',v.ready,v.ready?'Pretrained model installed':'','Check model'],
   generate:['vlm-frames','Prepare inspection frames','Render verified overhead frames from saved expert or student rollouts.',
    v.can_prepare,v.frames&&`${v.frames} saved overhead frames`,'Prepare frames'],
   evaluate:['vlm-benchmark','Evaluate visual answers','Compare the model’s frame verdicts with simulator geometry on a balanced sample.',
    v.ready&&v.can_benchmark,score&&`${score.correct}/${score.examples} matched simulator geometry`,'Evaluate VLM']
  };
 }
 if (!usesSceneControls() || !supportsQwenTraining(s)) return {};
 const ready = {check:true,generate:s.oracle?.success_rate>=.9,curate:Boolean(s.generation),
  train:Boolean(s.curation?.selected_episodes&&s.model_ready),evaluate:s.adapter_ready&&s.model_ready};
 return Object.fromEntries(pipelineCards.filter(({key}) => method==='qwen'||key==='check').map(({key,title,detail}) =>
  [key,[key,title,detail,ready[key],savedSummary(key,s),key==='train'&&s.adapter_ready?'Retrain student':JOBS[key].button]]));
}

function renderPipeline(s){
 const stages = methodStages(s), jobNames = Object.values(stages).map(stage => stage[0]);
 const visualMethod = method === 'vla' || method === 'vlm';
 const showStages = jobNames.length > 0;
 const pipeline = document.querySelector('.pipeline');
 pipeline.hidden = !s.experiment || (!showStages && !visualMethod);
 $('pipeline-title').textContent = ({expert:'Expert verification',qwen:'Learning pipeline',vla:'VLA learning pipeline',vlm:'VLM inspection pipeline'})[method] || 'Learning pipeline';
 const note = $('pipeline-note');
 note.hidden = !visualMethod;
 note.textContent = method === 'vlm' ?
  (!s.vlm?.supported ? 'Visual inspection supports the standard bowl and raised-shelf block.' :
   !s.vlm.ready ? 'Install the local pretrained model to run inspection. Setup instructions are in docs/VLM.md.' :
   'This workflow evaluates a pretrained model. Open Gallery to inspect or recheck individual frames.') :
  !s.vla?.supported_scene ? 'Camera-based action training currently supports the standard bowl.' :
  !s.vla.workflow?.ready ? 'Set up the isolated VLA environment to run these stages.' :
  'Pass a held-out rollout evaluation before running the camera policy in the viewer.';
 $('job').hidden = !showStages;
 pipeline.querySelector('.stages').hidden = !showStages;
 const vlaOutcome = s.vla?.workflow?.evaluation_outcome;
 const current=uiJob(s), visibleJob=jobNames.includes(current.name), jobBusy=Boolean(current.running),
  busy=jobBusy||s.switching||s.vla?.running, history=s.jobs||{};
 for(const {key, card, heading, description, order} of pipelineCards){
  const stage = stages[key];
  card.hidden = !stage;
  if (!stage) continue;
  const [name, title, detail, ready, saved, action] = stage;
  heading.textContent = title;
  description.textContent = detail;
  card.querySelector('.n').textContent = method === 'vlm' ? ({check:'01 / MODEL',generate:'02 / FRAMES',evaluate:'03 / EVALUATE'})[key] : order;
  const meta=JOBS[name], button=card.querySelector('button[data-job]');
  button.dataset.job = name;
  const record=current.name===name?current:(history[name]||{});
  const running = current.name === name && jobBusy;
  const status = running ? 'running' : ['failed','cancelled'].includes(record.status) ?
   record.status : (name === 'vla-evaluate' && vlaOutcome) || (saved ? 'complete' : ready ? 'ready' : 'pending');
  card.dataset.status=status;card.setAttribute('aria-busy',String(running));
  const statusText = running ? (current.pending ? 'Starting' : 'Running') :
   {complete:'✓ Complete',passed:'✓ Passed','did-not-pass':'Did not pass',failed:'Job failed',cancelled:'Stopped',ready:'Ready',pending:'Pending'}[status];
  $('stage-status-'+key).textContent=statusText;
  $(resultIds[key]).textContent=saved||'No saved result yet.';
  const progress=record.progress||{},total=Number(progress.total),done=Number(progress.current),determinate=total>0&&Number.isFinite(done);
  const bar=$('stage-progress-'+key);bar.hidden=!running;bar.classList.toggle('indeterminate',!determinate);
  if(determinate){const percentage=Math.max(0,Math.min(100,100*done/total));bar.firstElementChild.style.width=percentage+'%';bar.setAttribute('aria-valuemin','0');bar.setAttribute('aria-valuemax',String(total));bar.setAttribute('aria-valuenow',String(done))}
  else{bar.removeAttribute('aria-valuenow');bar.firstElementChild.style.width='35%'}
  bar.setAttribute('aria-valuetext',progress.message||statusText);
  button.disabled=busy||s.running||pendingRun||!ready;
  const label = running ? (current.pending ? 'Starting…' : meta.working +
   (determinate ? ` ${done.toLocaleString()} / ${total.toLocaleString()}` : '…')) : action;
  buttonLabel(button,label,running);
  button.title=running?(progress.message||meta.working):busy?'Wait for the current stage to finish.':
   s.running?'Stop the live episode first.':!ready?'Finish the preceding stages first.':
   name==='train'&&s.adapter_ready?'Train from the base model on the curated data. The previous run will be archived.':action;
 }
 if(!current.name||!visibleJob){$('job').style.display='none';return}
 const meta=JOBS[current.name]||{title:'Pipeline stage'},status=current.status||(current.running?'running':current.exit_code===0?'complete':'idle');
 $('job').style.display='block';$('job').dataset.status=status;$('job').setAttribute('aria-busy',String(jobBusy));$('job-spinner').hidden=!jobBusy;
 const evaluationResult = current.name === 'vla-evaluate' && status === 'complete' ? vlaOutcome : null;
 $('jobTitle').textContent = evaluationResult === 'did-not-pass' ? 'VLA evaluation did not pass' : meta.title+(jobBusy?(current.pending?' is starting':' is running'):status==='complete'?' complete':status==='failed'?' failed':status==='cancelled'?' stopped':'');
 const message=current.progress?.message;
 $('job-summary').textContent=status==='failed'?(current.error||'This stage could not finish. Open Run details for the error.'):message&&message!=='Starting…'?message:jobBusy?'Preparing this stage. Progress will appear here.':(Object.values(stages).find(stage=>stage[0]===current.name)?.[4]||'This stage has finished.');
 const log=(current.log||'').trim();$('job-details').hidden=!log;$('jobLog').textContent=log;
 if($('job-details').open)$('jobLog').scrollTop=$('jobLog').scrollHeight;
}
async function job(name){
 if(pendingJob||uiJob(state).running)return;
 requestError=null;pendingJob={name,id:null};renderWorkspace(state);
 try{const result=await post('/api/job',{name});pendingJob.id=result.job_id}
 catch(error){pendingJob=null;requestError={name,status:'failed',running:false,error:error.message};renderWorkspace(state)}
}
let experimentCatalog=[],pendingExperiment=null;
const selectMeasure=document.createElement('canvas').getContext('2d');
function fitPickers() {
 for (const picker of [$('experiment-picker'), methodPicker]) {
  const option = picker.selectedOptions[0];
  if (!option) continue;
  selectMeasure.font = getComputedStyle(picker).font;
  const spinnerWidth = picker.id === 'experiment-picker' && !$('experiment-loading').hidden ? 24 : 0;
  const available = picker.closest('.experiment-fields').clientWidth - spinnerWidth;
  picker.style.width = Math.min(Math.ceil(selectMeasure.measureText(option.textContent).width) + 52, Math.max(0, available)) + 'px';
 }
}
addEventListener('resize', fitPickers);
document.fonts?.ready.then(fitPickers);
let requestedExperiment = new URLSearchParams(location.search).get('experiment');

async function loadExperimentCatalog() {
 const response = await fetch('/api/experiments');
 if (!response.ok) throw Error('Could not load experiments');
 const data = await response.json();
 experimentCatalog = data.experiments;
 const picker = $('experiment-picker');
 picker.replaceChildren();
 for (const item of experimentCatalog) {
  const option = document.createElement('option');
  option.value = item.id;
  option.textContent = item.name + (item.available === false ? ' · asset unavailable' : '');
  option.disabled = item.available === false;
  option.title = item.unavailable_reason || '';
  picker.append(option);
 }
 picker.value = state.experiment?.id || data.active;
 fitPickers();
}

function syncExperiment(s) {
 if (!s.experiment) return;
 if (pendingExperiment === s.experiment.id || (s.error && !s.switching)) pendingExperiment = null;
 const switching = Boolean(s.switching || pendingExperiment);
 const locked = switching || s.running || pendingRun || pendingVla || s.vla?.running || uiJob(s).running;
 $('experiment-picker').value = pendingExperiment || s.experiment.id;
 $('experiment-picker').disabled = Boolean(locked);
 $('experiment-loading').hidden = !switching;
 methodPicker.disabled = Boolean(locked);
 document.querySelector('.experiment-bar').setAttribute('aria-busy', String(switching));
 $('experiment-note').textContent = switching ? 'Loading scene…' : s.experiment.description;
 $('task-instructions').textContent = s.experiment.instructions;
 $('robot-label').textContent = (s.experiment.robot_name || 'SO-101') + ' / MuJoCo';
 const query = new URLSearchParams({experiment: s.experiment.id, policy: method});
 const labUrl = '/?' + query;
 document.querySelector('[data-nav="lab"]').href = labUrl;
 document.querySelector('.brand').href = labUrl;
 if (!switching && (!requestedExperiment || requestedExperiment === s.experiment.id) && location.search !== '?' + query) {
  history.replaceState(null, '', labUrl);
 }
 for (const page of ['gallery', 'reports']) document.querySelector(`[data-nav="${page}"]`).href = '/' + page + '?' + query;
 $('footer-gallery').href = '/gallery?' + query;
 $('footer-report').href = '/reports?' + query;
 $('footer-curation').hidden = $('footer-divider').hidden = method !== 'qwen' || s.experiment.physics?.learning_enabled === false;
 $('footer-curation').href = (s.output_url || '/outputs') + '/curation_report.html';
 if (requestedExperiment && experimentCatalog.length && !switching && !locked) {
  const selected = experimentCatalog.find(item => item.id === requestedExperiment && item.available !== false)?.id || 'so101-bowl';
  requestedExperiment = null;
  if (selected !== s.experiment.id) selectExperiment(selected);
 }
 fitPickers();
}

async function selectExperiment(id) {
 if (id === state.experiment?.id) return;
 requestError = null;
 pendingExperiment = id;
 renderWorkspace(state);
 try { await post('/api/experiment', {id}); }
 catch { pendingExperiment = null; renderWorkspace(state); }
}

function selectMethod(value, refresh = true) {
 if (value !== method && method in seeds) seeds[method] = $('seed').value;
 method = value;
 methodPicker.value = value;
 if (method in seeds) $('seed').value = seeds[method];
 $('seed').min = method === 'vla' ? 20000 : 0;
 $('seed').max = method === 'vla' ? 29999 : 999999;
 $('seed-label').textContent = method === 'vla' ? 'Development scene seed' : 'Scene seed';
 $('method-note').textContent = methodPicker.selectedOptions[0].dataset.description;
 $('error').textContent = '';
 if (refresh) renderWorkspace(state);
}
methodPicker.addEventListener('change', () => selectMethod(methodPicker.value));
selectMethod(method, false);
loadExperimentCatalog().catch(error => $('error').textContent = error.message);

let loadingFrame = false, frameRevision = null;
function frameSource(s) {
 if (visualRun(s)) return {key: 'vla:' + (pendingVla ? 'pending' : s.vla?.id), visual: true};
 return {key: 'scene:' + (pendingExperiment || s.experiment?.id), visual: false};
}
function syncFrame(s) {
 const source = frameSource(s);
 if (viewport.dataset.frameKey !== source.key) {
  viewport.dataset.frameKey = source.key;
  viewport.dataset.source = source.visual ? 'vla' : 'scene';
  const image = $('frame');
  if (image.src.startsWith('blob:')) URL.revokeObjectURL(image.src);
  image.removeAttribute('src');
  image.style.visibility = 'hidden';
  image.alt = source.visual ? 'Live VLA front and wrist cameras' : 'Live robot simulation';
  $('loading').style.display = '';
  $('loading').lastChild.textContent = source.visual ? ' Loading visual policy…' : ' Preparing the simulation…';
  frameRevision = null;
 }
}
async function frame() {
 const source = frameSource(state), v = state.vla;
 if (loadingFrame || !state.ready || pendingExperiment || state.switching ||
 (source.visual && (pendingVla || !v?.has_frame || (!v.running && frameRevision === v.frame_revision)))) return;
 loadingFrame = true;
 try {
  const path = source.visual ? '/api/vla/frame.jpg' : '/frame.jpg';
  const response = await fetch(path + '?t=' + Date.now());
  if (!response.ok) return;
  const blob = await response.blob();
  if (viewport.dataset.frameKey !== source.key) return;
  const image = $('frame'), old = image.src;
  image.src = URL.createObjectURL(blob);
  image.style.visibility = 'visible';
  $('loading').style.display = 'none';
  if (old.startsWith('blob:')) URL.revokeObjectURL(old);
  if (source.visual) frameRevision = v.frame_revision;
 } catch {} finally { loadingFrame = false; }
}

function updateConnection(s){
 const indicator=document.querySelector('.connection-indicator');
 const job=uiJob(s),working=s.running||s.switching||job.running||s.vla?.running;
 let label=s.vla?.running?'VLA test running · on device':s.running?'Episode running':s.switching?'Loading experiment…':job.running?(JOBS[job.name]?.working||'Working')+' · on device':s.ready?'Lab online · on device':'Preparing simulation…';
 if(s.error&&!s.ready)label='Lab unavailable';
 if($('connection').textContent!==label)$('connection').textContent=label;
 indicator.dataset.state=s.error&&!s.ready?'offline':working||!s.ready?'busy':'online';
 indicator.title=label;$('dot').classList.toggle('pulse',Boolean(working));
}
function renderMeasurements(s) {
 const isVla = method === 'vla', isVlm = method === 'vlm';
 const classic = usesSceneControls() && s.experiment?.physics?.learning_enabled !== false;
 const vla = s.vla?.workflow || {}, vlm = s.vlm?.benchmark;
 const supported = isVla ? s.vla?.supported_scene : isVlm ? s.vlm?.supported : classic;
 $('metrics-card').hidden = !supported;
 $('evaluation-links').hidden = !(isVla || isVlm);
 const query = new URLSearchParams({experiment:s.experiment?.id || 'so101-bowl', policy:method});
 $('evaluation-gallery').href = '/gallery?' + query;
 $('evaluation-gallery').textContent = isVlm ? 'Inspect saved frames ↗' : 'Gallery ↗';
 $('evaluation-report').href = '/reports?' + query;
 $('expert-metric').hidden = false;
 $('student-metric').hidden = method === 'expert';
 const expertTest = s.results?.expert, expert = expertTest || s.oracle, student = s.results?.student;
 let first = [expertTest ? 'EXPERT TEST' : 'EXPERT CHECK', ratio(expert) || '—',
  expertTest ? 'Held-out scenes' : expert ? 'Randomized scene check' : 'Expert check pending'];
 let second = ['STUDENT TEST', ratio(student) || '—',
  student?.complete_benchmark ? '100 held-out scenes' : student ? 'Partial evaluation' : 'Evaluation pending'];
 let note = method === 'qwen' ? 'Student scores use scenes reserved for evaluation.' : '';
 if (isVla) {
  const score = vla.unseen || vla.familiar;
  first = ['VISUAL EXPERT CHECK', ratio(vla.expert) || '—', 'Fixed-tick scripted expert'];
  second = ['VLA PLACEMENTS', ratio(score) || '—', score ?
   `${vla.unseen ? 'Held-out' : 'Familiar'} scenes · ${score.collisions} collision${score.collisions === 1 ? '' : 's'}` : 'Evaluation pending'];
  note = s.vla.ready ? 'A qualified visual policy is available for live episodes.' :
   vlaRunRequirement(s.vla);
 } else if (isVlm) {
  first = ['VLM MATCHES', vlm ? `${vlm.correct}/${vlm.examples}` : '—', 'Agreement with simulator geometry'];
  second = ['MEAN INFERENCE', vlm?.latency_ms != null ? `${(vlm.latency_ms/1000).toFixed(1)} s` : '—',
   vlm ? `${vlm.uncertain} uncertain answers` : 'Evaluation pending'];
  note = 'This test checks object fit inside the outline. Gallery contains the images and inspection traces.';
 }
 for (const [ids, data] of [[['expert-metric-label','expertMetric','expertCaption'],first],
                             [['student-metric-label','studentMetric','studentCaption'],second]]) {
  ids.forEach((id,i) => $(id).textContent = data[i]);
 }
 $('metrics-note').hidden = !note;
 $('metrics-note').textContent = note;
}

function renderEpisode(s) {
 const o = s.observation || {}, v = s.vla || {};
 const visual = visualRun(s);
 const match = usesSceneControls() && s.policy === currentPolicy();
 const hammer = ['hammer_nail','hammer_pine','impact_nail'].includes(o.task);
 const suffix = hammer && usesSceneControls() ? ` · NAIL ${Math.round(o.nail_depth*1000)} / ${Math.round(o.target_depth*1000)} mm` :
  o.target_velocity ? ` · PLATFORM ${Math.round(Math.abs(o.target_velocity[1])*1000)} mm/s` : '';
 $('trace-card').hidden = !usesSceneControls() && method !== 'vla';
 $('trace-heading').textContent = method === 'vla' ? 'Replan trace' : 'Action trace';
 $('run-report').hidden = !(visualEpisode(s) && v.report_url && !pendingVla);
 if (!$('run-report').hidden) $('run-report').href = v.report_url;
 $('scene').textContent = visual ? (pendingVla || !v.has_frame ? 'Preparing policy cameras' :
  `SEED ${v.seed} · TICK ${v.tick || 0} / ${v.max_ticks || 240}`) :
  s.ready ? `SEED ${s.seed} · ${match ? 'DECISION ' + (o.step || 0) + ' · ' : ''}${s.camera?.toUpperCase() || 'FRONT'}${suffix}` : 'Waiting for scene';
 let message = '';
 if (visualEpisode(s)) {
  message = pendingVla || v.phase === 'loading' ? 'Loading visual policy…' : v.phase === 'stopping' ? 'Stopping…' :
   v.running ? ('Executing joint targets' + (typeof v.inference_ms === 'number' ? ` · ${v.inference_ms.toFixed(0)} ms last inference` : '')) : v.phase === 'failed' ? (v.error || 'The rollout failed.') :
   v.phase === 'stopped' ? 'Episode stopped.' : v.success ? 'Placement completed.' :
   'Placement not completed · ' + String(v.outcome || 'episode ended').replaceAll('_', ' ');
 } else if (match) {
  message = s.result === 'experiment_complete' ? 'Engineering trial complete · materials uncalibrated' : s.success === true ?
   (hammer ? `✓ Nail driven ${Math.round(o.nail_depth*1000)} mm · hammer clear` :
    o.task === 'cut_fruit' ? '✓ Cut complete · knife clear' : o.task === 'tile_adhesive_bedding' ? '✓ Adhesive and bedding complete · reduced model' :
    o.task === 'tile_dry_transfer' ? '✓ Four dry tiles placed · installation materials pending' : '✓ Placement completed successfully') :
   s.success === false ? 'Episode ended · ' + s.result : s.running ? 'Executing ' + (s.history.at(-1)?.action || 'first action') : '';
 }
 $('result').textContent = message;
 $('step').textContent = method === 'vla' ? (visual ? `${v.tick || 0} ticks` : '') : `${match ? o.step || 0 : 0} / ${s.experiment?.simulation?.max_steps || 15}`;
 const trace = method === 'vla' ? (visual ? v.trace || [] : []) : match ? s.history || [] : [];
 $('actions').replaceChildren();
 if (!trace.length) {
  const item = document.createElement('li');
  item.textContent = method === 'vla' ?
   'A live run shows replans here. Saved evaluation traces are in Gallery.' : 'Run an episode to see each decision.';
  $('actions').append(item);
 }
 trace.forEach((action, i) => {
  const item = document.createElement('li'), n = document.createElement('span'), value = document.createElement('span');
  n.className = 'num';
  n.textContent = method === 'vla' ? String(action.tick) : String(i + 1).padStart(2, '0');
  value.textContent = method === 'vla' ? `Replan · ${action.milliseconds.toFixed(1)} ms` +
   (action.first_target_rad ? ' · ' + action.first_target_rad.map(v=>v.toFixed(2)).join(', ') + ' rad' : '') : actionLabel(action);
  item.append(n, value);
  $('actions').append(item);
 });
 $('actions').scrollTop = $('actions').scrollHeight;
}

function renderWorkspace(s) {
 syncExperiment(s);
 renderViewer(s);
 renderTrainingShortcut(s);
 renderPlayback(s);
 renderPipeline(s);
 renderMeasurements(s);
 renderEpisode(s);
 renderPhysics(s, s.observation || {});
 syncFrame(s);
}

async function poll() {
 try {
  const response = await fetch('/api/status');
  if (!response.ok) throw Error('Lab unavailable');
  state = await response.json();
  if (state.running || state.error) pendingRun = false;
  if (state.vla?.running && method !== 'vla') selectMethod('vla');
  if (state.running && !state.vla?.running && method !== (state.policy === 'student' ? 'qwen' : 'expert')) {
   selectMethod(state.policy === 'student' ? 'qwen' : 'expert');
  }
  if (!drag && !orbitTimer && state.orbit) orbit = state.orbit;
  if (state.error) $('error').textContent = state.error;
  updateConnection(state);
  renderWorkspace(state);
  await frame();
 } catch {
  $('connection').textContent = 'Reconnecting to lab…';
  document.querySelector('.connection-indicator').dataset.state = 'offline';
  $('dot').classList.remove('pulse');
 } finally { setTimeout(poll, 700); }
}
poll();
setInterval(frame, 110);

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
