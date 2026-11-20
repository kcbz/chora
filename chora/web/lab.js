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
   if (operation === 'stop') {
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
function renderTrainingShortcut(s) {
 const visualTraining = method === 'vla' && s.vla?.supported_scene;
 const qwenTraining = usesSceneControls() && supportsQwenTraining(s);
 const shortcut = $('training-shortcut');
 shortcut.hidden = !visualTraining && !qwenTraining;
 shortcut.disabled = Boolean((method === 'expert' && methodPicker.disabled) || s.switching || pendingExperiment);
 shortcut.textContent = visualTraining ? 'Training details ↓' : 'Go to training ↓';
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
 const availability = method === 'vlm' ? 'Scene inspection is planned. A vision-language model has yet to be connected.' :
  method === 'planner' ? 'Task planning is planned. A pretrained action model and skills have yet to be connected.' :
  isVisual && !v.supported_scene ? 'VLA currently supports Bowl · Standard platform.' :
  isVisual && !canRun ? (v.unavailable_reason || 'A visual policy is not available for this scene.') :
  method === 'qwen' ? qwenStatus(s) : '';
 $('method-status').hidden = !availability;
 $('method-status').textContent = availability;
 $('seed-field').hidden = !usesSceneControls() && !canRun && !running;
 $('episode-buttons').hidden = !usesSceneControls() && !canRun && !running;
 $('seed').disabled = running || blocked;
 $('run').disabled = !s.ready || running || blocked || !canRun;
 $('reset').hidden = !usesSceneControls();
 $('reset').disabled = !s.ready || running || blocked;
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
function renderPipeline(s){
 const showStages = supportsQwenTraining(s) && usesSceneControls();
 const pipeline = document.querySelector('.pipeline');
 pipeline.hidden = (!showStages && method !== 'vla') || !s.experiment;
 $('pipeline-title').textContent = method === 'expert' ? 'Expert verification' : 'Learning pipeline';
 const note = $('pipeline-note');
 note.hidden = method !== 'vla';
 note.textContent = !s.vla?.supported_scene ? 'Visual training currently supports the standard bowl scene.' :
  s.vla.ready ? 'The evaluated visual policy can run above. Its development evaluation is in Reports.' :
  'Visual training uses camera recordings and joint targets. Recording, training, and evaluation are run from the command line.';
 $('job').hidden = !showStages;
 pipeline.querySelector('.stages').hidden = !showStages;
 const current=uiJob(s),visibleJob=showStages&&(!current.name?.startsWith('vla-'))&&(method==='qwen'||current.name==='check'),jobBusy=Boolean(current.running),busy=jobBusy||s.switching||s.vla?.running,history=s.jobs||{};
 const resultIds={check:'verifyText',generate:'collectText',curate:'curateText',train:'trainText',evaluate:'evalText'};
 for(const [name,id] of Object.entries(resultIds)){
  const meta=JOBS[name],card=document.querySelector(`[data-stage="${name}"]`),button=card.querySelector('button[data-job]');
  card.hidden = !showStages || (method === 'expert' && name !== 'check');
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
  const ready=name==='generate'?s.oracle?.success_rate>=.9:name==='curate'?Boolean(s.generation):name==='train'?Boolean(s.curation?.selected_episodes&&s.model_ready&&!s.adapter_ready):name==='evaluate'?s.adapter_ready&&s.model_ready:true;
  button.disabled=busy||s.running||pendingRun||!ready;
  let label=running?(current.pending?'Starting…':meta.working+(determinate?` ${done.toLocaleString()} / ${total.toLocaleString()}`:'…')):meta.button+' ↗';
  if(name==='train'&&s.adapter_ready&&!running)label='Student ready';
  buttonLabel(button,label,running);
  button.title=running?(progress.message||meta.working):busy?'Wait for the current stage to finish.':s.running?'Stop the live episode first.':name==='train'&&s.adapter_ready?'A saved student is already available in the Policy selector.':!ready?'Finish the preceding stages first.':meta.button;
 }
 if(!current.name||!visibleJob){$('job').style.display='none';return}
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
 const expert = s.oracle, student = s.results?.student;
 const visual = method === 'vla' && s.vla?.ready && s.vla?.supported_scene ?
  s.vla.candidates?.[0]?.evaluation : null;
 const skillTrial = s.experiment?.physics?.learning_enabled === false;
 $('metrics-card').hidden = !(visual || (usesSceneControls() && !skillTrial));
 $('expert-metric').hidden = method === 'vla';
 $('student-metric').hidden = method === 'expert';
 $('expertMetric').textContent = expert ? `${expert.successes}/${expert.episodes}` : '—';
 $('expertCaption').textContent = expert ? 'Randomized scene check' : 'Expert check pending';
 $('student-metric-label').textContent = visual ? 'VISUAL POLICY' : 'STUDENT TEST';
 $('studentMetric').textContent = visual ? `${visual.unseen_successes}/${visual.unseen_episodes}` :
  student ? `${student.successes}/${student.episodes}` : '—';
 $('studentCaption').textContent = visual ? 'Unseen development scenes' : student ?
  (student.complete_benchmark ? '100 held-out scenes' : 'Partial evaluation') : 'Evaluation pending';
 $('metrics-note').hidden = method !== 'qwen';
}

function renderEpisode(s) {
 const o = s.observation || {}, v = s.vla || {};
 const visual = visualRun(s);
 const match = usesSceneControls() && s.policy === currentPolicy();
 const hammer = ['hammer_nail','hammer_pine','impact_nail'].includes(o.task);
 const suffix = hammer && usesSceneControls() ? ` · NAIL ${Math.round(o.nail_depth*1000)} / ${Math.round(o.target_depth*1000)} mm` : '';
 $('trace-card').hidden = !usesSceneControls();
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
 $('step').textContent = `${match ? o.step || 0 : 0} / ${s.experiment?.simulation?.max_steps || 15}`;
 const trace = match ? s.history || [] : [];
 $('actions').replaceChildren();
 if (!trace.length) {
  const item = document.createElement('li');
  item.textContent = 'Run an episode to see each decision.';
  $('actions').append(item);
 }
 trace.forEach((action, i) => {
  const item = document.createElement('li'), n = document.createElement('span'), value = document.createElement('span');
  n.className = 'num';
  n.textContent = String(i + 1).padStart(2, '0');
  value.textContent = actionLabel(action);
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
