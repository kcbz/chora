const gallery = JSON.parse(document.getElementById('visual-gallery-data').textContent);
const dialog = document.getElementById('episode-dialog');
const frame = document.getElementById('visual-frame');
const description = document.getElementById('visual-description');
const trace = document.getElementById('visual-trace');
const inspect = document.getElementById('inspect-visual-frame');
const report = document.getElementById('visual-raw-report');
const traceDetails = document.getElementById('visual-trace-details');
let selected = null;
let inspecting = false;

function showAnswer(answer) {
 if (!answer) {
  description.textContent = 'No model answer has been saved for this frame.';
  traceDetails.hidden = true;
  return;
 }
 description.textContent = `Model: ${answer.answer} · Simulator: ${answer.expected} · ${answer.correct ? 'match' : 'mismatch'}`;
 document.getElementById('visual-trace-label').textContent = 'Inspection trace';
 traceDetails.hidden = false;
 trace.textContent = [answer.question, 'Raw answer: ' + answer.raw_answer,
  'Inference: ' + answer.latency_ms + ' ms', 'Model: ' + answer.model].join('\n');
}

function openFrame(row) {
 selected = row;
 document.getElementById('episode-title').textContent = `${row.label} · Seed ${row.seed}`;
 frame.hidden = !row.image;
 if (row.image) frame.src = row.image; else frame.removeAttribute('src');
 dialog.querySelector('.episode-media').hidden = !row.image;
 report.hidden = !row.report;
 if (row.report) report.href = row.report;
 inspect.hidden = gallery.method !== 'vlm' || !gallery.ready;
 inspect.disabled = inspecting;
 if (gallery.method === 'vlm') {
  showAnswer(row.answer);
 } else {
  description.textContent = `${row.success ? 'Placement completed' : String(row.reason).replaceAll('_',' ')} · ${row.ticks} ticks · ${row.collisions} collision${row.collisions === 1 ? '' : 's'}`;
  document.getElementById('visual-trace-label').textContent = 'Action replans';
  traceDetails.hidden = false;
  trace.textContent = (row.inferences || []).map(i =>
   `Tick ${i.tick} · ${i.milliseconds.toFixed(1)} ms` + (i.first_target_rad ?
    ' · first joint targets: ' + i.first_target_rad.map(v=>v.toFixed(2)).join(', ') + ' rad' : '')).join('\n') || 'No replans were recorded.';
 }
 dialog.showModal();
}
document.getElementById('close-visual-episode').addEventListener('click', () => dialog.close());
document.querySelectorAll('[data-visual-entry]').forEach(tile => tile.addEventListener('click', () => {
 const row = gallery.rows.find(entry => entry.id === tile.dataset.visualEntry);
 if (row) openFrame(row);
}));

inspect.addEventListener('click', async () => {
 if (inspecting || !selected) return;
 const row = selected;
 inspecting = true;
 inspect.disabled = true;
 inspect.textContent = 'Inspecting…';
 description.textContent = 'Running the model on this overhead image…';
 const args = {experiment: gallery.experiment, episode: row.id};
 try {
  const response = await fetch('/api/vlm/inspect', {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(args)});
  const job = await response.json();
  if (!response.ok) throw Error(job.error);
  for (let i=0; i<100; i++) {
   await new Promise(resolve => setTimeout(resolve, 500));
   const result = await fetch('/api/vlm/inspection?' + new URLSearchParams(args));
   if (!result.ok) throw Error('Could not read the inspection result.');
   const status = await result.json();
   if (status.running || status.job_id !== job.job_id) continue;
   if (status.error) throw Error(status.error);
   if (!status.answer) throw Error('No new answer was saved.');
   row.answer = status.answer;
   if (selected === row && dialog.open) showAnswer(row.answer);
   return;
  }
  throw Error('Inspection is still running. Reopen this frame when it finishes.');
 } catch (error) {
  if (selected === row && dialog.open) description.textContent = error.message;
 } finally {
  inspecting = false;
  inspect.disabled = false;
  inspect.textContent = 'Inspect frame';
 }
});
