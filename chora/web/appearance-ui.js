(()=>{
const $=id=>document.getElementById(id);
const appearance=window.ChoraAppearance;
function refreshAppearance(){
 const p=appearance.get(),now=new Date(),active=appearance.apply(now);
 $('appearance-mode').value=p.mode;
 for(const [id,value] of [['light-palette',p.light],['dark-palette',p.dark],['light-start',p.lightStart],['dark-start',p.darkStart]]){
  if(document.activeElement!==$(id)&&!(id.endsWith('-start')&&$('appearance-error').textContent))$(id).value=value;
 }
 document.querySelectorAll('[data-appearance-mode]').forEach(b=>{
  const selected=b.dataset.appearanceMode===p.mode;b.classList.toggle('active',selected);b.setAttribute('aria-pressed',String(selected));
 });
 const title=active.slot==='light'?'Light':'Dark';
 $('appearance-summary').textContent=(p.mode==='auto'?'Auto · ':'')+title+' · '+appearance.palettes[active.theme];
 $('appearance-clock').textContent='Device time '+now.toLocaleTimeString([], {hour:'2-digit',minute:'2-digit'})+' · '+(p.mode==='auto'?'Auto is showing ':'Showing ')+appearance.palettes[active.theme]+'.';
}
function changeAppearanceMode(mode){
 if(mode==='customize'){
  refreshAppearance();$('appearance-settings').showModal();return;
 }
 appearance.update({mode});refreshAppearance();
}
function choosePalette(slot,theme){appearance.update({[slot]:theme});refreshAppearance()}
function finishAppearanceSettings(){if($('appearance-error').textContent){$('light-start').reportValidity();return}$('appearance-settings').close()}
function changeAppearanceSchedule(){
 const light=$('light-start'),dark=$('dark-start');
 const valid=light.value&&dark.value&&light.value!==dark.value;
 const error=valid?'':'Choose two different switching times.';
 light.setCustomValidity(error);dark.setCustomValidity(error);$('appearance-error').textContent=error;
 if(valid){appearance.update({lightStart:light.value,darkStart:dark.value});refreshAppearance()}
}
refreshAppearance();
setInterval(refreshAppearance,15000);
document.addEventListener('visibilitychange',()=>{if(!document.hidden)refreshAppearance()});
window.addEventListener('focus',refreshAppearance);
window.addEventListener('storage',e=>{if(e.key===appearance.key||e.key===null){appearance.reload();refreshAppearance()}});


Object.assign(window,{changeAppearanceMode,choosePalette,finishAppearanceSettings,changeAppearanceSchedule});
})();
