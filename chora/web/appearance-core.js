window.ChoraAppearance=(()=>{
 const key='chora.appearance';
 const palettes={porcelain:'Porcelain',cobalt:'Cobalt',graphite:'Graphite',midnight:'Midnight',sage:'Sage'};
 const defaults={mode:'auto',light:'porcelain',dark:'graphite',lightStart:'07:00',darkStart:'19:00'};
 const validTime=t=>typeof t==='string'&&/^([01]\d|2[0-3]):[0-5]\d$/.test(t);
 const minutes=t=>Number(t.slice(0,2))*60+Number(t.slice(3));
 function normalize(value){
  const p=value&&typeof value==='object'?value:{};
  const out={mode:['light','dark','auto'].includes(p.mode)?p.mode:defaults.mode,
   light:Object.hasOwn(palettes,p.light)?p.light:defaults.light,
   dark:Object.hasOwn(palettes,p.dark)?p.dark:defaults.dark,
   lightStart:validTime(p.lightStart)?p.lightStart:defaults.lightStart,
   darkStart:validTime(p.darkStart)?p.darkStart:defaults.darkStart};
  if(out.lightStart===out.darkStart){out.lightStart=defaults.lightStart;out.darkStart=defaults.darkStart}
  return out;
 }
 function read(){try{return normalize(JSON.parse(localStorage.getItem(key)))}catch{return normalize(null)}}
 let prefs=read();
 function resolve(now=new Date()){
  let slot=prefs.mode;
  if(slot==='auto'){
   const current=now.getHours()*60+now.getMinutes(),start=minutes(prefs.lightStart),end=minutes(prefs.darkStart);
   const daytime=start<end?current>=start&&current<end:current>=start||current<end;
   slot=daytime?'light':'dark';
  }
  return {slot,theme:prefs[slot]};
 }
 function apply(now=new Date()){
  const active=resolve(now);
  document.documentElement.dataset.theme=active.theme;
  document.documentElement.dataset.appearance=prefs.mode;
  return active;
 }
 function update(patch){prefs=normalize({...prefs,...patch});try{localStorage.setItem(key,JSON.stringify(prefs))}catch{}return apply()}
 function reload(){prefs=read();return apply()}
 apply();
 return {key,palettes,get:()=>({...prefs}),resolve,apply,update,reload};
})();
