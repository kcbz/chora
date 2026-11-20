"""Loopback-only lab UI. Simulation and rendering belong to one worker thread."""
from __future__ import annotations

import json
from contextlib import nullcontext
import os
import mimetypes
import queue
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs

from .playback import PlaybackClock, validate_speed
from .common import OUTPUT, ROOT, read_json
from .ui import WEB
from .experiments import DEFAULT, catalog, create_experiment, load_config, output_dir, output_url, public_definition


class Stopped(Exception):
    pass

def scene_orbit(experiment):
    cfg = load_config(experiment)
    robot = cfg['experiment']['robot']
    pose={'azimuth':135,'elevation':-25,'distance':.65,'lookat':[.12,0,.16]}
    if robot in {'fr3','ur10e','fr3_duo','fr3_pair'}:pose.update(distance=1.8,lookat=[.4,0,.4])
    elif robot=='xyz_tool_rig':pose.update(distance=1.15,lookat=[0,.1,.27])
    elif cfg["simulation"].get("table_style") == "shelf" or cfg["simulation"].get("platform_motion"):
        pose.update(distance=.76, lookat=[.15,-.015,.145])
    return pose


class Lab:
    def __init__(self):
        from .vla.ui_service import VLAService
        self.vla = VLAService(ROOT)
        self.experiment=os.environ.get("CHORA_EXPERIMENT",DEFAULT)
        self.commands = queue.Queue()
        self.stop = threading.Event()
        self.lock = threading.Lock()
        self.image = b""
        self.state = {"ready":False,"running":False,"policy":"expert","camera":"front","seed":0,"history":[]}
        self.job = None
        self.job_name = None
        self.job_file = None
        self.playback_speed = 1
        self.camera_name = "front"
        self.orbit = scene_orbit(self.experiment)
        threading.Thread(target=self.worker,daemon=True).start()

    @property
    def output(self):
        return output_dir(self.experiment)

    def select_experiment(self,experiment):
        with self.lock:
            from .jobs import current_job
            if experiment not in {x["id"] for x in catalog()}:raise ValueError("Choose a fixed scene from the selector")
            load_config(experiment)
            meta=public_definition(experiment)
            if not meta['available']:raise ValueError(meta['unavailable_reason'])
            if self.vla.running or self.state["running"] or self.state.get("switching") or current_job(self.job,self.job_name,self.job_file,output=self.output).get("running"):
                raise ValueError("Finish the running episode or pipeline stage before switching experiments")
            self.state.update(switching=True,error=None)
            self.commands.put({"operation":"experiment","experiment":experiment})

    def worker(self):
        from .sim import make_simulation
        from .policy import make_expert
        from .visualization import Camera
        camera = None
        try:
            sim = make_simulation(load_config(self.experiment))
            camera = Camera(sim,1280,960)
            def publish(s):
                camera.set_orbit(self.orbit)
                frame = camera.jpeg(s,self.camera_name)
                with self.lock:
                    self.image = frame
                    self.state.update(observation=s.observe(),result=s.result,seed=s.seed,ready=True)
            publish(sim)
            while True:
                command = self.commands.get()
                self.stop.clear()
                try:
                    if command["operation"]=="experiment":
                        selected=command["experiment"]
                        directory=output_dir(selected)
                        if not (directory/"eval_gallery.html").exists():
                            result=subprocess.run([sys.executable,"-m","chora","report"],cwd=ROOT,env={**os.environ,"CHORA_EXPERIMENT":selected},capture_output=True,text=True,timeout=30)
                            if result.returncode:
                                raise RuntimeError("Could not prepare this experiment's report pages")
                        next_sim=make_simulation(load_config(selected))
                        next_camera=Camera(next_sim,1280,960)
                        camera.close();sim=next_sim;camera=next_camera
                        self.experiment=selected
                        self.orbit=scene_orbit(selected)
                        self.job=None;self.job_name=None;self.job_file=None
                        self.state.update(history=[],running=False,success=None,error=None,policy="expert",achieved_speed=None)
                        publish(sim)
                        continue
                    self.camera_name = command.get("camera",self.camera_name)
                    self.state["camera"] = self.camera_name
                    if command["operation"]=="camera":
                        publish(sim)
                        continue
                    seed = int(command.get("seed",0))
                    sim.reset(seed)
                    self.state.update(history=[],running=command["operation"]=="run",success=None,error=None,phase="preparing",achieved_speed=None)
                    publish(sim)
                    if command["operation"]=="run":
                        if self.vla.running:raise ValueError("Finish the VLA test first")
                        self.state.update(running=True,phase="loading_policy")
                        name = command.get("policy","expert")
                        if name=="student":
                            from .learn import Student
                            policy = Student(adapter_path=self.output/"adapter")
                        else:
                            policy = make_expert(seed,load_config(self.experiment))
                        self.state.update(running=True,policy=name,phase="playing")
                        history=[]
                        # Callbacks check pacing/stop every 50 ms of simulated
                        # time. Rendering is throttled in wall time at every speed.
                        sim.frame_stride=max(1,round(.05/sim.model.opt.timestep))
                        playback=PlaybackClock(float(sim.data.time),time.monotonic())
                        last_render=[0.]
                        def frame(s):
                            if self.stop.is_set():raise Stopped
                            now=time.monotonic()
                            delay=playback.delay(float(s.data.time),now,self.playback_speed)
                            if self.stop.wait(delay):raise Stopped
                            now=time.monotonic()
                            self.state['achieved_speed']=playback.commit(float(s.data.time),now)
                            self.state['simulation_seconds']=float(s.data.time)
                            if now-last_render[0]>=.10:
                                publish(s);last_render[0]=time.monotonic()
                        sim.on_frame = frame
                        while not sim.finished:
                            action = policy.act(sim.observe(),history)
                            sim.step(action)
                            history.append(action)
                            self.state["history"] = history.copy()
                            publish(sim)
                        self.state["success"] = sim.success
                except Stopped:
                    self.state["result"] = "stopped"
                except Exception as e:
                    self.state["error"] = str(e)
                finally:
                    sim.on_frame = None
                    self.state["running"] = False
                    self.state["phase"] = "idle"
                    self.state["switching"] = False
        except Exception as e:
            self.state["error"] = f"Viewer failed: {e}"
        finally:
            if camera: camera.close()

    def launch(self,name):
        from .jobs import current_job
        physics=load_config(self.experiment).get('physics',{})
        if physics.get('learning_enabled') is False:raise ValueError(physics.get('learning_readiness','This material trial has no supported scoring/student action protocol yet. Use Run episode to inspect it.'))
        allowed = {"check":["expert"],"generate":["generate"],"curate":["curate"],
                   "train":["train"],"evaluate":["evaluate","--record","3"]}
        if name=="train" and (self.output/"adapter/adapters.safetensors").exists():
            allowed["train"].append("--replace")
        if name not in allowed:
            raise ValueError("Unknown job")
        with self.lock:
            if self.state.get("switching") or current_job(self.job,self.job_name,self.job_file,output=self.output).get("running"):
                raise ValueError("A pipeline job is already running")
            if self.job and self.job.poll() is None:
                raise ValueError("A pipeline job is already running")
            if self.state["running"] or self.vla.running:
                raise ValueError("Stop live playback or the VLA test before starting a pipeline job")
            self.output.mkdir(parents=True,exist_ok=True)
            self.job_name = name
            path = self.output/f"{name}.log"
            self.job_file = path
            with path.open("w") as log:
                self.job = subprocess.Popen([sys.executable,"-u","-m","chora",*allowed[name]],cwd=ROOT,env={**os.environ,"CHORA_EXPERIMENT":self.experiment},stdout=log,stderr=subprocess.STDOUT)
                self.job.chora_started_at = time.time()
            return str(self.job.pid)

    def start_vla(self, checkpoint, seed):
        from .jobs import current_job
        with self.lock:
            if self.experiment != DEFAULT:
                raise ValueError("Select Bowl · Standard platform for these VLA checkpoints")
            if self.state["running"] or self.state.get("switching") or current_job(self.job,self.job_name,self.job_file,output=self.output).get("running"):
                raise ValueError("Finish the running episode or pipeline stage before testing the VLA")
            return self.vla.start(checkpoint, seed)

    def close(self):
        self.vla.close()

    def status(self):
        with self.lock:
            state = dict(self.state)
        state["playback_speed"] = self.playback_speed
        state["orbit"] = dict(self.orbit)
        state["experiment"] = public_definition(self.experiment)
        state["output_url"] = output_url(self.experiment)
        state["oracle"] = read_json(self.output/"oracle.json")
        state["generation"] = read_json(self.output/"generation.json")
        curated = read_json(self.output/"curation.json")
        state["curation"] = {k:v for k,v in (curated or {}).items() if not k.endswith("seeds")}
        state["training"] = read_json(self.output/"training.json")
        state["results"] = read_json(self.output/"results.json",{})
        state["adapter_ready"] = (self.output/"adapter/adapters.safetensors").exists()
        state["model_ready"] = (ROOT/"assets/qwen/config.json").exists()
        state["vla"] = self.vla.status()
        state["vla"]["supported_scene"] = self.experiment == DEFAULT
        from .jobs import current_job, stage_history
        state["job"] = current_job(self.job,self.job_name,self.job_file,output=self.output)
        state["jobs"] = stage_history(output=self.output)
        return state


def serve(port=8765):
    lab = Lab()
    class Handler(BaseHTTPRequestHandler):
        def setup(self):
            self.request.settimeout(10)
            super().setup()
        def local_request(self):
            # Binding loopback alone does not protect against DNS rebinding.
            hosts=self.headers.get_all('Host',[])
            port=self.server.server_port
            allowed={f'127.0.0.1:{port}',f'localhost:{port}'}
            if port==80:allowed.update({'127.0.0.1','localhost'})
            if len(hosts)!=1 or hosts[0].lower() not in allowed:
                self.respond({'error':'Local Host required'},status=403);return False
            origins=self.headers.get_all('Origin',[])
            if len(origins)>1 or (origins and origins[0] not in {f'http://127.0.0.1:{port}',f'http://localhost:{port}'}):
                self.respond({'error':'Local requests only'},status=403);return False
            return True
        def log_message(self,*args):
            pass
        def respond(self,body,content="application/json",status=200):
            if isinstance(body,(dict,list)):
                body = json.dumps(body).encode()
            elif isinstance(body,str):
                body = body.encode()
            self.send_response(status)
            self.send_header("Content-Type",content)
            self.send_header("Content-Length",str(len(body)))
            self.send_header("Cache-Control","no-store")
            self.send_header("X-Content-Type-Options","nosniff")
            self.send_header("X-Frame-Options","DENY")
            self.send_header("Content-Security-Policy","frame-ancestors 'none'; base-uri 'none'; object-src 'none'")
            self.send_header("Referrer-Policy","no-referrer")
            self.end_headers()
            try: self.wfile.write(body)
            except (BrokenPipeError,ConnectionResetError): pass
        def do_GET(self):
            if not self.local_request():return
            path, _, query = self.path.partition("?")
            if path=='/materials/tiling':
                from .tiling_workbench import render
                return self.respond(render(),'text/html; charset=utf-8')
            if path=='/api/tiling-materials':
                from .tiling_materials import catalog as materials,workbench,reference_panel
                try:
                    args=parse_qs(query)
                    if set(args)-{'age_s','exposure_s','speed_m_s','gap_m'}:raise ValueError('Unknown material parameter')
                    prediction=workbench(**{k:float(v[-1]) for k,v in args.items()})
                    return self.respond(dict(catalog=materials(),prediction=prediction,reference_panel=reference_panel()))
                except (ValueError,TypeError) as e:return self.respond({'error':str(e)},status=400)
            if path in {"/", "/gallery", "/reports"}:
                from .ui import control_method, policy_results, render_shell
                args = parse_qs(query)
                policy = control_method(args.get("policy", ["expert"])[0])
                if path == "/":
                    html = render_shell((WEB / "lab-template.html").read_text(),
                                        experiment=lab.experiment, policy=policy)
                else:
                    experiment = args.get("experiment", [lab.experiment])[0]
                    if experiment not in {item["id"] for item in catalog()}:
                        return self.respond({"error": "Unknown scene"}, status=404)
                    visual = lab.vla.status() if policy == "vla" and experiment == DEFAULT else None
                    html = policy_results(experiment, policy, path.lstrip("/"), visual)
                return self.respond(html, "text/html; charset=utf-8")
            if path.startswith("/static/"):
                name=path[len("/static/"):]
                if name in {"app.css","appearance-core.js","appearance-ui.js","lab.js","logo-octahedron.svg"}:
                    file=WEB/name
                    return self.respond(file.read_bytes(),mimetypes.guess_type(file)[0] or "application/octet-stream")
            if path=="/api/experiments":
                return self.respond({"experiments":catalog(),"active":lab.experiment})
            if path=="/api/status":
                return self.respond(lab.status())
            if path=="/api/vla/frame.jpg":
                frame=lab.vla.frame()
                return self.respond(frame,"image/jpeg",200 if frame else 503)
            if path=="/frame.jpg":
                return self.respond(lab.image,"image/jpeg",200 if lab.image else 503)
            if path.startswith("/outputs/"):
                from urllib.parse import unquote
                base=ROOT/"outputs"
                file = (base/unquote(path[len("/outputs/"):])).resolve()
                if file.is_relative_to(base.resolve()) and file.is_file():
                    if file.name in {"eval_gallery.html","experiment_report.html","curation_report.html"}:
                        from .ui import render_shell
                        relative=file.relative_to(base.resolve()).parts
                        experiment=relative[1] if relative[0]=="experiments" else DEFAULT
                        active="gallery" if file.name=="eval_gallery.html" else "reports"
                        return self.respond(render_shell(file.read_text(),active,experiment),"text/html; charset=utf-8")
                    return self.respond(file.read_bytes(),mimetypes.guess_type(file)[0] or "application/octet-stream")
            self.respond({"error":"Not found"},status=404)
        def do_POST(self):
            if not self.local_request():return
            if len(self.headers.get_all('Content-Type',[]))!=1 or self.headers.get_content_type()!='application/json':
                return self.respond({'error':'Content-Type must be application/json'},status=415)
            try:
                length = int(self.headers.get("Content-Length",0))
                if not 0<length<4096:
                    raise ValueError("Invalid request length")
                command = json.loads(self.rfile.read(length))
                if not isinstance(command,dict):raise ValueError('Command must be a JSON object')
                if self.path=="/api/vla":
                    if command.get("operation") == "run" and set(command) == {"operation","checkpoint","seed"}:
                        run_id = lab.start_vla(command["checkpoint"], command["seed"])
                        return self.respond({"ok":True,"id":run_id})
                    if command == {"operation":"stop"}:
                        lab.vla.stop()
                        return self.respond({"ok":True})
                    raise ValueError("Invalid VLA command")
                if self.path=="/api/control":
                    operation = command.get("operation")
                    if operation not in {"run","reset","stop","camera","orbit","speed"}:
                        raise ValueError("Unknown operation")
                    if command.get("camera","front") not in {"front","top","orbit"}:
                        raise ValueError("Unknown camera")
                    if command.get("policy","expert") not in {"expert","student"}:
                        raise ValueError("Unknown policy")
                    if not 0<=int(command.get("seed",0))<=999999:
                        raise ValueError("Seed must be 0..999999")
                    with getattr(lab,"lock",nullcontext()):
                        if operation in {"run","reset"} and getattr(getattr(lab,"vla",None),"running",False):
                            raise ValueError("Stop the VLA test before running or resetting the main viewer")
                        if operation=="run":
                            from .jobs import current_job
                            if lab.state.get("switching") or current_job(lab.job,lab.job_name,lab.job_file,output=lab.output).get("running"):
                                raise ValueError("Wait for the pipeline job before live playback")
                            if lab.state["running"]:
                                raise ValueError("An episode is already running")
                            if command.get("policy")=="student" and not (lab.output/"adapter/adapters.safetensors").exists():
                                raise ValueError("This experiment has no trained student yet. Use the expert or collect and train first.")
                        if operation=="speed":
                            lab.playback_speed=validate_speed(command.get("speed"))
                        elif operation=="orbit":
                            import math
                            pose = command.get("orbit",{})
                            az,el,dist = [float(pose[k]) for k in ("azimuth","elevation","distance")]
                            lookat = [float(x) for x in pose["lookat"]]
                            if len(lookat)!=3 or not all(math.isfinite(x) for x in [az,el,dist,*lookat]):
                                raise ValueError("Camera pose must contain finite numbers")
                            if not -89<=el<=-5 or not .2<=dist<=2 or any(abs(x)>2 for x in lookat):
                                raise ValueError("Camera pose exceeds view limits")
                            lab.orbit = {"azimuth":az%360,"elevation":el,"distance":dist,"lookat":lookat}
                            lab.camera_name = "orbit"
                            lab.state["camera"] = "orbit"
                            if not lab.state["running"] and lab.commands.qsize()<2:
                                lab.commands.put({"operation":"camera","camera":"orbit"})
                        elif operation=="stop":
                            lab.stop.set()
                        elif operation=="camera" and lab.state["running"]:
                            lab.camera_name = command["camera"]
                            lab.state["camera"] = command["camera"]
                        else:
                            if operation!="camera": lab.stop.set()
                            if lab.commands.qsize()>2:
                                raise ValueError("Viewer is busy")
                            if operation=="run":
                                lab.state.update(running=True,phase="preparing",policy=command.get("policy","expert"))
                            lab.commands.put(command)
                elif self.path=="/api/job":
                    job_id = lab.launch(command.get("name"))
                    return self.respond({"ok":True,"job_id":job_id})
                elif self.path=="/api/experiments":
                    return self.respond({"error":"Scene copies are disabled. Choose a fixed scene."},status=405)
                elif self.path=="/api/experiment":
                    lab.select_experiment(command.get("id"))
                else:
                    return self.respond({"error":"Not found"},status=404)
                self.respond({"ok":True})
            except (ValueError,TypeError,KeyError,OverflowError,json.JSONDecodeError) as e:
                self.respond({"error":str(e)},status=400)
    server = ThreadingHTTPServer(("127.0.0.1",port),Handler)
    print(f"Chora by Coherence: http://127.0.0.1:{server.server_port}",flush=True)
    try: server.serve_forever()
    except KeyboardInterrupt: pass
    finally:
        server.server_close()
        if hasattr(lab,"close"):lab.close()
