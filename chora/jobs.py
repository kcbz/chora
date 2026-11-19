"""Normalize process state so an old saved result never looks like a live job."""
import os

from .common import OUTPUT, read_json


def alive(pid):
    if not isinstance(pid,int) or pid<=0:
        return False
    try:
        os.kill(pid,0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def normalize(record):
    result = dict(record or {})
    result.setdefault("id",str(result.get("pid","")))
    if result.get("running") and not alive(result.get("pid")):
        result.update(running=False,status="failed",error="The process ended before reporting completion.")
    result.setdefault("status","running" if result.get("running") else "complete")
    result.setdefault("progress",{})
    return result


def current_job(process=None,name=None,log_path=None,output=None):
    output=output or OUTPUT
    active = normalize(read_json(output/"active_job.json",{}))
    if process is not None:
        code = process.poll()
        own = active if active.get("pid")==process.pid else {}
        newer_external = (active.get("pid")!=process.pid and active.get("name") and
                          active.get("started_at",0)>getattr(process,"chora_started_at",0))
        if code is None or (not newer_external and not active.get("running")):
            active = {**own,"id":str(process.pid),"pid":process.pid,"name":name,
                      "running":code is None,"exit_code":code,
                      "status":"running" if code is None else "complete" if code==0 else "failed"}
    if not active.get("name"):
        return {"name":None,"running":False,"status":"idle","progress":{},"log":""}
    if log_path is None or active.get("name")!=name:
        log_path = output/f"{active['name']}.log"
    active["log"] = log_path.read_text(errors="replace")[-4000:] if log_path and log_path.exists() else ""
    if active["status"]=="failed" and not active.get("error"):
        active["error"] = active["log"].strip().splitlines()[-1] if active["log"].strip() else "This stage did not finish. See run details."
    return active


def stage_history(output=None):
    output=output or OUTPUT
    return {name:normalize(read_json(output/"jobs"/f"{name}.json",{}))
            for name in ("check","generate","curate","train","evaluate")}
