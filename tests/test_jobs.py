import os
from pathlib import Path

import pytest

from chora import common, jobs


@pytest.fixture
def isolated(tmp_path,monkeypatch):
    monkeypatch.setattr(common,"OUTPUT",tmp_path)
    monkeypatch.setattr(jobs,"OUTPUT",tmp_path)
    return tmp_path


def test_job_keeps_progress_and_completion_across_restarts(isolated):
    @common.exclusive_job("check")
    def run():
        common.job_progress(17,50,"Checked 17 of 50 scenes")
        live = jobs.current_job()
        assert live["running"] and live["status"]=="running"
        assert live["progress"]["current"]==17
        return 48
    assert run()==48
    finished = jobs.current_job()
    assert not finished["running"] and finished["status"]=="complete"
    assert finished["progress"]["current"]==17
    assert jobs.stage_history()["check"]["finished_at"]>=finished["started_at"]


def test_failed_job_is_not_reported_as_complete(isolated):
    @common.exclusive_job("curate")
    def run():
        raise ValueError("Not enough clean examples")
    with pytest.raises(ValueError):run()
    failed = jobs.current_job()
    assert failed["status"]=="failed" and not failed["running"]
    assert failed["error"]=="Not enough clean examples"


def test_new_process_is_running_while_old_result_remains(isolated):
    common.save_json(isolated/"active_job.json",{"name":"generate","pid":123,"running":False,"status":"complete"})
    class Process:
        pid=os.getpid()
        def poll(self):return None
    live = jobs.current_job(Process(),"check",isolated/"check.log")
    assert live["name"]=="check" and live["running"] and live["status"]=="running"
    assert live["id"]==str(os.getpid())


def test_dead_worker_is_a_failure_not_a_stuck_spinner(isolated,monkeypatch):
    monkeypatch.setattr(jobs,"alive",lambda pid:False)
    common.save_json(isolated/"active_job.json",{"name":"train","pid":999,"running":True})
    state = jobs.current_job()
    assert state["status"]=="failed" and not state["running"]


def test_progress_cannot_overwrite_another_process(isolated):
    initial={"name":"train","pid":123,"running":True}
    common.save_json(isolated/"active_job.json",initial)
    common.job_progress(100,1200,"Wrong process")
    assert common.read_json(isolated/"active_job.json")==initial


def test_newer_external_result_replaces_previous_ui_job(isolated):
    common.save_json(isolated/"active_job.json",{"name":"curate","pid":456,"running":False,"status":"complete","started_at":200})
    class Process:
        pid=123
        chora_started_at=100
        def poll(self):return 0
    state=jobs.current_job(Process(),"check",isolated/"check.log")
    assert state["name"]=="curate" and state["status"]=="complete"
