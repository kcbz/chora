"""VLA UI process/catalog checks without model or graphics dependencies."""
import json
import queue
import subprocess
import threading
import urllib.error
import urllib.request

import pytest

from chora import server
from chora.vla.ui_service import VLAService


def candidate(root, name='dense-pilot', outside=False):
    dataset = root/'outputs/vla/datasets/pilot'
    dataset.mkdir(parents=True, exist_ok=True)
    (dataset/'dataset.json').write_text(json.dumps(dict(protocol='chora-vla-pilot-chunks-v1', fingerprint='data')))
    run = root/'outputs/vla/training'/name
    checkpoint = root/'outside' if outside else run/'checkpoints/final'
    checkpoint.mkdir(parents=True, exist_ok=True)
    run.mkdir(parents=True, exist_ok=True)
    (checkpoint/'model.safetensors').write_bytes(b'test-only')
    (checkpoint/'config.json').write_text(json.dumps(dict(protocol='chora-vla-skeleton-v1',
        metadata=dict(dataset_fingerprint='data',step=1000),normalization=dict(action_center=[0]*6))))
    (run/'report.json').write_text(json.dumps(dict(status='complete',mode='pilot_training',final=dict(checkpoint=str(checkpoint)))))
    return checkpoint


class Process:
    returncode = None
    def poll(self): return self.returncode
    def terminate(self): self.returncode = -15


def test_catalog_whitelists_paths_and_ignores_invalid_json(tmp_path):
    candidate(tmp_path)
    candidate(tmp_path, 'escape', outside=True)
    malformed = tmp_path/'outputs/vla/training/malformed'
    malformed.mkdir();(malformed/'report.json').write_text('null')
    service=VLAService(tmp_path)
    assert [r['id'] for r in service.candidates()]==['dense-pilot']
    assert service.status()['ready'] is False


def test_start_uses_fixed_environment_and_rejects_user_paths(tmp_path,monkeypatch):
    checkpoint=candidate(tmp_path)
    python=tmp_path/'.venv-vla/bin/python';python.parent.mkdir(parents=True);python.touch()
    calls=[];process=Process()
    def spawn(args,**kwargs): calls.append((args,kwargs));return process
    monkeypatch.setattr(subprocess,'Popen',spawn)
    service=VLAService(tmp_path)
    for invalid in ['../outside','missing']:
        with pytest.raises(ValueError,match='checkpoint'):service.start(invalid,20003)
    for invalid in [0,True,30000,20003.5]:
        with pytest.raises(ValueError,match='seed'):service.start('dense-pilot',invalid)
    run_id=service.start('dense-pilot',20003)
    assert calls[0][0][0]==str(python)
    assert str(checkpoint) in calls[0][0] and '--ui-progress' in calls[0][0]
    assert calls[0][1]['cwd']==tmp_path
    assert service.status()['phase']=='loading'
    with pytest.raises(ValueError,match='already'):service.start('dense-pilot',20003)
    service.stop();assert service.directory.with_suffix('.stop').exists()
    assert service.status()['phase']=='stopping'
    service.directory.mkdir()
    (service.directory/'report.json').write_text(json.dumps(dict(trials=[dict(reason='stopped',success=False)])))
    process.returncode=0
    result=service.status();assert result['phase']=='stopped' and result['id']==run_id


def test_finished_task_failure_is_not_reported_as_success(tmp_path):
    service=VLAService(tmp_path);service.directory=tmp_path/'run';service.directory.mkdir()
    service.run_id='run';service.log=tmp_path/'run.log';service.process=Process();service.process.returncode=0
    (service.directory/'report.json').write_text(json.dumps(dict(status='complete',trials=[dict(reason='tick_limit',success=False)])))
    result=service.status()
    assert result['phase']=='complete' and result['success'] is False and result['outcome']=='tick_limit'
    service.process.returncode=1;service.log.write_text('Traceback\nmissing model')
    assert service.status()['phase']=='failed' and service.status()['error']=='missing model'


@pytest.fixture
def api(monkeypatch):
    class Lab:
        experiment='so101-bowl'
        image=b''
        class VLA:
            def stop(self):self.stopped=True
            def frame(self):return b'jpeg'
        vla=VLA()
        def start_vla(self,checkpoint,seed):
            if checkpoint!='known':raise ValueError('Choose an available VLA checkpoint')
            self.started=(checkpoint,seed);return 'run'
        def status(self):return {}
    lab=Lab();monkeypatch.setattr(server,'Lab',lambda:lab)
    created=queue.Queue();factory=server.ThreadingHTTPServer
    def construct(*args,**kwargs):
        instance=factory(*args,**kwargs);created.put(instance);return instance
    monkeypatch.setattr(server,'ThreadingHTTPServer',construct)
    thread=threading.Thread(target=server.serve,args=(0,),daemon=True);thread.start()
    instance=created.get(timeout=5)
    yield f'http://127.0.0.1:{instance.server_port}',lab
    instance.shutdown();thread.join(timeout=5)


def test_vla_endpoints_validate_commands_and_publish_frames(api):
    base,lab=api
    def request(payload):
        return urllib.request.urlopen(urllib.request.Request(base+'/api/vla',data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'}),timeout=3)
    with request(dict(operation='run',checkpoint='known',seed=20003)) as r:assert json.load(r)['id']=='run'
    assert lab.started==('known',20003)
    for payload in [dict(operation='run',checkpoint='../outside',seed=20003),dict(operation='stop',path='/tmp'),dict(operation='other')]:
        with pytest.raises(urllib.error.HTTPError) as e:request(payload)
        assert e.value.code==400
    with request(dict(operation='stop')):pass
    assert lab.vla.stopped
    with urllib.request.urlopen(base+'/api/vla/frame.jpg') as r:assert r.read()==b'jpeg'


def test_lab_rejects_unsupported_scenes_and_conflicting_jobs(monkeypatch):
    from chora import jobs
    lab=server.Lab.__new__(server.Lab)
    lab.lock=threading.Lock();lab.state=dict(running=False,switching=False)
    lab.job=lab.job_name=lab.job_file=None
    lab.commands=queue.Queue()
    class VLA:
        running=False
        def start(self,*args):return 'started'
    lab.vla=VLA()
    monkeypatch.setattr(jobs,'current_job',lambda *a,**kw:dict(running=False))
    lab.experiment='so101-bowl-low-left'
    with pytest.raises(ValueError,match='Standard'):lab.start_vla('known',20003)
    lab.experiment='so101-bowl'
    lab.state['running']=True
    with pytest.raises(ValueError,match='Finish'):lab.start_vla('known',20003)
    lab.state['running']=False
    monkeypatch.setattr(jobs,'current_job',lambda *a,**kw:dict(running=True))
    with pytest.raises(ValueError,match='Finish'):lab.start_vla('known',20003)
    monkeypatch.setattr(jobs,'current_job',lambda *a,**kw:dict(running=False))
    assert lab.start_vla('known',20003)=='started'
    lab.vla.running=True
    with pytest.raises(ValueError,match='VLA'):lab.launch('check')
    with pytest.raises(ValueError,match='Finish'):lab.select_experiment('so101-bowl')
    assert lab.commands.empty()


def test_recommendation_requires_unseen_successes_and_matching_weights(tmp_path):
    import hashlib
    checkpoint=candidate(tmp_path)
    python=tmp_path/'.venv-vla/bin/python';python.parent.mkdir(parents=True);python.touch()
    dataset=tmp_path/'outputs/vla/datasets/pilot/dataset.json'
    value=json.loads(dataset.read_text());value['episodes']=[dict(seed=21000+i,split='train') for i in range(10)]
    dataset.write_text(json.dumps(value))
    folder=tmp_path/'outputs/vla/rollouts/test';folder.mkdir(parents=True)
    def report(seeds,success=True):
        payload=dict(status='complete',checkpoint=str(checkpoint),
                     checkpoint_sha256=hashlib.sha256((checkpoint/'model.safetensors').read_bytes()).hexdigest(),
                     trials=[dict(seed=seed,reason='success' if success else 'tick_limit',success=success,collisions=0) for seed in seeds])
        (folder/'report.json').write_text(json.dumps(payload))
    service=VLAService(tmp_path)
    report(range(21000,21010))
    assert service.status()['recommended_id'] is None
    report(range(25000,25010),success=False)
    state=service.status();assert state['recommended_id'] is None
    assert state['candidates'][0]['label'].startswith('Failed (0/10)')
    report(range(25000,25010))
    assert service.status()['recommended_id']=='dense-pilot'
    file=folder/'report.json';payload=json.loads(file.read_text());payload['execute_horizon']=1;file.write_text(json.dumps(payload))
    assert service.status()['recommended_id'] is None
    payload['execute_horizon']=4;file.write_text(json.dumps(payload))
    assert service.status()['recommended_id']=='dense-pilot'
    (checkpoint/'model.safetensors').write_bytes(b'changed weights')
    assert service.status()['recommended_id'] is None
