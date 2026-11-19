"""Exercise the actual HTTP handler with a harmless lab; no graphics/model loads."""
import json
import queue
import threading
import urllib.request
import urllib.error
import pytest
from chora import server

@pytest.fixture
def local_server(monkeypatch):
    class Lab:
        experiment='so101-bowl'
        playback_speed=1
        def status(self):return {'ready':True,'playback_speed':self.playback_speed}
    lab=Lab();monkeypatch.setattr(server,'Lab',lambda:lab)
    created=queue.Queue();original=server.ThreadingHTTPServer
    def construct(*args,**kwargs):
        instance=original(*args,**kwargs);created.put(instance);return instance
    monkeypatch.setattr(server,'ThreadingHTTPServer',construct)
    worker=threading.Thread(target=server.serve,args=(0,),daemon=True);worker.start()
    instance=created.get(timeout=5)
    yield f'http://127.0.0.1:{instance.server_port}',lab
    instance.shutdown();worker.join(timeout=5)

def request(base,path='/api/status',body=None,headers=None):
    return urllib.request.urlopen(urllib.request.Request(base+path,data=body,headers=headers or {}),timeout=3)

@pytest.mark.parametrize('host',['attacker.example','localhost.attacker.example','127.0.0.1.attacker.example','127.0.0.1:1'])
def test_foreign_hosts_rejected_before_reading_local_state(local_server,host):
    base,lab=local_server
    with pytest.raises(urllib.error.HTTPError) as e:request(base,headers={'Host':host})
    assert e.value.code==403

def test_local_get_and_clickjacking_headers(local_server):
    base,lab=local_server
    with request(base,'/') as r:
        assert r.status==200
        assert r.headers['X-Frame-Options']=='DENY'
        assert "frame-ancestors 'none'" in r.headers['Content-Security-Policy']
        assert r.headers['Referrer-Policy']=='no-referrer'

def test_foreign_origin_cannot_read_or_change_state(local_server):
    base,lab=local_server
    for body in [None,b'{"operation":"speed","speed":4}']:
        with pytest.raises(urllib.error.HTTPError) as e:request(base,'/api/control' if body else '/api/status',body,{'Origin':'https://attacker.example','Content-Type':'application/json'})
        assert e.value.code==403
    assert lab.playback_speed==1

def test_only_json_object_commands_are_accepted(local_server):
    base,lab=local_server
    with pytest.raises(urllib.error.HTTPError) as e:request(base,'/api/control',b'{"operation":"speed","speed":4}',{'Content-Type':'text/plain'})
    assert e.value.code==415 and lab.playback_speed==1
    for body in [b'[]',b'null',b'"speed"',b'{broken']:
        with pytest.raises(urllib.error.HTTPError) as e:request(base,'/api/control',body,{'Content-Type':'application/json'})
        assert e.value.code==400
    with request(base,'/api/control',b'{"operation":"speed","speed":2}',{'Content-Type':'application/json; charset=utf-8'}) as r:assert r.status==200
    assert lab.playback_speed==2
