"""Optional checks against an already-running lab; no experiment is changed."""
import json
import os
import urllib.error
import urllib.request

import pytest

pytestmark = pytest.mark.skipif(os.environ.get("CHORA_HTTP_TEST")!="1",reason="Start the lab and set CHORA_HTTP_TEST=1")
BASE = os.environ.get("CHORA_HTTP_BASE","http://127.0.0.1:8765")

def test_tiling_material_api_preserves_source_status_and_separate_windows():
    with urllib.request.urlopen(BASE+'/api/tiling-materials?age_s=2500&exposure_s=2401') as response:
        d=json.load(response)
    assert d['prediction']['tile']['dimensions_m']==[.1524,.1524,.0127]
    assert d['prediction']['tile']['mass_status']=='derived_from_assumed_density_not_measured'
    assert d['prediction']['adhesive']['workable']
    assert not d['prediction']['adhesive']['placement_window_open']
    assert d['prediction']['squeeze_force_n'] is None
    assert d['reference_panel']['scope']=='numerical_material_sequence_no_robot_motion'

def test_tiling_material_api_rejects_nonfinite_and_unknown_inputs():
    for query in ['age_s=nan','gap_m=0','unknown=1']:
        with pytest.raises(urllib.error.HTTPError) as error:urllib.request.urlopen(BASE+'/api/tiling-materials?'+query)
        assert error.value.code==400

def test_material_workbench_reuses_shared_brand_and_appearance():
    with urllib.request.urlopen(BASE+'/materials/tiling') as response:html=response.read().decode()
    assert 'Materials, with provenance.' in html
    assert 'id="material-controls"' in html and 'id="appearance-settings"' in html
    assert 'href="/static/app.css"' in html and 'viewBox="0 0 28 28"' in html
    assert '{{' not in html


def test_lab_serves_real_state():
    with urllib.request.urlopen(BASE+"/api/status") as response:
        state = json.load(response)
    assert state["ready"]
    cfg=state["experiment"]["simulation"]
    expected=cfg["table_height"]+(.01 if state["experiment"]["task"]=="hammer_nail" else .03 if state["experiment"]["task"] in {"hammer_pine","cut_fruit"} else 0)
    if state["experiment"]["robot"] in {"xyz_tool_rig","fr3","ur10e","fr3_pair"}:expected=cfg["target"][2]
    if state['experiment']['robot']=='fr3_duo':expected=.388
    assert state["observation"]["target"][2] == pytest.approx(expected)
    if state["oracle"]:assert state["oracle"]["episodes"] > 0


def test_camera_is_a_real_jpeg():
    with urllib.request.urlopen(BASE+"/frame.jpg") as response:
        assert response.headers["Content-Type"]=="image/jpeg"
        body = response.read()
    assert body[:2]==b"\xff\xd8" and len(body)>1000


def test_static_paths_cannot_escape_outputs():
    with pytest.raises(urllib.error.HTTPError) as error:
        urllib.request.urlopen(BASE+"/outputs/%2e%2e/configs/default.yaml")
    assert error.value.code==404


def test_cross_origin_commands_are_rejected():
    request = urllib.request.Request(BASE+"/api/control",data=b'{"operation":"stop"}',
        headers={"Origin":"https://untrusted.example","Content-Type":"application/json"})
    with pytest.raises(urllib.error.HTTPError) as error:
        urllib.request.urlopen(request)
    assert error.value.code==403


def test_bad_command_does_not_run():
    request = urllib.request.Request(BASE+"/api/control",data=b'{"operation":"execute_shell"}',
        headers={"Content-Type":"application/json"})
    with pytest.raises(urllib.error.HTTPError) as error:
        urllib.request.urlopen(request)
    assert error.value.code==400


def test_invalid_orbit_pose_is_rejected():
    data = json.dumps({"operation":"orbit","orbit":{"azimuth":0,"elevation":0,"distance":-1,"lookat":[0,0,0]}}).encode()
    request = urllib.request.Request(BASE+"/api/control",data=data,headers={"Content-Type":"application/json"})
    with pytest.raises(urllib.error.HTTPError) as error:
        urllib.request.urlopen(request)
    assert error.value.code==400


def test_orbit_control_preserves_episode_state():
    with urllib.request.urlopen(BASE+"/api/status") as response:
        before = json.load(response)
    if before["running"]:
        pytest.skip("Do not interrupt an episode being watched")
    pose = {"azimuth":45,"elevation":-35,"distance":.65,"lookat":[.12,0,.16]}
    def post(data):
        request = urllib.request.Request(BASE+"/api/control",data=json.dumps(data).encode(),headers={"Content-Type":"application/json"})
        with urllib.request.urlopen(request) as response:
            return json.load(response)
    try:
        assert post({"operation":"orbit","orbit":pose})["ok"]
        with urllib.request.urlopen(BASE+"/api/status") as response:
            after = json.load(response)
        assert after["camera"]=="orbit"
        assert after["orbit"]==pose
        assert after["observation"]==before["observation"]
    finally:
        post({"operation":"orbit","orbit":before["orbit"]})
        post({"operation":"camera","camera":before["camera"]})


def test_playback_controls_and_measured_speed_are_available():
    with urllib.request.urlopen(BASE+'/api/status') as response:s=json.load(response)
    assert s['playback_speed'] in [1,2,4]
    with urllib.request.urlopen(BASE+'/') as response:html=response.read().decode()
    assert 'data-playback-speed="4"' in html and 'Fastest' not in html
    assert 'id="playback-timing"' in html


def test_invalid_playback_speed_is_rejected_without_changing_scene():
    for speed in [True,0,-1,3,'4']:
        request=urllib.request.Request(BASE+'/api/control',data=json.dumps(dict(operation='speed',speed=speed)).encode(),headers={'Content-Type':'application/json'})
        with pytest.raises(urllib.error.HTTPError) as error:urllib.request.urlopen(request)
        assert error.value.code==400
