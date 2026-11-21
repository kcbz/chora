import json
import re

from chora import report


def test_gallery_uses_recorded_results_and_escapes_actions(tmp_path,monkeypatch):
    monkeypatch.setattr(report,"OUTPUT",tmp_path)
    summary = {"expert":{"successes":2,"episodes":3,"mean_steps_to_success":9.0}}
    (tmp_path/"results.json").write_text(json.dumps(summary))
    folder = tmp_path/"evaluation/expert"
    folder.mkdir(parents=True)
    episode = {"seed":100000,"success":False,"final_observation":{"last_result":"invalid_action"},
               "video":None,"steps":[{"action":{"action":"<script>alert(1)</script>"},"result":"invalid_action"}]}
    (folder/"episodes.json").write_text(json.dumps([episode]))
    report.build_report()
    html = (tmp_path/"eval_gallery.html").read_text()
    payload=json.loads(re.search(r'<script id="episode-data" type="application/json">(.*?)</script>',html,re.S).group(1))
    assert payload["policies"]["expert"]["successes"]==2
    assert payload["policies"]["expert"]["episodes"]==3
    assert payload["episodes"]["expert-100000"]["steps"][0]["action"]["action"]=="<script>alert(1)</script>"
    assert 'class="episode-wall"' in html
    assert "<script>alert(1)</script>" not in html
    assert "invalid_action" in html
    assert "| expert | 2 | 3 | 9.00 |" in (tmp_path/"REPORT.md").read_text()
    for name in ("eval_gallery.html","experiment_report.html","curation_report.html"):
        document = (tmp_path/name).read_text()
        assert 'src="/static/appearance-core.js"' in document
        assert 'id="appearance-mode"' in document
        assert 'id="appearance-settings"' in document
        assert 'href="/static/app.css"' in document
        assert 'viewBox="0 0 28 28"' in document
        assert "{{" not in document


def test_scoped_gallery_uses_existing_media_from_the_selected_scene(tmp_path):
    from chora.gallery import render_gallery

    folder = tmp_path / "evaluation/expert"
    (folder / "thumbs").mkdir(parents=True)
    (folder / "thumbs/seed_1.jpg").touch()
    (folder / "seed_1.mp4").touch()
    episode = {"success": False, "final_observation": {"last_result": "not_placed"}, "steps": []}
    records = [
        dict(episode, seed=1, thumbnail="evaluation/expert/thumbs/seed_1.jpg",
             video="evaluation/expert/seed_1.mp4"),
        dict(episode, seed=2, thumbnail="evaluation/expert/thumbs/seed_2.jpg"),
    ]
    (folder / "episodes.json").write_text(json.dumps(records))

    for scene, prefix in [("so101-bowl", "/outputs"),
                          ("so101-bowl-low-left", "/outputs/experiments/so101-bowl-low-left")]:
        html = render_gallery({"expert": {"successes": 0, "episodes": 2}},
                              tmp_path, experiment=scene, method="expert")
        data = json.loads(re.search(r'<script id="episode-data" type="application/json">(.*?)</script>',
                                    html, re.S).group(1))["episodes"]
        assert data["expert-1"]["thumbnail"] == prefix + "/evaluation/expert/thumbs/seed_1.jpg"
        assert data["expert-1"]["video"] == prefix + "/evaluation/expert/seed_1.mp4"
        assert data["expert-2"]["thumbnail"] is None


def test_vla_report_distinguishes_training_from_failed_evaluation(tmp_path, monkeypatch):
    from chora import experiments
    from chora.common import save_json
    from chora.ui import policy_results
    from chora.vla.workflow import status

    monkeypatch.setattr(experiments, "output_dir", lambda _: tmp_path)
    save_json(tmp_path / "vla/workflow.json", {"training": "training/test", "familiar": "rollouts/test"})
    save_json(tmp_path / "vla/training/test/report.json", {"status": "complete", "final": {"step": 1200}})
    path = tmp_path / "vla/rollouts/test/report.json"
    trials = [
        {"seed": 20000, "success": True, "ticks": 110, "collisions": 0},
        {"seed": 20001, "success": False, "ticks": 126, "collisions": 1, "reason": "collision"},
    ]
    save_json(path, {"status": "complete", "trials": trials})

    assert status(tmp_path)["evaluation_outcome"] == "did-not-pass"
    html = policy_results(experiments.DEFAULT, "vla", "reports", {"ready": False})
    assert "1,200 updates saved" in html
    assert "Familiar evaluation did not pass" in html
    assert "20001</th><td>Collision</td><td>126</td><td>1</td>" in html
    assert "Held-out testing has yet to run." in html
    assert "Locked" in html
    assert '/gallery?experiment=so101-bowl&amp;policy=vla' in html
