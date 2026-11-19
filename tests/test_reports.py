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
