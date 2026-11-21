from chora.common import save_json
from chora.visual_inspection import cases, inspect_case, public_cases


def test_inspection_uses_only_verified_overhead_frames_and_compares_geometry(tmp_path):
    path = "evaluation/student/inspection/seed_100000.jpg"
    frame = tmp_path / path
    frame.parent.mkdir(parents=True)
    frame.write_bytes(b"image")
    observation = {"bowl": [.175, .09, .057], "target": [.175, .09, .055],
                   "target_half_size": [.043, .037]}
    save_json(tmp_path / "evaluation/student/episodes.json", [
        dict(seed=100000, thumbnail_verified=True, inspection_frame=path, final_observation=observation),
        dict(seed=100001, thumbnail_verified=False, inspection_frame=path, final_observation=observation),
    ])
    published = public_cases("so101-bowl", tmp_path)
    assert published == [dict(id="student-100000", seed=100000, policy="student", image="/outputs/" + path)]
    assert public_cases("so101-bowl-moving-platform", tmp_path) == []
    row, = cases("so101-bowl", tmp_path)

    class Inspector:
        def ask(self, image, object_name):
            assert image == frame and object_name == "bowl"
            return "Inside the target?", "outside.", 50

    report = inspect_case(Inspector(), row, tmp_path)
    assert report["answer"] == "no"
    assert report["expected"] == "yes"
    assert report["correct"] is False
