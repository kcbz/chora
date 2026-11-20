import json

import numpy as np
import pytest

from chora.vla.dataset import ChunkDataset, prepare
from chora.vla.record import PROTOCOL, sha256


@pytest.fixture
def recordings(tmp_path):
    root = tmp_path / "recordings"
    root.mkdir()
    episodes = []
    for i, offset in enumerate((1., 2., 1000.)):
        seed, n = 20000+i, 21+i
        path = root / f"episode-{seed}.npz"
        values = np.arange(n+1)[:, None] * .01 + np.ones((n+1, 6)) * offset
        np.savez_compressed(path, actions=values[:-1], joint_positions=values,
                            joint_velocities=np.zeros_like(values), timestamps=np.arange(n+1)*.04,
                            rgb_front=np.zeros((n+1, 28, 28, 3), dtype=np.uint8),
                            rgb_wrist=np.full((n+1, 28, 28, 3), 100, dtype=np.uint8))
        episodes.append({"seed": seed, "success": True, "sha256": sha256(path)})
    manifest = {"protocol": PROTOCOL, "status": "complete", "images_recorded": True,
                "fingerprint": "same-simulation", "image_size": 28, "control_hz": 25,
                "joint_names": [f"joint{i}" for i in range(6)], "camera_names": ["front", "wrist"],
                "instruction": "Place the bowl.", "episodes": episodes}
    (root / "manifest.json").write_text(json.dumps(manifest))
    return root


def test_split_precedes_windows_and_normalization_uses_train_only(recordings, tmp_path):
    directory = prepare([recordings], tmp_path / "dataset")
    train, val = ChunkDataset(directory, "train"), ChunkDataset(directory, "validation")
    assert {e["seed"] for e in train.episodes}.isdisjoint({e["seed"] for e in val.episodes})
    assert len(train.episodes) == 2 and len(val.episodes) == 1
    expected = np.concatenate([train.episode_arrays(i)["actions"] for i in range(2)]).mean(0)
    np.testing.assert_allclose(train.norm["action_center"], expected)
    # Seed 20002 is the deliberately extreme validation episode.
    assert val.episodes[0]["seed"] == 20002
    assert max(train.norm["action_center"]) < 3
    assert train.norm == val.norm


def test_chunk_tail_masks_and_history_do_not_cross_reset(recordings, tmp_path):
    directory = prepare([recordings], tmp_path / "dataset")
    dataset = ChunkDataset(directory, "train", chunk=16)
    for episode_index, episode in enumerate(dataset.episodes):
        last = dataset.windows.index((episode_index, episode["ticks"]-1))
        sample = dataset[last]
        assert sample["valid"].sum() == 1
        assert not sample["targets"][1:].any()
        arrays = dataset.episode_arrays(episode_index)
        np.testing.assert_allclose(sample["targets"][0],
                                   (arrays["actions"][-1]-dataset.norm["action_center"])/dataset.norm["action_scale"], rtol=1e-5)
        first = dataset[dataset.windows.index((episode_index, 0))]
        assert first["history_tick"] == first["tick"] == 0


def test_duplicate_or_corrupted_recordings_are_rejected(recordings, tmp_path):
    with pytest.raises(ValueError, match="Duplicate"):
        prepare([recordings, recordings], tmp_path / "duplicate")
    with (recordings / "episode-20000.npz").open("ab") as f:
        f.write(b"modified")
    with pytest.raises(ValueError, match="checksum"):
        prepare([recordings], tmp_path / "bad")


def test_dataset_manifest_tampering_and_overwrite_rejected(recordings, tmp_path):
    directory = prepare([recordings], tmp_path / "dataset")
    with pytest.raises(FileExistsError):
        prepare([recordings], directory)
    file = directory / "dataset.json"
    data = json.loads(file.read_text())
    data["normalization"]["action_center"][0] = 1234
    file.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="integrity"):
        ChunkDataset(directory, "train")


def test_relative_arm_roundtrip_and_train_only_statistics(recordings,tmp_path):
    from chora.vla.actions import decode_actions
    directory=prepare([recordings],tmp_path/'relative',action_representation='relative_arm',chunk=4)
    dataset=ChunkDataset(directory,'train',chunk=4)
    sample=dataset[3]
    arrays=dataset.episode_arrays(sample['episode_index'])
    encoded=sample['targets']*dataset.norm['action_scale']+dataset.norm['action_center']
    decoded=decode_actions(encoded,arrays['joint_positions'][sample['tick']],'relative_arm')
    np.testing.assert_allclose(decoded[sample['valid']],arrays['actions'][sample['tick']:sample['tick']+4],atol=1e-6)
    assert max(np.abs(dataset.norm['action_center'][:5]))<.1
    # Gripper remains absolute and uses only the two training episodes.
    assert dataset.norm['action_center'][5]<3
    with pytest.raises(ValueError,match='Chunk'):
        ChunkDataset(directory,'train',chunk=16)
