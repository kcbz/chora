# Visual inspection

Choose **VLM · Visual inspection** for the standard bowl or raised-shelf block,
select a saved episode, and click **Inspect frame**. The model gets the saved
overhead final image and one fixed question about whether the object fits inside the
marked target. Chora shows the answer beside the simulator's geometric answer.
The model cannot move the arm. A final frame cannot establish stable placement
over time, so the geometric answer measures horizontal fit only.

The model is the local [MLX 4-bit Qwen3-VL-2B-Instruct variant](https://huggingface.co/mlx-community/Qwen3-VL-2B-Instruct-4bit) pinned in
`chora/visual_inspection.py`. It runs on Apple Silicon in an optional environment.
From the workspace root, set it up once:

```bash
uv venv --python .venv/bin/python .venv-vlm
uv pip install --python .venv-vlm/bin/python -r requirements-vlm-lock.txt
.venv-vlm/bin/python -m chora.visual_inspection download
```

Run the pilot benchmark with six inside and six outside examples drawn from each
scene's verified student evaluation:

```bash
.venv-vlm/bin/python -m chora.visual_inspection benchmark --per-class 6
CHORA_EXPERIMENT=so101-part-raised-shelf .venv-vlm/bin/python -m chora.visual_inspection benchmark --per-class 6
```

The report records the model revision, overhead view, each frame, model and simulator answers,
and inference latency. Uncertain or unparseable answers count as misses. Scores
appear in **Reports** when the VLM method is selected. This small set is a
diagnostic of the image question, not a qualification for autonomous inspection.


The overhead pilot matched **11/12** Standard bowl frames and **9/12**
raised-shelf block frames. Each set contains six examples on each side of the
fit criterion. The first front-view prompt answered yes on every example
and scored 6/12 on each set. Camera angle and prompt both changed for the
overhead pilot. Inspect answers against simulator geometry before using them
to make decisions.
