#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
if [ ! -d .venv ]; then
  python3.12 -m venv .venv
fi
.venv/bin/python -m pip install -r requirements-lock.txt
.venv/bin/python -m pip install -e . --no-deps
if [ ! -d assets/menagerie/.git ]; then
  git clone --depth 1 --filter=blob:none --sparse https://github.com/google-deepmind/mujoco_menagerie.git assets/menagerie
fi
# Active scene adapters and the optional robot evaluation study.
git -C assets/menagerie sparse-checkout set robotstudio_so101 franka_fr3_v2 franka_fr3_duo universal_robots_ur10e franka_emika_panda universal_robots_ur5e kuka_iiwa_14
git -C assets/menagerie fetch --depth 1 origin 370c43965323c0944b433b202aac5058b5f907a9
git -C assets/menagerie checkout --detach 370c43965323c0944b433b202aac5058b5f907a9
.venv/bin/python scripts/fetch_assets.py
.venv/bin/python -m chora doctor
