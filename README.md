<p align="center">
  <img src="chora/web/logo-octahedron.svg" width="88" height="88" alt="Chora — wireframe octahedron logo">
</p>

<h1 align="center">Chora</h1>
<p align="center"><sub>by coherence</sub></p>
<p align="center"><strong>A space for robotic skills to take shape.</strong></p>

<p align="center">
  <a href="#environment-and-setup">Setup</a> ·
  <a href="#usage">Usage</a> ·
  <a href="#testing">Testing</a> ·
  <a href="docs/README.md">Documentation</a>
</p>

---

Chora is a local robot-learning lab for building and inspecting manipulation
tasks in MuJoCo. It combines scripted experts, recorded demonstrations and local
Qwen imitation learning with a browser interface for simulation and evaluation.

## Why Chora?

**Chora** draws from the Greek **χώρα** (*chōra*): **place, space, or room**.
[Greek definition · LSJ](https://atlas.perseus.tufts.edu/dictionaries/entry/urn%3Acite2%3Ascaife-viewer%3Adictionaries.v1%3Alsj-n115193/)

For Coherence, the name describes a space where robotic skills take shape through
simulation, demonstration and learning. The wireframe octahedron—a Platonic
solid—reflects the geometry and structure of the worlds we model; its open frame
evokes a space to build within.

## Current scope

The supported learning baseline is SO-101 bowl placement. Other scenes explore
hammering, cutting and coordinated two-arm tile placement. These are material and
controller prototypes with explicit model limitations; the tiling student is not
yet trained. Qwen currently consumes structured simulator state, not camera images.

## Environment and setup

Requirements:

- Apple Silicon Mac; initially tested on an M3 with 16 GB memory.
- Python 3.12 and Git.
- Internet access for initial dependencies and assets, with several GB of free disk.

```bash
git clone https://github.com/kcbz/chora.git
cd chora
./setup.sh
```

Setup creates an isolated `.venv`, installs the pinned dependencies, checks out
robot models and retrieves checksum-verified tiling textures. MuJoCo runs the
physics; macOS OpenGL renders the scene, and MLX handles local model training and
inference. No paid model API or OpenAI API key is required.

```bash
.venv/bin/python -m chora doctor
```

Some cutting assets are intentionally not distributed because their licensing is
unresolved. See [asset setup](docs/ASSETS.md) for availability, optional reference
PDFs and wheel/workspace configuration.

## Usage

```bash
.venv/bin/python -m chora serve
```

Open [the local lab](http://127.0.0.1:8765), select a scene and policy, and click
**Run episode**. A trained student is available only when its local adapter exists.
You can also launch the lab by double-clicking `Launch Chora.command` in Finder.
Stop the server with Ctrl-C in its Terminal window.

- Drag to orbit, Shift-drag or right-drag to pan, and scroll to zoom.
- Use **Front view** and **Top view** for preset camera angles.
- Change **1×, 2× or 4×** beside the simulation during playback. These are target
  speeds; the measured multiplier shows achieved throughput.
- Use **Appearance** in the navbar for theme settings, and **Gallery/Reports** for
  locally generated recordings and evaluation results.

To open the coordinated tiling scene directly:

```bash
.venv/bin/python -m chora --experiment fr3-tile-installation serve
```

Learning commands apply to the supported bowl experiment. The detailed workflow
for expert checks, demonstration collection, curation, training and evaluation is
in [Experiments](docs/EXPERIMENTS.md). Material prototypes have their learning
controls disabled until their task-specific pipelines are ready.

## Testing

```bash
.venv/bin/python -m pytest -q
```

Optional tests require the corresponding graphics/GPU access or running lab:

```bash
# With the lab running
CHORA_HTTP_TEST=1 .venv/bin/python -m pytest -q tests/test_http.py

# With native graphics access
CHORA_RENDER_TEST=1 .venv/bin/python -m pytest -q tests/test_simulation.py

# On a Mac with Metal/MLX access
CHORA_MLX_TEST=1 .venv/bin/python -m pytest -q tests/test_learning.py
```

Run from a normal Terminal session if a sandbox cannot access graphics, Metal or
localhost ports. The default suite does not enable the opt-in rendering/MLX tests.

## Project layout and documentation

| Location | Purpose |
| --- | --- |
| `chora/` | Simulation, controllers, policies, learning pipeline and shared web UI |
| `configs/` | Scene, training, material and asset settings |
| `scripts/` | Asset preparation, recordings and developer utilities |
| `tests/` | Regression and security checks |
| `assets/` | Retained source assets and local downloads; third-party terms apply |
| `outputs/` | Ignored local datasets, adapters, recordings, reports and logs |
| [docs/](docs/README.md) | Technical reference, model limits and development history |

See [development history](docs/DEVELOPMENT_HISTORY.md) for changes and historical
results, and [security notes](docs/SECURITY.md) for the local server's boundaries.

## Authors

- Kodiak Berkoff
- Dhruv Sirohi

## License status

Licensing for a future release is **TBD**. For now, Coherence-authored code remains
proprietary under the current [LICENSE](LICENSE); no open-source license is granted.
Third-party components and assets have separate terms recorded in
[third-party notices](docs/THIRD_PARTY.md).
