<p align="center">
  <img src="chora/web/logo-octahedron.svg" width="88" height="88" alt="Chora — wireframe octahedron logo">
</p>

<h1 align="center">Chora</h1>
<p align="center"><sub>by coherence</sub></p>
<p align="center"><strong>A space for robotic skills to take shape.</strong></p>

<p align="center">
  <a href="#environment-and-setup">Setup</a> ·
  <a href="#launch-the-ui">Launch the UI</a> ·
  <a href="#usage">Usage</a> ·
  <a href="#testing">Testing</a> ·
  <a href="docs/README.md">Documentation</a>
</p>

---

Chora is a local robot-learning lab for building and inspecting manipulation
tasks in MuJoCo. It combines scripted experts, recorded demonstrations and local
Qwen imitation learning with a browser interface for simulation and evaluation.

## Launch the UI

After [setup](#environment-and-setup), run this from the `chora` project folder:

```bash
.venv/bin/python -m chora serve
```

Then open **[http://127.0.0.1:8765/](http://127.0.0.1:8765/)** in your browser.
Keep the Terminal window running while using the UI. Press **Ctrl-C** there to
stop the server. You can also double-click `Launch Chora.command` in Finder.

| Command | Purpose |
| --- | --- |
| `doctor` | Prints environment/package information and basic asset availability, then exits |
| `serve` | Starts the local web server that provides the browser UI |

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

To inspect the installed environment and basic asset availability:

```bash
.venv/bin/python -m chora doctor
```

Launch the browser interface with [`serve`](#launch-the-ui).

Some cutting assets are intentionally not distributed because their licensing is
unresolved. See [asset setup](docs/ASSETS.md) for availability, optional reference
PDFs and wheel/workspace configuration.

## Usage

With [the local lab running](#launch-the-ui), select a scene and policy, and click
**Run episode**. A trained student is available only when its local adapter exists.

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

### Test the visual VLA

1. Select **Bowl · Standard platform** in the scene dropdown.
2. Scroll to the **VLA development diagnostics** panel below the main simulation.
3. Choose a checkpoint by its measured development result, keep development seed
   **20003**, and click **Test VLA**. Failed/untested checkpoints remain available
   for explicit diagnostics; they are not selected as a usable default.
4. Watch the scene/wrist camera preview. Use **Stop** to cancel, or **Open run
   results** to inspect the completed test.

VLA tests use the separate `.venv-vla` environment and locally trained weights;
see [VLA setup](docs/VLA.md). The panel explains when either is unavailable.
The panel says **No usable VLA yet** until a checkpoint passes the documented
unseen-scene check. Training completion alone does not qualify. These outcomes
are separate from Qwen's scores.
After updating server code, stop the server with Ctrl-C and run `serve` again.

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
