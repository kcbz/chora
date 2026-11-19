# Workspace setup and asset policy

Chora code is proprietary; see [LICENSE](../LICENSE). Third-party code and assets retain their
own licenses. Publishing this repository does not grant an open-source license
to Coherence's code.

## Fresh checkout

Run `./setup.sh` on an Apple Silicon Mac with Python 3.12. It installs the pinned
environment, checks out the SO-101, FR3, FR3 Duo and UR10e models plus the optional
robot-study models at the recorded Menagerie revision, and fetches the runtime
tiling textures. Menagerie retains its upstream licenses in its local checkout.
Qwen weights and experiment outputs are deliberately local; download/train them
with the documented CLI when needed.

Runtime tiling textures are downloaded from approved HTTPS hosts, verified
against SHA-256 hashes in `configs/assets.json`, and installed atomically. PNG
files are regenerated with the pinned Pillow version. No file with a differing
local hash is silently overwritten.

```sh
.venv/bin/python scripts/fetch_assets.py          # required tiling textures
.venv/bin/python scripts/fetch_assets.py --check  # offline integrity check
.venv/bin/python scripts/fetch_assets.py --documents # optional source PDFs
```

Reference PDFs are retrieved from publishers for local inspection. They are not
bundled in the repository or relicensed. Source URLs, checksums and interpreted
material records remain in Git. Downloading a texture supplies appearance data,
not measured mechanical properties.

## Knife and optional unused assets

The previously imported knife mesh, texture and USD have unresolved upstream
attribution/license terms. They are excluded from the publishable history and
kept locally on the development machine. There is no automatic download or
procedural substitute. Cutting scenes are marked unavailable when these files
are missing. Enable them only after supplying an appropriately licensed tool
asset and reviewing its geometry/controller integration. The upstream notice
is retained for provenance, not as a license grant for the omitted asset.

Unused processed tomato USD pieces and generated JPG-to-PNG produce copies are
also excluded. Produce PNGs are regenerated from the retained CC0 JPGs at runtime.
The licensed YCB hammer and CC0 produce source files retain their provenance in
[THIRD_PARTY.md](THIRD_PARTY.md). Original downloaded files on the development machine are not
deleted by the publication cleanup.

## Source workspace versus wheel

A wheel contains Python modules and all HTML/CSS/JavaScript/SVG UI resources. The
large simulation data and workspace YAML configurations are deliberately outside
the wheel. Editable installation from the checkout is the supported default.
When using a wheel, set `CHORA_HOME` to a prepared checkout:

```sh
CHORA_HOME=/absolute/path/to/chora python -m chora serve
```

Installed UI resources come from the installed package; assets, configurations,
model weights and outputs come from that workspace. A wheel alone is not a
self-contained download of the simulation assets.
