# Third-party components

- **SO-101 model:** Google DeepMind MuJoCo Menagerie, `robotstudio_so101`.
  https://github.com/google-deepmind/mujoco_menagerie/tree/main/robotstudio_so101
  Tested revision: `370c43965323c0944b433b202aac5058b5f907a9`.
  The model directory is Apache-2.0 licensed; its original license and attribution
  remain in `assets/menagerie/robotstudio_so101/LICENSE` and `README.md`.
  Upstream robot design: The Robot Studio / Hugging Face SO-101.
  Scene additions and studio lighting are described in `chora/sim.py`.
- **MuJoCo:** https://github.com/google-deepmind/mujoco (Apache-2.0).
- **MLX / MLX-LM:** https://github.com/ml-explore/mlx-lm (MIT).
  Chora calls the installed library; its original source and notices remain in
  `.venv/lib/python3.12/site-packages/`.
- **Qwen3-1.7B:** https://huggingface.co/Qwen/Qwen3-1.7B (Apache-2.0).
  4-bit MLX conversion: https://huggingface.co/mlx-community/Qwen3-1.7B-4bit.
  Downloaded model configuration and tokenizer are retained in `assets/qwen`.
- Other installed dependencies retain their own licenses in `.venv`.

Chora's name is a provisional working name. No name-availability claim is made.

## Imported tools used by the XYZ trials

- **YCB 048_hammer**: Yale-CMU-Berkeley Object and Model Set, textured Google
  16k scan. CC BY 4.0. Source: https://www.ycbbenchmarks.com/ and
  https://registry.opendata.aws/ycb-benchmarks/ . Retrieved 2026-11-18 from
  https://ycb-benchmarks.s3.amazonaws.com/data/google/048_hammer_google_16k.tgz .
  Original files remain in `assets/ycb/048_hammer/google_16k`. Chora rotates and
  translates vertices at load time to align the striking face, and flips the
  OBJ V coordinate to MuJoCo's in-memory convention. Visual mesh is not a
  measured collision/constitutive model. License: https://creativecommons.org/licenses/by/4.0/ .
- **RoboCasa knife_22 chef's knife**, packaged as `knife22r` by
  EmbodiedSWE/robobench-assets. Source:
  https://huggingface.co/datasets/EmbodiedSWE/robobench-assets/tree/main/robobench/suites/cutting/assets/knife22r .
  USD visual surface converted to OBJ by `scripts/import_kitchen_knife.py`;
  collision proxy, holder and fixture are Chora additions. The original USD and texture remain local-only; they are excluded from the
  publishable history. The upstream `NOTICE` remains in `assets/kitchen_knife`.
  Upstream identifies it as Objaverse-derived with per-object CC terms; the
  mirror does not identify its original artist or exact CC variant. No claim
  is made that the mesh is covered by Chora's or the mirror's Apache license.
  Original attribution/terms must be resolved before redistributing this asset.
  No knife binary is included in the publishable repository.
- **Pixar OpenUSD / usd-core 26.11**: https://github.com/PixarAnimationStudios/OpenUSD .
  Optional one-time converter only. Runtime uses the already converted OBJ.

## Robot evaluation assets

The separate study in `outputs/robot_evaluation/` uses additional unmodified
Menagerie directories from the same pinned commit
`370c43965323c0944b433b202aac5058b5f907a9`: `franka_emika_panda`,
`franka_fr3_v2`, `franka_fr3_duo`, `universal_robots_ur5e`,
`universal_robots_ur10e`, and `kuka_iiwa_14`. Their original licenses, attribution,
conversion notes and limitations remain in each model directory. This download
does not register them as active Chora task adapters or transfer task scores.

## Produce appearance assets

- Poly Haven **Food Apple 01** and **Bananas**: https://polyhaven.com/a/food_apple_01 and https://polyhaven.com/a/bananas — CC0 1.0. Original glTF, buffers, textures, API manifests and per-file hashes are preserved in `assets/produce/`. Appearance is uniformly scaled/oriented; cut-plane clipping and MuJoCo skin binding are Chora adaptations. No calibrated material properties are supplied by these assets.
- **Tomato 2**, Moshe Caine: https://sketchfab.com/3d-models/201d42a9e5e54f1b836eff39a892ed88 — CC BY 4.0, author/license verified through public Sketchfab API metadata. Two processed USD pieces from EmbodiedSWE/robobench-assets are retained for inspection, with source hashes and attribution. They are not used by the current simulator; the tomato surface remains procedural.

### Tile and mortar surface appearance

`assets/tiling_surfaces/` contains CC0 assets from Poly Haven:

- [Terracotta Floor Tiles](https://polyhaven.com/a/terracotta_floor_tiles), Dimitrios Savva: diffuse and displacement maps. PNG diffuse copies are lossless format conversions for MuJoCo; individual tile interiors are selected through mesh UVs, without the photographed grout joints.
- [Scuffed Cement](https://polyhaven.com/a/scuffed_cement), Dario Barresi (processing), Dimitrios Savva (photography): diffuse map, used as a cement appearance reference on the mortar mesh.

These are appearance analogues, not scans of QuarryBasics 31IS or fresh LATICRETE
254 PLATINUM. Texture maps provide no rheological, friction, adhesion or fracture
data. Visual tile bevel/relief amplitudes are chosen rendering detail; collision
envelopes and material properties are separate. Source URLs/checksums are recorded
in `assets/tiling_surfaces/sources.json`.

Manufacturer reference PDFs are not bundled or relicensed. Their source URLs,
checksums and interpreted material records remain in the repository; optional
local retrieval is documented in docs/ASSETS.md. Large runtime tiling textures
are likewise retrieved with checksum verification rather than stored in Git.
