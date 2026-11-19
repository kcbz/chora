# Tiling baseline: source records and simulation-only physics

The target remains a two-arm floor/wall tiling cell. Physical calibration and new real-world data collection are explicitly deferred at the user's request. Manufacturer data anchor the baseline; missing mechanical properties are separate editable simulation assumptions. No hardware-validation claim is made.

## Selected products

**Tile:** Metropolitan Ceramics / Ironrock QuarryBasics Ironspot **31IS Mayflower Red, 6 x 6 x 1/2 inch**, an unglazed ceramic quarry tile. Manufacturer actual dimensions convert to **152.4 x 152.4 x 12.7 mm**. The manufacturer recommends a nominal **3/8-inch (9.525 mm) joint**. Original manufacturer documents and hashes are in `assets/material_sources/metro_31is/`.

- [Current brochure](https://metroceramics.com/wp-content/uploads/2026/07/Metropolitan-Deluxe-Brochure-March-20262.pdf), printed pages 2-3 and 7: product, actual face dimensions, thickness, joint recommendation and family-level test results.
- [Performance specifications](https://metroceramics.com/wp-content/uploads/2023/07/Metropolitan-Ceramics-Test-Results-2023.pdf): manufacturer reports **800 lbf or better (3558.58 N)** for the 1/2-inch-thick family under ASTM C648. This is a standardized breaking-test result, not a safe robot force, a SKU/batch strength distribution, a constitutive law, or a drop threshold.
- The published maximum 3% five-hour-boil absorption is not a water-uptake-rate model. Published wet DCOF over 0.60 is a specific tribometer result, not cup/sponge/adhesive contact friction.
- Manufacturer documents inspected do **not** publish net tile mass, elastic modulus, fracture toughness, impact response or robot-grip friction. The SDS explicitly lists relative density as unavailable.

**Provisional simulation mass:** uniform-envelope volume times **assumed 2200 kg/m3** density gives **0.6489277344 kg**. The assumed density sensitivity interval 1800-2600 kg/m3 implies approximately 0.531-0.767 kg. These are assumptions, not measured tile-to-tile variation. Back relief, rounded edges and manufacturing tolerances are not resolved by this cuboid.

**Adhesive:** LATICRETE **254 PLATINUM grey**, US/Canada datasheet **DS-677-1026**. [Manufacturer PDF](https://cdnmdm.laticrete.com/Datasheet/NA/14/Datasheet_NA_en_14.pdf).

- Typical wet density **1650 kg/m3**; published working properties include **2 h pot life** and **40 min open time**.
- Minimum **24 h before grouting at 21 C**, according to the installation instructions.
- The 28-day tensile-bond range **1.8-2.6 MPa** is retained as reference evidence, never assigned to freshly mixed paste.
- These values do not establish yield stress, flow curves, thixotropic kinetics, early tack, wall creep or a complete cure curve.

**Grout:** LATICRETE **PERMACOLOR Grout**, datasheet **DS-250-1026**. [Manufacturer PDF](https://cdnmdm.laticrete.com/Datasheet/NA/5600/Datasheet_NA_en_5600.pdf).

- Pot life **45-60 min at 21 C**, final set **2-3 h**, manufacturer joint range **1.5-12 mm** (rounded metric).
- The model chooses the lower pot-life anchor as a conservative simulated action cutoff. This is an explicit model choice.
- Initial cleaning is approximately **20-30 min at 21 C**; the cleanup subtask starts at 25 min. This is a manufacturer timing anchor, not a cleanability curve.
- Published 28-day flexural strength is **8.5-9.2 MPa**, compressive strength **35.7-36.8 MPa**, and shrinkage **0.51-0.62 mm/m**. These mature properties are reference data, not fresh-paste strengths.
- Wet density **2000 kg/m3**, yield stress, flow coefficients and wiping/collection coefficients are assumptions, not manufacturer measurements.

Downloaded source files are preserved with URL, retrieval date and SHA-256 records under `assets/material_sources/`. Published information is recorded by test/condition rather than being treated as a universal material label.

## Implemented now

`configs/materials/tiling_baseline.yaml` is the canonical product/assumption record. Every property declares units, evidence status, source/context or assumption rationale. `chora/tiling_materials.py` is shared by simulation, API, reports and workbench.

1. **Age-dependent paste response:** an assumed Herschel-Bulkley shear law, a first-order structural breakdown/recovery state, and an irreversible age multiplier. Pot life and exposure/open-time clocks are distinct. The baseline environment is held at 21 C; no calibrated temperature/humidity or chemical-hydration solver is claimed.
2. **Squeeze-film diagnostic:** circular-equivalent continuum estimate based on area, gap, approach speed and the assumed paste response. This is a numerical material-workbench calculation, not a force secretly applied beneath native robot-placed tiles.
3. **2 x 2 material accounting sequence:** finite adhesive reservoir, four under-tile volumes, squeeze-out, separate grout mixing time, finite joint capacity and face residue. Excess material moves to another compartment. The sequence enforces the published minimum grouting age. Coverage is explicitly a volume-based upper bound, not a resolved wetting/void map. This reference sequence has no robot motion.
4. **Two-arm cleanup subtask:** native FR3 Duo with actual-size product cuboid and the estimated mass. A spatial finite-cell surface layer replaces the old contact-dot-only demonstration. Removal requires actual native sponge contact and relative motion, with finite collection capacity. Assumed paste shear applies equal-and-opposite reactions to the tool and tile. Volume moves from residue cells to the sponge ledger; it is not simply deleted. The scene begins with prepared workholding and a prescribed residue layer.

The surface model is a reduced transport approximation. It does not resolve free-surface grout flow, individual grains, pore-scale absorption, evaporating water, grout setting chemistry or collected-material inertia changes. The native tile remains rigid. Its surviving a drop is **not evidence of damage resistance**. Tool fixtures remain explicitly custom and unverified.

## What the software checks establish

Tests check source/assumption separation, unit conversions, shear thinning, structural recovery, aging, independent batch/open-time gates, adhesive/grout volume conservation, grouting sequence, contact-dependent collection, finite capacity, expired material handling, finite bounded robot torques, contact forces and solver warnings. They are numerical/implementation checks, not validation of product behavior in the real world.

The first native cleanup pass is intentionally reported with its actual remaining material. It is not scored as a completed cleanup merely because the scripted raster finished.

## Remaining before a floor/wall installation expert

- Couple spatial adhesive ridges, bedding flow, gaps and voids to tile motion and robot forces.
- Implement floor placement and vertical-wall adhesion/slip/support as actual robot/environment interactions. A reference material sequence is not a substitute for this.
- Add actual application/grouting tools and their interaction models; existing mounts are custom fixtures.
- Add an explicitly parameterized damage/fracture model if tile breakage is to be evaluated. ASTM C648 alone does not provide its parameters.
- Define a scored simulation task and student action protocol; then create an expert and train/evaluate against the chosen simulation assumptions.

Physical calibration is **not** a prerequisite for these simulation-only stages. It remains separate from the implementation and numerical checks, and would be needed for later hardware claims.

## Open and reproduce

- Live workbench: `http://127.0.0.1:8765/materials/tiling`
- Native cleanup scene: `http://127.0.0.1:8765/?experiment=fr3-duo-tile-wipe`
- Machine-readable record/reference sequence: `outputs/tiling_materials/data.json`
- Assumed age sweep: `outputs/tiling_materials/assumed_model_age_sweep.csv`

```
.venv/bin/python -m pytest -q tests/test_tiling_materials.py tests/test_articulated_tasks.py
PYTHONPATH=. .venv/bin/python scripts/record_articulated_trials.py fr3-duo-tile-wipe
```

The original SO-101 bowl data, selected adapter and benchmark remain separate.
