# Real objects, physical properties, and Chora's tiling direction

## Current implementation

- The original SO-101 bowl benchmark stays unchanged, alongside two fixed platform variants.
- FR3 cooking trials now use the native seven-joint Menagerie FR3 v2, with the imported chef knife in a short rigid flange adapter. The controller stops the downward cut at a **55 N observed-force threshold** and then retracts. This is a discrete stop threshold: transient forces can overshoot it. Food often moves or the force stop triggers before separation. No cutting success score is claimed.
- Hammering uses the native six-joint UR10e, with its base set back 250 mm and the scanned YCB claw hammer installed at the flange. Axial task force accelerates the tool in air and is removed before contact. Joint gravity/orientation control remains active: this is **not an all-motors-off strike**. The effective-mass incoming-energy estimate is not a hardware impact rating.
- **FR3 Duo / Hold and wipe tile:** native 14-joint common-mount model, left underside suction holder, right installed sponge. Both arms move under bounded torque actuators. A free tile is held by finite MuJoCo contact adhesion and friction, with no weld and no object pose updates after initialization. Raster wiping reports contact force, coverage, and relative tile displacement.
- The tile begins in prepared contact with the cup. This does not demonstrate picking a tile from a stack, sealing a real cup, or placing a tile on adhesive. Surface dots mark sampled contact coverage; **there is no grout-removal solver**. Sponge softness uses soft contact, not a calibrated foam or absorbent-body model.
- The shared adapter/controller is `chora/arm_control.py`. It preserves the source links, joint limits, inertias and actuator force bounds; uses a payload-aware bounded torque controller with Cartesian IK trajectory references; and changes no source robot files. Tool geometry and food mechanics remain shared.

Robot files: MuJoCo Menagerie revision `370c43965323c0944b433b202aac5058b5f907a9`. The simulation models and torque maxima do not establish real hardware impact tolerance, continuous-duty capacity, tool certification, or manufacturer acceptance of this mounting arrangement.

## Where to get objects and properties

| Source | Useful content | What it does not establish |
|---|---|---|
| [YCB object set](https://registry.opendata.aws/ycb-benchmarks/) and [primary paper](https://people.eecs.berkeley.edu/~pabbeel/papers/2015-RAM-YCB-object-set.pdf) | Textured scans of tools and household objects, including hammer, wood block and fasteners; major dimensions and masses. Downloads are CC BY 4.0. Our hammer scan already comes from YCB. | No complete force/fracture/grasp calibration. **YCB fruit objects are plastic replicas**, not real food specimens. Their properties cannot stand in for ripe produce. |
| [BEHAVIOR objects](https://behavior.stanford.edu/behavior_components/objects.html) and [annotation process](https://behavior.stanford.edu/behavior_components/annotation_process.html) | Interactive household assets with physical/semantic annotations and object states. | Annotated values are not automatically measurements of a particular specimen. [Dataset assets are encrypted and loaded through DatasetObject](https://behavior.stanford.edu/getting_started/quickstart.html); this is not a simple portable MuJoCo asset library. Access/license and runtime compatibility need checking per use. |
| [Poly Haven](https://polyhaven.com/license) | CC0 PBR textures, models and lighting for wood, stone and surfaces. | Appearance data do not supply friction, fracture, rheology, wetting or validated contact mechanics. |
| [USDA Wood Handbook](https://research.fs.usda.gov/fpl/wood-handbook) | Species-dependent density, elasticity, moisture effects, strength and fastening references. | No universal pine nail-driving curve. Species, grain direction, moisture, nail diameter/coating, pilot hole, insertion speed and splitting matter. |
| [DiSECt cutting research](https://diff-cutting-sim.github.io/) | FEM/damage modeling and a method for fitting knife-force/deformation data. | Not a universal downloadable apple/tomato/banana material, and not a drop-in macOS MuJoCo module. Simulated datasets are not measurements of our produce. |
| Manufacturer CAD, product datasheets, and physical samples | Exact tool/tile dimensions, nominal mass and material grade; adhesive/grout product instructions and test specifications. [Example grout product documentation](https://www.laticrete.com/en/products/spectralock-1). | CAD alone does not establish blade sharpness, inertia, wet friction, cutting toughness, or the full time-dependent flow/removal law of mixed grout. |

Our knife remains the imported RoboCasa knife_22 asset through Robobench; preserve its upstream notice. Its exact original asset license still needs resolution before redistribution. Apple and banana now use textured Poly Haven geometry (CC0), uniformly scaled and bound to the coarse FEM envelope. Their interiors remain procedural and material numbers are illustrative, not specimen measurements. Tomato remains procedural; its licensed source scan has been identified, but the available processed pieces are not an intact drop-in asset.

## A physical asset needs more than a mesh

Track geometry and visual provenance separately from physical provenance. For each parameter record value, units, source (manufacturer / measured / fitted / assumed), uncertainty, sample/product identity, test conditions, calibration date and validity range. Source file hashes and license notices belong with imported assets. Do not label an entire object 'realistic' because one property came from a datasheet.

`configs/materials/tile_wipe.yaml` is the first explicit tiling material record. Its tile dimensions/mass, friction, cup force and sponge response are all marked **engineering assumptions**. They are used by the running two-arm scene. No laboratory calibration records exist yet.

The actual rigid tool contact geometry is simplified: a hammer-face proxy and a convex knife-blade profile. Rendered scan detail does not imply matching collision detail or known mass distribution. Installed brackets use assumed mass, dimensions and stiffness. Foods use deformable bulk/skin mechanics with a predefined cohesive plane; tomato liquid, arbitrary crack propagation and specimen-specific ripeness are not implemented.

## The tiling work sequence and what to validate

For conventional installations, separate bedding adhesive from joint grout: prepare substrate → spread/comb tile adhesive → place and bed tile → allow the adhesive to set per product instructions → fill joints with grout → wipe residue. See [manufacturer adhesive guidance](https://www.laticrete.com/en/support-and-training/downloads-and-faq/faqs/planning-a-job/thinsets). Product-specific exceptions should be explicit, rather than assuming grout is the bedding material.

1. **Workholding and cleanup contact (implemented first primitive):** suction capacity, tile slip/tilt, sponge load, sampled path coverage. Next measure real tile, cup and wet/dry sponge response.
2. **Dry placement:** tile pickup/seal verification, collision clearance, pose accuracy, joint gaps, lippage, release and regrasp. Start on floors, then include wall support and gravity-driven slip.
3. **Adhesive spreading and bedding:** combed ridges, trowel geometry/contact, rheology, squeezed volume, bonded coverage and sag. Match instrumented force and deposit measurements.
4. **Grouting and cleanup:** joint fill, yield stress/shear thinning, mixing/working time, sponge saturation and residue transfer. Match mass/volume balance and observed residue, rather than deleting a texture on contact.

Use a small instrumented physical test cell to fit these material models, then evaluate on different paths, speeds and material batches. A force/torque sensor at the tool and logged vacuum pressure are especially useful. Larger floor/wall tiles and troweling loads must be screened for payload **including the tools**, wrist moments, reach, mounting, duty and contact loads; FR3 Duo is the current coordination research platform, not a final construction-cell selection.

The present rigid/contact scenes run locally on the M3 in MuJoCo. High-fidelity grout/adhesive flow will require a separate solver or coupling evaluation; switching render assets cannot supply that physics.

## Reproduce

```
PYTHONPATH=. .venv/bin/python scripts/record_articulated_trials.py fr3-duo-tile-wipe
PYTHONPATH=. .venv/bin/python scripts/record_articulated_trials.py ur10e-hammer
PYTHONPATH=. .venv/bin/python scripts/record_articulated_trials.py fr3-cut-apple
```

Recordings and unscored observations are isolated under `outputs/experiments/<id>/material_trial.*`. The original adapter, dataset and scored bowl results are not overwritten.

## Mounting correction and remaining physical gaps

A non-unit UR attachment quaternion was used directly in a rotation matrix, doubling parts of the flange offset. It is now normalized before composition. Both dual-arm stems were short of their flange by 20–35 mm and now meet the flange. Regression checks verify compiled world-space connections across different joint poses. These corrections remove floating geometry; **the fixtures are still custom unverified designs**, not downloaded or mechanically qualified mounting hardware. The lab now labels them that way.

The knife collision proxy now has an explicit 30 µm truncated edge, tapering to a 1.2 mm body over 15 mm. These are assumed simulation values, not a measured edge radius, bevel or sharpness certificate. A calibrated cut needs measured edge geometry and force–displacement curves, not only a detailed visual mesh.

The nail still has one guided axial degree of freedom. Consequently it cannot bend, buckle or enter obliquely; the rigid board cannot splinter or develop cracks. Predicting those failures requires a freely oriented deformable/plastic nail and a grain-dependent wood damage/fracture model, with calibrated contact, insertion and impact response. Species, moisture, grain, defects and steel properties must be represented. Random failure events do not supply these mechanics. See the [USDA fastening reference](https://research.fs.usda.gov/treesearch/62253).
