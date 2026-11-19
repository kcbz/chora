# Shared tiling scene and Qwen interface

## Goal and current scope

One persistent two-arm environment should support individual stages and the
complete installation sequence: inspect/plan, prepare, apply adhesive, prepare
and acquire tiles, place/bed, inspect/clear joints, wait for setting, and grout/
finish. The scripted expert and Qwen should share observations and bounded
primitives. The existing Qwen method is imitation learning; RL is separate work.

The active **FR3 pair · Adhesive and tile bedding** scene now implements the
application/pickup/bedding milestone. It is not a completed installation job.
It uses two separately mounted native FR3 arms, four free rigid QuarryBasics
31IS tiles, a rack, a rigid horizontal substrate and a 2 × 2 pattern. Product
geometry and the recommended 9.525 mm joint spacing come from the manufacturer;
tile mass is estimated from assumed density. This is an elevated workcell patch,
not yet a wall or a moving robot installation on a room-sized floor.

## Robot sequence

Arm B uses an installed, simplified custom metering/comb head with a finite
500 mL premixed adhesive supply. It moves down beside each target and traverses
the footprint. A metered dose leaves the reservoir only along the actual tool
path near substrate height. Alternating notches define the model's ridge cross
section. This combines metering and combing; it is not a simulated conventional
hand trowel collecting and working a dollop. The hopper/tool uses fixed lumped
payload mass; fluid mass/inertia transfer is omitted.

A acquires the current tile while B approaches and applies its adhesive patch.
The explicit `parallel` primitive sends both arms' references through one native
physics timeline; it does not call two blocking moves in sequence. The arm
controller preflights the combined IK trajectory, bounds motor torques and checks
native inter-arm contact. Measured simultaneous TCP movement and its overlap with
actual dispensing are exposed separately.

B then closes its valve and moves to a nearby clearance position, not its home
pose. A transports, beds and releases the tile. The next cycle overlaps rack
return/pickup with the next application. B returns home only after its final
application. This is coordinated pickup/application with a clearance dependency
before placement, not simultaneous occupation of the same patch. The raised
feed manifold, narrow blade backplate and elevated stem/neck keep the prototype
head above already placed neighbors; native tool/tile contact stops the run.

The complete reference now has 53 actions. Clocks, material volume, body poses
and failures persist throughout; no objects are reset or teleported between
cycles. The serial 73-action reference is preserved in the output archive.

The vacuum and applicator are connected custom installed fixtures, not vendor
CAD, verified hardware, frictionally picked-up tools or autonomous tool changes.
Motions use bounded joint torques and Cartesian reference trajectories. Bedding
uses a slow position-controlled approach with load monitoring/guards; it is not
a closed-loop pressure-regulation controller. Force limits are software bounds,
not ceramic damage thresholds. Applicator contact with a tile stops the run.

## Mortar model and its limits

`chora/mortar_layer.py` implements a spatial height field, 16 × 16 cells per
footprint. The nominal combed ridge height is 8 mm with alternating 9.525 mm
ridges/voids, giving about 93 mL per patch. Parameters and limitations are recorded
in `configs/materials/mortar_layer.yaml`; manufacturer facts and constitutive
assumptions remain separated in `configs/materials/tiling_baseline.yaml`.

Tile underside/height-field intersection generates distributed support forces:
linear elastic stress from penetration, normal damping, and assumed
Herschel–Bulkley shear limited relative to the normal load. Stress above an
assumed confinement-dependent yield law relaxes ridge height. Displaced volume
fills available voids across that tile footprint; surplus enters an explicit
outside-footprint squeeze-out compartment. Resultant force and moment act on
the free rigid tile in MuJoCo, with reaction against the fixed support. The
applicator also receives assumed HB drag during metering.

This is a conservative reduced foundation/transport model, **not a resolved
3D fluid or mortar FEM solver**. In-footprint redistribution is lumped rather
than a predicted local flow front. Squeeze-out is rendered uniformly around the
perimeter to show its compartment volume, not a predicted extrusion shape.
Paste gravity and fluid inertia, evaporation, absorption, water separation,
hydrated chemistry, tensile/cured bonding, tile bending and fracture are absent.
The paste's global resting structural state is not a spatial shear-history
solver. These omissions must not be inferred from visually plausible motion.

Adhesive age advances with every physics tick. Deposition stops after the
published pot-life anchor; success also requires the published open-time anchor
not to have elapsed since each patch was spread. These fixed 21 °C time cutoffs
are not calibrated environment-dependent curing curves. No grouting-age advance
or grout finishing is part of this milestone.

## Checks and UI

A tile must be released, slow, upright and supported by the modeled mortar,
within 3 mm horizontal and 2 mm vertical target error. The yaw threshold is
about 1.15° and tilt threshold about 2.56°. Bedding also requires more than 85%
modeled grid contact, an unexpired adhesive window and conserved material volume.
These are chosen software criteria, not installation standards or certified
coverage. A full milestone result requires all four tiles; missing mortar
cannot pass. `installation_complete` remains false even when this milestone
returns `adhesive_bedding_complete`.

The live material card exposes reservoir, deposited volume, age, per-tile load,
modeled contact fraction, squeeze-out and balance error. Reports share the lab's
appearance system. Record with:

```sh
PYTHONPATH=. .venv/bin/python scripts/record_articulated_trials.py fr3-tile-installation --fps 10
```

Outputs are isolated under `outputs/experiments/fr3-tile-installation`.
`application_overlap.png` captures real pickup/application overlap;
`first_tile_bedded.png` captures the first completed placement. Earlier dry-transfer
assets are preserved in `dry_transfer_baseline/`. Dataset fingerprints identify
the new scene, mortar and visual code, surface asset manifest, parameter records
and actual robot source. Original
bowl demonstrations and the trained bowl adapter remain separate.

## Remaining integration and learning

`chora/tiling_task.py` supplies the eight-stage scheduler, curriculum modes,
action schema and full-job prompt. Its environment-check interface is not yet
bound to all stages of this workcell. The current milestone uses the distinct
`chora-tile-adhesive-bedding-v2` protocol; it cannot pass nonexistent inspection,
material preparation, joint cleanup, setting or grout stages.

Next work includes explicit preparation/inspection and recovery actions, joint
cleanup, time/setting behavior and grout finishing in this same scene, then the
scheduler bindings. Tool changing needs an executed and verified adapter; a tool
request cannot swap geometry for free. Single-stage fixtures must retain
provenance, while chained runs keep the same scene state.

Before training, add varied development/held-out fixtures, score the expert,
and extend recording, curation, model-output parsing and student execution for
this protocol. The reference currently repeats one deterministic fixture:
repetition is a regression check, not generalization evidence. Learning controls
remain disabled. Qwen will initially choose bounded primitives from structured
state rather than consume camera images or command motor torques directly.
Physical calibration is deferred and is not a prerequisite for these simulation
milestones; simulation success must not be presented as hardware validation.

## Visual surface representation

Tiles use individual UV regions of Poly Haven's CC0 Terracotta Floor Tiles map,
excluding its photographed joints, plus a cosmetic 0.8 mm bevel and small height
relief. These are an appearance analogue, not a scan of the selected SKU. The
rigid contact box, dimensions and mass remain separate from this cosmetic mesh.

The mortar renderer uses a continuously updated 65 × 65 vertex mesh per patch,
interpolated from the actual 16 × 16 material volume heights. Normals and GPU mesh
buffers update with this surface; invisible cell boxes are no longer the visible
material. A CC0 cement reference supplies surface detail with chosen reflectance.
It is not a photograph of fresh adhesive or a measured wet optical model. Empty
areas are below the substrate rather than painted in before deposition.
Squeeze-out still uses the declared perimeter compartment representation.

## Recorded simulation checks

The archived serial reference completed 73 actions in 210.749 simulated seconds
and reached a joint torque cap on a long applicator retract. The coordinated
reference completed 53 actions in 140.708 simulated seconds, about 33% shorter.
The headless check recorded 60.371 seconds of simultaneous arm movement, including
14.708 seconds of arm-A motion during actual metering. All four tiles met the
placement/bedding checks, with no detected robot collisions or solver warnings.
Peak commanded actuator fraction was about 0.592, below the caps. Material volume
was conserved. These are deterministic simulation regression results, not
hardware/material validation or a varied expert benchmark. See the current
recording and trace for the results after visual asset integration.
