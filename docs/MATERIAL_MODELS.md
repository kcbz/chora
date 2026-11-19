# Material and tool models

## Current articulated robot trials

The active selector now uses native FR3, UR10e and FR3 Duo adapters. See
[real objects and tiling](REAL_OBJECTS_AND_TILING.md) for their controllers,
contact assumptions, force stops and calibration status. The original SO-101
bowl task and trained model are unchanged.

## Archived XYZ tool trials

The previous tool selector used `gantry-hammer-impact` and `gantry-cut-*`, with
a custom, force-limited XYZ platform and installed tools. They replace the
SO-101 tool scenes below; original datasets/scores are retained as historical
prototypes, not transferred to the replacements.

The hammer is the complete YCB scan. Chora uses a separate, aligned face contact
proxy and assumed 0.7 kg translating assembly. A 0.35 J air acceleration is
followed by a motor-off impact. Downward drive is also prohibited near the nail
during retraction. The guided nail uses 120 + 3000*depth N dry friction, with a
0.4 ms friction time constant to avoid the original soft-friction transient.
Tests bound insertion work by incoming energy and check submillimetre alignment.
These are illustrative rig/wood parameters, not validated pine driving data.

The chef knife is an imported RoboCasa visual mesh with a thin collision proxy.
Food uses 150 moving nodes and 3D tetrahedral elasticity, two surface membranes
for skin, and 25 cohesive interface bonds. Native MuJoCo flex forces deform flesh
and skin; contact work advances a local cohesive front on one predefined plane.
A continuous skin is rendered before damage; split surfaces only appear once
bonds release. Apple bulk E=3 MPa, tomato E=15 kPa, banana E=150 kPa are uncalibrated
model choices. The Apple E matches an example simulated material in DiSECt,
not a fit to these objects. Skin parameters are also uncalibrated.

The model is a small-strain hyperelastic soft solid. It does not represent
flowing tomato juice, ripeness, true plastic crushing, arbitrary fracture, knife
edge wear, or a validated force-displacement response. Node displacement includes
fragment motion, so it is not labelled a material strain measurement. There is
no success-rate/training pipeline for these engineering trials. Recorded data
and live diagnostics support inspection without treating the result as validated
cooking competence.

MuJoCo 4.0.1 uses its discrete integrator for implicit flex elasticity, with
150 solver iterations and tolerance 1e-6. Tests check solver warnings as well as
force response and deformation. The platform is a custom simulation design,
not a certified commercial robot model.

## Archived SO-101 prototypes


These scenes are engineering prototypes, not validated predictions of real wood
or fruit. `chora/materials.py` holds all assumptions in SI units; each new scene's
report, dataset fingerprint, and lab show its model and calibration status.

## Tool grip and robot

Both the hammer and knife are free rigid bodies. They start in a prepared pinch
grasp. No weld or attachment joins the tool to the robot. Finger collisions and
Coulomb friction support flat-sided hardwood handles, with explicit rubber-like
pad geometries (assumed friction coefficient 0.8). Both fingers must contact the tool, handle
slip must remain below 12 mm, and the tool must clear the work at completion.
Opening the gripper drops the tool. Tool pickup is not implemented.

The original Menagerie robot's actuator torque bounds (2.94 Nm in this model)
are retained. These are simulation-model bounds, not independently measured
continuous torque limits of the user's arm. The miniature hammer head is about
96 g; the knife blade about 29 g, computed from geometry and rigid steel density.
Steel is rigid: elastic modulus is recorded as reference metadata, but there is
no steel plasticity, nail bending, edge dulling, or hardness-dependent wear.
Solver contact compliance is numerical, not a measured steel Young's modulus.

## Pine nailing

A 30 mm board, 2 mm nail and 20 mm insertion goal replace the earlier generic
slider demo. A guided nail represents a fixture-supported starting nail. The
resistance model is `F = pressure * tip_area + shank_shear * circumference * depth`.
The example bearing pressure is 5 MPa and shank shear 30 kPa. These are explicitly
uncalibrated assumptions, not published measurements of this exact pine/nail pair.
A fixture supports the nail's small weight, avoiding pre-impact creep. The nail
is not collision-meshed against wood; the insertion law represents its material
interaction. No wood splitting or bending is simulated. Successful old demo
scores do not apply to this new scene; arm-limited incomplete insertions are valid.

Wood hardness alone does not define insertion force. To calibrate this model we
need force-versus-depth/velocity traces for the chosen nail diameter, tip shape,
wood species, density, grain direction and moisture, plus impact speed/energy.
Withdrawal-force formulas must not be substituted for driving-force data.

## Cutting

A tapered steel blade, wood board, shallow cradle and distinct food geometry are
included. Banana is cut across its length; apple and tomato are bisected. Skin
and flesh have distinct illustrative resistance values. A force applied to the
free knife and an opposite force applied to the food feed back into robot/grip
dynamics. Positive cutting work accumulates only while an aligned blade moves
down inside the food. Separation requires both near-full penetration (1 mm
geometric terminal tolerance) and the integral of the layered resistance law.
Food consists of two rigid halves connected before release on one predefined
plane. Released halves move through forces, never pose changes. A small release force
is limited by an energy budget of 2% of the accumulated cutting work. This is a reduced
order cohesive-work surrogate, not arbitrary fracture topology, FEM deformation,
juice flow, or a validated model of skin puncture and flesh elasticity.

Skin flecks, wood grain, apple core/seeds, tomato flesh/locules/seeds, and banana
peel/interior are deterministic procedural texture maps. They are not photographs,
scans, or a photorealistic material library. Knife steel uses renderer shading.

Calibrating cutting requires measured blade forces and trajectories for the
specified edge angle/sharpness, slice speed and draw motion, fruit variety,
ripeness, temperature and skin condition. Appearance realism additionally needs
licensed scan/PBR assets and a renderer that supports their intended maps.

## References and calibration direction

- MuJoCo contact forces and friction: https://mujoco.readthedocs.io/en/stable/computation/
- MuJoCo modeling and actuator limits: https://mujoco.readthedocs.io/en/latest/modeling.html
- DiSECt uses finite elements/damage and calibrates cutting against measured forces: https://diff-cutting-sim.github.io/
- DiSECt's published apple material example is a simulated dataset, not validation of our layered law: https://github.com/NVlabs/DiSECt/blob/main/dataset/README.md

Do not relabel prototype scores as real-hardware competence. Fitting the above
models and testing on held-out physical measurements is required before that claim.
