# Robot-arm selection for Chora — 18 November 2026

Recommendation: develop the Franka FR3 family for general manipulation and
bimanual cooking research. Evaluate UR10e for longer/heavier tool and construction
work. Keep the existing miniature SO-101 bowl benchmark unchanged. This is a
provisional simulation shortlist, not a robot purchase or hardware impact approval.

Open `http://127.0.0.1:8765/outputs/robot_evaluation/index.html` for the full report,
native-model images, model-level results and source links.

| Task | First candidate | One or two arms? |
| --- | --- | --- |
| Existing miniature bowl experiment | SO-101 | One; preserve trained baseline |
| Full-size objects, measuring cup, pouring | FR3 | One if the receiving bowl rests on the bench; two for active vessel holding or mixing |
| Cutting | FR3 with a short rigid knife adapter and a task controller | A second FR3 can stabilize/reorient food; a passive fixture is a simpler baseline |
| Higher cutting forces / longer tool offset | UR5e or UR10e | Second arm optional; depends on workholding |
| Steel hammer / construction tools | UR10e is the next simulation candidate | One tool arm plus board clamp initially; impact and tool-mount validation still outstanding |
| Part holding plus insertion/fastening, handover, jar opening | FR3 Duo / paired FR3 | Two coordinated arms |

## What was evaluated

Pinned Google DeepMind MuJoCo Menagerie commit:
`370c43965323c0944b433b202aac5058b5f907a9`.

Five single-arm models: Panda (no-hand variant), FR3 v2, UR5e, UR10e and KUKA iiwa
14. Each was tested against 62 shared tabletop TCP targets (12 bowl, 18 knife,
12 hammer and 20 tilted pouring poses). Three layout/tool-mount variants add
another 186 pose cases, for 496 single-arm pose cases in total. The native FR3
Duo model was tested at 14 paired target poses. Five native position-controller
hold tests used an assumed tool mass/inertia and a constant test wrench.

Checks include bounded multistart IK, native collision geometry plus bench/floor,
static gravity/tool/contact loads, enabled actuator-force limits, and a local
velocity feasibility probe for hammer poses. Pose tolerance is 2 mm and 2 degrees.

These are endpoint screens, not planned trajectories or executed manipulation
tasks. Grippers, tool collisions, grasps, physical material response and dual-arm
contact coordination were not validated. Numerical IK failures are not proofs
of impossibility, and chosen IK solutions were not optimized for torque margin.

## Findings that change the implementation choice

- Panda and FR3 reached all 12 bowl, 18 cutting and 12 hammer targets in the shared
  layout. Their torque limits, not reach alone, constrained the long tools.
- In the selected knife configurations with 170 mm lateral offset, both Panda
  and FR3 stayed below model torque maxima at 20/40 N vertical reaction plus
  10 N lateral draw force. At 60 N, only 3/18 selected poses remained below the
  model limits. These are hypothetical load cases, not measured fruit forces.
- Reducing the FR3 knife offset to 80 mm, with the corresponding shorter center
  of mass offset, put all 18 knife cases below model torque maxima at 60 N.
  The worst ratio was approximately 0.78 of the model maximum, before any
  continuous-duty, transient or hardware-specific margin is considered.
- Moving the UR10e base 250 mm away from the table changed its knife screening
  from 12/18 to 18/18 collision-free poses and hammer screening from 4/12 to
  12/12. All reached poses in those two cases stayed below the chosen model's
  static force maxima. Its pouring coverage changed too: placement is task-dependent.
- FR3 Duo reached all 14 paired poses without arm collisions: five pouring/holding,
  three cutting/holding, three handover and three assembly pairs. This is not a
  completed bimanual task, and it omits end-effectors and object collisions.
- The pinned iiwa model's actuator force limits are disabled, despite README text
  discussing limits. Its static capacity was deliberately not scored. Audit/fix
  that model before using its apparent strength as evidence.
- The native position-servo hold tests were not task-ready, including the short
  FR3 mount. They are stress tests with a constant applied wrench, not cutting
  contact. A proper payload-aware controller and force-control strategy are
  required; importing an arm model is not enough.

## Why not choose the largest arm for everything?

FR3 provides seven axes, joint torque sensing and a native bimanual-model option.
That is a strong basis for contact-rich research and coordinated workholding.
The UR models offer more static torque margin for the tested long-offset loads,
but a larger reach requires a different cell layout and does not establish
better dexterity or material interaction. Panda is a useful mature simulation
reference with an existing hand model; FR3 is the preferred main family for the
next implementation. SO-101 remains useful for its original small-object task.

ALOHA is a later option for light bimanual learning demonstrations, rather than
the default for these full-size tools and liquid loads. YuMi's 0.5 kg per-arm
payload targets light assembly and is not the primary choice for this shortlist.

## Impact boundary

A static 100 N probe is not a hammer impact rating. The cited Franka impact
experiment used an ABS printed hammer, foam material and pilot holes, so it does
not validate a steel claw hammer driving into pine. Dynamic wrist loads, impact
impulse, tool-holder/gripper behavior, alignment, braking and material response
must be evaluated before any hardware hammering claim. A second arm is useful
for board manipulation between operations; it is not needed merely to replace
a straightforward board clamp.

## Version and source boundaries

FR3 is not the older Panda. The FR3 v2 and dual FR3 assets are distinct models.
Franka's vendor page currently lists Menagerie for single FR3 but not a released
MuJoCo reference-kit setup for FR3 Duo; the Menagerie dual model is a community
conversion of the vendor URDF with its own documented assumptions.

UR5e/UR10e names here refer to pinned model versions and 2024 datasheets. Current
UR naming and payload revisions must not be silently assigned to old simulation
assets. Simulator actuator maxima are not continuous-duty, force-sensor precision
or certified impact limits. No prices or procurement conclusions were evaluated.

## Reproduce

The downloaded source model directories retain their licenses and were not edited.

```bash
# Native models must first be included in the existing sparse checkout:
git -C assets/menagerie sparse-checkout add franka_emika_panda franka_fr3_v2 franka_fr3_duo universal_robots_ur5e universal_robots_ur10e kuka_iiwa_14

# Separate study; leaves active Chora scenes and model weights unchanged:
PYTHONPATH=. .venv/bin/python scripts/run_arm_evaluation.py --render
.venv/bin/python -m pytest -q tests/test_arm_evaluation.py
```

All raw results, source XML hashes, joint solutions, contact diagnostics, loads,
assumptions and rendered models are in `outputs/robot_evaluation/`.

## Primary references

- [Menagerie pinned revision](https://github.com/google-deepmind/mujoco_menagerie/tree/370c43965323c0944b433b202aac5058b5f907a9)
- [Franka FR3 datasheet](https://franka.de/hubfs/Datasheet_FrankaResearch3_R02212_2.4.pdf?hsLang=en)
- [Franka robot/controller limits](https://frankarobotics.github.io/docs/robot_specifications.html)
- [Franka model availability](https://franka.de/models-and-simulation)
- [UR5e 2024 datasheet](https://www.universal-robots.com/media/1807465/ur5e_e-series_datasheets_web.pdf)
- [UR10e 2024 datasheet](https://www.universal-robots.com/media/1807466/ur10e_e-series_datasheets_web.pdf)
- [Current UR model-name changes](https://www.universal-robots.com/insights/faq/)
- [KUKA iiwa product and data](https://www.kuka.com/en-se/products/robotics-systems/industrial-robots/lbr-iiwa)
- [Franka impact experiment](https://arxiv.org/html/2402.05502v2)
- [ABB YuMi](https://one.robotics.abb.com/en/p/IRB-14000-YuMi-Dual-Arm)
- [ALOHA 2](https://aloha-2.github.io/assets/aloha2.pdf)
- [ViperX working-payload guidance](https://docs.trossenrobotics.com/interbotix_xsarms_docs/specifications/vx300s.html)
