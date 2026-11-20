# Documentation

The [project README](../README.md) covers setup, usage, testing and authors.

- [VLA integration plan](VLA_INTEGRATION_PLAN.md): camera-based policies, M3 feasibility, latency evaluation and two-arm tiling milestones.
- [VLA development](VLA.md): fixed-tick camera demonstrations, recording format and verified replay.
- [VLA M3 feasibility](VLA_M3_FEASIBILITY.md): neural model smoke tests, inference/camera timings and memory measurements.
- [VLA pilot training](VLA_TRAINING.md): episode splits, action chunks, frozen feature caching, training and resume.
- [VLA failed-pilot analysis](VLA_FAILURE_ANALYSIS.md): reported oscillation, replacement experiment and actual failed task results.
- [VLA policy diagnostics](VLA_POLICY_DIAGNOSTICS.md): joint/sequence conditioning fixes and actual bounded rollout results.
- [Development workflow](CONTRIBUTING.md): task-sized commits, relevant checks and pushing completed work.
- [Experiments](EXPERIMENTS.md): learning commands, protocols, outputs and developer reference.
- [Development history](DEVELOPMENT_HISTORY.md): changes, milestones and historical results.
- [Assets](ASSETS.md): downloads, checksums, omitted assets and wheel/workspace setup.
- [Security](SECURITY.md): local-server boundaries and publication safeguards.
- [Third-party notices](THIRD_PARTY.md): sources, attribution and licensing restrictions.
- [Material models](MATERIAL_MODELS.md): mechanics approximations and known limitations.
- [Robot evaluation](ROBOT_ARM_EVALUATION.md): selection study and its limits.
- [Real objects and tiling](REAL_OBJECTS_AND_TILING.md): object sources and task direction.
- [Tiling baseline](TILING_BASELINE.md): published product data versus model assumptions.
- [Tiling task interface](TILING_TASK_INTERFACE.md): shared scene, coordination and remaining stages.

Licensing for a future release is TBD. The current proprietary notice remains in
[LICENSE](../LICENSE); third-party terms are separate.
