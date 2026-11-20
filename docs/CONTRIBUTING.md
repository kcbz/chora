# Development workflow

Finish a task, run its relevant checks, review the diff, then commit and push to
the configured upstream. Keep independent tasks in separate commits with concise,
change-focused messages. Preserve unrelated user changes rather than sweeping
everything into a commit. Report which checks passed and what remains incomplete.

Use the regular environment for the lab and Qwen work, and `.venv-vla` for model
and VLA checks. The original environment intentionally skips optional torch tests.

```bash
.venv/bin/python -m pytest -q
.venv-vla/bin/python -m pytest -q tests/test_vla_*.py
git diff --check
git status --short
```

Choose checks appropriate to the change. Graphics/Metal/browser checks need a
normal local session with device access. Inspect changed UI in the browser,
including a narrow viewport and the supported appearance modes.

Generated datasets, weights, caches, environments and runtime logs stay out of
Git. Source, configuration, tests and general documentation belong in the repo;
changes, experiment results and limitations belong in `docs/`. Preserve original
benchmark artifacts when adding experimental policies.
