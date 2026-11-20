# Skill Orchestrator pilot

Start on the two-arm tiling workcell in simulation. Use the local base Qwen
model as a planner. Give it concise scene state, the job goal, and the outcome
of the previous skill. It selects a named skill with validated arguments,
such as prepare adhesive for the next cell, acquire a tile, or bed a tile.
The planner's prompt and action schema are separate from the existing
state-based imitation adapter.

Execute the skills with the existing scripted motion controllers first.
The workcell decides arm ownership and clearance, and checks the result
using tile pose, grip state, and simulated bedding measurements. Feed those
checks back to the planner so it can retry a pickup, proceed, or stop with
a reason. Start with a two-tile job and measure completion, interventions,
and recovery on held-out initial conditions against the scripted sequence.

Once a VLA passes the same skill benchmark, it can implement a skill behind
the same interface. A VLM can later supply visual observations for checks
that Chora currently makes from simulator state. The selector exposes the
planned mode while its planner and skill interface are in development.
