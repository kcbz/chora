"""Fixed-rate SO-101 adapter; student observations contain no object state."""
from __future__ import annotations

import copy
import math

import mujoco
import numpy as np

from ..common import config
from ..sim import Simulation

INSTRUCTION = "Place the bowl upright inside the marked target and release it."


class BowlTickEnv:
    def __init__(self, cfg=None, *, hz=25, images=True, image_size=224, max_ticks=500):
        self.cfg = copy.deepcopy(cfg or config())
        exp = self.cfg.get("experiment", {})
        if exp.get("robot", "so101") != "so101" or exp.get("task", "bowl_pick_place") != "bowl_pick_place":
            raise ValueError("VLA tick recording currently supports SO-101 bowl placement only")
        dt = float(self.cfg["simulation"]["timestep"])
        if isinstance(hz, bool) or not math.isfinite(hz) or hz <= 0:
            raise ValueError("Control frequency must be positive and finite")
        self.substeps = round(1 / hz / dt)
        if self.substeps < 1 or not math.isclose(self.substeps * dt, 1 / hz, abs_tol=1e-10):
            raise ValueError("Control period must be an integer number of physics steps")
        if isinstance(max_ticks, bool) or not isinstance(max_ticks, int) or not 1 <= max_ticks <= 2000:
            raise ValueError("max_ticks must be 1..2000")
        if isinstance(image_size, bool) or not isinstance(image_size, int) or not 32 <= image_size <= 512:
            raise ValueError("image_size must be 32..512")
        self.hz, self.period, self.max_ticks = hz, 1 / hz, max_ticks
        self.image_size = image_size
        self.sim = Simulation(self.cfg, policy_cameras=True)
        m = self.sim.model
        self.actuator_ids = np.arange(m.nu)
        if not np.all(m.actuator_trntype == mujoco.mjtTrn.mjTRN_JOINT):
            raise ValueError("Expected direct joint actuators")
        self.joint_ids = m.actuator_trnid[:, 0].copy()
        if not np.all(np.isin(m.jnt_type[self.joint_ids], [mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE])):
            raise ValueError("Expected scalar joints")
        self.qadr = m.jnt_qposadr[self.joint_ids].copy()
        self.vadr = m.jnt_dofadr[self.joint_ids].copy()
        self.joint_names = [m.joint(int(j)).name for j in self.joint_ids]
        if self.joint_names != ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]:
            raise ValueError("Unexpected SO-101 actuator ordering; update the robot adapter explicitly")
        self.actuator_names = [m.actuator(int(a)).name for a in self.actuator_ids]
        self.bounds = m.actuator_ctrlrange.copy()
        if not np.all(m.actuator_ctrllimited):
            raise ValueError("Expected bounded position actuators")
        self.camera_names = ("front", "wrist")
        self.renderer = mujoco.Renderer(m, height=image_size, width=image_size) if images else None
        self.reset(0)

    def reset(self, seed):
        self.sim.reset(seed)
        mujoco.mj_forward(self.sim.model, self.sim.data)
        self.tick = 0
        self.start_time = float(self.sim.data.time)
        self.stable_seconds = 0.0
        self.done = self.success = False
        self.reason = "running"
        return self.observe()

    def observe(self):
        obs = {"timestamp": float(self.sim.data.time), "tick": self.tick,
               "joint_positions": self.sim.data.qpos[self.qadr].copy(),
               "joint_velocities": self.sim.data.qvel[self.vadr].copy(),
               "instruction": INSTRUCTION, "images": {}}
        if self.renderer is not None:
            for name in self.camera_names:
                self.renderer.update_scene(self.sim.data, camera=name)
                obs["images"][name] = self.renderer.render().copy()
        return obs

    def step_tick(self, action):
        if self.done:
            raise RuntimeError("Episode ended; reset before stepping")
        raw = np.asarray(action)
        if raw.dtype.kind not in "fiu" or raw.shape != (len(self.actuator_ids),):
            raise ValueError("Action must be a numeric vector matching the actuator order")
        target = raw.astype(np.float64)
        if not np.isfinite(target).all():
            raise ValueError("Action must be finite")
        if np.any(target < self.bounds[:, 0]) or np.any(target > self.bounds[:, 1]):
            raise ValueError("Joint target outside actuator limits")
        s = self.sim
        s.data.ctrl[self.actuator_ids] = target
        s.gripper_open = bool(target[-1] > .1)
        collision = False
        base = s.model.body("base").id
        for _ in range(self.substeps):
            mujoco.mj_step(s.model, s.data)
            s.physics_steps += 1
            hit = any(0 in pair and any(b in s.arm_bodies and b != base for b in pair) for pair in s.contacts())
            collision |= hit
            self.stable_seconds = self.stable_seconds + s.model.opt.timestep if s.is_placed() else 0.0
        self.tick += 1
        # Synchronize derived poses with the final integrated qpos before capture.
        # Do this even without rendering so recording cannot change the physics.
        mujoco.mj_forward(s.model, s.data)
        self.success = self.stable_seconds + 1e-10 >= s.cfg["settle_seconds"]
        if collision:
            self.done, self.success, self.reason = True, False, "collision"
            s.collision_count += 1
        elif self.success:
            self.done, self.reason = True, "success"
        elif self.tick >= self.max_ticks:
            self.done, self.reason = True, "tick_limit"
        return self.observe(), self.done, self.info()

    def info(self):
        # Scoring-only state, excluded from observe().
        return {"success": self.success, "reason": self.reason, "ticks": self.tick,
                "collisions": self.sim.collision_count, "bowl_position": self.sim.bowl.tolist()}

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None


class TickExpert:
    """Translate the existing planner into references executed at real control ticks.

    IK is solved before each Cartesian movement. These are targets actually applied
    to the physical simulation, not reconstructed macro-action endpoint labels.
    """
    def __init__(self, env, seed):
        from ..policy import Expert
        self.env, self.planner = env, Expert(seed)
        self.history = []
        self.pending = []
        self.exhausted = False
        self.phase = "reset"

    def act(self):
        env, s = self.env, self.env.sim
        if not self.pending:
            macro = self.planner.act(s.observe(), self.history)
            self.history.append(macro)
            kind = self.phase = macro["action"]
            hold = s.data.ctrl.copy()
            if kind == "move_to":
                target = np.array([macro[k] for k in "xyz"])
                start = s.tcp
                n = math.ceil(s.cfg["move_seconds"] / env.period)
                q = s.data.qpos[env.qadr[:5]].copy()
                for f in np.linspace(0, 1, n):
                    q = s.solve_ik(start + (target - start) * f, q)
                    if q is None:
                        raise RuntimeError("Tick expert generated an unreachable path")
                    reference = hold.copy()
                    reference[:5] = q
                    self.pending.append(reference)
                self.pending.extend([self.pending[-1].copy() for _ in range(math.ceil(.12 / env.period))])
            elif kind in {"open_gripper", "close_gripper"}:
                from ..sim import OPEN, CLOSED
                hold[-1] = OPEN if kind == "open_gripper" else CLOSED
                self.pending = [hold.copy() for _ in range(math.ceil(.35 / env.period))]
            else:
                self.exhausted = True
                self.pending = [hold.copy() for _ in range(math.ceil(s.cfg["settle_seconds"] / env.period))]
        return self.pending.pop(0)
