"""Official SO-101 dynamics, numerical IK, and physical rim grasping.

No attachment, teleportation, or hidden grasp constraint is used. The bowl is a
free body throughout the episode. Actions use metres; model prompts use mm.
"""
from __future__ import annotations

import math

import mujoco
import numpy as np
from scipy.optimize import least_squares

from .common import ROOT, config

BOWL_RADIUS = 0.027
BOWL_HEIGHT = 0.025
PAN_ORIGIN = np.array([0.0388353, 0.0])
OPEN = 0.40
CLOSED = -0.17


def validate_action(action):
    if not isinstance(action, dict) or "action" not in action:
        raise ValueError("Expected an action object")
    name = action["action"]
    fields = {"action", "x", "y", "z"} if name == "move_to" else {"action"}
    if name not in {"move_to", "open_gripper", "close_gripper", "done"}:
        raise ValueError("Unknown action")
    if set(action) != fields:
        raise ValueError("Action has missing or extra fields")
    if name == "move_to":
        for key in "xyz":
            value = action[key]
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
                raise ValueError("Coordinates must be finite numbers")
    return action


def build_model(cfg, *, policy_cameras=False):
    source = ROOT / "assets/menagerie/robotstudio_so101/so101.xml"
    if not source.exists():
        raise FileNotFoundError("Official SO-101 assets missing. Run ./setup.sh")
    s = mujoco.MjSpec.from_file(str(source))
    s.modelname = "Chora SO-101 physical pick and place"
    s.option.timestep = cfg["timestep"]
    s.option.iterations = 50
    s.body("base").pos = [0, 0, 0.12]
    s.body("gripper").add_site(name="tcp", pos=[0, 0, -0.098], size=[0.003]*3, rgba=[0.2, 0.9, 0.7, 0.0])
    # Retain the official yellow robot materials; colours distinguish the arm,
    # terracotta bowl, and green goal instead of blending them into the UI.
    s.visual.headlight.ambient = [.40,.40,.40]
    s.visual.headlight.diffuse = [.65,.65,.65]
    s.add_texture(name="studio_sky",type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
                  builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
                  rgb1=[.24,.31,.35],rgb2=[.07,.10,.12],width=512,height=3072)
    w = s.worldbody
    w.add_light(pos=[0, -0.5, 1.5], dir=[0, 0, -1], diffuse=[0.85]*3)
    w.add_light(pos=[0.7, 0.5, 1], dir=[-0.4, 0, -1], diffuse=[0.5]*3)
    w.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[1, 1, .01], rgba=[.23, .28, .30, 1])
    w.add_geom(name="pedestal", type=mujoco.mjtGeom.mjGEOM_BOX, pos=[0, 0, .06], size=[.046, .045, .06], rgba=[.20, .25, .25, 1])
    h = cfg["table_height"]
    table_x,table_y=cfg.get("table_center",[.175,.09])
    w.add_geom(name="table", type=mujoco.mjtGeom.mjGEOM_BOX, pos=[table_x,table_y,h/2], size=[.083, .074, h/2], rgba=[.33, .39, .37, 1])
    tx, ty, _ = cfg["target"]
    hx, hy = cfg["target_half_size"]
    for axis in range(2):
        for sign in [-1, 1]:
            p = [tx, ty, h+.0003]
            p[axis] += sign*[hx, hy][axis]
            size = [.001, hy, .0003] if axis == 0 else [hx, .001, .0003]
            w.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, pos=p, size=size, rgba=[.64, .88, .50, 1], contype=0, conaffinity=0)
    bowl = w.add_body(name="bowl", pos=[.18, -.08, .004])
    bowl.add_freejoint(name="bowl_free")
    if cfg.get("object_kind","bowl")=="bowl":
        bowl.add_geom(name="bowl_bottom", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[BOWL_RADIUS, .002, 0], mass=.014, rgba=[.86, .54, .34, 1], friction=[1.2,.01,.001])
        for i in range(24):
            a = 2*math.pi*i/24
            bowl.add_geom(name=f"bowl_rim_{i}", type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                          pos=[BOWL_RADIUS*math.cos(a), BOWL_RADIUS*math.sin(a), .012],
                          size=[.002, .010, 0], mass=.0005, rgba=[.9,.61,.4,0], friction=[1.5,.01,.001], condim=6)
        # A continuous hollow visual shell over the unchanged contact primitives.
        # It carries no mass and has no collision participation.
        vertices,faces = [],[]
        segments = 64
        for radius,z in ((.027,.002),(.029,.024),(.025,.024),(.023,.004)):
            for i in range(segments):
                angle = 2*math.pi*i/segments
                vertices.extend([radius*math.cos(angle),radius*math.sin(angle),z])
        for ring in range(4):
            for i in range(segments):
                a,b = ring*segments+i,ring*segments+(i+1)%segments
                c,d = ((ring+1)%4)*segments+i,((ring+1)%4)*segments+(i+1)%segments
                faces.extend([a,b,c,b,d,c])
        s.add_mesh(name="bowl_visual_shell",uservert=vertices,userface=faces,smoothnormal=True)
        bowl.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH,meshname="bowl_visual_shell",mass=0,
                      contype=0,conaffinity=0,rgba=[.83,.48,.28,1])
        # An orientation mark with no collision geometry.
        bowl.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, pos=[BOWL_RADIUS,0,.023], size=[.003]*3,
                      mass=0, rgba=[.98,.85,.65,1], contype=0, conaffinity=0)
    else:
        kind=cfg["object_kind"]
        half=.008 if kind=="part" else .009
        colour=[.63,.35,.22,1] if kind=="part" else [.73,.82,.43,1]
        bowl.add_geom(name="transfer_object",type=mujoco.mjtGeom.mjGEOM_BOX,size=[half,half,half],mass=.018,rgba=colour,friction=[1.8,.01,.001],condim=6)
    front = np.array([.49,-.49,.36])
    z = front-np.array([.11,0,.18]); z /= np.linalg.norm(z)
    x = np.cross([0,0,1],z); x /= np.linalg.norm(x)
    y = np.cross(z,x)
    w.add_camera(name="front", pos=front, xyaxes=np.r_[x,y], fovy=42)
    w.add_camera(name="top", pos=[.13,0,.66], xyaxes=[1,0,0,0,1,0], fovy=48)
    if policy_cameras:
        # Optical axis follows the gripper toward its TCP; camera has no mass/contact.
        s.body("gripper").add_camera(name="wrist", pos=[0, -.045, -.015],
                                      xyaxes=[1, 0, 0, 0, 1, 0], fovy=70)
    model = s.compile()
    model.vis.global_.offwidth = 1280
    model.vis.global_.offheight = 960
    return model


class Simulation:
    def __init__(self, cfg=None, *, policy_cameras=False):
        full_cfg=cfg or config()
        if full_cfg.get("experiment",{}).get("robot","so101")!="so101":
            raise ValueError("This robot requires its own simulation/controller adapter")
        if full_cfg.get("experiment",{}).get("task")=="hammer_nail":
            raise ValueError("Use make_simulation for the hammer task adapter")
        self.cfg = full_cfg["simulation"]
        self.object_kind=self.cfg.get("object_kind","bowl")
        self.object_rest_height=.002 if self.object_kind=="bowl" else .008 if self.object_kind=="part" else .009
        self.object_radius=BOWL_RADIUS if self.object_kind=="bowl" else math.sqrt(2)*self.object_rest_height
        self.model = build_model(self.cfg, policy_cameras=policy_cameras)
        self.data = mujoco.MjData(self.model)
        self.ik_data = mujoco.MjData(self.model)
        self.tcp_id = self.model.site("tcp").id
        self.gripper_id = self.model.body("gripper").id
        self.bowl_id = self.model.body("bowl").id
        self.bowl_address = self.model.joint("bowl_free").qposadr[0]
        self.bowl_dof = self.model.joint("bowl_free").dofadr[0]
        self.arm_bodies = set(range(1, self.bowl_id))
        self.on_frame = None
        self.frame_stride = 10
        self.reset(0)

    @property
    def tcp(self):
        return self.data.site_xpos[self.tcp_id].copy()

    @property
    def bowl(self):
        return self.data.xpos[self.bowl_id].copy()

    def solve_ik(self, target, initial=None):
        target = np.array(target, float)
        if not np.all(np.isfinite(target)) or not (.005 <= target[2] <= .19):
            return None
        radial = np.linalg.norm(target[:2]-PAN_ORIGIN)
        if not .075 < radial < .28:
            return None
        start = self.data.qpos[:4].copy() if initial is None else np.asarray(initial)[:4].copy()
        limits = self.model.jnt_range[:4]
        start = np.clip(start, limits[:,0]+.0001, limits[:,1]-.0001)
        def residual(q):
            self.ik_data.qpos[:4] = q
            self.ik_data.qpos[4] = .0487
            mujoco.mj_forward(self.model, self.ik_data)
            z = self.ik_data.xmat[self.gripper_id].reshape(3,3)[:,2]
            return np.r_[self.ik_data.site_xpos[self.tcp_id]-target, .10*(z-[0,0,1])]
        result = least_squares(residual, start, bounds=(limits[:,0]+.0001,limits[:,1]-.0001), max_nfev=45, ftol=1e-7, xtol=1e-7, gtol=1e-7)
        if np.linalg.norm(residual(result.x)) > .0015:
            return None
        return np.r_[result.x, .0487]

    def reset(self, seed):
        self.seed = int(seed)
        self.rng = np.random.default_rng(seed)
        mujoco.mj_resetData(self.model, self.data)
        a = self.bowl_address
        self.data.qpos[a:a+3] = [self.rng.uniform(*self.cfg["spawn_x"]), self.rng.uniform(*self.cfg["spawn_y"]), self.object_rest_height+.002]
        yaw = self.rng.uniform(-math.pi,math.pi)
        self.data.qpos[a+3:a+7] = [math.cos(yaw/2),0,0,math.sin(yaw/2)]
        home = self.solve_ik([.18,-.015,.14], [0,-.4,.7,1.1])
        if home is None:
            raise RuntimeError("Home pose is unreachable")
        self.data.qpos[:5] = home
        self.data.qpos[5] = OPEN
        self.data.ctrl[:] = self.data.qpos[:6]
        self.gripper_open = True
        self.step_index = 0
        self.result = "reset"
        self.finished = False
        self.success = False
        self.physics_steps = 0
        self.collision_count = 0
        self.action = {"action":"reset"}
        self._advance(.3)
        self.initial_bowl = self.bowl.tolist()
        return self.observe()

    def _advance(self, seconds):
        for _ in range(round(seconds/self.model.opt.timestep)):
            mujoco.mj_step(self.model, self.data)
            self.physics_steps += 1
            if self.on_frame is not None and self.physics_steps % self.frame_stride == 0:
                self.on_frame(self)

    def contacts(self):
        return [(int(self.model.geom_bodyid[c.geom1]), int(self.model.geom_bodyid[c.geom2])) for c in self.data.contact]

    def observe(self):
        rot = self.data.xmat[self.bowl_id].reshape(3,3)
        grasped = not self.gripper_open and any(self.bowl_id in pair and any(b in self.arm_bodies for b in pair) for pair in self.contacts())
        observation={"bowl": [round(float(x),4) for x in self.bowl],
                "bowl_yaw": round(math.atan2(rot[1,0],rot[0,0]),3),
                "target": self.cfg["target"], "target_half_size": self.cfg["target_half_size"],
                "tcp": [round(float(x),4) for x in self.tcp], "gripper_open": self.gripper_open,
                "grasped": grasped, "step": self.step_index, "last_result": self.result}
        if self.object_kind!="bowl":
            observation["object"]=observation.pop("bowl")
            observation["object_yaw"]=observation.pop("bowl_yaw")
            observation["object_kind"]=self.object_kind
        return observation

    def is_placed(self):
        p = self.bowl
        t = self.cfg["target"]
        half = np.array(self.cfg["target_half_size"])
        # The entire bowl must fit inside the marked region.
        inside = np.all(np.abs(p[:2]-t[:2])+self.object_radius < half)
        upright = self.data.xmat[self.bowl_id].reshape(3,3)[2,2] > math.cos(math.radians(15))
        resting = abs(p[2]-(t[2]+self.object_rest_height)) < .006
        slow = np.linalg.norm(self.data.qvel[self.bowl_dof:self.bowl_dof+6]) < .025
        touching = any(self.bowl_id in pair and any(b in self.arm_bodies for b in pair) for pair in self.contacts())
        return bool(inside and upright and resting and slow and self.gripper_open and not touching)

    def _stable_success(self):
        for _ in range(round(self.cfg["settle_seconds"]/self.model.opt.timestep)):
            self._advance(self.model.opt.timestep)
            if not self.is_placed():
                return False
        return True

    def step(self, action):
        if self.finished:
            raise RuntimeError("Episode has ended; reset before taking another action")
        self.step_index += 1
        self.action = action
        try:
            validate_action(action)
        except ValueError:
            self.result = "invalid_action"
        else:
            kind = action["action"]
            if kind == "move_to":
                target = np.array([action[k] for k in "xyz"],float)
                start = self.tcp
                # Validate the entire straight path before moving anything.
                n = max(12, int(np.linalg.norm(target-start)/.004))
                path = []
                q = self.data.qpos[:5].copy()
                for fraction in np.linspace(0,1,n):
                    q = self.solve_ik(start+(target-start)*fraction,q)
                    if q is None:
                        break
                    path.append(q)
                if q is None:
                    self.result = "rejected_unreachable"
                else:
                    self.result = "ok"
                    for q in path:
                        self.data.ctrl[:5] = q
                        self._advance(self.cfg["move_seconds"]/n)
                        # Contact with the bowl is necessary for physical grasping.
                        # Arm contact with the table/floor aborts the move.
                        if any(0 in pair and any(b in self.arm_bodies and b != self.model.body("base").id for b in pair) for pair in self.contacts()):
                            self.data.ctrl[:5] = self.data.qpos[:5]
                            self.result = "collision"
                            self.collision_count += 1
                            break
                    self._advance(.12)
            elif kind in {"open_gripper","close_gripper"}:
                self.gripper_open = kind == "open_gripper"
                self.data.ctrl[5] = OPEN if self.gripper_open else CLOSED
                self._advance(.35)
                self.result = "ok" if self.gripper_open or self.observe()["grasped"] else "failed_grasp"
            else:
                self.success = self._stable_success()
                self.result = "success" if self.success else "not_placed"
                self.finished = True
        if self.step_index >= self.cfg["max_steps"] and not self.finished:
            self.finished = True
            self.result = "step_limit"
        return self.observe()


def make_simulation(cfg=None):
    cfg=cfg or config()
    task=cfg.get("experiment",{}).get("task")
    robot=cfg.get('experiment',{}).get('robot')
    if task in {'tile_dry_transfer','tile_adhesive_bedding'}:
        from .tile_installation import TileInstallationSimulation
        return TileInstallationSimulation(cfg)
    if robot=='fr3_duo':
        from .dual_tile import DualTileSimulation
        return DualTileSimulation(cfg)
    if robot=='fr3':
        from .arm_food import ArmFoodSimulation
        return ArmFoodSimulation(cfg)
    if robot=='ur10e':
        from .arm_hammer import ArmHammerSimulation
        return ArmHammerSimulation(cfg)
    if task=="impact_nail":
        from .gantry import GantryImpact
        return GantryImpact(cfg)
    if task=="soft_cut":
        from .soft_food import SoftFoodSimulation
        return SoftFoodSimulation(cfg)
    if task=="hammer_pine":
        from .tool_physics import PhysicalHammerSimulation
        return PhysicalHammerSimulation(cfg)
    if task=="cut_fruit":
        from .cutting import CuttingSimulation
        return CuttingSimulation(cfg)
    if task=="hammer_nail":
        from .hammer import HammerSimulation
        return HammerSimulation(cfg)
    return Simulation(cfg)
