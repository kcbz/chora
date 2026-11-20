from __future__ import annotations

import numpy as np

from .sim import BOWL_RADIUS, PAN_ORIGIN


def move(p):
    return {"action":"move_to", **{k:round(float(v),3) for k,v in zip("xyz",p)}}


class Expert:
    """A demonstration planner using only observations, history, and a seed.

    Grasp the near rim, lift, transport, place, release, and retreat. The
    observation-to-action interface is shared by the expert and learned model.
    """
    def __init__(self, seed):
        self.rng = np.random.default_rng(seed+23457)
        self.phase = 0
        self.grasp_offset = None
        self.place = None
        self.height = self.rng.uniform(.125,.145)

    def act(self, obs, history):
        b = np.array(obs["bowl"])
        if self.grasp_offset is None:
            radial = b[:2]-PAN_ORIGIN
            radial /= np.linalg.norm(radial)
            self.grasp_offset = -radial*BOWL_RADIUS
            self.place = np.array(obs["target"],float)
            self.place[:2] += self.rng.uniform([-.006,-.004],[.006,.004])
            self.height = max(self.height, self.place[2] + .052)
        if self.phase in (4, 5) and "target_at_release" in obs:
            self.place = np.array(obs["target_at_release"], float)
        p = b.copy()
        p[:2] += self.grasp_offset
        p[2] = .022
        # Approximate the carried bowl-to-TCP offset, learned only from current
        # observations after lifting, rather than an internal simulator handle.
        if self.phase == 4:
            self.carried_offset = np.array(obs["tcp"])-b
        target = self.place.copy()
        target[:2] += getattr(self,"carried_offset", np.r_[self.grasp_offset,0])[:2]
        if self.phase in (5, 6, 7):
            # Wrist yaw follows the arm pan, so a carried offset rotates during
            # transport. Correct the placement from the observed bowl position.
            if self.phase == 5:
                self.release_xy = np.array(obs["tcp"][:2]) + self.place[:2] - b[:2]
            target[:2] = self.release_xy
        actions = [move([*p[:2],self.height]), move(p), {"action":"close_gripper"},
                   move([*p[:2],self.height]), move([*target[:2],self.height]),
                   move([*target[:2],self.place[2]+.021]), {"action":"open_gripper"},
                   move([*target[:2],self.height]), {"action":"done"}]
        action = actions[min(self.phase,len(actions)-1)]
        self.phase += 1
        return action


class StagingExpert:
    """Top-down transfer of a solid part or ingredient, using observations only."""
    def __init__(self,seed):
        self.phase=0
        self.height=.135
        self.grasp=None
        self.release_xy=None
        self.half=None

    def act(self,obs,history):
        obj=np.array(obs["object"]);tcp=np.array(obs["tcp"]);target=np.array(obs["target"])
        if self.grasp is None:
            radial=obj[:2]-PAN_ORIGIN;radial/=np.linalg.norm(radial)
            self.grasp=obj[:2]-.003*radial
            self.half=.008 if obs["object_kind"]=="part" else .009
        if self.phase==3:
            self.grasp=tcp[:2].copy()
        if self.phase==4:
            self.carry_offset=tcp-obj
        transport=target[:2]+getattr(self,"carry_offset",np.zeros(3))[:2]
        if self.phase==5:
            self.release_xy=tcp[:2]+target[:2]-obj[:2]
            self.release_z=target[2]+self.half+(tcp[2]-obj[2])-.001
        release=self.release_xy if self.release_xy is not None else transport
        actions=[move([*self.grasp,self.height]),move([*self.grasp,self.half+.004]),{"action":"close_gripper"},
                 move([*self.grasp,self.height]),move([*transport,self.height]),
                 move([*release,getattr(self,"release_z",target[2]+self.half+.004)]),
                 {"action":"open_gripper"},move([*release,self.height]),{"action":"done"}]
        action=actions[min(self.phase,8)];self.phase+=1
        return action


def make_expert(seed,cfg=None):
    if cfg is None:
        from .common import config
        cfg=config()
    if cfg.get('experiment',{}).get('task')=='tile_adhesive_bedding':
        from .tile_installation import TileBeddingExpert
        return TileBeddingExpert(seed)
    if cfg.get('experiment',{}).get('task')=='tile_dry_transfer':
        from .tile_installation import TileTransferExpert
        return TileTransferExpert(seed)
    if cfg.get('experiment',{}).get('task')=='dual_tile_wipe':
        from .dual_tile import DualTileExpert
        return DualTileExpert(seed)
    if cfg.get("experiment",{}).get("task")=="impact_nail":
        from .gantry import ImpactExpert
        return ImpactExpert(seed)
    if cfg.get("experiment",{}).get("task")=="soft_cut":
        from .soft_food import SoftFoodExpert
        return SoftFoodExpert(seed)
    if cfg.get("experiment",{}).get("task")=="cut_fruit":
        from .cutting import CuttingExpert
        return CuttingExpert(seed)
    if cfg.get("experiment",{}).get("task") in {"hammer_nail","hammer_pine"}:
        from .hammer import HammerExpert
        return HammerExpert(seed)
    return Expert(seed) if cfg["simulation"].get("object_kind","bowl")=="bowl" else StagingExpert(seed)


SYSTEM = ('Control an SO-101 robot to place the bowl fully inside the target, upright. '
          'Coordinates and sizes are integer millimetres; yaw is radians. '
          'Output exactly one JSON object, no explanation. Allowed: '
          '{"action":"move_to","x":int,"y":int,"z":int}, '
          '{"action":"open_gripper"}, {"action":"close_gripper"}, {"action":"done"}. '
          'Grasp the near rim, lift above the table, move to target, lower, release, retreat, then done. '
          'move_to z must be 5..190 mm. A scripted IK controller executes Cartesian moves.')


def to_mm(value):
    result = dict(value)
    for key in ("bowl","object","tcp","actual_tool_tip","blade_edge","target","target_half_size","target_at_release","target_velocity"):
        if key in result:
            result[key] = [round(x*1000) for x in result[key]]
    for key in ("x","y","z","depth","nail_depth","target_depth","food_height"):
        if key in result:
            result[key] = round(result[key]*1000)
    return result


def messages(obs, history, action=None):
    if obs.get('task')=='tile_installation':
        from .tiling_task import messages as tiling_messages
        return tiling_messages(obs,history,action)
    import json
    user = {"observation":to_mm(obs),"history":[to_mm(x) for x in history[-2:]]}
    system=SYSTEM
    if obs.get("task") in {"hammer_nail","hammer_pine"}:
        system=('Control an SO-101 mounted hammer to drive a guided nail at least to target_depth, then retract and finish. '
                'Coordinates, depth and sizes are integer millimetres; duration is seconds. '
                'Output exactly one JSON object. Allowed: {"action":"move_to","x":int,"y":int,"z":int}, '
                '{"action":"strike","depth":int,"duration":number}, {"action":"retract"}, {"action":"done"}. '
                'Align above the observed nail at z=125 mm before each strike. Strike depth 2..12 mm, duration 0.06..0.18 s. '
                'Retract after each strike, inspect depth, and realign before striking again. Finish when depth reaches target and tool is clear.')
    elif obs.get("task")=="cut_fruit":
        system=('Control an SO-101 gripped knife to cut the observed fruit on its board, then retract. '
                'Coordinates, depths and food_height are integer millimetres; duration is seconds. Output one JSON object. '
                'Allowed: {"action":"move_to","x":int,"y":int,"z":int}, {"action":"slice","depth":int,"duration":number}, '
                '{"action":"retract"}, {"action":"done"}. Align above the fruit at target z + food_height + 3 mm. '
                'Slice down through it at duration 1.4 seconds, then retract and finish. Slice depth 5..55 mm, duration 0.5..3 seconds. '
                'Inspect cut_complete and tool_held; a cut requires sufficient penetration and fracture work.')
    elif "target_at_release" in obs:
        system += (' The platform slides sideways. Target is its current center, target_velocity is in mm/s, '
                   'and target_at_release projects its center after a descent and opening. '
                   'Use the projected center to plan the release, preserving the offset from bowl center to gripper.')
    elif obs.get("object_kind"):
        label="construction part" if obs["object_kind"]=="part" else "ingredient portion"
        system=system.replace("bowl",label).replace("Grasp the near rim","Grasp the object from above")
    msgs = [{"role":"system","content":system},{"role":"user","content":json.dumps(user,separators=(",",":"))}]
    if action is not None:
        msgs.append({"role":"assistant","content":json.dumps(to_mm(action),separators=(",",":"))})
    return msgs
