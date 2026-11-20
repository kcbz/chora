from __future__ import annotations

import io
import time
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont


class Camera:
    def __init__(self, sim, width=1280, height=960):
        self.renderer = mujoco.Renderer(sim.model, height=height, width=width)
        self.platform_id = None
        if getattr(sim, "object_kind", None) in {"bowl", "part", "ingredient"}:
            self.platform_id = sim.model.geom("table").id
        self.orbit = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(self.orbit)
        self.set_orbit({"azimuth":135,"elevation":-25,"distance":.65,"lookat":[.12,0,.16]})

    def set_orbit(self,params):
        self.orbit.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.orbit.azimuth = params["azimuth"]
        self.orbit.elevation = params["elevation"]
        self.orbit.distance = params["distance"]
        self.orbit.lookat[:] = params["lookat"]

    def render(self, sim, camera="front", overlay=True):
        dirty=getattr(sim,'dirty_meshes',None)
        if dirty:
            if self.renderer._gl_context:self.renderer._gl_context.make_current()
            for mesh_id in dirty:mujoco.mjr_uploadMesh(sim.model,self.renderer._mjr_context,mesh_id)
            dirty.clear()
        self.renderer.update_scene(sim.data, camera=self.orbit if camera=="orbit" else camera)
        if self.platform_id is not None:
            # Style the presentation view without changing recorded VLA inputs.
            scene = self.renderer.scene
            for geom in scene.geoms[:scene.ngeom]:
                if geom.objtype == mujoco.mjtObj.mjOBJ_GEOM and geom.objid == self.platform_id:
                    geom.rgba[:] = (.23, .42, .76, 1)
                    break
        frame = self.renderer.render().copy()
        if not overlay:
            return frame
        image = Image.fromarray(frame)
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default(size=max(12,image.width//80))
        draw.rectangle((0,0,image.width,55),fill=(17,25,27))
        draw.text((16,9),f"CHORA / COHERENCE    seed {sim.seed}    decision {sim.step_index}",font=font,fill=(217,230,217))
        draw.text((16,32),f"{sim.action.get('action','reset')}  /  {sim.result}",font=font,fill=(173,203,165))
        return np.asarray(image)

    def jpeg(self, sim, camera="front", overlay=True):
        f = io.BytesIO()
        Image.fromarray(self.render(sim,camera,overlay=overlay)).save(f,format="JPEG",quality=94,subsampling=0)
        return f.getvalue()

    def close(self):
        self.renderer.close()


class Recorder:
    def __init__(self, sim, path, fps=20):
        self.sim = sim
        self.camera = Camera(sim)
        Path(path).parent.mkdir(parents=True,exist_ok=True)
        self.writer = imageio.get_writer(str(path),fps=fps,codec="libx264",pixelformat="yuv420p",macro_block_size=16)
        self.previous_callback = sim.on_frame
        sim.frame_stride = max(1,round(1/(fps*sim.model.opt.timestep)))
        sim.on_frame = self.frame
        self.frame(sim)

    def frame(self, sim):
        self.writer.append_data(self.camera.render(sim))

    def close(self):
        try:
            self.frame(self.sim)
        finally:
            self.sim.on_frame = self.previous_callback
            self.writer.close()
            self.camera.close()


def watch_live(sim, policy):
    import mujoco.viewer
    history = []
    with mujoco.viewer.launch_passive(sim.model,sim.data) as viewer:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        viewer.cam.fixedcamid = sim.model.camera("front").id
        last = time.monotonic()
        def sync(s):
            nonlocal last
            if not viewer.is_running():
                raise KeyboardInterrupt
            viewer.sync()
            remaining = s.frame_stride*s.model.opt.timestep-(time.monotonic()-last)
            if remaining>0:
                time.sleep(remaining)
            last = time.monotonic()
        sim.on_frame = sync
        try:
            while not sim.finished and viewer.is_running():
                action = policy.act(sim.observe(),history)
                sim.step(action)
                history.append(action)
        finally:
            sim.on_frame = None
