"""What a backend is allowed to know about the running app."""
from __future__ import annotations

import logging
from typing import Callable, Optional

import numpy as np

log = logging.getLogger("weather_fx")

SESSION_ROOT = "/WeatherFX"


class WeatherContext:
    def __init__(self, manager):
        self._manager = manager
        self.anchor_provider: Optional[Callable[[], object]] = None
        #: How far the wind has carried the cloud field, stage axes, **metres**. The manager sets
        #: it from its weather time on every step (a function of time, not a running sum), so
        #: manual time drives it deterministically. Every consumer of the cloud (the volumes, the
        #: dome, the sun's shadow, a sensor model) offsets by this.
        self.cloud_drift_m = np.zeros(3)
        #: The sky's colour just above the horizon **in renderer units** (the dome's texture times
        #: its intensity), published by the sky effect after each bake; None while no sky is
        #: authored. RTX fog colours are in those units too, so a fog that should match the sky
        #: needs this -- a fog colour of about 1 next to a sky of about 150 renders black.
        self.sky_horizon_rgb = None
        #: The dome's exposure and white-balance gains, ``{"exposure": float, "gains": (r, g, b)}``,
        #: published by the sky effect after each bake: whatever else draws sky in the frame (the
        #: per-pixel cloud layer) must be exposed and balanced the same way.
        self.sky_dome = None

    # ---------------------------------------------------------------- stage
    @staticmethod
    def stage():
        import omni.usd

        return omni.usd.get_context().get_stage()

    def meters_per_unit(self) -> float:
        from pxr import UsdGeom

        stage = self.stage()
        return float(UsdGeom.GetStageMetersPerUnit(stage)) if stage else 1.0

    def up_axis(self) -> int:
        from pxr import UsdGeom

        stage = self.stage()
        if stage is None:
            return 2
        return 2 if UsdGeom.GetStageUpAxis(stage) == UsdGeom.Tokens.z else 1

    @property
    def state(self):
        return self._manager.state_ref

    @property
    def time(self) -> float:
        return self._manager.time

    # ---------------------------------------------------------------- anchor
    def anchor(self) -> np.ndarray:
        """World position (stage units) the particle volumes follow.

        Priority: custom provider > general.follow_prim > active viewport camera > origin.
        Note: with Fabric, a camera moved by physics may not update its USD transform.
        Use ``WeatherController.set_anchor_provider`` in that case.
        """
        if self.anchor_provider is not None:
            try:
                return np.asarray(self.anchor_provider(), dtype=float).reshape(3)
            except Exception:
                log.exception("weather_fx: anchor provider failed")
        path = self.state.general.follow_prim or self._active_camera_path()
        pos = self._world_position(path) if path else None
        return pos if pos is not None else np.zeros(3)

    @staticmethod
    def _active_camera_path() -> Optional[str]:
        try:
            from omni.kit.viewport.utility import get_active_viewport

            viewport = get_active_viewport()
            return str(viewport.camera_path) if viewport else None
        except Exception:
            return None

    def _world_position(self, path: str) -> Optional[np.ndarray]:
        from pxr import Usd, UsdGeom

        stage = self.stage()
        if stage is None:
            return None
        prim = stage.GetPrimAtPath(path)
        if not prim or not prim.IsValid() or not prim.IsA(UsdGeom.Xformable):
            return None
        m = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        t = m.ExtractTranslation()
        return np.array([t[0], t[1], t[2]], dtype=float)
