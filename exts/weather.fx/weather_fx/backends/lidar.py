"""Planned lidar backend (not registered in v0.1).

Plan:
* Rain/snow particles are real geometry, so RTX lidar already gets hits on them.
  Add a toggle to hide them from lidar if you only want attenuation.
* Fog: post-process the point cloud from the RTX lidar annotator:
  - intensity *= exp(-2 * beta * range)  (two-way Beer-Lambert, beta from fog.visibility_m)
  - drop returns whose attenuated intensity falls below a detection threshold
  - add near-range backscatter returns (see Hahner et al., ICCV 2021, fog simulation on lidar)
* Rain: range-dependent attenuation from rain rate and random dropouts.
"""
from .base import SensorBackend


class LidarBackend(SensorBackend):
    name = "lidar"

    def __init__(self, sensor_prim_path: str):
        self.sensor_prim_path = sensor_prim_path

    def apply_state(self, state, changed):
        raise NotImplementedError("Lidar weather backend is planned for v0.3; see ROADMAP.md")
