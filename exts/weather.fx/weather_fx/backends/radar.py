"""Planned radar backend (not registered in v0.1).

Plan: attenuate RTX radar detections by rain rate (two-way path loss),
add rain clutter detections near the sensor, and leave fog mostly
transparent, which is the main reason radar is used in bad weather.
"""
from .base import SensorBackend


class RadarBackend(SensorBackend):
    name = "radar"

    def __init__(self, sensor_prim_path: str):
        self.sensor_prim_path = sensor_prim_path

    def apply_state(self, state, changed):
        raise NotImplementedError("Radar weather backend is planned for v0.3; see ROADMAP.md")
