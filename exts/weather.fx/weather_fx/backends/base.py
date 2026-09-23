"""Interface every sensor backend implements.

The viewport backend is the only one shipped in v0.1. Lidar and radar
backends will implement the same four methods and read the same
WeatherState, so one UI/script call drives every sensor consistently.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Set


class SensorBackend(ABC):
    name: str = "base"

    def attach(self, context) -> None:
        """Called once when registered. ``context`` is a runtime.WeatherContext."""
        self.context = context

    @abstractmethod
    def apply_state(self, state, changed: Set[str]) -> None:
        """React to a state change. ``changed`` lists the modified section names."""

    def update(self, dt: float, t: float) -> None:
        """Called every tick while weather is enabled."""

    def detach(self) -> None:
        """Undo every change this backend made (settings, prims, lights)."""

    def stats(self) -> dict:
        return {}


class Effect(ABC):
    """A single visual effect inside a backend (fog, rain, ...). Same lifecycle as a backend."""

    name: str = "effect"

    def attach(self, context) -> None:
        self.context = context

    @abstractmethod
    def apply_state(self, state, changed: Set[str]) -> None: ...

    def update(self, dt: float, t: float) -> None: ...

    def detach(self) -> None: ...

    def stats(self) -> dict:
        return {}
