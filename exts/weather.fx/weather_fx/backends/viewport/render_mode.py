"""Which renderer is running, and so how the clouds are drawn.

RTX's path tracer renders heterogeneous volumes; its real-time mode, on the builds measured (see
the thermal project's ADR 0144), does not. So under ``clouds.render_path = "auto"`` the clouds are
3D volumes while the path tracer runs and are painted into the sky dome otherwise, and switching
the viewport's renderer switches them. The field is the same object either way.
"""
from __future__ import annotations

from typing import Optional

#: The carb setting the viewport's renderer menu writes.
RENDER_MODE_SETTING = "/rtx/rendermode"
#: Values of that setting that mean the (offline-quality) path tracer.
PATH_TRACED_MODES = frozenset({"PathTracing"})


def choose_cloud_path(render_path: str, render_mode: Optional[str]) -> str:
    """``"volume"`` or ``"dome"`` for a requested path and the renderer's mode string."""
    if render_path in ("volume", "dome"):
        return render_path
    return "volume" if render_mode in PATH_TRACED_MODES else "dome"


def current_render_mode() -> Optional[str]:
    try:
        import carb.settings
    except ImportError:
        return None
    value = carb.settings.get_settings().get(RENDER_MODE_SETTING)
    return str(value) if value is not None else None


def cloud_path(state) -> str:
    """The path the clouds should take right now for ``state``."""
    return choose_cloud_path(state.clouds.render_path, current_render_mode())


class RenderModeWatcher:
    """Reports when the viewport's renderer changes, polled from an effect's ``update``."""

    def __init__(self):
        self._last: Optional[str] = current_render_mode()

    def changed(self) -> bool:
        mode = current_render_mode()
        if mode != self._last:
            self._last = mode
            return True
        return False
