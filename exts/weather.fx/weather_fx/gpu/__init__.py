"""GPU evaluation of the cloud function (NVIDIA Warp). Optional: ``core`` never imports this.

Warp ships with Isaac Sim as the Kit extension ``omni.warp.core`` rather than as a Python
package on the interpreter's path, so :func:`ensure_warp` finds that extension's folder and adds
it when a plain ``import warp`` fails. Outside Isaac Sim, ``pip install warp-lang`` works too.
"""
from __future__ import annotations

import glob
import os
import sys

__all__ = ["ensure_warp", "warp_available"]

#: Where Kit keeps its downloaded extensions; the newest ``omni.warp.core`` there wins.
_EXTENSION_CACHES = (
    os.path.expanduser("~/.local/share/ov/data/exts/v2"),
    os.path.expanduser("~/.local/share/ov/data/exts"),
)


def ensure_warp():
    """Import and return ``warp``, finding Isaac Sim's copy if it is not on the path."""
    try:
        import warp  # noqa: F401
    except ModuleNotFoundError:
        found = []
        for cache in _EXTENSION_CACHES:
            found.extend(glob.glob(os.path.join(cache, "omni.warp.core-*")))
        if not found:
            raise
        found.sort(key=lambda path: _version_key(os.path.basename(path)))
        sys.path.append(found[-1])
        import warp  # noqa: F401
    return sys.modules["warp"]


def _version_key(name: str):
    try:
        version = name.split("omni.warp.core-", 1)[1].split("+", 1)[0]
        return tuple(int(part) for part in version.split("."))
    except (IndexError, ValueError):
        return (0,)


def warp_available() -> bool:
    try:
        ensure_warp()
    except Exception:
        return False
    return True
