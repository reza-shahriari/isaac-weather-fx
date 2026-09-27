"""Write a cloud grid as an OpenVDB fog volume, with the OpenVDB that ships inside Kit.

Isaac Sim carries a full OpenVDB binding in the ``omni.volume`` extension. Inside Kit, with that
extension enabled, ``import openvdb`` normally just works; when it does not, the two things Kit
would have done are done here -- preload the shared libraries the binding links against and put
the extension's directory on ``sys.path``. The recipe is the one the thermal-camera project
measured (``irsim_isaac.env.ensure_openvdb_on_path``).
"""
from __future__ import annotations

import ctypes
import importlib
import logging
import pathlib
import sys
from typing import Any, Optional, Sequence

log = logging.getLogger("weather_fx")

_PRELOAD = ("libtbb.so.12", "libtbbmalloc.so.2", "libopenvdb.so.12.0")


def import_openvdb() -> Optional[Any]:
    """The ``openvdb`` module, or ``None`` if this install has none."""
    try:
        import openvdb  # type: ignore

        return openvdb
    except ImportError:
        pass
    directory = _omni_volume_directory()
    if directory is None:
        return None
    for name in _PRELOAD:
        for library in (directory / "bin" / name, directory / name):
            if library.is_file():
                try:
                    ctypes.CDLL(str(library), mode=ctypes.RTLD_GLOBAL)
                except OSError:
                    log.debug("weather_fx: could not preload %s", library)
                break
    for candidate in (directory, directory / "bin"):
        if str(candidate) not in sys.path:
            sys.path.append(str(candidate))
    importlib.invalidate_caches()
    try:
        import openvdb  # type: ignore

        return openvdb
    except ImportError:
        log.warning("weather_fx: omni.volume found at %s but openvdb would not import", directory)
        return None


def _omni_volume_directory() -> Optional[pathlib.Path]:
    try:
        import omni.kit.app

        manager = omni.kit.app.get_app().get_extension_manager()
        if not manager.is_extension_enabled("omni.volume"):
            manager.set_extension_enabled_immediate("omni.volume", True)
        ext_id = manager.get_enabled_extension_id("omni.volume")
        path = manager.get_extension_path(ext_id) if ext_id else None
        if not path:
            return None
        root = pathlib.Path(path)
        # The binding sits somewhere under the extension; find the directory that holds it.
        for found in sorted(root.rglob("openvdb*.so")) + sorted(root.rglob("openvdb*.pyd")):
            return found.parent
        package = next(iter(root.rglob("openvdb/__init__.py")), None)
        return package.parent.parent if package else root
    except Exception:
        log.debug("weather_fx: omni.volume lookup failed", exc_info=True)
        return None


def write_fog_volume(
    openvdb: Any,
    path: pathlib.Path,
    values: Any,
    voxel: Sequence[float],
    first_centre: Sequence[float],
    tolerance: float = 1e-3,
) -> int:
    """Write ``values`` (``(nx, ny, nz)`` float32) as a fog volume named ``density``.

    ``voxel`` and ``first_centre`` are in **stage units**: index ``(0, 0, 0)`` is the centre of the
    first voxel. The transform is written as a matrix (row vectors, translation in the last row)
    rather than composed from operations, because the order of a scale against a translation is
    exactly the kind of thing that silently puts a cloud underground.

    ``tolerance`` is the value treated as empty. A cloud field is mostly empty space, and storing
    that space as tiny numbers rather than absence makes a sparse grid dense. Returns the active
    voxel count.
    """
    import numpy as np

    grid = openvdb.FloatGrid()
    grid.name = "density"
    grid.gridClass = openvdb.GridClass.FOG_VOLUME
    grid.transform = openvdb.createLinearTransform([
        [float(voxel[0]), 0.0, 0.0, 0.0],
        [0.0, float(voxel[1]), 0.0, 0.0],
        [0.0, 0.0, float(voxel[2]), 0.0],
        [float(first_centre[0]), float(first_centre[1]), float(first_centre[2]), 1.0],
    ])
    grid.copyFromArray(np.ascontiguousarray(values, dtype=np.float32), tolerance=float(tolerance))
    grid.addStatsMetadata()
    path.parent.mkdir(parents=True, exist_ok=True)
    openvdb.write(str(path), grids=[grid])
    return int(grid.activeVoxelCount())
