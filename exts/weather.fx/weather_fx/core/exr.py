"""A minimal OpenEXR writer: single part, scanline, uncompressed, 32-bit float.

An environment map has to reach the renderer in a high-dynamic-range format, and the sky spans
six decades between noon and a moonless night -- far outside what any 8-bit container holds. EXR
is what RTX dome lights take.

Written against the format specification with :mod:`struct` and nothing else, rather than pulling
in ``OpenEXR`` or ``imageio``: this package's only dependency is numpy, and a renderer-side
extension that cannot be imported in a plain Python process is a renderer-side extension that
cannot be unit tested.

Deliberately narrow. Uncompressed and float32 only -- a 1024-row map is 25 MB, written once per
bake, and neither the size nor the write time is anywhere near the cost of generating it. Adding
ZIP compression would save disk and cost correctness risk in the one piece of this codebase that
has to be byte-exact against a specification nobody here can debug from the other side.
"""
from __future__ import annotations

import pathlib
import struct
from typing import Any, Union

import numpy as np

__all__ = ["write_exr"]

_MAGIC = 20000630
_VERSION = 2  # single-part scanline, no long names, no tiles
_PIXEL_TYPE_FLOAT = 2


def _attribute(name: str, kind: str, payload: bytes) -> bytes:
    return (
        name.encode("ascii")
        + b"\0"
        + kind.encode("ascii")
        + b"\0"
        + struct.pack("<i", len(payload))
        + payload
    )


def write_exr(path: Union[str, pathlib.Path], image: Any) -> pathlib.Path:
    """Write ``(height, width, 3)`` linear RGB to an uncompressed float32 EXR.

    Channels are stored in the alphabetical order the format requires -- **B, G, R** -- which is
    the detail that makes a hand-written EXR come out with its red and blue swapped. A sky written
    that way looks like a sunset at noon and is entirely plausible until someone checks.
    """
    data = np.asarray(image, dtype=np.float32)
    if data.ndim != 3 or data.shape[2] != 3:
        raise ValueError(f"an EXR needs (height, width, 3); got {data.shape}")
    if not np.all(np.isfinite(data)):
        raise ValueError("an EXR may not carry NaN or infinity: fix the source, not the file")
    height, width = data.shape[:2]
    target = pathlib.Path(path)

    header = struct.pack("<I", _MAGIC) + struct.pack("<I", _VERSION)
    channels = b""
    for name in ("B", "G", "R"):
        channels += (
            name.encode("ascii")
            + b"\0"
            + struct.pack("<i", _PIXEL_TYPE_FLOAT)
            + struct.pack("<B", 0)          # pLinear
            + b"\0\0\0"                      # reserved
            + struct.pack("<ii", 1, 1)       # x and y sampling
        )
    channels += b"\0"
    header += _attribute("channels", "chlist", channels)
    header += _attribute("compression", "compression", struct.pack("<B", 0))
    box = struct.pack("<iiii", 0, 0, width - 1, height - 1)
    header += _attribute("dataWindow", "box2i", box)
    header += _attribute("displayWindow", "box2i", box)
    header += _attribute("lineOrder", "lineOrder", struct.pack("<B", 0))
    header += _attribute("pixelAspectRatio", "float", struct.pack("<f", 1.0))
    header += _attribute("screenWindowCenter", "v2f", struct.pack("<ff", 0.0, 0.0))
    header += _attribute("screenWindowWidth", "float", struct.pack("<f", 1.0))
    header += b"\0"

    # Every scanline is one chunk: 8 bytes of offset table entry each, then y + size + pixels.
    row_bytes = width * 3 * 4
    chunk_bytes = 4 + 4 + row_bytes
    offsets_start = len(header)
    first_chunk = offsets_start + 8 * height
    offsets = struct.pack(
        f"<{height}Q", *(first_chunk + row * chunk_bytes for row in range(height))
    )

    # B, G, R planes per scanline, each contiguous -- not interleaved pixels.
    planes = np.stack([data[..., 2], data[..., 1], data[..., 0]], axis=1)
    body = bytearray()
    for row in range(height):
        body += struct.pack("<ii", row, row_bytes)
        body += planes[row].astype("<f4").tobytes()

    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "wb") as handle:
        handle.write(header)
        handle.write(offsets)
        handle.write(bytes(body))
    return target
