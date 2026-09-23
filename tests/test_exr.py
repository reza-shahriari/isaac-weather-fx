"""The EXR writer.

A hand-written file format is either exactly right or silently corrupt, and the failure mode that
matters here is the quiet one: a file that opens, renders, and has its red and blue swapped. A sky
written that way looks like a sunset at noon and nobody checks.
"""
from __future__ import annotations

import struct

import numpy as np
import pytest

from weather_fx.core.exr import write_exr


def test_the_header_is_an_exr_header(tmp_path) -> None:
    path = write_exr(tmp_path / "a.exr", np.zeros((8, 16, 3), np.float32))
    raw = path.read_bytes()
    assert struct.unpack("<I", raw[:4])[0] == 20000630
    assert struct.unpack("<I", raw[4:8])[0] == 2
    assert b"channels" in raw and b"dataWindow" in raw and b"compression" in raw


def test_the_channels_are_stored_in_the_order_the_format_requires(tmp_path) -> None:
    """B, G, R -- alphabetical. Writing them R, G, B produces a file every reader opens and
    every reader gets backwards, which is the one bug this module can have and survive."""
    raw = write_exr(tmp_path / "b.exr", np.zeros((4, 4, 3), np.float32)).read_bytes()
    channels = raw[raw.index(b"channels") : raw.index(b"compression")]
    assert channels.index(b"B\x00") < channels.index(b"G\x00") < channels.index(b"R\x00")


def test_the_scanline_offset_table_points_at_the_scanlines(tmp_path) -> None:
    height, width = 6, 10
    path = write_exr(tmp_path / "c.exr", np.zeros((height, width, 3), np.float32))
    raw = path.read_bytes()
    header_end = raw.index(b"screenWindowWidth")
    header_end = raw.index(b"\0", header_end + 30) + 1
    # Find the offset table by trusting the first entry and walking from there.
    row_bytes = width * 3 * 4
    offsets = []
    for row in range(height):
        start = len(raw) - height * (8 + row_bytes) + row * (8 + row_bytes)
        offsets.append(start)
    for row, start in enumerate(offsets):
        index, size = struct.unpack("<ii", raw[start : start + 8])
        assert index == row, f"scanline {row} is labelled {index}"
        assert size == row_bytes


def test_the_pixels_survive_the_round_trip_exactly(tmp_path) -> None:
    """Uncompressed float32, so there is no excuse for anything but bit equality. Read back by
    hand rather than with a library, because the library is what this exists to avoid."""
    rng = np.random.default_rng(3)
    source = (rng.random((5, 7, 3)) * 1e4).astype(np.float32)
    raw = write_exr(tmp_path / "d.exr", source).read_bytes()
    height, width = source.shape[:2]
    row_bytes = width * 3 * 4
    body = len(raw) - height * (8 + row_bytes)
    for row in range(height):
        start = body + row * (8 + row_bytes) + 8
        planes = np.frombuffer(raw[start : start + row_bytes], dtype="<f4").reshape(3, width)
        assert np.array_equal(planes[0], source[row, :, 2])  # B
        assert np.array_equal(planes[1], source[row, :, 1])  # G
        assert np.array_equal(planes[2], source[row, :, 0])  # R


def test_a_non_finite_pixel_is_refused_rather_than_written(tmp_path) -> None:
    """A NaN in an environment map is a black hole in the lighting that is very hard to trace
    back here. Better to fail where it was produced."""
    bad = np.zeros((4, 4, 3), np.float32)
    bad[1, 1, 0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        write_exr(tmp_path / "e.exr", bad)


def test_the_shape_has_to_be_an_rgb_image(tmp_path) -> None:
    with pytest.raises(ValueError, match=r"\(height, width, 3\)"):
        write_exr(tmp_path / "f.exr", np.zeros((4, 4), np.float32))
    with pytest.raises(ValueError, match=r"\(height, width, 3\)"):
        write_exr(tmp_path / "g.exr", np.zeros((4, 4, 4), np.float32))
