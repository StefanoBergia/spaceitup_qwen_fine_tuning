"""d2 endings (spec 2026-10-13 Addendum T2): where a route goes out of sight, its ending, the visible re-anchor.

Synthetic maps as tests/test_route_turns.py: 5 m x 5 m (or 10 m long) at 5 cm, camera at (2.5, 0.2) facing +z, so
+x is the rover's LEFT. fpv_paths.json is written by hand: its flags say where the route is out of sight.
"""
import json

import numpy as np
import pytest

from rover_vlm.route_events import FPV_STEP_M, HIDDEN_BLIP_M, fpv_hidden_flags, hidden_tail_m


def _flags(s):
    return [c == "H" for c in s]


def test_constants():
    assert (HIDDEN_BLIP_M, FPV_STEP_M) == (0.3, 0.05)


@pytest.mark.parametrize("flags,tail", [
    ("..........", None),          # the end is in view
    ("......HHHH", 0.15),          # 4 hidden points: the final hidden run starts 3 steps back
    ("..HH.HHHHH", 0.35),          # a 1-point visible blip (0.05 m) between hidden runs counts as hidden
    ("HH......HHH", 0.1),          # a 6-point visible run (0.3 m) is real: the final run starts at index 8
    ("HHHHH", 0.2),                # hidden from the first point: the whole route (Review Focus 4)
    ("....HHHH.", None),           # a short visible end is still a visible end
])
def test_hidden_tail_m(flags, tail):
    got = hidden_tail_m(_flags(flags), step=0.05)
    assert got == (None if tail is None else pytest.approx(tail))


def test_hidden_tail_m_of_no_flags_is_none():
    assert hidden_tail_m([]) is None


def test_fpv_hidden_flags_reads_one_candidates_runs(tmp_path):
    (tmp_path / "fpv_paths.json").write_text(json.dumps({"image_size": [512, 512], "candidates": [
        {"runs": [{"hidden": False, "uv": [[0, 0]] * 3}, {"hidden": True, "uv": [[0, 0]] * 2}]},
        {"runs": [{"hidden": False, "uv": [[0, 0]] * 4}]}]}))
    assert fpv_hidden_flags(tmp_path, 0) == [False, False, False, True, True]
    assert fpv_hidden_flags(tmp_path, 1) == [False] * 4
    assert fpv_hidden_flags(tmp_path / "nowhere", 0) is None
