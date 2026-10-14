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


from rover_vlm import route_events as RE                                                  # noqa: E402
from rover_vlm.landmarks import OCC                                                       # noqa: E402
from rover_vlm.route_events import (                                                      # noqa: E402
    EVENTS_SCHEMA, REASONS, behind_anchor, describe_route, describe_xz, detect_turn, load_frame, route_geometry,
)
from test_route_events import _long_state                                                 # noqa: E402
from test_route_turns import BEHIND, _behind_state, _cabinet, _free, write_sample         # noqa: E402

# right of the box ahead (0.25 m clear), then back left behind it: the end is hidden by the box on the map
BEHIND_END = [(2.5, 0.2), (1.9, 1.2), (1.9, 2.6), (2.2, 3.4)]


def write_fpv(sd, route, tail_m):
    """fpv_paths.json for candidate 0: the route's points every FPV_STEP_M, the last tail_m of them hidden."""
    n = int(round(route_geometry(np.asarray(route, float))["arc"][-1] / FPV_STEP_M)) + 1
    h = int(round(tail_m / FPV_STEP_M))
    runs = [{"hidden": False, "uv": [[0, 0]] * (n - 1 - h)}, {"hidden": True, "uv": [[0, 0]] * (h + 1)}]
    (sd / "fpv_paths.json").write_text(json.dumps({"image_size": [512, 512], "candidates": [{"runs": runs}]}))
    return sd


def _box(frame):
    return f"{frame.name}:b{int(frame.lab[40, 50])}"          # the box ahead of _behind_state


def test_schema_and_reason():
    assert EVENTS_SCHEMA == "d2-endings" and "no_behind_anchor" in REASONS


def test_hidden_end_gets_a_behind_ending(tmp_path):
    sd = write_fpv(write_sample(tmp_path / "s", _behind_state(), [BEHIND_END]), BEHIND_END, 0.6)
    _, frame = load_frame(sd)
    out = describe_route(sd, 0)
    assert out["reason"] is None
    assert [(e["type"], e["side"], e["objects"]) for e in out["events"]] == [("pass", "right", [_box(frame)])]
    st = out["stop"]
    assert (st["type"], st["relation"], st["object"], st["amount"]) == ("behind", "behind", _box(frame), None)
    L = route_geometry(np.asarray(BEHIND_END, float))["arc"][-1]
    assert st["disappear_arc_m"] == pytest.approx(L - 0.6, abs=0.06)
    assert out["objects"][st["object"]]["kind_hint"] == "object"        # an event member: its event role wins


def test_without_fpv_paths_the_end_counts_as_visible(tmp_path):
    out = describe_route(write_sample(tmp_path / "s", _behind_state(), [BEHIND_END]), 0)
    assert out["stop"]["type"] == "near" and set(out["stop"]) == {"type", "relation", "amount", "distance_m", "object"}


def test_hidden_turn_after_the_disappearance_is_the_ending(tmp_path):
    sd = write_fpv(write_sample(tmp_path / "s", _behind_state(), [BEHIND]), BEHIND, 2.2)   # turn at ~3.0 m of 4.97
    out = describe_route(sd, 0)
    assert out["reason"] is None
    assert [(e["type"], e.get("relation")) for e in out["events"]] == [("pass", None), ("turn", "behind")]
    st = out["stop"]
    assert (st["type"], st["relation"], st["object"]) == ("turn", None, None)
    assert st["disappear_arc_m"] == pytest.approx(4.97 - 2.2, abs=0.06)


def test_a_turn_before_the_disappearance_stays_an_event_and_the_ending_follows(tmp_path):
    sd = write_fpv(write_sample(tmp_path / "s", _behind_state(), [BEHIND]), BEHIND, 1.8)   # disappears at ~3.17 m
    _, frame = load_frame(sd)
    out = describe_route(sd, 0)
    assert [e["type"] for e in out["events"]] == ["pass", "turn"]
    assert (out["stop"]["type"], out["stop"]["object"]) == ("behind", _box(frame))


def test_no_nameable_occluder_drops_the_route(tmp_path):
    route = [(2.5, 0.2), (2.5, 3.0)]
    sd = write_fpv(write_sample(tmp_path / "s", _cabinet(_free()), [route]), route, 1.0)
    assert describe_route(sd, 0)["reason"] == "no_behind_anchor"


def test_obstacle_events_after_the_disappearance_are_dropped(tmp_path, monkeypatch):
    s = _long_state()
    s[86:94, 54:66] = OCC                                   # crate x 2.7..3.3, z 4.3..4.7: a second pass at ~4.15 m
    route = [(2.5, 0.2), (2.5, 6.0)]
    meta, frame = load_frame(write_sample(tmp_path / "s", s, [route]))
    crate = int(frame.lab[90, 60])
    assert [e["type"] for e in describe_xz(meta, frame, np.array(route))["events"]] == ["pass", "pass"]  # scene check
    vis, outline = RE._anchor_outline(frame, crate, RE._blob_cells(frame)[crate][1][0])
    monkeypatch.setattr(RE, "behind_anchor", lambda f, r, a: {"blob": crate, "vis_xz": vis, "outline": outline})
    out = describe_xz(meta, frame, np.array(route), hidden_tail=2.5)        # disappears at 3.3 m
    assert len(out["events"]) == 1 and out["events"][0]["arc_m"] < 3.3
    assert out["stop"]["type"] == "behind" and out["stop"]["object"] == f"{frame.name}:b{crate}"


def test_behind_anchor_reads_past_the_disappearance_then_falls_back_to_the_end(tmp_path):
    _, frame = load_frame(write_sample(tmp_path / "s", _behind_state(), [BEHIND_END]))
    box = int(frame.lab[40, 50])
    L = route_geometry(np.asarray(BEHIND_END, float))["arc"][-1]
    assert behind_anchor(frame, np.array(BEHIND_END), L - 0.6)["blob"] == box       # read 0.15 m past it
    assert behind_anchor(frame, np.array(BEHIND_END), 0.2)["blob"] == box           # in view there: the end's occluder
    _, open_frame = load_frame(write_sample(tmp_path / "o", _cabinet(_free()), [[(2.5, 0.2), (2.5, 3.0)]]))
    assert behind_anchor(open_frame, np.array([(2.5, 0.2), (2.5, 3.0)]), 2.0) is None


def test_a_route_hidden_from_its_first_point_does_not_crash(tmp_path):        # Review Focus 4
    sd = write_fpv(write_sample(tmp_path / "s", _behind_state(), [BEHIND_END]), BEHIND_END, 3.42)
    out = describe_route(sd, 0)
    assert out["reason"] in (None, "no_events_bent", "no_behind_anchor")
    if out["reason"] is None:
        assert out["events"] == [] and out["stop"]["disappear_arc_m"] == pytest.approx(0.0, abs=0.06)


def test_describe_route_passes_its_fpv_tail(tmp_path):
    sd = write_fpv(write_sample(tmp_path / "s", _behind_state(), [BEHIND_END]), BEHIND_END, 0.6)
    meta, frame = load_frame(sd)
    tail = hidden_tail_m(fpv_hidden_flags(sd, 0), FPV_STEP_M)
    assert describe_route(sd, 0) == describe_xz(meta, frame, np.array(BEHIND_END), turn=detect_turn(BEHIND_END),
                                                hidden_tail=tail)


from rover_vlm.route_events import REANCHOR_M, build_events, candidates, stop_phrase   # noqa: E402

LONG = [(2.5, 0.2), (2.5, 6.0)]       # passes the post at z 1.0, then ~5 m to the end: "well past the post"


def _wide_box(z_row):
    s = _long_state()
    s[z_row:z_row + 4, 30:70] = OCC   # a 2 m wide box x 1.5..3.5 (wall-like), 0.2 m deep
    return s


def _stop(tmp_path, state, route):
    _, frame = load_frame(write_sample(tmp_path / "s", state, [route]))
    cs = candidates(frame, np.array(route))
    return frame, stop_phrase(frame, np.array(route), build_events(frame, cs), cs)


def test_reanchor_radius():
    assert REANCHOR_M == 1.5


def test_visible_well_past_reanchors_to_an_object_within_1_5_m(tmp_path):
    frame, st = _stop(tmp_path, _wide_box(144), LONG)          # box face 1.23 m ahead of the end
    assert (st["type"], st["relation"], st.get("reanchored")) == ("near", "in front of", True)
    assert st["anchor"]["blob"] == int(frame.lab[145, 50]) and st["anchor"]["kind"] == "wall"
    row = describe_route(write_sample(tmp_path / "d", _wide_box(144), [LONG]), 0)["stop"]
    assert row["reanchored"] is True and row["relation"] == "in front of"


def test_well_past_stays_when_nothing_is_within_1_5_m(tmp_path):
    _, st = _stop(tmp_path, _wide_box(154), LONG)              # 1.73 m ahead
    assert (st["type"], st["relation"], st["amount"]) == ("distance", "past", "well") and "reanchored" not in st
    row = describe_route(write_sample(tmp_path / "d", _wide_box(154), [LONG]), 0)["stop"]
    assert set(row) == {"type", "relation", "amount", "distance_m", "object"}


def test_a_little_past_is_never_reanchored(tmp_path):
    s = _long_state(100)
    s[88:92, 30:70] = OCC                                      # 1.2 m ahead of the end below
    _, st = _stop(tmp_path, s, [(2.5, 0.2), (2.5, 3.2)])
    assert (st["type"], st["amount"]) == ("distance", "a little") and "reanchored" not in st
