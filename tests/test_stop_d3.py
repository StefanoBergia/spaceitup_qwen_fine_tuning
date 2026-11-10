"""d3 visible stops: an object relation within CLOSE_M, else a place (spec 2026-11-10-stop-phrases-d3-design.md).

Scenes as tests/test_route_turns.py: 100 x 100 grid at 0.05 m, camera (2.5, 0.2) looking +z (its right is -x),
routes straight along +z to (2.5, 3.0) unless stated.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from rover_vlm.landmarks import FREE, OCC, UNKNOWN
from rover_vlm.route_events import (CLOSE_M, EVENTS_SCHEMA, EVENTS_SCHEMA_D3, REASONS, check_events_schema,
                                    describe_route, load_frame, stop_d3, stop_kind)
from test_route_endings import BEHIND_END, write_fpv
from test_route_turns import _behind_state, _free, write_sample

END = [(2.5, 0.2), (2.5, 3.0)]


def _stop(tmp_path, state, route=END):
    _, frame = load_frame(write_sample(tmp_path / "s", state, [route]))
    return frame, stop_d3(frame, np.asarray(route, float))


def _with(rows, cols):
    s = _free()
    s[rows, cols] = OCC
    return s


def test_constants():
    assert CLOSE_M == 1.5 and "no_stop_place" in REASONS and EVENTS_SCHEMA_D3 == "d3-stops"


def test_in_front_of_an_object_ahead(tmp_path):
    frame, st = _stop(tmp_path, _with(slice(72, 80), slice(44, 56)))      # 0.6 x 0.4 m box, face 0.6 m ahead
    assert (st["type"], st["relation"]) == ("object", "in front of")
    assert st["anchor"]["blob"] == int(frame.lab[75, 50]) and st["anchor"]["kind"] == "object"


DETOUR = [(2.5, 0.2), (1.8, 1.6), (1.8, 2.6), (2.5, 3.0)]               # round a post on the camera ray, end behind it


def test_behind_an_object_between_camera_and_goal(tmp_path):
    _, st = _stop(tmp_path, _with(slice(42, 46), slice(48, 52)), DETOUR)   # post x 2.4-2.6, z 2.1-2.3, in the end's column
    assert (st["type"], st["relation"]) == ("object", "behind")


def test_object_beside_the_route_end_is_not_behind(tmp_path):     # final review I1
    """A post 0.4-0.6 m beside the last metre of the route is not between the camera and the end in the image."""
    _, st = _stop(tmp_path, _with(slice(46, 50), slice(38, 42)))          # post x 1.9-2.1, z 2.3-2.5
    assert st.get("relation") != "behind"


def test_next_to_an_object_beside_the_goal(tmp_path):
    _, st = _stop(tmp_path, _with(slice(56, 64), slice(35, 39)))          # post x 1.75-1.95, z 2.8-3.2
    assert (st["type"], st["relation"]) == ("object", "next to")


def test_diagonal_object_is_not_next_to(tmp_path):
    _, st = _stop(tmp_path, _with(slice(68, 76), slice(35, 39)))          # post x 1.75-1.95, z 3.4-3.8
    assert st["type"] == "place"


def test_furniture_is_not_a_wall_for_the_place(tmp_path):
    """An object that fits no relation is furniture, not the wall of "stop against the wall"."""
    _, st = _stop(tmp_path, _with(slice(68, 76), slice(35, 39)))
    assert st["place"] == "room_middle"


def test_walls_are_not_object_anchors(tmp_path):
    _, st = _stop(tmp_path, _with(slice(78, 82), slice(30, 70)))          # 2 m wide box, face 0.9 m ahead
    assert st["type"] == "place" and st["place"] == "wall"


def test_object_beyond_close_m_is_ignored(tmp_path):
    _, st = _stop(tmp_path, _with(slice(92, 98), slice(44, 56)))          # face 1.6 m ahead
    assert st["type"] == "place"


def test_d3_stop_record(tmp_path):
    row = describe_route(write_sample(tmp_path / "s", _free(), [[(2.5, 0.2), (2.5, 2.5)]]), 0, stops="d3")
    st = row["stop"]
    assert row["reason"] is None
    assert (st["type"], st["place"], st["object"], st["relation"]) == ("place", "room_middle", None, None)
    assert set(st["measures"]) == {"free_share", "clear_m"}


def test_d3_object_stop_record_names_its_object(tmp_path):
    row = describe_route(write_sample(tmp_path / "s", _with(slice(72, 80), slice(44, 56)), [END]), 0, stops="d3")
    st = row["stop"]
    assert (st["type"], st["relation"], st["place"]) == ("object", "in front of", None)
    assert st["object"] in row["objects"] and row["objects"][st["object"]]["kind_hint"] == "object"


def test_no_place_drops_the_route_as_no_stop_place(tmp_path):
    s = np.full((100, 100), UNKNOWN, np.uint8)
    s[:, 46:54] = FREE                                                       # only a strip along the route seen
    row = describe_route(write_sample(tmp_path / "s", s, [END]), 0, stops="d3")
    assert row["reason"] == "no_stop_place"


def test_hidden_ends_keep_their_t2_endings_under_d3(tmp_path):
    sd = write_fpv(write_sample(tmp_path / "s", _behind_state(), [BEHIND_END]), BEHIND_END, 0.6)
    assert describe_route(sd, 0, stops="d3") == describe_route(sd, 0, stops="d2")
    assert describe_route(sd, 0, stops="d3")["stop"]["type"] == "behind"


def test_d2_stays_the_default(tmp_path):
    sd = write_sample(tmp_path / "s", _with(slice(72, 80), slice(44, 56)), [END])
    assert describe_route(sd, 0) == describe_route(sd, 0, stops="d2")
    assert describe_route(sd, 0)["stop"]["type"] == "near"


def test_stop_kind():
    assert stop_kind({"type": "object", "relation": "next to"}) == "object:next to"
    assert stop_kind({"type": "place", "place": "corner"}) == "place:corner"
    assert stop_kind({"type": "distance", "relation": "past"}) is None
    assert stop_kind({"type": "behind", "relation": "behind"}) is None
    assert stop_kind(None) is None


@pytest.mark.parametrize("schema", [EVENTS_SCHEMA, "d3-stops"])
def test_schema_check_accepts_d2_and_d3(tmp_path, schema):
    (tmp_path / "events_meta.json").write_text(json.dumps({"schema": schema}))
    check_events_schema(Path(tmp_path) / "events_train.jsonl")


def test_schema_check_refuses_others(tmp_path):
    (tmp_path / "events_meta.json").write_text(json.dumps({"schema": "x"}))
    with pytest.raises(SystemExit):
        check_events_schema(Path(tmp_path) / "events_train.jsonl")
