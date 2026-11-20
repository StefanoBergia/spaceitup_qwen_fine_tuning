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
    assert stop_kind({"type": "wall", "relation": "in front of"}) == "wall:in front of"
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



# --- Addendum A1: the wall as a backup when no place fits ----------------------------------------------------------
WIDE_AHEAD = (slice(78, 82), slice(30, 70))                               # 2 m wide box, face 0.9 m ahead (wall-like)


def test_wall_ahead_backs_up_when_no_place_fits(tmp_path, monkeypatch):
    import rover_vlm.route_events as RE
    monkeypatch.setattr(RE, "place_at", lambda grid, route: None)
    frame, st = _stop(tmp_path, _with(*WIDE_AHEAD))
    assert (st["type"], st["relation"]) == ("wall", "in front of")
    assert st["anchor"]["blob"] == int(frame.lab[80, 50]) and st["anchor"]["kind"] == "wall"


def test_a_place_wins_over_the_wall_backup(tmp_path):
    _, st = _stop(tmp_path, _with(*WIDE_AHEAD))
    assert st["type"] == "place"


def test_the_backup_needs_the_wall_ahead(tmp_path, monkeypatch):
    import rover_vlm.route_events as RE
    monkeypatch.setattr(RE, "place_at", lambda grid, route: None)
    _, st = _stop(tmp_path, _with(slice(20, 90), slice(30, 33)))         # a long wall beside the route, not ahead
    assert st is None


def test_wall_backup_record(tmp_path, monkeypatch):
    import rover_vlm.route_events as RE
    monkeypatch.setattr(RE, "place_at", lambda grid, route: None)
    row = describe_route(write_sample(tmp_path / "s", _with(*WIDE_AHEAD), [END]), 0, stops="d3")
    st = row["stop"]
    assert (st["type"], st["relation"], st["place"]) == ("wall", "in front of", None)
    assert row["objects"][st["object"]]["kind_hint"] == "wall"


# --- Addendum A2: furniture is not "the wall" ----------------------------------------------------------------------
# Boxes are (x0, x1, z0, z1, top) in metres; the depth image is ray-cast from them (planar metres, fit = identity).
BACK_WALL = (0.0, 5.0, 4.85, 5.0, 4.0)
SOFA = (2.0, 3.0, 2.0, 2.8, 0.8)                     # low, 1.8-2.6 m ahead of the camera
LONG_SOFA = (1.5, 3.5, 3.6, 4.4, 0.8)                # 2 m x 0.8 m: wall-sized on the map, 0.6 m ahead of END's end


def _render(cam, boxes, H=128, W=128):
    """Planar depth of a level camera over a flat floor among upright boxes, marched every 0.01 m; 0 = no hit."""
    from rover_vlm.landmarks import CAM_HEIGHT_M
    a = np.broadcast_to(2 * (np.arange(W) + 0.5) / W - 1, (H, W))
    b = np.broadcast_to((2 * (np.arange(H) + 0.5) / H - 1)[:, None], (H, W))
    out = np.zeros((H, W))
    for f in np.arange(0.3, 8.0, 0.01):
        hgt = CAM_HEIGHT_M - b * f
        x = cam.pos[0] + f * cam.h[0] + a * f * cam.right[0]
        z = cam.pos[1] + f * cam.h[1] + a * f * cam.right[1]
        hit = hgt <= 0
        for x0, x1, z0, z1, top in boxes:
            hit = hit | ((x >= x0) & (x <= x1) & (z >= z0) & (z <= z1) & (hgt <= top))
        out[(out == 0) & hit] = f
    return out


def _cells_in(box):
    """(rows, cols) slices of the grid cells whose centres lie in a box's footprint."""
    x0, x1, z0, z1, _ = box
    return slice(int(np.ceil(z0 / 0.05 - 0.5)), int(np.floor(z1 / 0.05 - 0.5)) + 1), \
        slice(int(np.ceil(x0 / 0.05 - 0.5)), int(np.floor(x1 / 0.05 - 0.5)) + 1)


def _scene(tmp_path, mapped, rendered=(), base=None, route=END):
    """A frame whose map holds the `mapped` boxes and whose depth image shows them plus `rendered`."""
    s = _free() if base is None else base.copy()
    for box in mapped:
        s[_cells_in(box)] = OCC
    _, frame = load_frame(write_sample(tmp_path / "s", s, [route]))
    frame.depth = _render(frame.cam, list(mapped) + list(rendered))
    frame.depth_fit = (1.0, 0.0, 0.0)
    return frame


def _xz(frame, box):
    m = np.zeros(frame.grid.s.shape, bool)
    m[_cells_in(box)] = True
    return frame.grid.xz(np.argwhere(m))


def test_cell_heights_tell_low_furniture_from_a_wall(tmp_path):
    from rover_vlm.route_events import HEIGHT_LOW, HEIGHT_TALL, cell_heights
    frame = _scene(tmp_path, [SOFA, BACK_WALL])
    sofa = cell_heights(frame, _xz(frame, SOFA))
    wall = cell_heights(frame, _xz(frame, (0.5, 1.5, 4.85, 5.0, 4.0)))      # the back wall's stretch the sofa does not hide
    assert (sofa == HEIGHT_LOW).mean() >= 0.8
    assert (wall == HEIGHT_TALL).mean() >= 0.8


def test_cell_heights_unknown_without_a_usable_depth(tmp_path):
    from rover_vlm.route_events import DEPTH_FIT_MAX_RESID_M, HEIGHT_UNKNOWN, cell_heights
    frame = _scene(tmp_path, [SOFA, BACK_WALL])
    xz = _xz(frame, SOFA)
    frame.depth_fit = (1.0, 0.0, DEPTH_FIT_MAX_RESID_M + 0.1)
    assert (cell_heights(frame, xz) == HEIGHT_UNKNOWN).all()
    frame.depth_fit = None
    assert (cell_heights(frame, xz) == HEIGHT_UNKNOWN).all()
    frame.depth = None
    assert (cell_heights(frame, xz) == HEIGHT_UNKNOWN).all()


def test_low_furniture_ahead_is_an_object_not_the_wall(tmp_path):
    """A wall-sized sofa 0.6 m ahead: without heights it is "by the wall"; the depth shows it low -> in front of it."""
    frame = _scene(tmp_path, [LONG_SOFA], rendered=[BACK_WALL])
    st = stop_d3(frame, np.asarray(END, float))
    assert (st["type"], st["relation"]) == ("object", "in front of")
    assert st["anchor"]["blob"] == int(frame.lab[80, 50]) and st["anchor"]["kind"] == "object"


def test_without_depth_the_wall_sized_sofa_stays_by_the_wall(tmp_path):
    frame = _scene(tmp_path, [LONG_SOFA], rendered=[BACK_WALL])
    frame.depth = None
    st = stop_d3(frame, np.asarray(END, float))
    assert (st["type"], st["place"]) == ("place", "wall")


def test_a_tall_wall_ahead_stays_by_the_wall(tmp_path):
    frame = _scene(tmp_path, [(1.5, 3.5, 3.6, 3.8, 2.5)], rendered=[BACK_WALL])
    st = stop_d3(frame, np.asarray(END, float))
    assert (st["type"], st["place"]) == ("place", "wall")


def test_by_the_wall_stays_when_the_recheck_finds_no_stop(tmp_path):
    """Low furniture diagonal from the end in unexplored space: once it is not a wall, no relation, place or wall ahead
    fits -> the stop stays "by the wall" (the fix never drops a route)."""
    base = np.full((100, 100), UNKNOWN, np.uint8)
    base[:65, 40:60] = FREE                                                 # a strip along the route seen
    frame = _scene(tmp_path, [(0.1, 1.6, 3.6, 4.5, 0.8)], rendered=[BACK_WALL], base=base)   # 1.1 m off, diagonal
    st = stop_d3(frame, np.asarray(END, float))
    assert (st["type"], st["place"]) == ("place", "wall")


def test_a_hallway_between_a_wall_and_low_furniture_stays_a_hallway(tmp_path):
    """Only "by the wall" is looked at again: the user's A/B kept corridors formed by furniture and a wall."""
    from rover_vlm.route_events import low_furniture
    frame = _scene(tmp_path, [(1.6, 1.75, 0.5, 4.5, 2.5), (3.25, 3.5, 1.5, 4.5, 0.8)], rendered=[BACK_WALL])
    assert low_furniture(frame, np.array(END[-1])) == {int(frame.lab[60, 67])}       # the sofa side is low
    st = stop_d3(frame, np.asarray(END, float))
    assert (st["type"], st["place"]) == ("place", "hallway")
