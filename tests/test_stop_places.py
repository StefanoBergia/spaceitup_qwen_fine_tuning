"""Stop places from the occupancy grid (d3, spec docs/superpowers/specs/2026-11-10-stop-phrases-d3-design.md).

Scenes: a 100 x 100 grid at 0.05 m (5 m x 5 m; row <- z, col <- x), route straight along +z from (2.5, 0.2).
"""

import numpy as np

from rover_vlm.landmarks import FREE, OCC, UNKNOWN, Grid
from rover_vlm.stop_places import PLACES, _walls, place_at, ray_hits


def _free():
    return np.full((100, 100), FREE, np.uint8)


def _route(goal_z):
    return [(2.5, 0.2), (2.5, goal_z)]


def _place(state, goal_z):
    out = place_at(Grid(state, 0.05, 0.0, 0.0), _route(goal_z))
    return None if out is None else out["place"]


def _corridor(x_left_col, x_right_col, end_row=80):
    s = _free()
    s[:end_row, x_left_col] = OCC
    s[:end_row, x_right_col] = OCC
    s[end_row:end_row + 3, x_left_col:x_right_col + 1] = OCC        # the end wall
    return s


def _door_wall():
    s = _free()
    s[40:43, :] = OCC                 # wall z 2.0-2.15 across the room
    s[40:43, 40:60] = FREE            # 1.0 m gap x 2.0-3.0
    return s


def test_room_middle_in_open_space():
    assert _place(_free(), 2.5) == "room_middle"


def test_room_middle_needs_known_free_space():                      # Review Focus 1
    s = np.full((100, 100), UNKNOWN, np.uint8)
    s[:, 46:54] = FREE                                               # only a 0.4 m strip was seen
    assert _place(s, 2.5) is None


def test_goal_at_grid_edge_is_safe():                                # Review Focus 2
    out = place_at(Grid(_free(), 0.05, 0.0, 0.0), _route(4.95))
    assert out is None or out["place"] in PLACES


def test_wall_beside_the_goal():
    s = _free()
    s[:, 32:35] = OCC                                                # wall x 1.6-1.75, 0.78 m right of the goal
    assert _place(s, 2.5) == "wall"


def test_corner():
    s = _free()
    s[:, 32:35] = OCC                                                # wall on the right, 0.78 m
    s[66:69, :] = OCC                                                # wall ahead, 0.8 m
    assert _place(s, 2.5) == "corner"


def test_hallway_end():
    assert _place(_corridor(35, 65), 3.0) == "hallway_end"          # 1.5 m wide, end wall 1.0 m ahead


def test_dead_end_corridor_is_not_a_corner():
    assert _place(_corridor(35, 65), 3.0) != "corner"


def test_wide_corridor_is_not_a_hallway():
    assert _place(_corridor(20, 80), 3.0) != "hallway_end"          # 3.0 m wide


def test_doorway():
    assert _place(_door_wall(), 2.05) == "doorway"


def test_past_the_doorway_is_not_a_doorway():
    assert _place(_door_wall(), 3.0) != "doorway"


def test_gap_behind_the_goal_is_not_a_corner():
    """Two wall stretches either side of a gap behind the goal are one wall with a hole, not a corner."""
    assert _place(_door_wall(), 3.0) != "corner"


def test_one_flat_wall_is_not_a_corner():
    s = _free()
    s[:, 40:43] = OCC                                                # wall x 2.0-2.15, 0.35 m right
    assert _place(s, 2.5) != "corner"


def test_ray_hits_reports_unknown_not_free():
    s = _free()
    s[60:70, :] = UNKNOWN                                            # unseen band from z 3.0
    hits = ray_hits(Grid(s, 0.05, 0.0, 0.0), (2.5, 2.5), (0.0, 1.0))
    ahead = [h for h in hits if abs(h[0]) < 1e-6][0]
    assert ahead[1] == "unknown" and 0.4 < ahead[2] < 0.6


def test_ray_hits_angle_convention():
    s = _free()
    s[:, 32:35] = OCC                                                # wall at -x = the right of a +z heading
    hits = {round(a): (o, d) for a, o, d in ray_hits(Grid(s, 0.05, 0.0, 0.0), (2.5, 2.5), (0.0, 1.0))}
    assert hits[90][0] == "occ" and hits[-90][0] != "occ"


def test_measures_are_small_and_rounded():
    for state, z in ((_free(), 2.5), (_corridor(35, 65), 3.0), (_door_wall(), 2.05)):
        m = place_at(Grid(state, 0.05, 0.0, 0.0), _route(z))["measures"]
        assert len(m) <= 6 and all(isinstance(v, float) for v in m.values())


def test_doorway_within_reach_before_the_gap():                       # final review I2a: spec "within ~0.75 m"
    assert _place(_door_wall(), 1.5) == "doorway"                     # 0.5 m before the wall


def test_doorway_needs_known_walls_around_the_gap():                  # final review I2b: unknown is no evidence
    s = np.full((100, 100), UNKNOWN, np.uint8)
    s[:, 39:62] = FREE                                                 # a 1.2 m wide strip seen along the route
    s[56:65, 38] = OCC                                                 # its side walls known only for z 2.8-3.25
    s[56:65, 62] = OCC
    assert _place(s, 3.0) != "doorway"


def test_a_plateau_across_the_fan_wrap_is_one_wall():                 # final review M1
    hits = [(22.5 * k - (360.0 if 22.5 * k > 180 else 0.0), "occ", 0.9) for k in range(16)]
    hits[0], hits[15] = (0.0, "occ", 0.7), (-22.5, "occ", 0.7)          # ahead and 22.5 deg left: the same flat wall
    assert len(_walls(hits, 3.0)) == 1


def test_opening_in_the_wall_ahead_is_not_a_corner():                 # final review M2
    s = _free()
    s[68:71, :] = OCC                                                  # wall ahead z 3.4-3.55, 0.9 m from the goal ...
    s[68:71, 38:62] = FREE                                             # ... with a 1.2 m opening
    s[92:95, :] = OCC                                                  # the next room's wall, 2.1 m from the goal
    assert _place(s, 2.5) != "corner"
