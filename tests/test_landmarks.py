"""Grounded landmarks: geometry on synthetic occupancy grids + one real-sample check."""
import json
from pathlib import Path

import numpy as np
import pytest

from rover_vlm.landmarks import (
    FREE, OCC, Camera, Grid, components, convex_hull, densify,
)


def test_camera_centre_and_sides():
    cam = Camera([0.0, 0.0], [0.0, 1.0])
    uv, fwd = cam.uv([[0.0, 2.0]])
    assert fwd[0] == pytest.approx(2.0)
    assert uv[0, 0] == pytest.approx(0.5)
    assert uv[0, 1] == pytest.approx(0.5 + 0.5 * 0.8 / 2.0)
    right, _ = cam.uv([cam.pos + 2.0 * cam.h + 0.5 * cam.right])
    left, _ = cam.uv([cam.pos + 2.0 * cam.h - 0.5 * cam.right])
    assert right[0, 0] > 0.5 > left[0, 0]
    top, _ = cam.uv([[0.0, 2.0]], height=0.8)
    assert top[0, 1] == pytest.approx(0.5)             # camera height -> horizon


def test_camera_in_frame():
    cam = Camera([0.0, 0.0], [0.0, 1.0])
    uv, fwd = cam.uv([[0.0, 2.0], [0.0, -2.0], [0.0, 0.1]])
    assert cam.in_frame(uv, fwd).tolist() == [True, False, False]


def test_grid_roundtrip_and_ray():
    s = np.full((40, 40), FREE, np.uint8)
    s[20, 10:30] = OCC                                   # wall at z = 1.0..1.05
    g = Grid(s, 0.05, 0.0, 0.0)
    rc = g.rc([[0.52, 1.02]])[0]
    assert (round(rc[0]), round(rc[1])) == (20, 10)
    assert np.allclose(g.xz([[20, 10]])[0], [0.525, 1.025])
    assert g.cell((20, 15)) == OCC and g.cell((-1, 0)) == 0
    assert g.first_occupied([1.0, 0.2], [1.0, 1.8]) == (20, 20)
    assert g.first_occupied([0.2, 0.2], [0.2, 1.8]) is None


def test_densify_components_hull():
    d = densify(np.array([[0.0, 0.0], [1.0, 0.0]]), 0.25)
    assert len(d) == 5 and np.allclose(d[-1], [1.0, 0.0])
    m = np.zeros((6, 6), bool); m[0:2, 0:2] = True; m[4:6, 4:6] = True
    lab, n = components(m)
    assert n == 2 and lab[0, 0] != lab[5, 5]
    h = convex_hull([(0, 0), (1, 0), (1, 1), (0, 1), (0.5, 0.5)])
    assert len(h) == 4 and (0.5, 0.5) not in h


from rover_vlm.landmarks import (  # noqa: E402
    REASONS, select_target,
)


def _write(d, state, route, *, flavor="detour", pose=(2.5, 0.2), heading=(0.0, 1.0), crossing=None):
    """A synthetic sample dir: 5 m x 5 m map at 5 cm, camera at `pose` facing +z."""
    d.mkdir(parents=True, exist_ok=True)
    np.savez(d / "state.npz", state=state, mpp=0.05, x_min=0.0, z_min=0.0)
    cands = [{"label": "left", "polyline_xz": [list(map(float, p)) for p in route]}]
    meta = {"pose_xz": list(pose), "heading_xz": list(heading), "flavor": flavor,
            "chord": {"goal": [route[-1][0], 0.0, route[-1][1]], "floor_height": 0.0,
                      "crossing_xz": crossing},
            "candidates": cands}
    (d / "meta.json").write_text(json.dumps(meta))
    return d


def _free():
    return np.full((100, 100), FREE, np.uint8)


# image-left for heading +z: right = (-h_z, h_x) = (-1, 0), so left = (+1, 0)
LEFT = np.array([1.0, 0.0])


def _box_route():
    return [(2.5, 0.2), tuple(np.array([2.5, 1.5]) + 0.9 * LEFT),
            tuple(np.array([2.5, 3.5]) + 0.9 * LEFT), (2.5, 4.6)]


def test_object_the_route_bends_around(tmp_path):
    s = _free()
    s[45:56, 45:56] = OCC                                # 0.55 m box centred (2.5, 2.5)
    t = select_target(_write(tmp_path / "s", s, _box_route()), 0)
    assert t["target"] == "object" and t["reason"] is None
    assert np.allclose(t["world_xz"], [2.5, 2.5], atol=0.3)
    assert len(t["polygon"]) >= 3


def test_nearest_obstacle_passed_not_the_largest(tmp_path):
    s = _free()
    s[45:56, 45:56] = OCC                                # box 0.6 m inside the route
    s[28:32, 60:62] = OCC                                # small post x 3.0..3.1, z 1.4..1.6, 0.3 m inside
    t = select_target(_write(tmp_path / "s", s, _box_route()), 0)
    assert t["target"] == "object"
    assert np.allclose(t["world_xz"], [3.05, 1.5], atol=0.15)


def test_obstacle_on_the_outside_of_the_bend_is_not_a_target(tmp_path):
    s = _free()
    s[20:80, 74:76] = OCC                                # wall x 3.7..3.8 alongside, outside the bend
    t = select_target(_write(tmp_path / "s", s, _box_route()), 0)
    assert (t["target"], t["reason"]) == ("none", "no_visible_target")


def test_close_obstacle_counts_although_its_floor_is_below_the_image(tmp_path):
    s = _free()
    s[11:13, 45:47] = OCC                                # pillar x 2.25..2.35, z 0.55..0.65 (0.4 m ahead)
    route = [(2.5, 0.2), (3.4, 1.5), (3.4, 3.5), (2.5, 4.6)]
    t = select_target(_write(tmp_path / "s", s, route), 0)
    assert t["target"] == "object"
    assert np.allclose(t["world_xz"], [2.3, 0.6], atol=0.1)
    assert t["bbox"][1] < 1.0                            # its column reaches into the frame


def test_hidden_obstacle_skipped_for_the_one_in_front(tmp_path):
    s = _free()
    s[45:56, 50:56] = OCC                                # box x 2.5..2.8, z 2.25..2.8
    s[20:22, 51:60] = OCC                                # screen x 2.55..3.0 at z 1.0, in front of it
    t = select_target(_write(tmp_path / "s", s, _box_route()), 0)
    assert t["target"] == "object"
    assert 0.9 < t["world_xz"][1] < 1.2                  # the screen, not the box behind it


def test_bends_both_ways_obstacle_passed_on_the_labels_side(tmp_path):
    # the route first swings right (x 2.2), then bows left overall (label "left" = +x). The side
    # word must be true for the named obstacle: a post the route passes with the post on its
    # right (route goes left of it) is a target; one passed the other way round is not.
    route = [(2.5, 0.2), (2.2, 1.0), (2.2, 1.6), (3.4, 2.6), (3.4, 3.6), (2.5, 4.6)]
    s = _free()
    s[24:28, 38:40] = OCC                                # post x 1.9..2.0, z 1.2..1.4: on the route's right
    t = select_target(_write(tmp_path / "a", s, route), 0)
    assert t["target"] == "object" and np.allclose(t["world_xz"], [1.95, 1.3], atol=0.1)
    s = _free()
    s[24:28, 48:50] = OCC                                # post x 2.4..2.5: on the route's left
    t = select_target(_write(tmp_path / "b", s, route), 0)
    assert (t["target"], t["reason"]) == ("none", "no_visible_target")


def test_nothing_visible(tmp_path):
    route = [(2.5, 0.2), (3.4, 2.0), (2.5, 4.6)]
    t = select_target(_write(tmp_path / "s", _free(), route), 0)
    assert (t["target"], t["reason"]) == ("none", "no_visible_target")


def test_bad_inputs_are_reasons_not_crashes(tmp_path):
    d = _write(tmp_path / "s", _free(), [(2.5, 0.2), (2.5, 4.6)])
    assert select_target(d, 7)["reason"] == "bad_candidate"
    (d / "state.npz").unlink()
    assert select_target(d, 0)["reason"] == "no_state"
    assert set(REASONS) == {"bad_candidate", "no_state", "no_visible_target", "target_too_small"}


REAL = Path("/nfs/projects/spaceitup/rover_navigation/data/habitat_generated/train/"
            "00366-fxbzYAGkrtm/samples/00366-fxbzYAGkrtm_c012")


@pytest.mark.skipif(not REAL.exists(), reason="shared Habitat dataset not mounted")
def test_projection_matches_stored_goal_pixel():
    meta = json.loads((REAL / "meta.json").read_text())
    fpv = json.loads((REAL / "fpv_paths.json").read_text())
    g = np.array(meta["chord"]["goal"])[[0, 2]]
    uv, _ = Camera(meta["pose_xz"], meta["heading_xz"]).uv([g])
    W, H = fpv["image_size"]
    assert uv[0, 0] * W == pytest.approx(fpv["goal"]["uv"][0], abs=1.0)
    assert uv[0, 1] * H == pytest.approx(fpv["goal"]["uv"][1], abs=1.0)


def test_landmark_row_carries_record_fields(tmp_path):
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "prep", Path(__file__).resolve().parent.parent / "scripts" / "prepare_habitat_instr.py")
    prep = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prep)
    s = _free()
    s[45:56, 45:56] = OCC
    d = _write(tmp_path / "scene_c000", s, _box_route())
    rec = {"id": "scene_c000_L", "image": [str(d / "fpv_enhanced.png")],
           "habitat_meta": {"candidate_index": 0, "side": "left", "kind": "instructed", "pair_id": None}}
    row = prep.landmark_row(rec)
    assert row["id"] == "scene_c000_L" and row["side"] == "left" and row["target"] == "object"
    json.dumps(row)                                      # serialisable as a JSONL row


from rover_vlm.landmark_check import (  # noqa: E402
    OPENING_WORDS, beside_categories, canonical, name_matches, target_category,
)


def test_name_matching():
    assert canonical("couch") == canonical("sofa")
    assert name_matches("grey sofa", "couch")
    assert name_matches("kitchen island", "kitchen island")
    assert name_matches("wooden cabinet", "cabinet")
    assert not name_matches("armchair", "table")
    assert {"doorway", "hallway"} <= OPENING_WORDS


def test_target_category_majority_inside_polygon():
    ids = np.zeros((10, 10), np.uint16)
    ids[2:8, 2:8] = 5
    ids[0:10, 0:2] = 9
    cats = json.dumps({"5": "couch", "9": "wall", "0": "floor"})
    assert target_category(ids, cats, [[0.2, 0.2], [0.8, 0.2], [0.8, 0.8], [0.2, 0.8]]) == "couch"
    assert "wall" in beside_categories(ids, cats, [0.2, 0.2, 0.8, 0.8])


def test_semantic_report_counts_agreement_and_mismatches(tmp_path):
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "audit", Path(__file__).resolve().parent.parent / "scripts" / "audit_instruction_labels.py")
    audit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(audit)
    ids = np.zeros((10, 10), np.uint16)
    ids[2:8, 2:8] = 5
    np.savez(tmp_path / "s_c000.npz", ids=ids, categories=json.dumps({"5": "couch", "0": "floor"}))
    box = [[0.2, 0.2], [0.8, 0.2], [0.8, 0.8], [0.2, 0.8]]
    lms = {i: {"sample_dir": "/x/s_c000", "target": "object", "polygon": box, "bbox": [0.2, 0.2, 0.8, 0.8]}
           for i in ("a", "b")}
    lms["c"] = {"sample_dir": "/x/other_c001", "target": "object", "polygon": box, "bbox": None}
    rows = [{"id": "a", "passed": True, "reply": {"name": "grey sofa"}, "pair_id": None},
            {"id": "b", "passed": True, "reply": {"name": "bookshelf"}, "pair_id": "p"},
            {"id": "c", "passed": True, "reply": {"name": "sofa"}, "pair_id": None}]
    report, mism = audit.semantic_report(rows, lms, tmp_path)
    assert report["cells"]["object/single"] == {"n": 1, "agree": 1, "rate": 1.0}
    assert report["cells"]["object/pair"]["agree"] == 0
    assert report["no_semantic_frame"] == 1
    assert [m["id"] for m in mism] == ["b"] and mism[0]["hm3d"] == "couch"


def test_target_category_skips_the_wall_behind_an_object():
    # review: an object outline (floor to 1 m) often covers more wall than object
    ids = np.full((10, 10), 9, np.uint16)
    ids[5:8, 3:7] = 5
    cats = json.dumps({"5": "couch", "9": "wall"})
    box = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
    assert target_category(ids, cats, box) == "couch"
    assert target_category(ids, cats, box, name="white wall") == "wall"   # the name is the wall
