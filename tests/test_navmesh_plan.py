"""navmesh_plan: request planning with a fake PathFinder (habitat_sim is not imported here)."""
import importlib.util
from pathlib import Path

import numpy as np

_SPEC = importlib.util.spec_from_file_location(
    "navmesh_plan", Path(__file__).resolve().parent.parent / "scripts" / "navmesh_plan.py")
npl = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(npl)


class FakePF:
    """Navigable floor: the square 0 <= x, z <= 2 at y = 1.0."""
    def snap_point(self, p):
        return np.array([min(max(p[0], 0), 2), 1.0, min(max(p[2], 0), 2)], np.float32)


class NanPF(FakePF):
    def snap_point(self, p):
        return np.array([np.nan, np.nan, np.nan], np.float32)


def straight(pf, a, b):
    return [a, (a + b) / 2, b]


def test_plan_request_joins_legs_and_reports_reasons():
    route, why = npl.plan_request(FakePF(), [(0.2, 0.2), (1.0, 1.8), (1.8, 0.2)], 1.0, straight)
    assert why is None and route[0] == [0.2, 0.2] and route[-1] == [1.8, 0.2]
    assert [1.0, 1.8] in route and len(route) == 5                      # legs joined, shared via kept once
    assert npl.plan_request(FakePF(), [(0.2, 0.2), (2.5, 1.0)], 1.0, straight) == (None, "off_navmesh")
    assert npl.plan_request(NanPF(), [(0.2, 0.2), (1.0, 1.0)], 1.0, straight) == (None, "off_navmesh")
    assert npl.plan_request(FakePF(), [(0.2, 0.2), (1.0, 1.0)], None, straight) == (None, "no_floor")
    assert npl.plan_request(FakePF(), [(0.2, 0.2), (1.0, 1.0)], 1.0, lambda pf, a, b: None) == (None, "no_path")


def test_plan_all_keys_and_missing_scene():
    rows = [{"key": "f|a", "scene_id": "s1", "floor_y": 1.0, "waypoints_xz": [[0.2, 0.2], [1.0, 1.0]]},
            {"key": "g|b", "scene_id": "gone", "floor_y": 1.0, "waypoints_xz": [[0.2, 0.2], [1.0, 1.0]]}]
    out = npl.plan_all(rows, lambda s: FakePF() if s == "s1" else None, straight)
    assert out["f|a"]["reason"] is None and out["f|a"]["route_xz"][-1] == [1.0, 1.0]
    assert out["g|b"] == {"route_xz": None, "reason": "no_navmesh"}


def test_find_scene_glb_skips_semantic(tmp_path):
    d = tmp_path / "val" / "00803-abc"
    d.mkdir(parents=True)
    (d / "abc.semantic.glb").write_text("")
    (d / "abc.basis.glb").write_text("")
    assert npl.find_scene_glb(tmp_path, "00803-abc") == d / "abc.basis.glb"
    assert npl.find_scene_glb(tmp_path, "nope") is None
