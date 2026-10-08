"""Plan steering routes on the scene navmesh (habitat env; Addendum A1).

Runs in the `habitat` conda env (habitat_sim 0.3.3, Python 3.9), NOT the uv .venv. Call the env's
python directly — `conda run` inside this repo picks up .python-version (pyenv 3.12) and fails:

    export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
    ~/miniforge3/envs/habitat/bin/python scripts/navmesh_plan.py \
        --requests data/prepared_habitat_instr_v3/steer/requests_eval.jsonl \
        --out data/prepared_habitat_instr_v3/steer/routes_eval.json

Per scene the navmesh is baked at agent radius --clear-m (CPU, create_renderer=False, via
spaceitup-habitat's roverbench.scene.navmesh.recompute_navmesh; ~1 s per scene) and cached under
outputs/navmesh_cache/. Each request's waypoints are snapped onto the navmesh (within --snap-m,
at the frame's floor height) and joined by one geodesic per leg. Writes
{key: {"route_xz": [[x, z], ...] | null, "reason": null | "off_navmesh" | "no_path" | "no_floor" | "no_navmesh"}}.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
HM3D_ROOT = Path("/nfs/projects/spaceitup/rover_navigation/data/hm3d")
ROVERBENCH_SRC = REPO_ROOT.parent / "spaceitup-habitat" / "src"
CACHE = REPO_ROOT / "outputs" / "navmesh_cache"
STEER = REPO_ROOT / "data" / "prepared_habitat_instr_v3" / "steer"
CLEAR_M = 0.30      # = rover_vlm.steering.CLEAR_M (this script runs outside the uv env)
SNAP_M = 0.3        # = rover_vlm.steering.SNAP_M
MAX_Y_DELTA = 0.5


def snap_xz(pf, xz, floor_y, tol=SNAP_M):
    """The navmesh point nearest to floor point xz, or None when it is farther than tol (or NaN)."""
    p = np.array([xz[0], floor_y, xz[1]], np.float32)
    s = np.asarray(pf.snap_point(p), float)
    if (not np.all(np.isfinite(s)) or abs(s[1] - floor_y) > MAX_Y_DELTA
            or np.hypot(s[0] - xz[0], s[2] - xz[1]) > tol):
        return None
    return s


def plan_request(pf, waypoints_xz, floor_y, find_path, tol=SNAP_M):
    """(route_xz, None) through the snapped waypoints, one geodesic per leg (find_path(pf, a, b) ->
    list of 3-D points or None); else (None, reason)."""
    if floor_y is None:
        return None, "no_floor"
    pts = [snap_xz(pf, w, floor_y, tol) for w in waypoints_xz]
    if any(p is None for p in pts):
        return None, "off_navmesh"
    out = []
    for a, b in zip(pts[:-1], pts[1:]):
        leg = find_path(pf, a, b)
        if leg is None or len(leg) < 2:
            return None, "no_path"
        leg = [[round(float(q[0]), 3), round(float(q[2]), 3)] for q in leg]
        out += leg if not out else leg[1:]
    return out, None


def plan_all(rows, get_pf, find_path, tol=SNAP_M):
    out = {}
    for row in rows:
        pf = get_pf(row["scene_id"])
        if pf is None:
            out[row["key"]] = {"route_xz": None, "reason": "no_navmesh"}
            continue
        route, why = plan_request(pf, row["waypoints_xz"], row.get("floor_y"), find_path, tol)
        out[row["key"]] = {"route_xz": route, "reason": why}
    return out


def find_scene_glb(root, scene_id):
    """The scene's render mesh (<split>/<scene_id>/*.basis.glb), first split alphabetically."""
    hits = sorted(Path(root).glob(f"*/{scene_id}/*.basis.glb"))
    return hits[0] if hits else None


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--requests", type=Path, default=STEER / "requests_eval.jsonl")
    p.add_argument("--out", type=Path, default=STEER / "routes_eval.json")
    p.add_argument("--clear-m", type=float, default=CLEAR_M)
    p.add_argument("--snap-m", type=float, default=SNAP_M)
    p.add_argument("--hm3d-root", type=Path, default=HM3D_ROOT)
    p.add_argument("--roverbench-src", type=Path, default=ROVERBENCH_SRC)
    p.add_argument("--cache", type=Path, default=CACHE)
    args = p.parse_args()
    sys.path.insert(0, str(args.roverbench_src))
    import habitat_sim
    from roverbench.scene.navmesh import load_pathfinder, recompute_navmesh

    args.cache.mkdir(parents=True, exist_ok=True)
    pfs = {}

    def get_pf(scene_id):
        if scene_id not in pfs:
            nav = args.cache / f"{scene_id}_r{int(round(args.clear_m * 100)):03d}.navmesh"
            if nav.exists():
                pfs[scene_id] = load_pathfinder(str(nav))
            else:
                glb = find_scene_glb(args.hm3d_root, scene_id)
                pfs[scene_id] = None if glb is None else recompute_navmesh(
                    str(glb), agent_radius=args.clear_m, out_navmesh=str(nav))
            print(f"scene {scene_id}: {'ok' if pfs[scene_id] is not None else 'NO NAVMESH'}", flush=True)
        return pfs[scene_id]

    def find_path(pf, a, b):
        sp = habitat_sim.ShortestPath()
        sp.requested_start = np.asarray(a, np.float32)
        sp.requested_end = np.asarray(b, np.float32)
        return [np.asarray(q, float) for q in sp.points] if pf.find_path(sp) else None

    rows = [json.loads(x) for x in args.requests.read_text().splitlines() if x.strip()]
    routes = plan_all(rows, get_pf, find_path, args.snap_m)
    args.out.write_text(json.dumps(routes))
    counts = Counter("ok" if v["reason"] is None else v["reason"] for v in routes.values())
    print(f"{len(routes)} requests: {dict(counts)} -> {args.out}")


if __name__ == "__main__":
    main()
