#!/usr/bin/env python3
"""Render one HM3D SEMANTIC frame per labelset sample, for the grounded-landmark name check.

For every sample behind data/prepared_habitat_instr_v2/labelset_{train,eval}.json whose
scene ships HM3D semantics, replay the SEMANTIC sensor at the sample's stored camera pose
(same world frame, 512x512, hfov 90, camera 0.8 m) and save
outputs/landmark_semantics/<sample>.npz: `ids` (instance id per pixel) + `categories`
(JSON {id: category}). Resumes: existing files are skipped.

Runs in the `habitat` conda env (NOT uv) on the full A100 (MIG slices cannot render):
    sbatch slurm/render_semantic_frames.sbatch
Imports the generator's roverbench.sim from ../spaceitup-habitat/src; writes nothing there.
Checked by `scripts/audit_instruction_labels.py --mode semantic`.
"""
import argparse
import json
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
HABITAT_SRC = REPO_ROOT.parent / "spaceitup-habitat" / "src"
HM3D = Path("/nfs/projects/spaceitup/rover_navigation/data/hm3d")
sys.path.insert(0, str(HABITAT_SRC))
from roverbench.sim import make_simulator, set_agent_pose  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=REPO_ROOT / "data" / "prepared_habitat_instr_v2")
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "outputs" / "landmark_semantics")
    ap.add_argument("--config", default=str(HM3D / "hm3d_annotated_basis.scene_dataset_config.json"))
    ap.add_argument("--res", type=int, default=512)
    ap.add_argument("--limit-scenes", type=int, default=0)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    by_scene = defaultdict(set)
    for split in ("train", "eval"):
        for r in json.loads((args.data / f"labelset_{split}.json").read_text()):
            by_scene[r["habitat_meta"]["scene_id"]].add(str(Path(r["image"][0]).parent))
    semantic = {p.parent.name for p in HM3D.glob("*/*/*.semantic.glb")}
    scenes = sorted(s for s in by_scene if s in semantic)
    if args.limit_scenes:
        scenes = scenes[: args.limit_scenes]
    print(f"{len(by_scene)} scenes in the labelsets, {len(scenes)} with HM3D semantics", flush=True)

    done = fail = 0
    t0 = time.time()
    for n, scene in enumerate(scenes, 1):
        todo = [Path(d) for d in sorted(by_scene[scene]) if not (args.out / f"{Path(d).name}.npz").exists()]
        if not todo:
            continue
        try:
            sim = make_simulator(scene.split("-", 1)[1], None, width=args.res, height=args.res,
                                 camera_height_m=0.8, hfov_deg=90.0, with_depth=False,
                                 with_semantic=True, scene_dataset_config=args.config)
        except Exception:
            fail += len(todo)
            print(f"[{scene}] SIM FAILED\n{traceback.format_exc()}", flush=True)
            continue
        try:
            cats = json.dumps({int(o.semantic_id): o.category.name()
                               for o in sim.semantic_scene.objects if o is not None})
            for d in todo:
                meta = json.loads((d / "meta.json").read_text())
                fh = meta["chord"]["floor_height"]
                set_agent_pose(sim, [meta["pose_xz"][0], fh, meta["pose_xz"][1]],
                               [meta["heading_xz"][0], 0.0, meta["heading_xz"][1]])
                ids = np.asarray(sim.get_sensor_observations()["semantic"])
                ids = ids.astype(np.uint16 if ids.max() < 65536 else np.uint32)
                np.savez_compressed(args.out / f"{d.name}.npz", ids=ids, categories=cats)
                done += 1
        finally:
            sim.close()
        print(f"[{n}/{len(scenes)}] {scene}: {len(todo)} frames  total {done}  "
              f"{(time.time() - t0) / 60:.1f} min", flush=True)
    print(f"done: {done} frames written, {fail} failed", flush=True)


if __name__ == "__main__":
    main()
