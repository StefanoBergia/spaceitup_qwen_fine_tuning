"""Build the instruction-conditioned dataset from the round-4 splits. CPU only.

Stage 1 — pairs (before labelling): every original round-4 frame with a feasible left and
right route that enter the frame at the same point becomes two records (template
instruction), plus the two overlay images per record that Cosmos is shown.

    uv run scripts/prepare_habitat_instr.py pairs
    uv run scripts/prepare_habitat_instr.py pairs --limit 50 --out-dir /tmp/instr_try

Stage 2 — finalize (after scripts/label_instructions.py): keep the pairs whose two
records both passed, and write matched template/Cosmos datasets (same ids, same order,
only the instruction differs).

    uv run scripts/prepare_habitat_instr.py finalize \
        --instructions outputs/instr_labels/train_i1.jsonl outputs/instr_labels/eval_i1.jsonl

v2 — the general model (every round-4 frame, original and yaw-augmented; see
docs/superpowers/specs/2026-09-25-instruction-general-model-design.md):

    uv run scripts/prepare_habitat_instr.py general            # plain copies + what to label
    sbatch slurm/label_instructions.sbatch                      # Cosmos, unattended
    uv run scripts/prepare_habitat_instr.py finalize-general \
        --instructions outputs/instr_labels_v2/train_i3.jsonl outputs/instr_labels_v2/eval_i3.jsonl

Grounded landmarks (2026-09-28, docs/superpowers/specs/2026-09-28-grounded-landmarks-design.md):
one geometry-chosen target per labelset record, before i4 labelling. CPU, ~40 min, 4 workers
of single-threaded numpy:

    export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
    uv run scripts/prepare_habitat_instr.py landmarks
    uv run scripts/prepare_habitat_instr.py landmarks --limit 200      # smoke run

Route descriptions (2026-09-30, docs/superpowers/specs/2026-09-30-route-descriptions-design.md):
events + stop phrase per labelset record (both routes of a pair), plus one object row per
frame object for Cosmos to name. CPU, 4 single-threaded workers:

    export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
    uv run scripts/prepare_habitat_instr.py events --limit 200     # smoke run (prints timing)
    uv run scripts/prepare_habitat_instr.py events
    # after slurm/label_descriptions.sbatch (STAGE=full)
    uv run scripts/prepare_habitat_instr.py finalize-descriptions

Steering eval (2026-10-07, docs/superpowers/specs/2026-10-07-steering-eval-design.md): eval-only
pairs planned on each d1 eval frame's own map (CPU, 4 single-threaded workers), checked on the
navmesh in the habitat env, then one pair per viewpoint per slice:

    export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
    uv run scripts/prepare_habitat_instr.py steer --limit 40     # smoke run
    uv run scripts/prepare_habitat_instr.py steer
    conda run -n habitat python scripts/navmesh_check.py         # see its docstring
    uv run scripts/prepare_habitat_instr.py steer-select
    # after STAGE=steer sbatch slurm/label_descriptions.sbatch
    uv run scripts/prepare_habitat_instr.py finalize-steer

The split is round 4's (data/prepared_habitat_v4), which is already scene-disjoint;
that is re-asserted here. Yaw-augmented frames are skipped: the candidate projections
belong to the unrotated camera.
"""

import argparse
import json
import statistics
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

from rover_vlm.descriptions import as_form, finalize_descriptions, shuffled_texts
from rover_vlm.instructions import (
    assert_scene_disjoint, build_frame, build_pair, build_wrong_side, finalize_general, finalize_pairs,
    render_check_image, render_generation_image, rescore_row, with_instruction,
)
from rover_vlm import steering
from rover_vlm.landmarks import select_target
from rover_vlm.route_events import EVENTS_SCHEMA, collect_objects, describe_route
from rover_vlm.steering import frame_pairs

REPO_ROOT = Path(__file__).resolve().parent.parent
SOURCE = REPO_ROOT / "data" / "prepared_habitat_v4"
OUT_DIR = REPO_ROOT / "data" / "prepared_habitat_instr_v1"
SPLITS = {"train": "train_full", "eval": "eval"}
# manifest fields carried over from round 4; everything path-specific is rebuilt per side
CARRY = ("sample_id", "label_side", "margin", "decidability", "near_symmetric",
         "n_candidates", "detour_ratio")


def build_pairs(source_recs, out_dir, reasons):
    out = []
    for i, src in enumerate(source_recs):
        hm = src["habitat_meta"]
        if hm.get("source") != "original":
            continue
        pair = build_pair(Path(src["image"][0]).parent, reasons=reasons,
                          meta_extra={k: hm.get(k) for k in CARRY})
        if pair is None:
            continue
        for rec in pair:
            gen = out_dir / "overlays" / f"{rec['id']}_gen.jpg"
            chk = out_dir / "overlays" / f"{rec['id']}_check.jpg"
            render_generation_image(rec, gen)
            rec["habitat_meta"].update({"gen_image": str(gen), "check_image": str(chk),
                                        "check_expected": render_check_image(rec, chk)})
        out.extend(pair)
        if (i + 1) % 2000 == 0:
            print(f"  {i + 1}/{len(source_recs)} scanned, {len(out) // 2} pairs", flush=True)
    return out


def stage_pairs(args):
    reasons, splits = {}, {}
    for name, src_name in SPLITS.items():
        src = json.loads((args.source / f"{src_name}.json").read_text())
        if args.limit:
            src = src[: args.limit]
        splits[name] = build_pairs(src, args.out_dir, reasons)
        print(f"{name}: {len(splits[name]) // 2} pairs from {len(src)} round-4 records")
    assert_scene_disjoint(splits["train"], splits["eval"])
    args.out_dir.mkdir(parents=True, exist_ok=True)
    stats = {}
    for name, recs in splits.items():
        (args.out_dir / f"pairs_{name}.json").write_text(json.dumps(recs, indent=1))
        lefts = [r["habitat_meta"] for r in recs if r["habitat_meta"]["side"] == "left"]
        gaps = [hm["entry_gap"] for hm in lefts]
        stats[name] = {
            "pairs": len(lefts),
            "scenes": len({hm["scene_id"] for hm in lefts}),
            # which side round 4 called correct -- the side an instruction-blind model drifts to
            "label_side": dict(Counter(hm.get("label_side") for hm in lefts)),
            "check_target_is_1": sum(r["habitat_meta"]["check_expected"] == 1 for r in recs),
            "entry_gap_zero": sum(g == 0 for g in gaps),
            "entry_gap_median": statistics.median(gaps) if gaps else None,
        }
    meta = {"source": str(args.source), "skipped": reasons, "stats": stats}
    (args.out_dir / "pairs_meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


def stage_finalize(args):
    rows = {}
    for path in args.instructions:
        for line in Path(path).read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                if not row.get("error"):
                    rows[row["id"]] = row
    meta = {"instructions": [str(p) for p in args.instructions], "splits": {}}
    for name in SPLITS:
        pairs = json.loads((args.out_dir / f"pairs_{name}.json").read_text())
        template, cosmos, stats = finalize_pairs(pairs, rows)
        (args.out_dir / f"{name}_template.json").write_text(json.dumps(template, indent=1))
        (args.out_dir / f"{name}_cosmos.json").write_text(json.dumps(cosmos, indent=1))
        meta["splits"][name] = {**stats, "records": len(cosmos)}
        print(f"{name}: kept {stats['pairs_kept']}/{stats['pairs_in']} pairs -> {len(cosmos)} records")
    (args.out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"wrote -> {args.out_dir}")


V2_OUT = REPO_ROOT / "data" / "prepared_habitat_instr_v2"
V3_OUT = REPO_ROOT / "data" / "prepared_habitat_instr_v3"
D1_LABELS = REPO_ROOT / "outputs" / "instr_labels_v3"
WRONG_SIDE_N, HANDWRITTEN_N, V2_SEED = 300, 30, 7
HANDWRITTEN = REPO_ROOT / "configs" / "instructions_handwritten_v2.json"


def stage_general(args):
    import random
    reasons, stats = {}, {}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, src_name in SPLITS.items():
        src = json.loads((args.source / f"{src_name}.json").read_text())
        if args.limit:
            src = src[: args.limit]
        plain, labelset, wrong = [], [], []
        for i, rec in enumerate(src):
            extra = {k: rec["habitat_meta"].get(k) for k in CARRY}
            f = build_frame(rec, meta_extra=extra, reasons=reasons)
            plain.append(f["plain"])
            labelset.extend(f["labelset"])
            # single-route (1 record) or direct (0) frames; build_wrong_side rejects the rest
            if name == "eval" and not f["labelset"][1:]:
                w = build_wrong_side(rec, meta_extra=extra)
                if w is not None:
                    wrong.append(w)
            if (i + 1) % 2000 == 0:
                print(f"  {name}: {i + 1}/{len(src)} frames, {len(labelset)} to label", flush=True)
        if name == "eval":
            # the wrong-side probe needs no Cosmos label (template wording, no drawn route),
            # so it is written here rather than labelled
            random.Random(V2_SEED).shuffle(wrong)
            (args.out_dir / "probe_wrong_side.json").write_text(json.dumps(wrong[:WRONG_SIDE_N], indent=1))
            stats["probe_wrong_side"] = {"candidates": len(wrong), "written": len(wrong[:WRONG_SIDE_N])}
            orig_pairs = sorted({r["habitat_meta"]["pair_id"] for r in labelset
                                 if r["habitat_meta"].get("pair_id") and r["habitat_meta"].get("source") == "original"})
            random.Random(V2_SEED).shuffle(orig_pairs)
            (args.out_dir / "handwritten_pairs.json").write_text(json.dumps(sorted(orig_pairs[:HANDWRITTEN_N]), indent=1))
        (args.out_dir / f"plain_{name}.json").write_text(json.dumps(plain, indent=1))
        (args.out_dir / f"labelset_{name}.json").write_text(json.dumps(labelset, indent=1))
        paired = [r["habitat_meta"] for r in labelset if r["habitat_meta"].get("pair_id")]
        stats[name] = {
            "plain": len(plain), "to_label": len(labelset),
            "kinds": dict(Counter(r["habitat_meta"]["kind"] for r in labelset)),
            "pairs_by_source": dict(Counter(hm["source"] for hm in paired if hm["side"] == "left")),
            "pair_viewpoints": len({hm.get("parent_sample_id") or hm["pair_id"] for hm in paired}),
            "no_alt_path": sum(r["habitat_meta"].get("alt_path") is None for r in labelset),
        }
        print(f"{name}: {json.dumps(stats[name])}", flush=True)
    train = json.loads((args.out_dir / "labelset_train.json").read_text())
    eval_ = json.loads((args.out_dir / "labelset_eval.json").read_text())
    assert_scene_disjoint(train, eval_)
    (args.out_dir / "general_meta.json").write_text(
        json.dumps({"source": str(args.source), "skipped": reasons, "stats": stats}, indent=2))
    print(f"skipped: {reasons}")


def _label_rows(paths):
    rows = {}
    for path in paths:
        for line in Path(path).read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                if not row.get("error"):
                    rows[row["id"]] = row
    return rows


def stage_finalize_general(args):
    # re-judge with the current text rules, so a rule added after labelling (e.g.
    # side:egocentric) applies without re-running Cosmos
    raw = _label_rows(args.instructions)
    rows = {i: rescore_row(r) for i, r in raw.items()}
    newly = Counter(x for i, r in rows.items() if raw[i].get("passed") and not r.get("passed")
                    for x in r["rejects"])
    print(f"rescore: {sum(raw[i].get('passed') and not r.get('passed') for i, r in rows.items())} "
          f"previously passed rows now rejected {dict(newly)}")
    meta = {"instructions": [str(p) for p in args.instructions]}
    # an unlabelled record is not a rejection: a label job that timed out part-way would
    # otherwise yield a silently shrunken training set (v2 review I3)
    for name in SPLITS:
        labels = json.loads((args.out_dir / f"labelset_{name}.json").read_text())
        missing = sum(r["id"] not in rows for r in labels)
        print(f"{name}: {missing} of {len(labels)} records have no label row")
        if missing > 0.01 * len(labels) and not args.allow_missing:
            raise SystemExit(f"{name}: {missing} records unlabelled (> 1%) — finish the label job "
                             "(re-submit slurm/label_instructions.sbatch) or pass --allow-missing")
    d = args.out_dir
    for name in SPLITS:
        plain = json.loads((d / f"plain_{name}.json").read_text())
        labels = json.loads((d / f"labelset_{name}.json").read_text())
        out = finalize_general(plain, labels, rows)
        meta[name] = {**out["stats"], "records": len(out["train_or_all"])}
        if name == "train":
            (d / "train_general.json").write_text(json.dumps(out["train_or_all"], indent=1))
        else:
            for fname, recs in (("eval_plain", plain), ("eval_instructed", out["instructed"]),
                                ("eval_pairs", out["pairs"])):
                (d / f"{fname}.json").write_text(json.dumps(recs, indent=1))
                meta[name][fname] = len(recs)
            if HANDWRITTEN.exists():
                # built from the template records, so the probe does not depend on Cosmos
                # having passed these pairs
                texts = json.loads(HANDWRITTEN.read_text())
                chosen = set(json.loads((d / "handwritten_pairs.json").read_text()))
                picked = [r for r in labels if r["habitat_meta"].get("pair_id") in chosen]
                missing = [r["id"] for r in picked if r["id"] not in texts]
                assert not missing, f"{HANDWRITTEN.name} lacks {missing[:4]}"
                hw = [with_instruction(r, texts[r["id"]], "handwritten") for r in picked]
                (d / "probe_handwritten.json").write_text(json.dumps(hw, indent=1))
                meta[name]["probe_handwritten"] = len(hw)
            else:
                print(f"WARNING: {HANDWRITTEN} missing — probe_handwritten.json not written")
        print(f"{name}: {json.dumps(meta[name])}")
    (d / "meta_v2.json").write_text(json.dumps(meta, indent=2))


LANDMARK_WORKERS = 4   # shared 96-core box: modest, and each worker is single-threaded numpy


def landmark_row(rec):
    """One landmarks JSONL row for a labelset record (see rover_vlm.landmarks.select_target)."""
    hm = rec["habitat_meta"]
    sample_dir = Path(rec["image"][0]).parent
    try:
        t = select_target(sample_dir, hm["candidate_index"], side=hm["side"])
    except Exception as e:  # noqa: BLE001 — one bad sample is a counted reason, not a dead stage
        t = {"target": "none", "reason": "error", "polygon": [], "bbox": None,
             "world_xz": None, "diag": {"error": f"{type(e).__name__}: {e}"}}
    return {"id": rec["id"], "sample_dir": str(sample_dir), "candidate_index": hm["candidate_index"],
            "side": hm["side"], "kind": hm.get("kind"), "pair_id": hm.get("pair_id"), **t}


def stage_landmarks(args):
    d = args.out_dir
    meta = {}
    for name in SPLITS:
        recs = json.loads((d / f"labelset_{name}.json").read_text())
        if args.limit:
            recs = recs[: args.limit]
        with Pool(LANDMARK_WORKERS) as pool:
            rows = pool.map(landmark_row, recs, chunksize=32)
        out = d / f"landmarks_{name}.jsonl"
        out.write_text("".join(json.dumps(r) + "\n" for r in rows))
        counts = Counter(r["target"] if r["target"] != "none" else f"none:{r['reason']}" for r in rows)
        pairs = Counter(r["target"] for r in rows if r["pair_id"])
        meta[name] = {"records": len(rows), "targets": dict(counts), "pair_targets": dict(pairs)}
        print(f"{name}: {json.dumps(meta[name])} -> {out}")
    (d / "landmarks_meta.json").write_text(json.dumps(meta, indent=2))


EVENT_WORKERS = 4


def event_row(rec):
    """One events row for a labelset record (rover_vlm.route_events.describe_route)."""
    hm = rec["habitat_meta"]
    sample_dir = Path(rec["image"][0]).parent
    try:
        d = describe_route(sample_dir, hm["candidate_index"])
    except Exception as e:  # noqa: BLE001 — one bad sample is a counted reason, not a dead stage
        d = {"reason": "error", "events": [], "stop": None, "objects": {},
             "diag": {"error": f"{type(e).__name__}: {e}"}}
    return {"id": rec["id"], "sample_dir": str(sample_dir), "image": rec["image"][0],
            "candidate_index": hm["candidate_index"], "kind": hm.get("kind"),
            "pair_id": hm.get("pair_id"), "sample_id": hm.get("sample_id"), **d}


def stage_events(args):
    import time
    d = args.out_dir
    meta = {"schema": EVENTS_SCHEMA}     # label_descriptions.py and the audit refuse files without it
    for name in SPLITS:
        recs = json.loads((d / f"labelset_{name}.json").read_text())
        if args.limit:
            recs = recs[: args.limit]
        t0 = time.time()
        with Pool(EVENT_WORKERS) as pool:
            rows = pool.map(event_row, recs, chunksize=16)
        rows, objects = collect_objects(rows)
        (d / f"events_{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        (d / f"objects_{name}.jsonl").write_text("".join(json.dumps(o) + "\n" for o in objects))
        ok = [r for r in rows if r["reason"] is None]
        pairs = Counter(r["pair_id"] for r in ok if r["pair_id"])
        meta[name] = {
            "records": len(rows), "described": len(ok),
            "reasons": dict(Counter(r["reason"] for r in rows if r["reason"])),
            "events_per_route": dict(sorted(Counter(len(r["events"]) for r in ok).items())),
            "event_types": dict(Counter(e["type"] for r in ok for e in r["events"])),
            "stop_types": dict(Counter(r["stop"]["type"] for r in ok)),
            "pairs_both_described": sum(1 for n in pairs.values() if n == 2),
            "objects": len(objects),
            "depth_fallback": round(sum(o["depth_fallback"] for o in objects) / max(1, len(objects)), 3),
            "seconds": round(time.time() - t0, 1),
        }
        print(f"{name}: {json.dumps(meta[name])}", flush=True)
    (d / "events_meta.json").write_text(json.dumps(meta, indent=2))


STEER_DIR = V3_OUT / "steer"
STEER_WORKERS = 4
_STEER = {}


def _steer_init(names, fallback, kinds):
    _STEER.update(names=names, fallback=fallback, kinds=kinds)


def steer_frame(item):
    """(candidate pairs, reason counts, error or None) for one eval frame (pool worker)."""
    try:
        pairs, reasons = frame_pairs(item["sample_dir"], item["start_uv"], _STEER["names"],
                                     _STEER["fallback"], _STEER["kinds"],
                                     steering.map_planner())  # replaced in Task 8
        floor_y = steering.floor_height(item["sample_dir"])
        for p in pairs:
            p.update({k: item[k] for k in ("sample_id", "viewpoint", "is_aug", "scene_id", "image", "sample_dir")},
                     floor_y=floor_y)
    except Exception as e:  # noqa: BLE001 — one bad frame is a counted reason, not a dead stage
        return [], {"error": 1}, f"{item['sample_id']}: {type(e).__name__}: {e}"
    return pairs, dict(reasons), None


def _jsonl(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


def stage_steer(args):
    import time
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    items = steering.steer_frames(json.loads(args.frames.read_text()))
    if args.limit:
        items = items[: args.limit]
    names, fallback, kinds = steering.load_names(_jsonl(args.names), _jsonl(args.fallback_names))
    t0 = time.time()
    with Pool(STEER_WORKERS, initializer=_steer_init, initargs=(names, fallback, kinds)) as pool:
        results = pool.map(steer_frame, items, chunksize=8)
    reasons, errors, n_frames = Counter(), [], {"side": set(), "dest": set()}
    with (out / "candidates_eval.jsonl").open("w") as f:
        for pairs, rs, err in results:
            reasons.update(rs)
            if err:
                errors.append(err)
            for p in pairs:
                n_frames[p["slice"]].add(p["viewpoint"])
                f.write(json.dumps(p) + "\n")
    meta = {"frames": len(items), "seconds": round(time.time() - t0, 1),
            "viewpoints_with_candidates": {k: len(v) for k, v in n_frames.items()},
            "reasons": dict(sorted(reasons.items())), "errors": errors[:20], "n_errors": len(errors)}
    (out / "steer_meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


def desc_path(labels: Path, split: str, suffix: str = "") -> Path:
    """Composed-description file of one split: desc_<split><suffix>_d1.jsonl (suffix "_mix" = NAMER=mix)."""
    return labels / f"desc_{split}{suffix}_d1.jsonl"


def stage_finalize_descriptions(args):
    import random
    args.out_dir.mkdir(parents=True, exist_ok=True)
    meta, splits = {}, {}
    for name in SPLITS:
        recs = json.loads((args.events_dir / f"labelset_{name}.json").read_text())
        events = {r["id"]: r for r in map(json.loads, (args.events_dir / f"events_{name}.jsonl").open())}
        objects = {o["object_id"]: o for o in map(json.loads, (args.events_dir / f"objects_{name}.jsonl").open())}
        desc = _label_rows([desc_path(args.labels, name, args.desc_suffix)])
        out = finalize_descriptions(recs, events, desc, objects)
        if out["stats"]["unlabelled"] > 0.01 * max(1, out["stats"]["records"] - out["stats"]["dropped_geometry"]) \
                and not args.allow_missing:
            raise SystemExit(f"{name}: {out['stats']['unlabelled']} described routes have no label row (> 1%) — "
                             "finish slurm/label_descriptions.sbatch or pass --allow-missing")
        splits[name] = out
        meta[name] = out["stats"]
        print(f"{name}: {json.dumps(out['stats'])}", flush=True)
    assert_scene_disjoint(splits["train"]["kept"], splits["eval"]["kept"])
    d = args.out_dir
    train = splits["train"]["kept"]
    random.Random(V2_SEED).shuffle(train)
    (d / "train_desc.json").write_text(json.dumps(train, indent=1))
    ev = splits["eval"]
    files = {"eval_desc_entry": [as_form(r, "entry") for r in ev["kept"]],
             "eval_desc_goal": [as_form(r, "goal") for r in ev["kept"]],
             "eval_desc_pairs": [as_form(r, "entry") for r in ev["pairs"]],
             "eval_desc_shuffled": shuffled_texts([as_form(r, "entry") for r in ev["kept"]], seed=V2_SEED)}
    for fname, rs in files.items():
        (d / f"{fname}.json").write_text(json.dumps(rs, indent=1))
        meta["eval"][fname] = len(rs)
    meta["train"]["train_desc"] = len(train)
    (d / "meta_v3.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="stage", required=True)
    a = sub.add_parser("pairs", help="build pairs + overlays (before labelling)")
    a.add_argument("--source", type=Path, default=SOURCE)
    a.add_argument("--out-dir", type=Path, default=OUT_DIR)
    a.add_argument("--limit", type=int, default=None, help="cap round-4 records per split (testing)")
    b = sub.add_parser("finalize", help="keep fully passed pairs (after labelling)")
    b.add_argument("--out-dir", type=Path, default=OUT_DIR)
    b.add_argument("--instructions", type=Path, nargs="+", required=True, help="label JSONL files")
    c = sub.add_parser("general", help="v2: plain copies + records to label for every round-4 frame")
    c.add_argument("--source", type=Path, default=SOURCE)
    c.add_argument("--out-dir", type=Path, default=V2_OUT)
    c.add_argument("--limit", type=int, default=None, help="cap round-4 records per split (testing)")
    e = sub.add_parser("finalize-general", help="v2: assemble train/eval files after labelling")
    e.add_argument("--out-dir", type=Path, default=V2_OUT)
    e.add_argument("--instructions", type=Path, nargs="+", required=True, help="label JSONL files")
    e.add_argument("--allow-missing", action="store_true",
                   help="assemble even if more than 1%% of records have no label row")
    g = sub.add_parser("landmarks", help="v2: grounded landmark target per labelset record (CPU)")
    g.add_argument("--out-dir", type=Path, default=V2_OUT)
    g.add_argument("--limit", type=int, default=0, help="first N records per split (smoke run)")
    h = sub.add_parser("events", help="d1: route events + stop phrase per labelset record (CPU)")
    h.add_argument("--out-dir", type=Path, default=V2_OUT)
    h.add_argument("--limit", type=int, default=0, help="first N records per split (smoke run)")
    k = sub.add_parser("finalize-descriptions", help="d1: train/eval files from labelled descriptions (CPU)")
    k.add_argument("--events-dir", type=Path, default=V2_OUT)
    k.add_argument("--labels", type=Path, default=D1_LABELS)
    k.add_argument("--out-dir", type=Path, default=V3_OUT)
    k.add_argument("--desc-suffix", default="",
                   help="desc file tag: \"_mix\" reads desc_<split>_mix_d1.jsonl (NAMER=mix); default desc_<split>_d1.jsonl")
    k.add_argument("--allow-missing", action="store_true")
    m = sub.add_parser("steer", help="steering: candidate pairs per d1 eval frame (CPU)")
    m.add_argument("--frames", type=Path, default=V3_OUT / "eval_desc_entry.json")
    m.add_argument("--names", type=Path, default=D1_LABELS / "names_eval_gemma_d1.jsonl")
    m.add_argument("--fallback-names", type=Path, default=D1_LABELS / "names_eval_d1.jsonl")
    m.add_argument("--out-dir", type=Path, default=STEER_DIR)
    m.add_argument("--limit", type=int, default=0, help="first N frames (smoke run)")
    args = p.parse_args()
    {"pairs": stage_pairs, "finalize": stage_finalize, "general": stage_general,
     "finalize-general": stage_finalize_general, "landmarks": stage_landmarks, "events": stage_events, "steer": stage_steer,
     "finalize-descriptions": stage_finalize_descriptions}[args.stage](args)


if __name__ == "__main__":
    main()
