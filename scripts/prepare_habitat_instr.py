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

Steering eval (2026-10-07, docs/superpowers/specs/2026-10-07-steering-eval-design.md, Addendum A2):
eval-only stop pairs: each d1 eval frame's ground-truth route cut at two points whose stop phrases
differ. The defaults of steer / steer-select / finalize-steer / steer-objects are the d2 round's
(frames data/prepared_habitat_instr_v4/eval_desc_entry.json, names *_d2.jsonl, steer dir
data/prepared_habitat_instr_v4/steer); d1's set lives in data/prepared_habitat_instr_v3/steer (pass
explicit paths to redo it). CPU, 4 workers:

    export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
    uv run scripts/prepare_habitat_instr.py steer-objects
    STAGE=steer_names sbatch slurm/label_descriptions.sbatch            # GPU, Gemma
    uv run scripts/prepare_habitat_instr.py steer
    uv run scripts/prepare_habitat_instr.py steer-select
    # after STAGE=steer sbatch slurm/label_descriptions.sbatch
    uv run scripts/prepare_habitat_instr.py finalize-steer

The split is round 4's (data/prepared_habitat_v4), which is already scene-disjoint;
that is re-asserted here. Yaw-augmented frames are skipped: the candidate projections
belong to the unrotated camera.

Turn events and d2 (2026-10-13, docs/superpowers/specs/2026-10-13-turn-events-d2-design.md): events into v4 from
the v2 labelsets, seed the d2 label files from d1, label (GPU), finalize. CPU, 4 workers:

    export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
    uv run scripts/prepare_habitat_instr.py events --labelset-dir data/prepared_habitat_instr_v2 \
        --out-dir data/prepared_habitat_instr_v4
    uv run scripts/prepare_habitat_instr.py d2-seed
    ROUND=d2 DATA=data/prepared_habitat_instr_v4 NAMER=mix sbatch slurm/label_descriptions.sbatch   # gate
    uv run scripts/prepare_habitat_instr.py finalize-descriptions --round d2 --desc-suffix _mix

d2 endings (Addendum T2, 2026-10-14): rerun events into v4 (--force: the schema changed to d2-endings), check the
GT ceiling of the endings, re-seed:

    uv run scripts/prepare_habitat_instr.py events --labelset-dir data/prepared_habitat_instr_v2 \
        --out-dir data/prepared_habitat_instr_v4 --force
    uv run scripts/prepare_habitat_instr.py endings-check      # CEILING: behind >= 0.95
    uv run scripts/prepare_habitat_instr.py d2-seed --force
"""

import argparse
import json
import statistics
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

from rover_vlm.descriptions import (
    as_form, finalize_descriptions, route_objects, seed_d2_labels, shuffled_texts, d2_rejects)
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
V4_OUT = REPO_ROOT / "data" / "prepared_habitat_instr_v4"          # d2 (spec 2026-10-13)
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
            "pair_id": hm.get("pair_id"), "sample_id": hm.get("sample_id"),
            "fpv_flags": (sample_dir / "fpv_paths.json").exists(), **d}


def stage_events(args):
    import time
    d = args.out_dir
    meta_path = d / "events_meta.json"
    if meta_path.exists() and not args.force:
        old = json.loads(meta_path.read_text()).get("schema")
        if old != EVENTS_SCHEMA:
            raise SystemExit(f"refusing to overwrite {d}: its events are schema {old} (d1); write d2 events to "
                             "data/prepared_habitat_instr_v4 or pass --force")
    d.mkdir(parents=True, exist_ok=True)
    src = args.labelset_dir or d
    meta = {"schema": EVENTS_SCHEMA, "labelset_dir": str(src)}     # label_descriptions.py and the audit refuse files without it
    for name in SPLITS:
        recs = json.loads((src / f"labelset_{name}.json").read_text())
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
            "fpv_missing": sum(not r.get("fpv_flags") for r in rows),
            "reanchored": sum(bool((r.get("stop") or {}).get("reanchored")) for r in ok),
            "objects": len(objects),
            "depth_fallback": round(sum(o["depth_fallback"] for o in objects) / max(1, len(objects)), 3),
            "seconds": round(time.time() - t0, 1),
        }
        print(f"{name}: {json.dumps(meta[name])}", flush=True)
    (d / "events_meta.json").write_text(json.dumps(meta, indent=2))


STEER_DIR = V4_OUT / "steer"      # d2; d1's set lives in data/prepared_habitat_instr_v3/steer
STEER_WORKERS = 4
_STEER = {}


def _steer_init(names, fallback, kinds):
    _STEER.update(names=names, fallback=fallback, kinds=kinds)


def steer_frame(item):
    """(candidate pairs, reason counts, error or None) for one eval frame (pool worker)."""
    try:
        pairs, reasons = frame_pairs(item["sample_dir"], item["candidate_index"], _STEER["names"],
                                     _STEER["fallback"], _STEER["kinds"])
        for p in pairs:
            p.update({k: item[k] for k in ("sample_id", "viewpoint", "is_aug", "scene_id", "image", "sample_dir")})
    except Exception as e:  # noqa: BLE001 — one bad frame is a counted reason, not a dead stage
        return [], {"error": 1}, f"{item['sample_id']}: {type(e).__name__}: {e}"
    return pairs, dict(reasons), None


def _jsonl(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


def steer_objects_frame(args):
    item, skip = args
    try:
        return steering.object_rows(item["sample_dir"], item["image"], skip), None
    except Exception as e:  # noqa: BLE001 — one bad frame is a counted error, not a dead stage
        return [], f"{item['sample_id']}: {type(e).__name__}: {e}"


def stage_steer_objects(args):
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    items = steering.steer_frames(json.loads(args.frames.read_text()))
    if args.limit:
        items = items[: args.limit]
    named = {r["id"] for p in (args.names, args.fallback_names) for r in _jsonl(p) if r.get("passed")}
    with Pool(STEER_WORKERS) as pool:
        results = pool.map(steer_objects_frame, [(i, named) for i in items], chunksize=8)
    rows = [r for rs, _ in results for r in rs]
    errors = [e for _, e in results if e]
    (out / "objects_eval.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    meta = {"frames": len(items), "objects": len(rows), "already_named": len(named), "errors": errors[:20],
            "n_errors": len(errors)}
    (out / "objects_meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


def _steer_names(args):
    """(primary name rows, (names, fallback, kinds)): primary = d1 Gemma + steer Gemma rows."""
    prim = _jsonl(args.names)
    if args.steer_names.exists():
        prim += _jsonl(args.steer_names)
    else:
        print(f"WARNING: {args.steer_names} missing — only d1's names (run STAGE=steer_names first)")
    return prim, steering.load_names(prim, _jsonl(args.fallback_names))


def _run_steer_pool(args):
    items = steering.steer_frames(json.loads(args.frames.read_text()))
    if args.limit:
        items = items[: args.limit]
    prim, (names, fallback, kinds) = _steer_names(args)
    with Pool(STEER_WORKERS, initializer=_steer_init, initargs=(names, fallback, kinds)) as pool:
        return items, prim, pool.map(steer_frame, items, chunksize=8)


def stage_steer(args):
    import time
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    items, prim, results = _run_steer_pool(args)
    (out / "names_primary_eval.jsonl").write_text("".join(json.dumps(r) + "\n" for r in prim))
    reasons, errors, n_frames = Counter(), [], {}
    with (out / "candidates_eval.jsonl").open("w") as f:
        for pairs, rs, err in results:
            reasons.update(rs)
            if err:
                errors.append(err)
            for p in pairs:
                n_frames.setdefault(p["slice"], set()).add(p["viewpoint"])
                f.write(json.dumps(p) + "\n")
    meta = {"frames": len(items), "seconds": round(time.time() - t0, 1),
            "viewpoints_with_candidates": {k: len(v) for k, v in n_frames.items()},
            "reasons": dict(sorted(reasons.items())), "errors": errors[:20], "n_errors": len(errors)}
    (out / "steer_meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


def stage_steer_select(args):
    d = args.out_dir
    cands = _jsonl(d / "candidates_eval.jsonl")
    kept = steering.select_pairs(cands)
    rows = [row for p in kept for row in steering.pair_rows(p)]
    (d / "events_steer_eval.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    per = Counter(p["slice"] for p in kept)
    meta = {"schema": EVENTS_SCHEMA, "candidates": len(cands), "pairs": dict(per), "routes": len(rows),
            "originals": dict(Counter(p["slice"] for p in kept if not p["is_aug"])),
            "prefix_events": sum(bool(p["prefix_events"]) for p in kept)}
    (d / "events_meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))
    n, t = len(kept), per.get("turn", 0)
    ok = n >= steering.GATE0_MIN_PAIRS and t >= steering.GATE0_MIN_TURN_PAIRS
    print(f"GATE 0: {n} pairs ({t} turn, {per.get('stop', 0)} stop) -> {'PASS' if ok else 'FAIL'} "
          f"(need >= {steering.GATE0_MIN_PAIRS} and >= {steering.GATE0_MIN_TURN_PAIRS} turn)")


def desc_path(labels: Path, split: str, suffix: str = "", round_: str = "d1") -> Path:
    """Composed-description file of one split: desc_<split><suffix>_<round>.jsonl (suffix "_mix" = NAMER=mix)."""
    return labels / f"desc_{split}{suffix}_{round_}.jsonl"


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


COMPOSE_PER_MIN = 14     # Cosmos compose throughput on 3g.40gb (job 92229)


def stage_d2_seed(args):
    """d2: seed the d2 label files with the d1 rows that still hold, so the label job (--resume) names only new
    objects and composes only changed routes; behind endings of unchanged routes are rewritten from the d1 text on CPU
    (Addendum T2); write the changed eval routes for the gate (spec "Seeding")."""
    files = {s: {"gemma": args.labels / f"names_{s}_gemma_d2.jsonl", "cosmos": args.labels / f"names_{s}_d2.jsonl",
                 "desc": desc_path(args.labels, s, "_mix", "d2")} for s in SPLITS}
    existing = [str(p) for f in files.values() for p in f.values() if p.exists() and p.stat().st_size]
    if existing and not args.force:
        raise SystemExit(f"refusing to overwrite d2 label files (they may hold GPU work): {existing[:3]} — "
                         "pass --force only if they are seeds you mean to redo")
    meta = {}
    for s in SPLITS:
        new_ev = _jsonl(args.events_dir / f"events_{s}.jsonl")
        new_obj = _jsonl(args.events_dir / f"objects_{s}.jsonl")
        seed = seed_d2_labels(_jsonl(args.d1_events_dir / f"events_{s}.jsonl"), new_ev, new_obj,
                              {"gemma": _jsonl(args.labels / f"names_{s}_gemma_d1.jsonl"),
                               "cosmos": _jsonl(args.labels / f"names_{s}_d1.jsonl")},
                              _jsonl(desc_path(args.labels, s, "_mix")))
        _write_jsonl(files[s]["gemma"], seed["names"]["gemma"])
        _write_jsonl(files[s]["cosmos"], seed["names"]["cosmos"])
        _write_jsonl(files[s]["desc"], seed["desc"])
        wanted = {o for r in seed["changed"] for o in route_objects(r)}
        if s == "eval":
            _write_jsonl(args.events_dir / "events_eval_changed.jsonl", seed["changed"])
            _write_jsonl(args.events_dir / "objects_eval_changed.jsonl",
                         [o for o in new_obj if o["object_id"] in wanted])
        named = {r["id"] for r in seed["names"]["gemma"] if r.get("passed")}
        meta[s] = {"described": sum(r.get("reason") is None for r in new_ev),
                   "reused": sum(r.get("reused_from") == "d1" for r in seed["desc"]),
                   "rewritten": sum(r.get("rewritten_from") == "d1" for r in seed["desc"]),
                   "rewrite_failed": seed["rewrite_failed"],
                   "to_compose": len(seed["changed"]),
                   "changed_with_turn": sum(any(e["type"] == "turn" for e in r["events"]) for r in seed["changed"]),
                   "objects": len(new_obj), "objects_named_d1_gemma": len(named),
                   "objects_to_name": len(wanted - named),
                   "est_compose_hours": round(len(seed["changed"]) / COMPOSE_PER_MIN / 60, 1)}
    (args.events_dir / "seed_meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


def stage_finalize_descriptions(args):
    import random
    d2 = args.round != "d1"
    events_dir = args.events_dir or (V4_OUT if d2 else V2_OUT)
    labelset_dir = args.labelset_dir or V2_OUT
    out_dir = args.out_dir or (V4_OUT if d2 else V3_OUT)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta, splits = {}, {}
    for name in SPLITS:
        recs = json.loads((labelset_dir / f"labelset_{name}.json").read_text())
        events = {r["id"]: r for r in map(json.loads, (events_dir / f"events_{name}.jsonl").open())}
        objects = {o["object_id"]: o for o in map(json.loads, (events_dir / f"objects_{name}.jsonl").open())}
        desc = _label_rows([desc_path(args.labels, name, args.desc_suffix, args.round)])
        out = finalize_descriptions(recs, events, desc, objects, recheck=d2_rejects if d2 else None)
        if out["stats"]["unlabelled"] > 0.01 * max(1, out["stats"]["records"] - out["stats"]["dropped_geometry"]) \
                and not args.allow_missing:
            raise SystemExit(f"{name}: {out['stats']['unlabelled']} described routes have no label row (> 1%) — "
                             "finish slurm/label_descriptions.sbatch or pass --allow-missing")
        splits[name] = out
        meta[name] = out["stats"]
        print(f"{name}: {json.dumps(out['stats'])}", flush=True)
    assert_scene_disjoint(splits["train"]["kept"], splits["eval"]["kept"])
    d = out_dir
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
    (d / ("meta_d2.json" if d2 else "meta_v3.json")).write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


GATE2_MIN_COMPOSED = 0.85


def stage_finalize_steer(args):
    """Assemble the 4 eval_steer_stop_* files from steering rows + compose rows; print Gate 2."""
    d = args.steer_dir
    base = {}
    for r in json.loads(args.frames.read_text()):
        base.setdefault(r["habitat_meta"]["sample_id"], r)
    rows = [json.loads(x) for x in (d / "events_steer_eval.jsonl").read_text().splitlines() if x.strip()]
    out = steering.finalize_steer(base, rows, _label_rows([args.desc]) if args.desc.exists() else {})
    stats = out["stats"]
    pairs = sum(v for k, v in stats.items() if k.endswith(":pairs"))
    missing = sum(v for k, v in stats.items() if k.endswith(":unlabelled"))
    if missing > 0.01 * max(1, pairs) and not args.allow_missing:
        raise SystemExit(f"{missing} of {pairs} pairs have no compose row (> 1%) — finish "
                         "STAGE=steer sbatch slurm/label_descriptions.sbatch or pass --allow-missing")
    for stem, recs in out["files"].items():
        (args.out_dir / f"{stem}.json").write_text(json.dumps(recs, indent=1))
    meta = {"stats": stats, "files": {k: len(v) for k, v in out["files"].items()}}
    (d / "finalize_meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))
    n = sum(v for k, v in stats.items() if k.endswith(":pairs"))
    c = sum(v for k, v in stats.items() if k.endswith(":composed"))
    share = c / n if n else 0.0
    print(f"GATE 2: composed {c}/{n} = {share:.3f} -> {'PASS' if share >= GATE2_MIN_COMPOSED else 'FAIL'} "
          f"(need >= {GATE2_MIN_COMPOSED})")


CEILING_MIN = 0.95      # d2 T2: the GT paths must pass their own behind check at least this often


def stage_endings_check(args):
    """d2 T2 step 2 (CPU): every described route of a split checked against its own ending with its GT answer path
    (labelset record): per ending type n, GT follow rate, GT ending rate (behind: both test points). Writes
    endings_check_<split>.json next to the events and prints the ceiling gate (behind >= CEILING_MIN)."""
    from rover_vlm.descriptions import desc_check
    from rover_vlm.eval import behind_followed, description_follow
    from rover_vlm.instructions import _answer_path
    recs = {r["id"]: r for r in json.loads((args.labelset_dir / f"labelset_{args.split}.json").read_text())}
    objects = {o["object_id"]: o for o in _jsonl(args.events_dir / f"objects_{args.split}.jsonl")}
    acc = {}
    for ev in _jsonl(args.events_dir / f"events_{args.split}.jsonl"):
        if ev.get("reason") is not None or ev["id"] not in recs:
            continue
        chk = desc_check(ev, objects)
        path = _answer_path(recs[ev["id"]])
        kind = chk["ending"]["type"]
        f = description_follow(path, chk, path[-1][:2])
        a = acc.setdefault(kind, {"n": 0, "follow": 0, "ending": 0, "mid": 0})
        a["n"] += 1
        a["follow"] += f["follows"]
        if kind == "behind":
            a["ending"] += behind_followed(path, chk["ending"]["outline"])
            a["mid"] += behind_followed(path, chk["ending"]["outline"], point="midpoint")
        elif kind == "turn":
            a["ending"] += next((ok for e, ok in zip(chk["events"], f["events"]) if e["type"] == "turn"), 0)
        else:
            a["ending"] += f["stop"]
    out = {k: {"n": a["n"], "gt_follow_rate": round(a["follow"] / a["n"], 4),
               "gt_ending_rate": round(a["ending"] / a["n"], 4),
               **({"gt_ending_rate_midpoint": round(a["mid"] / a["n"], 4)} if k == "behind" else {})}
           for k, a in sorted(acc.items())}
    (args.events_dir / f"endings_check_{args.split}.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    b = out.get("behind", {})
    ok = (b.get("gt_ending_rate") or 0) >= CEILING_MIN
    print(f"CEILING: behind gt_ending_rate {b.get('gt_ending_rate')} (midpoint {b.get('gt_ending_rate_midpoint')}) "
          f"-> {'PASS' if ok else 'FAIL'} (need >= {CEILING_MIN})")


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
    h.add_argument("--labelset-dir", type=Path, default=None, help="where labelset_<split>.json are "
                   "(default: --out-dir); d2: data/prepared_habitat_instr_v2")
    h.add_argument("--limit", type=int, default=0, help="first N records per split (smoke run)")
    h.add_argument("--force", action="store_true", help="overwrite events of another schema (d1) in --out-dir")
    k = sub.add_parser("finalize-descriptions", help="d1: train/eval files from labelled descriptions (CPU)")
    k.add_argument("--events-dir", type=Path, default=None)
    k.add_argument("--labelset-dir", type=Path, default=None)
    k.add_argument("--round", default="d1", choices=["d1", "d2"])
    k.add_argument("--labels", type=Path, default=D1_LABELS)
    k.add_argument("--out-dir", type=Path, default=None)
    k.add_argument("--desc-suffix", default="",
                   help="desc file tag: \"_mix\" reads desc_<split>_mix_d1.jsonl (NAMER=mix); default desc_<split>_d1.jsonl")
    k.add_argument("--allow-missing", action="store_true")
    sd2 = sub.add_parser("d2-seed", help="d2: seed the d2 label files from d1 + the changed eval routes (CPU)")
    sd2.add_argument("--d1-events-dir", type=Path, default=V2_OUT)
    sd2.add_argument("--events-dir", type=Path, default=V4_OUT)
    sd2.add_argument("--labels", type=Path, default=D1_LABELS)
    sd2.add_argument("--force", action="store_true", help="overwrite existing d2 seed files")
    ec = sub.add_parser("endings-check", help="d2 T2: GT ceiling of each ending check on a split (CPU)")
    ec.add_argument("--events-dir", type=Path, default=V4_OUT)
    ec.add_argument("--labelset-dir", type=Path, default=V2_OUT)
    ec.add_argument("--split", default="eval", choices=list(SPLITS))
    m = sub.add_parser("steer", help="steering: stop pairs per d1 eval frame (CPU)")
    m.add_argument("--frames", type=Path, default=V4_OUT / "eval_desc_entry.json")
    m.add_argument("--names", type=Path, default=D1_LABELS / "names_eval_gemma_d2.jsonl")
    m.add_argument("--steer-names", type=Path, default=D1_LABELS / "names_steer_gemma_d2.jsonl")
    m.add_argument("--fallback-names", type=Path, default=D1_LABELS / "names_eval_d2.jsonl")
    m.add_argument("--out-dir", type=Path, default=STEER_DIR)
    m.add_argument("--limit", type=int, default=0, help="first N frames (smoke run)")
    sel = sub.add_parser("steer-select", help="steering: one stop pair per viewpoint -> events_steer_eval.jsonl (CPU)")
    sel.add_argument("--out-dir", type=Path, default=STEER_DIR)
    fs = sub.add_parser("finalize-steer", help="steering: the 4 eval_steer_stop_* files + Gate 2 (after compose, CPU)")
    fs.add_argument("--frames", type=Path, default=V4_OUT / "eval_desc_entry.json")
    fs.add_argument("--steer-dir", type=Path, default=STEER_DIR)
    fs.add_argument("--desc", type=Path, default=D1_LABELS / "desc_steer_eval_d2.jsonl")
    fs.add_argument("--out-dir", type=Path, default=V4_OUT)
    fs.add_argument("--allow-missing", action="store_true", help="proceed with unlabelled pairs")
    o = sub.add_parser("steer-objects", help="steering: naming rows for every visible non-wall eval object (CPU)")
    o.add_argument("--frames", type=Path, default=V4_OUT / "eval_desc_entry.json")
    o.add_argument("--names", type=Path, default=D1_LABELS / "names_eval_gemma_d2.jsonl")
    o.add_argument("--fallback-names", type=Path, default=D1_LABELS / "names_eval_d2.jsonl")
    o.add_argument("--out-dir", type=Path, default=STEER_DIR)
    o.add_argument("--limit", type=int, default=0, help="first N frames (smoke run)")
    args = p.parse_args()
    {"pairs": stage_pairs, "finalize": stage_finalize, "general": stage_general,
     "finalize-general": stage_finalize_general, "landmarks": stage_landmarks, "events": stage_events, "steer": stage_steer, "steer-objects": stage_steer_objects,
     "steer-select": stage_steer_select, "finalize-steer": stage_finalize_steer,
     "finalize-descriptions": stage_finalize_descriptions, "d2-seed": stage_d2_seed,
     "endings-check": stage_endings_check}[args.stage](args)


if __name__ == "__main__":
    main()
