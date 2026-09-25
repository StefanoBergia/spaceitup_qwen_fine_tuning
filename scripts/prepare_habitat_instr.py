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
        --instructions outputs/instr_labels_v2/train_i2.jsonl outputs/instr_labels_v2/eval_i2.jsonl

The split is round 4's (data/prepared_habitat_v4), which is already scene-disjoint;
that is re-asserted here. Yaw-augmented frames are skipped: the candidate projections
belong to the unrotated camera.
"""

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path

from rover_vlm.instructions import (
    assert_scene_disjoint, build_frame, build_pair, build_wrong_side, finalize_general, finalize_pairs,
    render_check_image, render_generation_image, with_instruction,
)

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
    rows = _label_rows(args.instructions)
    meta = {"instructions": [str(p) for p in args.instructions]}
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
    args = p.parse_args()
    {"pairs": stage_pairs, "finalize": stage_finalize, "general": stage_general,
     "finalize-general": stage_finalize_general}[args.stage](args)


if __name__ == "__main__":
    main()
