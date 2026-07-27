"""Filter a full trace-labelling run down to a clean training set.

    uv run scripts/filter_traces.py                       # both path_v6 splits
    uv run scripts/filter_traces.py outputs/traces_full/path_v6_eval.jsonl

CPU-only, deterministic, no network. Reads the raw outputs/traces_full/*.jsonl produced by
scripts/label_traces.py, re-scores leakage with the CURRENT rover_vlm.traces.leakage_spans
(the raw file stored whatever detector was live at run time), and writes a sibling
*.filtered.jsonl — NON-DESTRUCTIVELY: every input row is preserved, each annotated with

    keep         bool
    drop_reason  one of {error, no-trace, truncated, leak, drift} or null when kept
    leakage      re-scored spans (may differ from the stored field)

The raw files are never mutated, so re-running after tuning the detector is free. The clean
training input is simply [r for r in rows if r["keep"]].

Why these drop reasons, in priority order:
  error       the request never returned
  no-trace    nothing parseable came back (usually a truncated generation)
  truncated   finish_reason != stop — generation was cut off, so the trace is partial
  leak        leakage_spans() fired: the trace refers to an answer it was handed, which is
              worse than no trace because it teaches Qwen to defer to a route that will not
              exist at inference (see the handed-route family in rover_vlm.traces)
  drift       the echoed answer is valid JSON but its goal != ground truth, i.e. the reply
              wandered to a different sample; the trace cannot be trusted to describe THIS one

A reply that merely skipped the JSON envelope (unverifiable, not mismatched) is KEPT — its
trace text is fine and answer_payload() already keeps "unverifiable" distinct from "wrong".

Merging the kept traces into Qwen's <think> block and the TrajectoryCollator mask change
are the NEXT phase, not this one.
"""

import argparse
import json
from pathlib import Path

from rover_vlm.traces import answer_payload, leakage_spans

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT = [REPO_ROOT / "outputs" / "traces_full" / f"path_v6_{s}.jsonl"
           for s in ("train_full", "eval")]

REASONS = ["error", "no-trace", "truncated", "leak", "drift"]


def _drifted(row):
    """A *verifiable* answer the labeller already scored as not matching ground truth — the
    reply wandered to another sample. The real ground truth is not in the trace row, so this
    reads the label-time verdict (answer_matches_gt) rather than recomputing it. Unverifiable
    replies (no JSON) are not drift: the trace text is fine, just unechoed."""
    return answer_payload(row.get("answer")) is not None and not row.get("answer_matches_gt")


def filter_file(path):
    rows = [json.loads(line) for line in path.open() if line.strip()]
    counts = {r: 0 for r in REASONS}
    kept = 0
    out = []
    for row in rows:
        # Drift uses the label-time verdict (answer_matches_gt), since the real ground
        # truth is not in the trace row — see _drifted.
        if row.get("error"):
            keep, reason, leaks = False, "error", []
        elif not row.get("trace"):
            keep, reason, leaks = False, "no-trace", []
        elif row.get("finish_reason") not in ("stop", None):
            keep, reason, leaks = False, "truncated", leakage_spans(row["trace"])
        elif leakage_spans(row["trace"]):
            keep, reason, leaks = False, "leak", leakage_spans(row["trace"])
        elif _drifted(row):
            keep, reason, leaks = False, "drift", []
        else:
            keep, reason, leaks = True, None, leakage_spans(row["trace"])
        if keep:
            kept += 1
        else:
            counts[reason] += 1
        out.append({**row, "keep": keep, "drop_reason": reason, "leakage": leaks})

    out_path = path.with_suffix(".filtered.jsonl")
    with out_path.open("w") as f:
        for row in out:
            f.write(json.dumps(row) + "\n")
    return rows, out, out_path, kept, counts


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("files", nargs="*", type=Path, default=DEFAULT,
                   help="raw *.jsonl to filter (default: both path_v6 splits)")
    args = p.parse_args()

    grand_kept = grand_n = 0
    for path in args.files:
        if not path.exists():
            raise SystemExit(f"missing {path} — run scripts/label_traces.py first")
        rows, out, out_path, kept, counts = filter_file(path)
        n = len(rows)
        grand_kept += kept
        grand_n += n
        dropped = n - kept
        print(f"\n{path.name}  (n={n})")
        for reason in REASONS:
            if counts[reason]:
                print(f"   drop {reason:10} {counts[reason]}")
        print(f"   TOTAL DROP {dropped}  ({100 * dropped / n:.1f}%)"
              f"   ->  KEEP {kept}  ({100 * kept / n:.1f}%)")
        print(f"   wrote {out_path.name}  ({len(out)} rows, same as input)")

    if len(args.files) > 1:
        print(f"\ntotal: KEEP {grand_kept}/{grand_n}  ({100 * grand_kept / grand_n:.1f}%)")
    # A kept trace that still leaks would be a filter bug; assert the shipped set is clean.
    residual = sum(1 for path in args.files
                   for row in (json.loads(l) for l in path.with_suffix(".filtered.jsonl").open())
                   if row["keep"] and leakage_spans(row.get("trace")))
    print(f"\nresidual leaks in the kept set: {residual}  (must be 0)")
    if residual:
        raise SystemExit("kept set is not leak-free — filter logic is wrong")


if __name__ == "__main__":
    main()
