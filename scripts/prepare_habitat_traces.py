"""Merge Cosmos3 reasoning traces into the prepared Habitat path-regression training set.

    uv run scripts/prepare_habitat_traces.py            # default v2 train split
    uv run scripts/prepare_habitat_traces.py \
        --prepared data/prepared_habitat_v2/train_full.json \
        --traces   outputs/traces_full/path_v6_train_full.filtered.jsonl \
        --out      data/prepared_habitat_v2/train_full_traced.json

CPU-only, deterministic, no network. Joins each prepared conversation record to its kept
trace BY ID and attaches the reasoning as a top-level "reasoning" field. Downstream,
rover_vlm.training.TrajectoryDataset picks that field up and the collator renders it inside
Qwen3.5's <think> block (see that module's docstring).

Design decisions baked in here:
  * The label stays the AUTHORITATIVE ground-truth answer from the prepared record
    (conversations[1]["value"]). The trace row's own "answer" is the labeller's echo/reword
    and is NOT used — only its "trace" (the reasoning prose) is.
  * Only rows with keep == True from the filter are used; leaked/no-trace/drift/truncated
    rows carry no reasoning. By default their prepared records are DROPPED (a reasoning-trace
    fine-tune wants every sample to reason); pass --keep-untraced to instead include them
    answer-only (the collator trains those exactly as the pre-trace pipeline did).
  * A trace that somehow still contains <think>/</think>/<answer> tags would corrupt the
    chat-template rendering, so such rows are dropped with a warning (should be none).

Eval is intentionally NOT traced: at eval the model generates its own reasoning and we score
the parsed answer, so eval.json stays plain. Remember to run scripts/evaluate.py with
--enable-thinking against an adapter trained here.
"""

import argparse
import json
import re
from pathlib import Path

from rover_vlm.traces import leakage_spans

REPO_ROOT = Path(__file__).resolve().parent.parent
TAG_RE = re.compile(r"</?think>|</?answer>", re.IGNORECASE)


def _rel(path):
    """Display path relative to the repo when possible, else absolute (for --out elsewhere)."""
    try:
        return path.resolve().relative_to(REPO_ROOT)
    except ValueError:
        return path


def load_kept_traces(path):
    """id -> trace text, for keep==True rows only. Rows with tag-contaminated traces or an
    empty trace are skipped (and counted) so they cannot corrupt the rendered <think>."""
    traces, skipped_tags, skipped_empty = {}, 0, 0
    for line in path.open():
        if not line.strip():
            continue
        row = json.loads(line)
        if not row.get("keep"):
            continue
        trace = (row.get("trace") or "").strip()
        if not trace:
            skipped_empty += 1
            continue
        if TAG_RE.search(trace):
            skipped_tags += 1
            continue
        traces[row["id"]] = trace
    return traces, skipped_tags, skipped_empty


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--prepared", type=Path,
                   default=REPO_ROOT / "data/prepared_habitat_v2/train_full.json")
    p.add_argument("--traces", type=Path,
                   default=REPO_ROOT / "outputs/traces_full/path_v6_train_full.filtered.jsonl")
    p.add_argument("--out", type=Path,
                   default=REPO_ROOT / "data/prepared_habitat_v2/train_full_traced.json")
    p.add_argument("--keep-untraced", action="store_true",
                   help="include prepared records that have no kept trace (answer-only), "
                        "instead of dropping them")
    args = p.parse_args()

    for f in (args.prepared, args.traces):
        if not f.exists():
            raise SystemExit(f"missing {f}")

    records = json.loads(args.prepared.read_text())
    traces, skipped_tags, skipped_empty = load_kept_traces(args.traces)

    out = []
    matched = dropped_untraced = 0
    for rec in records:
        trace = traces.get(rec["id"])
        if trace is not None:
            out.append({**rec, "reasoning": trace})
            matched += 1
        elif args.keep_untraced:
            out.append({k: v for k, v in rec.items() if k != "reasoning"})
        else:
            dropped_untraced += 1

    # A trace id with no prepared record means the two files are from different rounds —
    # surface it rather than silently under-joining.
    orphan_traces = set(traces) - {r["id"] for r in records}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out))

    # The filter guarantees keep => leak-free, but re-assert on what we actually shipped.
    residual = sum(1 for r in out if r.get("reasoning") and leakage_spans(r["reasoning"]))

    print(f"prepared records:        {len(records)}")
    print(f"kept traces (by id):     {len(traces)}"
          f"   (skipped {skipped_empty} empty, {skipped_tags} tag-contaminated)")
    print(f"matched (with reasoning):{matched}")
    if args.keep_untraced:
        print(f"untraced kept answer-only:{len(out) - matched}")
    else:
        print(f"dropped (no kept trace): {dropped_untraced}")
    if orphan_traces:
        print(f"WARNING: {len(orphan_traces)} kept traces had no prepared record "
              f"(mismatched rounds?), e.g. {sorted(orphan_traces)[:3]}")
    print(f"\nwrote {_rel(args.out)}  ({len(out)} records)")
    print(f"residual leaks in reasoning: {residual}  (must be 0)")
    if residual:
        raise SystemExit("a shipped reasoning trace still leaks — re-run the filter first")


if __name__ == "__main__":
    main()
