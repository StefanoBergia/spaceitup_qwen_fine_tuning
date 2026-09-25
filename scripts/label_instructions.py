"""Label route instructions with Cosmos3-Nano: one per record, checked by text rules.

Runs on the LOGIN NODE against the endpoint published by slurm/serve_cosmos3.sbatch.
Input is a pairs file from `scripts/prepare_habitat_instr.py pairs`; each record carries
its two overlay images (habitat_meta.gen_image / check_image).

Per record, one call (prompt i2): the frame with the record's route bold magenta and the
other side's grey, plus the side the route takes (known exactly from the metadata); Cosmos
names the landmark the route passes. A record passes when the instruction parses and trips
no rule in instruction_rejects(text, side): correct side word present, opposite absent,
something named, no reference to the drawing.

(Prompt i1 also asked Cosmos to verify each instruction against two identically drawn
routes; the 100-pair gate showed it cannot judge left from right on drawn routes — 63% even
for the template instruction — so that call was dropped. See INSTRUCTION_PROMPT_VERSION.)

    # gate: first 100 eval pairs; exits 2 if the gate fails
    uv run scripts/label_instructions.py --split data/prepared_habitat_instr_v1/pairs_eval.json \
        --limit-pairs 100 --out outputs/instr_labels/gate_i2.jsonl --concurrency 8 --gate
    # full run (resumable)
    uv run scripts/label_instructions.py --split data/prepared_habitat_instr_v1/pairs_train.json \
        --out outputs/instr_labels/train_i2.jsonl --concurrency 8 --resume

    # re-judge a finished file with the current parser (no endpoint, no GPU)
    uv run scripts/label_instructions.py --split data/prepared_habitat_instr_v1/pairs_eval.json \
        --limit-pairs 100 --out outputs/instr_labels/gate_i1.jsonl --rescore --gate

    # v2 (every round-4 frame; overlays rendered in memory, sent inline)
    uv run scripts/label_instructions.py --split data/prepared_habitat_instr_v2/labelset_eval.json \
        --out outputs/instr_labels_v2/eval_i3.jsonl --concurrency 8 --resume
    # ... then one more attempt for the rejected rows (seed + 1)
    uv run scripts/label_instructions.py --split data/prepared_habitat_instr_v2/labelset_eval.json \
        --out outputs/instr_labels_v2/eval_i3.jsonl --concurrency 8 --resume --retry-rejected
    # record-level gate for single-route rows (no pairs in the split)
    ... --gate --min-accept 0.9 --min-pair-survival 0

Writes one JSON line per record, flushed as it goes, so a killed run keeps what it had.
"""

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from rover_vlm.instructions import (
    INSTRUCTION_PROMPT_VERSION, gate_failures, generation_prompt, instruction_rejects,
    is_side_only, parse_instruction, render_generation_jpeg, rescore_row, retry_ids, summarize_labels,
)
from rover_vlm.labelling_io import chat_with_image, connect, prepare_output, resolve_endpoint

REPO_ROOT = Path(__file__).resolve().parent.parent
ENDPOINT_FILE = REPO_ROOT / "outputs" / "cosmos3_endpoint.txt"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--split", type=Path, required=True, help="pairs_*.json from prepare_habitat_instr.py")
    p.add_argument("--out", type=Path, required=True, help="output JSONL")
    p.add_argument("--limit-pairs", type=int, default=None, help="label only the first N pairs")
    p.add_argument("--endpoint", default=None, help="override outputs/cosmos3_endpoint.txt")
    p.add_argument("--model", default=None, help="override the served model id")
    p.add_argument("--max-tokens", type=int, default=4096)
    p.add_argument("--seed", type=int, default=0, help="fixed so prompt revisions are comparable")
    p.add_argument("--concurrency", type=int, default=1, help="in-flight records (8 for a full run)")
    p.add_argument("--resume", action="store_true", help="continue a run; retries error rows")
    p.add_argument("--overwrite", action="store_true", help="allow replacing a non-empty output")
    p.add_argument("--show", action="store_true", help="print both raw replies per record")
    p.add_argument("--rescore", action="store_true",
                   help="re-parse the saved raw replies in --out with the current parser; no requests")
    p.add_argument("--gate", action="store_true", help="exit 2 unless the thresholds below hold")
    p.add_argument("--min-pair-survival", type=float, default=0.8)
    p.add_argument("--min-accept", type=float, default=None,
                   help="gate also on the share of accepted records (v2 single-route rows)")
    p.add_argument("--retry-rejected", action="store_true",
                   help="with --resume: label rejected, not-yet-retried rows once more, seed + 1")
    p.add_argument("--max-drawing-refs", type=int, default=0)
    args = p.parse_args()

    records = json.loads(args.split.read_text())
    if args.limit_pairs:
        keep = set(list(dict.fromkeys(r["habitat_meta"].get("pair_id") for r in records
                                      if r["habitat_meta"].get("pair_id")))[: args.limit_pairs])
        records = [r for r in records if r["habitat_meta"].get("pair_id") in keep]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.rescore:
        if not args.out.exists():
            raise SystemExit(f"--rescore needs an existing {args.out}")
        old = [json.loads(line) for line in args.out.read_text().splitlines() if line.strip()]
        new = [rescore_row(r) for r in old]
        tmp = args.out.with_suffix(".jsonl.tmp")
        tmp.write_text("".join(json.dumps(r) + "\n" for r in new))
        tmp.replace(args.out)
        changed = sum(a != b for a, b in zip(old, new))
        print(f"rescored {len(new)} rows in {args.out} ({changed} changed)")
        done, mode = {r["id"] for r in new}, "a"
    else:
        done, mode = prepare_output(args.out, resume=args.resume, overwrite=args.overwrite)
    seed = args.seed
    if args.retry_rejected:
        if not args.resume:
            raise SystemExit("--retry-rejected continues an existing file: pass --resume too")
        old = [json.loads(line) for line in args.out.read_text().splitlines() if line.strip()]
        again = retry_ids(old) & {r["id"] for r in records}
        tmp = args.out.with_suffix(".jsonl.tmp")
        tmp.write_text("".join(json.dumps(r) + "\n" for r in old if r["id"] not in again))
        tmp.replace(args.out)
        done -= again
        seed = args.seed + 1
        mode = "a"
        print(f"retrying {len(again)} rejected rows with seed {seed}")
    todo = [r for r in records if r["id"] not in done]
    print(f"split: {args.split}  records {len(records)}  to label {len(todo)}  "
          f"prompt {INSTRUCTION_PROMPT_VERSION}  concurrency {args.concurrency}")

    if todo:
        base_url = resolve_endpoint(args.endpoint, ENDPOINT_FILE)
        client, model = connect(base_url, args.model)
        print(f"endpoint: {base_url}\nmodel:    {model}\n")

        def label(rec):
            """One record, both calls. Never raises: a bad sample is a recorded failure."""
            hm = rec["habitat_meta"]
            row = {"id": rec["id"], "pair_id": hm.get("pair_id"), "kind": hm.get("kind"),
                   "side": hm["side"], "prompt_version": INSTRUCTION_PROMPT_VERSION}
            if args.retry_rejected:
                row["retried"] = True
            # v1 records point at overlay files; v2 renders the overlay in memory
            image = hm["gen_image"] if hm.get("gen_image") else render_generation_jpeg(rec)
            t0 = time.time()
            try:
                raw_gen, fin_gen = chat_with_image(client, model, image,
                                                   generation_prompt(hm["side"], has_alt=bool(hm.get("alt_path"))),
                                                   max_tokens=args.max_tokens, seed=seed)
            except Exception as e:  # noqa: BLE001 — one bad sample must not kill the run
                return {**row, "error": f"{type(e).__name__}: {e}"}
            text = parse_instruction(raw_gen)
            rejects = instruction_rejects(text, side=hm["side"])
            row.update({
                "instruction": text, "rejects": rejects,
                "side_only": is_side_only(text) if text else None,
                "passed": bool(text and not rejects),
                "finish": fin_gen, "latency_s": round(time.time() - t0, 2),
            })
            # keep the raw reply whenever it is the only way to explain the outcome
            if args.show or not row["passed"]:
                row["raw_gen"] = raw_gen
            return row

        started = time.time()
        with args.out.open(mode) as f, ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futures = [pool.submit(label, r) for r in todo]
            try:
                for i, fut in enumerate(as_completed(futures), 1):
                    row = fut.result()
                    f.write(json.dumps(row) + "\n")
                    f.flush()
                    rate = i / max(1e-9, time.time() - started)
                    if row.get("error"):
                        status = f"ERROR {row['error']}"
                    elif row["passed"]:
                        status = "ok"
                    else:
                        status = f"FAIL rejects={row['rejects']}"
                    print(f"[{i}/{len(todo)}] {row['id']}  {rate * 60:.0f}/min  "
                          f"eta {(len(todo) - i) / rate / 60:.0f}m  {row.get('instruction')!r}  {status}", flush=True)
                    if args.show:
                        print(f"--- reply ---\n{row.get('raw_gen')}\n")
            except KeyboardInterrupt:
                print(f"\ninterrupted — {args.out} holds everything finished so far. Re-run with --resume.")
                for fut in futures:
                    fut.cancel()
                raise

    rows = [json.loads(line) for line in args.out.read_text().splitlines() if line.strip()]
    wanted = {r["id"] for r in records}
    rows = [r for r in rows if r["id"] in wanted]
    summary = summarize_labels(rows)
    print("\nsummary:\n" + json.dumps(summary, indent=2))
    errors = sum(1 for r in rows if r.get("error"))
    if errors:
        print(f"{errors} request errors — re-run with --resume to retry them")
    if args.gate:
        fails = gate_failures(summary, min_pair_survival=args.min_pair_survival,
                              max_drawing_refs=args.max_drawing_refs, min_accept=args.min_accept)
        print("GATE: " + ("PASS" if not fails else "FAIL — " + "; ".join(fails)))
        if fails:
            sys.exit(2)


if __name__ == "__main__":
    main()
