"""Generate reasoning traces for a prepared split by asking Cosmos3-Nano to justify the
ground-truth answer.

Runs on the LOGIN NODE against the endpoint published by slurm/serve_cosmos3.sbatch (odin
reaches thor over TCP), so a prompt revision costs seconds rather than a 35 GB reload.

    # smoke test: 3 samples, full request/response printed
    uv run scripts/label_traces.py --task choice \
        --split data/prepared_habitat_choice_v2/eval.json --limit 3 --show

    # prompt-iteration round
    uv run scripts/label_traces.py --task path \
        --split data/prepared_habitat_v2/eval.json --limit 30 --prompt-version v2

    # full dataset — see slurm/label_full_path.sbatch, which runs this unattended
    uv run scripts/label_traces.py --task path --split data/prepared_habitat_v2/train_full.json \
        --out-dir outputs/traces_full --out-name path_v6_train_full --concurrency 8 --resume

Writes <out-dir>/<out-name>.jsonl, one line per sample, flushed as it goes so a killed run
keeps what it had. The summary at the end is the prompt-quality gate: leakage should be 0
and answers should match ground truth.

Two things make it safe to point at 9,140 samples. It refuses to truncate a non-empty
output file unless you say --resume or --overwrite; and --resume both skips what is already
done and *retries* rows that failed, since a failed request is unfinished work rather than a
result. --concurrency stays at 1 by default so prompt-iteration output keeps split order.

The prompts themselves live in src/rover_vlm/traces.py — edit there, not here.
"""

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from rover_vlm.traces import (
    PROMPT_VERSION,
    answer_matches,
    answer_payload,
    build_prompt,
    leakage_spans,
    parse_trace,
    trace_from_answer,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
ENDPOINT_FILE = REPO_ROOT / "outputs" / "cosmos3_endpoint.txt"
OUT_DIR = REPO_ROOT / "outputs" / "traces"


def resolve_endpoint(explicit):
    if explicit:
        return explicit
    if not ENDPOINT_FILE.exists():
        raise SystemExit(
            f"No endpoint at {ENDPOINT_FILE}.\n"
            "Start the server with 'sbatch slurm/serve_cosmos3.sbatch' and wait for it to\n"
            "report READY, or pass --endpoint http://<host>:<port>/v1 explicitly."
        )
    return ENDPOINT_FILE.read_text().strip()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", choices=["choice", "path"], required=True)
    p.add_argument("--split", type=Path, required=True, help="a prepared split JSON")
    p.add_argument("--limit", type=int, default=None, help="cap samples (use for smoke/iteration)")
    p.add_argument("--endpoint", default=None, help="override outputs/cosmos3_endpoint.txt")
    p.add_argument("--model", default=None, help="override the served model id")
    p.add_argument("--prompt-version", default=PROMPT_VERSION, help="tags the output filename")
    p.add_argument("--max-tokens", type=int, default=8192,
                   help="reasoning runs long; too low truncates mid-<think> and "
                        "loses the sample outright (2 of 30 did so at 4096)")
    p.add_argument("--seed", type=int, default=0, help="fixed so prompt revisions are comparable")
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    p.add_argument("--out-name", default=None,
                   help="output stem (default <task>_<prompt-version>); set it per split when "
                        "labelling more than one into the same directory")
    p.add_argument("--concurrency", type=int, default=1,
                   help="in-flight requests. 1 keeps output in split order for prompt "
                        "iteration; a full-dataset run wants 8 or so")
    p.add_argument("--resume", action="store_true",
                   help="append, skipping ids already in the output file — a long run that "
                        "was interrupted continues instead of starting over")
    p.add_argument("--overwrite", action="store_true",
                   help="allow replacing a non-empty output file")
    p.add_argument("--ids", default=None,
                   help="comma-separated sample ids to run (re-check specific failures)")
    p.add_argument("--show", action="store_true", help="print each prompt and raw reply")
    args = p.parse_args()
    if args.show and args.concurrency > 1:
        raise SystemExit("--show interleaves unreadably with --concurrency > 1")

    records = json.loads(args.split.read_text())
    if args.ids:
        want = {i.strip() for i in args.ids.split(',') if i.strip()}
        records = [r for r in records if r['id'] in want]
        missing = want - {r['id'] for r in records}
        if missing:
            raise SystemExit(f"ids not present in {args.split}: {sorted(missing)}")
    if args.limit:
        records = records[: args.limit]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / f"{args.out_name or f'{args.task}_{args.prompt_version}'}.jsonl"

    # A full run is hours of GPU time; silently truncating its output would be the most
    # expensive mistake this script can make.
    done = set()
    if out_path.exists() and out_path.stat().st_size:
        if args.resume:
            kept = [json.loads(line) for line in out_path.open() if line.strip()]
            # A failed request is unfinished work, not a result — resume retries it. The
            # file is rewritten without those rows first, so a retried sample ends up with
            # one line rather than an error line shadowed by a later success.
            failed = sum(1 for r in kept if r.get("error"))
            kept = [r for r in kept if not r.get("error")]
            done = {r["id"] for r in kept}
            tmp = out_path.with_suffix(".jsonl.tmp")
            tmp.write_text("".join(json.dumps(r) + "\n" for r in kept))
            os.replace(tmp, out_path)          # atomic: never leaves a half-written file
            records = [r for r in records if r["id"] not in done]
            print(f"resume:   {len(done)} done in {out_path.name}"
                  + (f", {failed} failed rows dropped for retry" if failed else "")
                  + f", {len(records)} left")
        elif not args.overwrite:
            raise SystemExit(
                f"{out_path} exists and is not empty. Pass --resume to continue it, or "
                f"--overwrite to replace it.")
    print(f"split:    {args.split} ({len(records)} to label, task={args.task}, "
          f"concurrency={args.concurrency})\n")
    if not records:
        print("nothing left to do")
        return

    try:
        import openai
    except ImportError:
        raise SystemExit("openai not installed — run 'uv sync' (it is a project dependency).")

    base_url = resolve_endpoint(args.endpoint)
    client = openai.OpenAI(api_key="EMPTY", base_url=base_url, timeout=600)
    try:
        model = args.model or client.models.list().data[0].id
    except Exception as e:  # noqa: BLE001 — a dead endpoint should read as one line, not a stack
        raise SystemExit(
            f"Cannot reach the server at {base_url} ({type(e).__name__}: {e}).\n"
            "Check the serve job is still RUNNING (squeue) and that\n"
            "outputs/cosmos3_endpoint.txt points at the node it is actually on."
        )
    print(f"endpoint: {base_url}\nmodel:    {model}")


    def label(rec):
        """One sample, start to finish. Never raises: a bad sample is a recorded failure."""
        prompt = build_prompt(args.task, rec)
        image_uri = Path(rec["image"][0]).resolve().as_uri()
        t0 = time.time()
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": image_uri}},
                    {"type": "text", "text": prompt},
                ]}],
                max_tokens=args.max_tokens,
                seed=args.seed,
            )
            raw = resp.choices[0].message.content or ""
            finish = resp.choices[0].finish_reason
        except Exception as e:  # noqa: BLE001 — one bad sample must not kill the run
            return rec, prompt, image_uri, None, {"id": rec["id"],
                                                  "error": f"{type(e).__name__}: {e}"}

        dt = time.time() - t0
        scratch, answer = parse_trace(raw)
        trace = trace_from_answer(answer)   # the deliverable; scratch is debug only
        leaks = leakage_spans(trace)
        matched = answer_matches(args.task, answer, rec)
        # No parseable JSON means we could not check drift — not that it drifted.
        verifiable = answer_payload(answer) is not None
        return rec, prompt, image_uri, raw, {
            "id": rec["id"], "task": args.task, "prompt_version": args.prompt_version,
            "trace": trace, "answer": answer, "answer_matches_gt": matched, "verifiable": verifiable,
            "leakage": leaks, "words": len(trace.split()) if trace else 0,
            "finish_reason": finish, "scratch_words": len((scratch or "").split()),
            "latency_s": round(dt, 2),
            # keep raw whenever it is the only way to explain the outcome
            "raw": raw if (args.show or trace is None or finish != "stop") else None,
        }

    n_ok = n_leak = n_match = n_err = n_unverifiable = 0
    word_counts = []
    n, started = len(records), time.time()
    mode = "a" if (args.resume and done) else "w"

    with out_path.open(mode) as f:
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futures = [pool.submit(label, rec) for rec in records]
            try:
                for i, fut in enumerate(as_completed(futures), 1):
                    rec, prompt, image_uri, raw, row = fut.result()
                    # Flushed per line so a killed run keeps every sample it paid for.
                    f.write(json.dumps(row) + "\n")
                    f.flush()

                    if row.get("error"):
                        n_err += 1
                        print(f"[{i}/{n}] {row['id']}  ERROR: {row['error']}")
                        continue

                    trace, leaks = row["trace"], row["leakage"]
                    n_ok += trace is not None
                    n_leak += bool(leaks)
                    n_match += row["answer_matches_gt"]
                    n_unverifiable += not row["verifiable"]
                    if trace:
                        word_counts.append(row["words"])

                    flags = []
                    if trace is None:
                        flags.append("NO-TRACE")
                    if leaks:
                        flags.append(f"LEAK{leaks}")
                    if not row["answer_matches_gt"]:
                        flags.append("ANSWER-MISMATCH" if row["verifiable"] else "UNVERIFIED(no-json)")
                    if row["finish_reason"] != "stop":
                        flags.append(f"finish={row['finish_reason']}")
                    # An ETA, because the alternative on a 9k-sample run is watching ids scroll.
                    rate = i / max(1e-9, time.time() - started)
                    eta = (n - i) / rate
                    print(f"[{i}/{n}] {row['id']}  {row['words']}w  {row['latency_s']}s  "
                          f"{rate * 60:.0f}/min  eta {eta / 60:.0f}m  "
                          f"{' '.join(flags) if flags else 'ok'}")

                    if args.show:
                        print(f"\n--- prompt ---\n{prompt}\n--- image ---\n{image_uri}"
                              f"\n--- raw reply ---\n{raw}\n{'-' * 70}\n")
            except KeyboardInterrupt:
                print("\ninterrupted — cancelling queued requests; "
                      f"{out_path} holds everything finished so far. Re-run with --resume.")
                for fut in futures:
                    fut.cancel()
                raise

    med = sorted(word_counts)[len(word_counts) // 2] if word_counts else 0
    elapsed = (time.time() - started) / 60
    print(f"\nwrote {out_path}   ({n} samples in {elapsed:.0f}m)")
    print(f"  parsed a trace   {n_ok}/{n}")
    print(f"  answer matches   {n_match}/{n}" + (f"   ({n_unverifiable} unverifiable: answer was not JSON)" if n_unverifiable else ""))
    print(f"  LEAKAGE          {n_leak}/{n}   <- must reach 0 before scaling up")
    print(f"  median words     {med}   (target 60-120)")
    if n_err:
        print(f"  request errors   {n_err}/{n}   <- re-run with --resume to retry just these")


if __name__ == "__main__":
    main()
