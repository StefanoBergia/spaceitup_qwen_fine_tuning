"""Label route descriptions (prompt d2) with Cosmos3-Nano: name objects, then compose sentences.

Runs against the endpoint slurm/serve_cosmos3.sbatch (or slurm/label_descriptions.sbatch)
publishes. Inputs come from `prepare_habitat_instr.py events`.

    # stage 1: one call per frame object (both routes of a pair share it), on a composite image
    uv run scripts/label_descriptions.py --stage name \
        --events data/prepared_habitat_instr_v2/events_eval.jsonl \
        --objects data/prepared_habitat_instr_v2/objects_eval.jsonl \
        --out outputs/instr_labels_v3/names_eval_d1.jsonl --concurrency 8 --resume
    # stage 2: one text-only call per described route
    uv run scripts/label_descriptions.py --stage compose \
        --events data/prepared_habitat_instr_v2/events_eval.jsonl \
        --names outputs/instr_labels_v3/names_eval_d1.jsonl \
        --out outputs/instr_labels_v3/desc_eval_d1.jsonl --concurrency 8 --resume
    # steering (Addendum A1): name every row of an objects file, no routes
    uv run scripts/label_descriptions.py --stage name --all-objects \
        --objects data/prepared_habitat_instr_v3/steer/objects_eval.jsonl \
        --out outputs/instr_labels_v3/names_steer_gemma_d1.jsonl --concurrency 8 --resume
    #   --fallback-names OTHER_names.jsonl: used per route when --names leaves an object unnamed or two
    #   objects share a head noun ("name:clash"); rows carry "namer": primary|fallback
    # gate: --limit 300 on both stages, --gate on compose (exit 2 when a threshold fails;
    # slurm/label_descriptions.sbatch keeps the name stage report-only)

Each object/route gets up to two Cosmos attempts (seed, seed + 1) inside one row; compose then
falls back to the template sentence. Rows are flushed as they finish, so a killed run keeps
what it had; --resume retries error rows only.
"""

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from rover_vlm.descriptions import (
    d1_gate_failures, label_object, label_route, name_rows_split, names_missing, objects_needing_fallback,
    route_objects, select_routes,
    summarize_d1,
)
from rover_vlm.labelling_io import chat_text, chat_with_image, connect, prepare_output, resolve_endpoint
from rover_vlm.route_events import check_events_schema

REPO_ROOT = Path(__file__).resolve().parent.parent
ENDPOINT_FILE = REPO_ROOT / "outputs" / "cosmos3_endpoint.txt"


DEFAULT_MAX_TOKENS = 4096  # gate r1: 2048 cut 96 of 140 compose replies before </answer>


def jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", choices=["name", "compose"], required=True)
    p.add_argument("--events", type=Path, help="events_*.jsonl (not with --all-objects)")
    p.add_argument("--all-objects", action="store_true",
                   help="stage name: name every row of --objects; no --events (steering, Addendum A1)")
    p.add_argument("--objects", type=Path, help="objects_*.jsonl (stage name)")
    p.add_argument("--names", type=Path, help="names_*_d1.jsonl from stage name (stage compose)")
    p.add_argument("--only-unusable-under", type=Path,
                   help="stage name: only name objects of routes the name rows in FILE cannot serve (mix fallback)")
    p.add_argument("--fallback-names", type=Path,
                   help="second names file (same format); tried when --names leaves an object unnamed or clashes")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--limit", type=int, default=None, help="first N described routes (and their objects)")
    p.add_argument("--endpoint", default=None)
    p.add_argument("--model", default=None)
    p.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS,
                   help="both stages; thinking replies need room to reach </answer>")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--gate", action="store_true", help="exit 2 unless accept >= 0.9 (and template share <= 0.1)")
    args = p.parse_args()

    if args.all_objects and args.stage != "name":
        raise SystemExit("--all-objects is for --stage name")
    if not args.all_objects:
        if not args.events:
            raise SystemExit("--events is required (unless --stage name --all-objects)")
        check_events_schema(args.events)
    routes = [] if args.all_objects else select_routes(jsonl(args.events), args.limit)
    if args.stage == "name":
        if not args.objects:
            raise SystemExit("--stage name needs --objects objects_*.jsonl")
        if args.all_objects:
            items = jsonl(args.objects)
            items = items[: args.limit] if args.limit else items
        elif args.only_unusable_under:
            if not args.only_unusable_under.exists():
                raise SystemExit(f"--only-unusable-under {args.only_unusable_under} not found")
            wanted = objects_needing_fallback(routes, jsonl(args.only_unusable_under))
            print(f"only-unusable-under: {len(wanted)} objects of {len(routes)} routes", flush=True)
        else:
            wanted = {o for r in routes for o in route_objects(r)}
        if not args.all_objects:
            items = [o for o in jsonl(args.objects) if o["object_id"] in wanted]
            missing = wanted - {o["object_id"] for o in items}
            if missing:
                raise SystemExit(f"{len(missing)} route objects have no object row (e.g. {sorted(missing)[:3]})")
        key = "object_id"
    else:
        if not (args.names and args.names.exists()):
            raise SystemExit("--stage compose needs --names from a finished stage name run")
        name_rows = jsonl(args.names)
        names, rejected = name_rows_split(name_rows)
        kinds = {r["id"]: r.get("need") for r in name_rows}   # kind_hint: walls get "past the end of"
        fb_names = fb_rejected = None
        if args.fallback_names:
            if not args.fallback_names.exists():
                raise SystemExit(f"--fallback-names {args.fallback_names} not found")
            fb_rows = jsonl(args.fallback_names)
            fb_names, fb_rejected = name_rows_split(fb_rows)
            for r in fb_rows:                      # same object ids -> same need; fill ids the primary lacks
                kinds.setdefault(r["id"], r.get("need"))
        items, key = routes, "id"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    done, mode = prepare_output(args.out, resume=args.resume, overwrite=args.overwrite)
    todo = [it for it in items if it[key] not in done]
    print(f"stage {args.stage}: {len(items)} items, {len(todo)} to label, concurrency {args.concurrency}")

    if todo:
        base_url = resolve_endpoint(args.endpoint, ENDPOINT_FILE)
        client, model = connect(base_url, args.model)
        print(f"endpoint: {base_url}\nmodel:    {model}\n")

        def ask_image(image, prompt, seed):
            return chat_with_image(client, model, image, prompt, max_tokens=args.max_tokens, seed=seed)

        def ask_text(prompt, seed):
            return chat_text(client, model, prompt, max_tokens=args.max_tokens, seed=seed)

        def work(it):
            t0 = time.time()
            if args.stage == "compose":   # a name that never got an answer is retryable, not a rejection
                gap = names_missing(it, names, rejected, fb_names, fb_rejected)
                if gap:
                    return {"id": it[key], "error": f"name_missing: {gap}"}
            try:
                row = (label_object(ask_image, it, args.seed) if args.stage == "name"
                       else label_route(ask_text, it, names, args.seed, kinds, fb_names))
            except Exception as e:  # noqa: BLE001 — one bad item must not kill the run
                return {"id": it[key], "error": f"{type(e).__name__}: {e}"}
            row["latency_s"] = round(time.time() - t0, 2)
            return row

        started = time.time()
        with args.out.open(mode) as f, ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futures = [pool.submit(work, it) for it in todo]
            for i, fut in enumerate(as_completed(futures), 1):
                row = fut.result()
                f.write(json.dumps(row) + "\n")
                f.flush()
                rate = i / max(1e-9, time.time() - started)
                status = (f"ERROR {row['error']}" if row.get("error") else "ok" if row["passed"]
                          else f"FAIL {row['rejects']}")
                shown = row.get("name") if args.stage == "name" else row.get("description")
                print(f"[{i}/{len(todo)}] {row['id']}  {rate * 60:.0f}/min  eta {(len(todo) - i) / rate / 60:.0f}m  "
                      f"{shown!r}  {status}", flush=True)

    rows = [r for r in jsonl(args.out) if r["id"] in {it[key] for it in items}]
    summary = summarize_d1(rows, args.stage)
    print("\nsummary:\n" + json.dumps(summary, indent=2))
    if summary["errors"]:
        print(f"{summary['errors']} request errors — re-run with --resume to retry them")
    if args.gate:
        fails = d1_gate_failures(summary, args.stage)
        print("GATE: " + ("PASS" if not fails else "FAIL — " + "; ".join(fails)))
        if fails:
            sys.exit(2)


if __name__ == "__main__":
    main()
