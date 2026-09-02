"""Six prompt revisions on one page: how the Cosmos3 trace prompt converged on v6.

Login node (CPU-only). Reads the finished JSONL in outputs/traces/ — no GPU, no server:

    uv run scripts/visualize_traces.py
    uv run scripts/visualize_traces.py --no-images --out outputs/traces_light.html

The evidence for choosing v6 is otherwise spread across eleven JSONL files and a comment
block in src/rover_vlm/traces.py. This puts three things side by side:

  gate metrics   trace yield, verifiable, answer match, leakage, word spread, truncation,
                 per prompt version per task — the numbers scripts/label_traces.py prints,
                 each column carrying a definition of what it does and does not mean.
  frame gallery  one frame, and for every version that covered it the prompt that was sent
                 beside the trace that came back. Earlier versions' sample ids are a strict
                 subset of v6's, so the comparison is exact. Prompts are re-rendered per
                 frame through that version's own code — a recovered prompt string belongs
                 to one sample and would show the wrong ground truth anywhere else.
  prompt diffs   consecutive prompts diffed line by line, with the failure that forced
                 each revision attached to the diff that embodies it.

Leakage is RE-SCORED with today's detector rather than read from the stored field: the
regex list was widened after the v1 run, so the stored numbers were produced by different
detectors and are not comparable to each other. Both are reported.

Writes a self-contained outputs/traces_report.html.
"""

import argparse
import difflib
import json
import re
import sys
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw

from rover_vlm.habitat_choice import DATASET_ROOT, render_choice_image
from rover_vlm.overlay import draw_goal, draw_path, embed_jpeg
from rover_vlm.traces import PROMPT_VERSION, build_prompt, leakage_spans

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT = REPO_ROOT / "outputs" / "traces_report.html"

TASKS = {
    "choice": ("Path classification", "data/prepared_habitat_choice_v2/eval.json"),
    "path": ("Path regression", "data/prepared_habitat_v2/eval.json"),
}
GT_BLUE = (42, 120, 214)
TARGET_WORDS = (60, 120)

# Why each revision happened. What each revision *did* is not written here — it is diffed
# straight from the recovered prompt text (see scripts/recover_trace_prompts.py), because
# prose drifts: the comment block in traces.py credits v3 with removing the furniture
# example, and the recovered prompts show that landed at v4, one run later. Each note is
# attached to its version's diff rather than rendered as a separate timeline.
CHANGELOG = [
    {
        "version": "v1", "scope": "choice",
        "failure": "Harvested Cosmos's &lt;think&gt; block as the trace, which is the wrong slot. "
                   "&lt;think&gt; is private deliberation about <em>our request</em>, so every trace "
                   "opened \"Okay, let's see. The user wants me to explain why candidate 1 is...\" — "
                   "meta-reasoning about a labelling job, not a rover reasoning about a room.",
    },
    {
        "version": "v2", "scope": "both",
        "failure": "Fixes the voice by making the reasoning a <em>product</em> in the answer JSON and "
                   "conceding &lt;think&gt; to Cosmos as scratch. Introduces two problems that take "
                   "four more revisions to clear: the scratch is explicitly unbounded, and the "
                   "illustrative furniture in the rules is about to get parroted.",
    },
    {
        "version": "v3", "scope": "path",
        "failure": "Echoing the whole waypoint list back cost ~200 tokens and pushed replies into "
                   "pretty-printed fenced JSON that lost the &lt;answer&gt; tags. Three numbers catch "
                   "the same drift, so the answer echoes only the goal.",
    },
    {
        "version": "v4", "scope": "both",
        "failure": "2 of 12 v2 path traces parroted \"the sofa and the kitchen counter\" straight out "
                   "of the prompt, inventing objects that were not in the frame — illustrative nouns "
                   "in a labelling prompt become hallucinations in the labels. Note this landed "
                   "<em>after</em> the v3 run, so v2 and v3 labels were both produced with the "
                   "example still in place.",
    },
    {
        "version": "v5", "scope": "both",
        "failure": "The unbounded scratch from v2 finally bit: on 2 of 20 choice samples Cosmos spent "
                   "~3,300 words deliberating and hit the token cap before ever closing "
                   "&lt;/think&gt;, so those samples yielded nothing at all.",
    },
    {
        "version": "v6", "scope": "both",
        "failure": "Shortening the scratch made Cosmos skip the JSON envelope and write the reasoning "
                   "as bare prose on 19 of 30 choice samples. The trace still survived via salvage, "
                   "but the echoed answer — the only check that it was looking at the right sample — "
                   "did not.",
    },
]

# What each column of the metrics table actually measures. These carry the two caveats a
# reader cannot recover from the numbers — that leakage is re-scored, and that v6's zero
# truncations are partly a raised token cap — so they sit on the column they qualify
# instead of in a preamble that is easy to scroll past.
METRIC_DEFS = [
    {"col": "Version", "title": "Prompt revision",
     "body": "Which revision of the labelling prompt produced this run. Not every version "
             "ran on both tasks: choice has no v3 and path has no v1, because a revision was "
             "only re-run where it changed something. v2diag was a 3-sample diagnostic "
             "re-run of v2, not a revision of its own."},
    {"col": "n", "title": "Samples in the run",
     "body": "Early versions were smoke tests. v1 is n=2 and settles nothing on its own — it "
             "is here because it is where the approach was wrong in an instructive way, not "
             "because its rates are comparable to v6's n=30."},
    {"col": "Trace", "title": "Yielded any reasoning at all",
     "body": "How many replies produced usable reasoning text. The parser is deliberately "
             "forgiving — missing &lt;answer&gt; tags, fenced JSON and bare prose are all "
             "salvaged — so a miss here means the reply was unusable even after salvage, "
             "usually because generation was cut off mid-thought."},
    {"col": "Verified", "title": "Echoed back the right answer — a drift check, not accuracy",
     "body": "Cosmos is <em>shown</em> the answer and asked to justify it, so this can never "
             "measure whether it is right. It checks that the reply was about the sample it "
             "was sent: the answer block re-states the choice (or the goal), and a mismatch "
             "means the reply wandered, which makes its trace untrustworthy even though the "
             "label is still correct. Most misses are a skipped JSON envelope, which is "
             "unverifiable rather than wrong — the two are kept apart."},
    {"col": "Leak (now)", "title": "The phase gate — must be zero",
     "body": "Traces that betray the answer was handed over (\"the correct one\", \"as "
             "stated\", \"the answer is\"). This matters more than anything else in the "
             "table: the trace becomes Qwen's inner monologue, and at inference Qwen will "
             "<em>not</em> know the answer, so a trace reasoning backwards from it teaches "
             "exactly the wrong reflex and is worse than no trace at all. Re-scored here "
             "with the current detector so all versions are judged alike."},
    {"col": "Leak (as run)", "title": "What the run recorded at the time",
     "body": "The detector's pattern list was widened after v1 produced leaks it missed, so "
             "each run's stored count came from a different detector and they cannot be "
             "compared with each other. Kept beside the re-scored column so the difference "
             "stays visible instead of being quietly reconciled."},
    {"col": "Truncated", "title": "Hit the token cap before finishing",
     "body": "Replies that stopped at the generation limit (finish_reason=length) and lost "
             "the sample outright. Reading this column needs one caveat: v6 ran with "
             "--max-tokens 8192 where v4 and v5 used 4096, so its zero reflects the raised "
             "cap as well as the tighter scratch bound in the prompt."},
    {"col": "Words", "title": "Length spread of the recovered traces",
     "body": "Box is p25–p75, the heavy tick is the median, the thin line is min–max, and "
             "the shaded band is the 60–120 word target the prompt asks for. The axis is "
             "capped at 250 so that band stays legible; an arrow at the right edge means "
             "values run past the cap (v1's median alone is 412)."},
    {"col": "Median", "title": "Median trace length",
     "body": "Over the samples that produced a trace, matching how scripts/label_traces.py "
             "reports it at the end of a run — so the page and the CLI cannot disagree."},
]


def prompt_diff(before, after):
    """Consecutive prompts as classified lines, so the page can render a real diff.

    The prompts are what actually changed between runs, so the diff is the comparison —
    the changelog prose only supplies the why.
    """
    out = []
    for line in difflib.unified_diff(before.splitlines(), after.splitlines(), lineterm="", n=2):
        if line.startswith(("---", "+++")):
            continue
        op = ("hunk" if line.startswith("@@")
              else "add" if line.startswith("+")
              else "del" if line.startswith("-") else "ctx")
        out.append({"op": op, "t": line if op == "hunk" else line[1:] if op != "ctx" else line[1:]})
    return out


def historical_builder(source, version, task):
    """Import a recovered traces.py snapshot and hand back its prompt builder.

    A recovered *prompt* is rendered for one sample only, so the per-frame view cannot
    reuse it — it has to re-render each frame through the prompt code of that moment, or
    every card would show the first sample's ground truth.
    """
    import importlib.util

    tmp = REPO_ROOT / "src" / "rover_vlm" / f"_hist_{task}_{version}.py"
    tmp.write_text(source)
    try:
        spec = importlib.util.spec_from_file_location(f"rover_vlm._hist_{task}_{version}", tmp)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
    finally:
        tmp.unlink(missing_ok=True)
    builder = getattr(mod, "build_prompt", None)
    return (lambda rec: builder(task, rec)) if builder else getattr(mod, f"{task}_trace_prompt")


def prompt_history(path, task, versions, notes=None):
    """The literal prompt behind each run, newest last, each diffed against the previous.

    Recovered by scripts/recover_trace_prompts.py from the session transcript; the repo
    itself only ever held v6. Returns [] when that file is absent, and the page then says
    so rather than showing a reconstruction that is not there.
    """
    if not path.exists():
        return []
    history = json.loads(path.read_text())
    chain, previous = [], None
    for v in versions:
        entry = history.get(f"{task}_{v}")
        if entry is None:
            chain.append({"version": v, "text": None, "diff": None, "from": None,
                          "why": (notes or {}).get(v)})
            continue
        text = entry["prompt"]
        chain.append({"version": v, "text": text, "sample": entry["sample"],
                      "words": len(text.split()),
                      "diff": prompt_diff(previous[1], text) if previous else None,
                      "from": previous[0] if previous else None,
                      # The changelog's "why" rides on the diff that embodies it, rather
                      # than repeating the whole timeline as its own section.
                      "why": (notes or {}).get(v)})
        previous = (v, text)
    return chain


def version_key(version):
    """Sort v1 < v2 < v2diag < v3 ... — the diagnostic re-run sits with its own version."""
    m = re.match(r"v(\d+)(.*)$", version)
    return (int(m.group(1)), m.group(2)) if m else (10**6, version)


def quantiles(values):
    """p25/median/p75 plus range, on an already-small sample. None if there is nothing."""
    if not values:
        return None
    s = sorted(values)
    pick = lambda q: s[min(len(s) - 1, int(q * len(s)))]  # noqa: E731
    return {"min": s[0], "p25": pick(0.25), "med": pick(0.5), "p75": pick(0.75), "max": s[-1]}


def summarize(records):
    """The gate metrics for one run, computed the way scripts/label_traces.py prints them.

    Leakage is counted twice: `leakStored` as each run recorded it at the time, and
    `leakNow` re-scored with the current detector. Only the second is comparable across
    versions, because the pattern list was widened after v1 produced leaks it missed.
    """
    n = len(records)
    ok = [r for r in records if r.get("trace")]
    errors = [r for r in records if r.get("error")]
    finish = {}
    for r in records:
        if not r.get("error"):
            finish[r.get("finish_reason") or "none"] = finish.get(r.get("finish_reason") or "none", 0) + 1
    return {
        "n": n,
        "traces": len(ok),
        "verifiable": sum(1 for r in records if r.get("verifiable")),
        "match": sum(1 for r in records if r.get("answer_matches_gt")),
        "leakStored": sum(1 for r in records if r.get("leakage")),
        "leakNow": sum(1 for r in ok if leakage_spans(r["trace"])),
        "errors": len(errors),
        "length": finish.get("length", 0),
        "words": quantiles([r["words"] for r in ok if r.get("words")]),
        "scratch": quantiles([r["scratch_words"] for r in records if r.get("scratch_words")]),
        "latency": quantiles([r["latency_s"] for r in records if r.get("latency_s")]),
        "finish": finish,
    }


def load_runs(traces_dir):
    """{task: {version: [record, ...]}} from outputs/traces/<task>_<version>.jsonl."""
    runs = {}
    for path in sorted(traces_dir.glob("*.jsonl")):
        task, _, version = path.stem.partition("_")
        if task not in TASKS or not version:
            print(f"  skipping {path.name} — not a <task>_<version>.jsonl for a known task")
            continue
        runs.setdefault(task, {})[version] = [json.loads(line) for line in path.open() if line.strip()]
    return runs


def frame_image(task, record, max_px, quality, sample_dir=None, tmp_dir=None):
    """The frame as a data URI. Path frames get the ground-truth route drawn on first.

    Choice frames are the training composites, which draw every candidate solid — the
    model is meant to infer occlusion rather than be told it. Given the source sample they
    are re-rendered with `dashed=True` instead, breaking occluded stretches into dashes.
    That is a reading aid for this page only: it is what lets you see that a candidate the
    trace calls blocked really does pass behind something.
    """
    if task == "choice" and sample_dir is not None:
        out = Path(tmp_dir) / f"{record['id']}.jpg"
        render_choice_image(sample_dir, out, dashed=True)
        return embed_jpeg(Image.open(out), max_px=max_px, quality=quality)

    img = Image.open(record["image"][0]).convert("RGB")
    if task == "path":
        gt = json.loads(record["conversations"][1]["value"])
        d = ImageDraw.Draw(img)
        W, H = img.size
        draw_path(d, gt["path"], GT_BLUE, W, H)
        draw_goal(d, gt["goal"], W, H)
    return embed_jpeg(img, max_px=max_px, quality=quality)


def build_frames(task, versions, runs, split_records, max_px, quality, with_images, builders,
                 sample_dirs=None, tmp_dir=None):
    """One card per frame: the image, its ground truth, and each version's prompt + trace."""
    latest = versions[-1]
    by_version = {v: {r["id"]: r for r in runs[v]} for v in versions}
    frames, undashed = [], 0
    for rec in runs[latest]:
        sample = split_records.get(rec["id"])
        if sample is None:
            print(f"  {rec['id']} is not in the split — skipping its card")
            continue
        entry = {"id": rec["id"], "traces": {}}
        if with_images:
            src = (sample_dirs or {}).get(rec["id"])
            undashed += task == "choice" and src is None
            entry["img"] = frame_image(task, sample, max_px, quality, src, tmp_dir)
        if task == "choice":
            entry["meta"] = sample["choice_meta"]
        else:
            entry["meta"] = json.loads(sample["conversations"][1]["value"])
        for v in versions:
            r = by_version[v].get(rec["id"])
            if r is None:
                continue
            build = builders.get(v)
            entry["traces"][v] = {
                "trace": r.get("trace"),
                "words": r.get("words") or 0,
                "leaks": leakage_spans(r.get("trace")),
                "verifiable": bool(r.get("verifiable")),
                "match": bool(r.get("answer_matches_gt")),
                "finish": r.get("finish_reason"),
                "scratch": r.get("scratch_words"),
                "error": r.get("error"),
                # Rendered through this version's own prompt code, on this frame.
                "prompt": build(sample) if build else None,
            }
        frames.append(entry)
    if undashed:
        print(f"  {undashed}/{len(frames)} choice frames had no source sample dir — those "
              f"show the solid training composite instead of the dashed reading aid")
    return frames


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--traces-dir", type=Path, default=REPO_ROOT / "outputs" / "traces")
    p.add_argument("--choice-split", type=Path,
                   default=REPO_ROOT / TASKS["choice"][1])
    p.add_argument("--path-split", type=Path, default=REPO_ROOT / TASKS["path"][1])
    p.add_argument("--max-px", type=int, default=560, help="long side of the embedded frames")
    p.add_argument("--quality", type=int, default=80, help="JPEG quality for the frames")
    p.add_argument("--no-images", action="store_true", help="metrics and text only, much smaller")
    p.add_argument("--dataset-root", type=Path, default=DATASET_ROOT,
                   help="Habitat source samples, for re-rendering choice frames dashed")
    p.add_argument("--solid", action="store_true",
                   help="show the choice training composites verbatim instead of dashing "
                        "the occluded stretches")
    p.add_argument("--prompt-history", type=Path,
                   default=REPO_ROOT / "scripts" / "trace_prompt_history.json",
                   help="recovered per-run prompts (scripts/recover_trace_prompts.py)")
    p.add_argument("--out", type=Path, default=OUT)
    args = p.parse_args()

    runs = load_runs(args.traces_dir)
    if not runs:
        raise SystemExit(f"no <task>_<version>.jsonl under {args.traces_dir} — run label_traces.py first")

    splits = {"choice": args.choice_split, "path": args.path_split}
    notes = {e["version"]: e["failure"] for e in CHANGELOG}
    data = {"promptVersion": PROMPT_VERSION, "targetWords": list(TARGET_WORDS),
            "metricDefs": METRIC_DEFS, "dashed": not args.solid, "tasks": {}, "prompts": {}}

    dashed = not (args.solid or args.no_images)
    sample_dirs = ({d.name: d for d in args.dataset_root.glob("*/samples/*/")}
                   if dashed else {})
    if dashed and not sample_dirs:
        print(f"  no samples under {args.dataset_root} — choice frames stay solid")

    tmp = tempfile.TemporaryDirectory()
    for task, (label, _) in TASKS.items():
        if task not in runs:
            print(f"  no runs for task {task!r} — omitted from the page")
            continue
        versions = sorted(runs[task], key=version_key)
        records = json.loads(splits[task].read_text())
        split_records = {r["id"]: r for r in records}

        history = json.loads(args.prompt_history.read_text()) if args.prompt_history.exists() else {}
        builders = {v: historical_builder(history[f"{task}_{v}"]["source"], v, task)
                    for v in versions if f"{task}_{v}" in history}

        data["tasks"][task] = {
            "label": label,
            "split": str(splits[task].relative_to(REPO_ROOT)),
            "versions": [
                dict(version=v, diagnostic=not v[1:].isdigit(), **summarize(runs[task][v]))
                for v in versions
            ],
            "frames": build_frames(task, versions, runs[task], split_records,
                                   args.max_px, args.quality, not args.no_images, builders,
                                   sample_dirs, tmp.name),
        }
        data["prompts"][task] = prompt_history(args.prompt_history, task, versions, notes)
        # The live prompt is rendered independently and must equal the recovered latest, or
        # the reconstruction has gone stale against the code it claims to reproduce.
        live = build_prompt(task, split_records[runs[task][versions[-1]][0]["id"]])
        newest = next((e for e in reversed(data["prompts"][task]) if e["text"]), None)
        if newest is None:
            data["prompts"][task] = [{"version": versions[-1], "text": live, "words": len(live.split()),
                                      "diff": None, "from": None, "why": notes.get(versions[-1])}]
            print(f"  no recovered prompt history for {task} — showing only the live prompt")
        elif newest["text"] != live:
            raise SystemExit(
                f"{task}: recovered {newest['version']} prompt differs from what traces.py builds "
                f"today. Re-run scripts/recover_trace_prompts.py, or the page will show a prompt "
                f"that no longer matches the code.")

    tmp.cleanup()

    html = (Path(__file__).parent / "_traces_report.html").read_text()
    # A trace containing "</script>" would end the block early; nothing does today, but the
    # page is generated from model output and must not depend on that staying true.
    html = html.replace("/*__DATA__*/null", json.dumps(data).replace("</", "<\\/"))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(html)

    print(f"wrote {args.out}  ({args.out.stat().st_size / 1024 / 1024:.1f} MB)")
    for task, t in data["tasks"].items():
        print(f"  {task}: {len(t['versions'])} versions, {len(t['frames'])} frames")
        for r in t["versions"]:
            w = r["words"]["med"] if r["words"] else 0
            print(f"    {r['version']:7} n={r['n']:3}  trace {r['traces']:3}  verified {r['match']:3}"
                  f"  leak {r['leakNow']:3} (stored {r['leakStored']})  med {w:4}w")



if __name__ == "__main__":
    main()
