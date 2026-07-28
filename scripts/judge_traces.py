"""LLM-as-judge scoring of a model's reasoning traces AGAINST THE GROUND TRUTH.

GPU (via sbatch, see slurm/judge_traces.sbatch):
    uv run scripts/judge_traces.py \
        --eval-dir outputs/eval_habitat_v2_traced --tag habitat_train_full_traced

The judge is shown the KNOWN ground truth for each frame (the true path direction, whether the
goal is hidden, how many path points are occluded — derived from the eval label) plus the
model's <think> reasoning, and grades ONLY whether the reasoning AGREES WITH THE GROUND TRUTH.
No image and no teacher trace are used: this is reasoning-vs-truth, so the score is objective
and checkable rather than a vibes rating of fluency. That is the point — a reference-free judge
tends to hand out uniform high marks; grading against known truth forces discrimination.

Verdict per frame (strict JSON): direction_correct, occlusion_correct, contradicts_gt (bools)
and score (1-5 overall agreement). Aggregate: direction/occlusion accuracy, contradiction rate,
mean score. Writes <out-dir>/<tag>/judge.jsonl (resumable) + judge_metrics.json.

Judge model: nvidia/Cosmos-Reason2-8B by default (cached, transformers, no vLLM) — a different
model from both the trace teacher and the student. It is Qwen-derived, so swap a non-Qwen judge
via --model-id to rule out self-preference. Needs a GPU; iterate with --max-samples.
"""

import argparse
import json
import re
import time
from pathlib import Path

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_JUDGE = "nvidia/Cosmos-Reason2-8B"

JUDGE_PROMPT = (
    "You are grading a rover's written reasoning about a scene, against the KNOWN ground "
    "truth for that scene.\n\n"
    "Ground truth:\n"
    "- The traversable path to the goal heads {direction}.\n"
    "- The goal is {goalstate}.\n"
    "- {n_occluded} of {n_path} path points are hidden behind obstacles.\n\n"
    "The rover's reasoning was:\n\"{reasoning}\"\n\n"
    "Grade ONLY whether the reasoning AGREES WITH THE GROUND TRUTH above. Ignore how fluent it "
    "is, and ignore descriptive details the ground truth does not cover. Reply with ONE strict "
    "JSON object and nothing else:\n"
    '{{"direction_correct": true or false (does it describe heading roughly the ground-truth '
    'way), "occlusion_correct": true or false (does it correctly say whether the goal is '
    'hidden), "contradicts_gt": true or false (does it assert anything that conflicts with the '
    'ground truth), "score": <integer 1-5, overall agreement with the ground truth>}}'
)

_OBJ = re.compile(r"\{.*\}", re.S)
_BOOLS = ("direction_correct", "occlusion_correct", "contradicts_gt")


def reasoning_of(generated):
    """The <think> reasoning from a predictions.json `generated` string (text before the first
    </think>; the whole string if the tag is absent)."""
    i = (generated or "").find("</think>")
    return (generated[:i] if i >= 0 else (generated or "")).strip()


def gt_summary(gt):
    """Human-readable ground truth from a {path, goal} label, or None if unusable. Direction is
    the net image-x shift from the path start to the goal (coarse but checkable); occlusion is
    the goal's visibility flag (0 = hidden)."""
    if not gt or not gt.get("goal"):
        return None
    goal = gt["goal"]
    path = gt.get("path") or []
    start = path[0] if path else goal
    dx = goal[0] - start[0]
    direction = ("to the left" if dx < -0.08 else "to the right" if dx > 0.08
                 else "roughly straight ahead")
    return {
        "direction": direction,
        "goalstate": "hidden behind an obstacle" if goal[2] == 0 else "visible and unobstructed",
        "goal_hidden": goal[2] == 0,
        "n_occluded": sum(1 for p in path if p[2] == 0),
        "n_path": len(path),
    }


def _as_bool(x):
    if isinstance(x, bool):
        return x
    if isinstance(x, (int, float)):
        return x != 0
    if isinstance(x, str):
        return x.strip().lower() in ("true", "yes", "1", "y", "t")
    return None


def parse_verdict(text):
    """Extract the JSON verdict; None if unusable or a field is the wrong type."""
    m = _OBJ.search(text or "")
    if not m:
        return None
    try:
        v = json.loads(m.group(0))
    except ValueError:
        return None
    if not isinstance(v, dict) or "score" not in v or not all(k in v for k in _BOOLS):
        return None
    out = {}
    for k in _BOOLS:
        b = _as_bool(v[k])
        if b is None:
            return None
        out[k] = b
    try:
        out["score"] = max(1, min(5, int(round(float(v["score"])))))
    except (ValueError, TypeError):
        return None
    return out


def aggregate(rows):
    scored = [r for r in rows if r.get("verdict")]
    n = len(scored)
    agg = {"n_total": len(rows), "n_scored": n,
           "parse_rate": n / len(rows) if rows else 0.0}
    if n:
        agg["direction_accuracy"] = sum(1 for r in scored if r["verdict"]["direction_correct"]) / n
        agg["occlusion_accuracy"] = sum(1 for r in scored if r["verdict"]["occlusion_correct"]) / n
        agg["contradiction_rate"] = sum(1 for r in scored if r["verdict"]["contradicts_gt"]) / n
        agg["score_mean"] = sum(r["verdict"]["score"] for r in scored) / n
    return agg


def load_done(path):
    done = {}
    if path.exists():
        for line in path.open():
            if line.strip():
                r = json.loads(line)
                done[r["id"]] = r
    return done


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--eval-dir", type=Path, required=True,
                   help="tree holding <tag>/predictions.json to judge")
    p.add_argument("--tag", default="habitat_train_full_traced")
    p.add_argument("--model-id", default=DEFAULT_JUDGE)
    p.add_argument("--out-dir", type=Path, default=None,
                   help="default: <eval-dir>/<tag>/ alongside predictions.json")
    p.add_argument("--max-new-tokens", type=int, default=384)
    p.add_argument("--max-samples", type=int, default=None, help="prompt iteration")
    args = p.parse_args()

    preds = json.loads((args.eval_dir / args.tag / "predictions.json").read_text())
    if args.max_samples:
        preds = preds[: args.max_samples]

    out_dir = args.out_dir or (args.eval_dir / args.tag)
    out_dir.mkdir(parents=True, exist_ok=True)
    jpath = out_dir / "judge.jsonl"
    done = load_done(jpath)
    todo = [r for r in preds if r["id"] not in done and gt_summary(r.get("gt")) is not None]
    print(f"judge={args.model_id}  tag={args.tag}  to score={len(todo)}  "
          f"(already done {len(done)})", flush=True)

    if not torch.cuda.is_available():
        raise SystemExit("no GPU — run this via sbatch on the cluster, not the login node")
    device, dtype = "cuda", torch.bfloat16
    processor = AutoProcessor.from_pretrained(args.model_id)
    model = AutoModelForImageTextToText.from_pretrained(args.model_id, dtype=dtype).to(device)
    model.eval()

    t0 = time.time()
    with jpath.open("a") as sink:
        for i, rec in enumerate(todo, 1):
            g = gt_summary(rec["gt"])
            prompt = JUDGE_PROMPT.format(reasoning=reasoning_of(rec.get("generated", "")).replace("\n", " "), **g)
            messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
            text = processor.apply_chat_template(messages, tokenize=False,
                                                 add_generation_prompt=True)
            batch = processor(text=[text], return_tensors="pt").to(device)
            with torch.no_grad():
                gen = model.generate(**batch, max_new_tokens=args.max_new_tokens, do_sample=False)
            decoded = processor.batch_decode(gen[:, batch["input_ids"].shape[1]:],
                                             skip_special_tokens=True)[0]
            sink.write(json.dumps({"id": rec["id"], "verdict": parse_verdict(decoded),
                                   "raw": decoded}) + "\n")
            sink.flush()
            if i % 25 == 0 or i == len(todo):
                print(f"  {i}/{len(todo)}  ({(time.time()-t0)/i:.1f}s/it)", flush=True)

    rows = list(load_done(jpath).values())
    agg = aggregate(rows)
    agg.update({"tag": args.tag, "judge_model": args.model_id})
    (out_dir / "judge_metrics.json").write_text(json.dumps(agg, indent=2))
    print("\njudge metrics:", json.dumps(agg, indent=2), flush=True)


if __name__ == "__main__":
    main()
