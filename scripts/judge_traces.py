"""VLM-as-judge scoring of a model's generated reasoning traces (the trustworthy tier).

GPU (via sbatch, see slurm/judge_traces.sbatch):
    uv run scripts/judge_traces.py \
        --eval-dir outputs/eval_habitat_v2_traced --tag habitat_train_full_traced

For each eval frame it shows the JUDGE model the image plus the student's <think> reasoning
and asks — reference-free, grounded in the picture itself, not in any teacher trace — whether
the reasoning is accurate about THIS image. This is the metric the literature trusts for
reasoning quality; the cheap ROUGE/occlusion numbers in the report are only a floor.

Judge: nvidia/Cosmos-Reason2-8B by default — a DIFFERENT model from both the trace teacher
(Cosmos3-Nano) and the student (Qwen3.5), already cached, loads via transformers
(AutoModelForImageTextToText, qwen3_vl arch), no vLLM. Caveat: it is Qwen-derived, so a mild
self-preference toward the Qwen student is possible; swap a non-Qwen judge with --model-id to
check. Each sample is scored 1-5 on faithfulness / occlusion / coherence plus a hallucinated-
object list; writes <out-dir>/<tag>/judge.jsonl (one row per id) and judge_metrics.json.

Resumable: rows already in judge.jsonl are skipped, so a timed-out job just resubmits.
CPU-less machines cannot run this — it needs the GPU; iterate on the prompt with --max-samples.
"""

import argparse
import json
import re
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_JUDGE = "nvidia/Cosmos-Reason2-8B"
EVAL_FILE = REPO_ROOT / "data" / "prepared_habitat_v2" / "eval.json"

JUDGE_PROMPT = (
    "<image>\n"
    "You are grading a rover's written reasoning about this first-person photo. The rover had to "
    "find a walkable path to a goal straight ahead and say which points are hidden behind "
    "obstacles. Here is the rover's reasoning, verbatim:\n\n"
    "\"{reasoning}\"\n\n"
    "Judge ONLY whether this reasoning is accurate about THIS image — not whether it sounds "
    "fluent. Reply with one strict JSON object and nothing else:\n"
    '{{"faithful": <1-5, integer: does it describe surfaces/objects actually visible here, '
    'inventing nothing>, "occlusion": <1-5: does it correctly identify what is or is not hidden '
    'behind obstacles>, "coherent": <1-5: internally consistent and on-task>, '
    '"hallucinations": [<short names of objects it mentions that are NOT in the image>]}}'
)

_OBJ = re.compile(r"\{.*\}", re.S)
SCORES = ("faithful", "occlusion", "coherent")


def reasoning_of(generated):
    """The <think> reasoning from a predictions.json `generated` string (text before the first
    </think>; the whole string if the tag is absent)."""
    i = (generated or "").find("</think>")
    return (generated[:i] if i >= 0 else (generated or "")).strip()


def parse_verdict(text):
    """Pull the JSON verdict out of the judge's output; None if unusable."""
    m = _OBJ.search(text or "")
    if not m:
        return None
    try:
        v = json.loads(m.group(0))
    except ValueError:
        return None
    if not isinstance(v, dict) or not all(k in v for k in SCORES):
        return None
    out = {}
    for k in SCORES:
        try:
            out[k] = max(1, min(5, int(round(float(v[k])))))
        except (ValueError, TypeError):
            return None
    h = v.get("hallucinations", [])
    out["hallucinations"] = [str(x) for x in h] if isinstance(h, list) else []
    return out


def aggregate(rows):
    scored = [r for r in rows if r.get("verdict")]
    n = len(scored)
    agg = {"n_total": len(rows), "n_scored": n,
           "parse_rate": n / len(rows) if rows else 0.0}
    if n:
        for k in SCORES:
            vals = [r["verdict"][k] for r in scored]
            agg[f"{k}_mean"] = sum(vals) / n
        agg["hallucination_rate"] = sum(1 for r in scored if r["verdict"]["hallucinations"]) / n
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
    p.add_argument("--eval-file", type=Path, default=EVAL_FILE, help="id -> image lookup")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="default: <eval-dir>/<tag>/ alongside predictions.json")
    p.add_argument("--max-new-tokens", type=int, default=384)
    p.add_argument("--max-samples", type=int, default=None, help="prompt iteration")
    args = p.parse_args()

    preds = json.loads((args.eval_dir / args.tag / "predictions.json").read_text())
    if args.max_samples:
        preds = preds[: args.max_samples]
    id_to_image = {r["id"]: r["image"][0] for r in json.loads(args.eval_file.read_text())}

    out_dir = args.out_dir or (args.eval_dir / args.tag)
    out_dir.mkdir(parents=True, exist_ok=True)
    jpath = out_dir / "judge.jsonl"
    done = load_done(jpath)
    todo = [p for p in preds if p["id"] not in done and p["id"] in id_to_image]
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
            reasoning = reasoning_of(rec.get("generated", ""))
            image = Image.open(id_to_image[rec["id"]]).convert("RGB")
            prompt = JUDGE_PROMPT.format(reasoning=reasoning.replace("\n", " "))
            messages = [{"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt.replace("<image>\n", "")}]}]
            text = processor.apply_chat_template(messages, tokenize=False,
                                                 add_generation_prompt=True)
            batch = processor(text=[text], images=[image], return_tensors="pt").to(device)
            with torch.no_grad():
                gen = model.generate(**batch, max_new_tokens=args.max_new_tokens, do_sample=False)
            decoded = processor.batch_decode(gen[:, batch["input_ids"].shape[1]:],
                                             skip_special_tokens=True)[0]
            verdict = parse_verdict(decoded)
            sink.write(json.dumps({"id": rec["id"], "verdict": verdict, "raw": decoded}) + "\n")
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
