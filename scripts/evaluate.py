"""Evaluate a model (base or base+LoRA adapter) on the held-out eval split.

GPU (via sbatch, see slurm/eval.sbatch):
    uv run scripts/evaluate.py --tag base                                  # zero-shot baseline
    uv run scripts/evaluate.py --adapter outputs/runs/lora_train_500/adapter --tag lora_500

Quick CPU sanity check (slow — use --max-samples):
    uv run scripts/evaluate.py --tag base --max-samples 2

Writes to outputs/eval/<tag>/:
    predictions.json — per-sample: prompt id, generated text, parsed waypoints, metrics
    metrics.json     — aggregate: parse rate, in-range rate, waypoint errors (mean/median)

All models are evaluated on the same fixed eval split (data/prepared/eval.json), so
tags are directly comparable; scripts/compare_evals.py builds the scaling table/plot.
"""

import argparse
import json
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

from rover_vlm.eval import (
    aggregate_choice_metrics,
    aggregate_metrics,
    aggregate_habitat_metrics,
    choice_metrics,
    habitat_metrics,
    normalize_prediction,
    parse_choice_answer,
    parse_path_answer,
    parse_waypoints,
    trajectory_metrics,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
IMAGE_ROOT = REPO_ROOT / "data" / "sharerobot" / "trajectory"
MODEL_ID = "Qwen/Qwen3.5-2B"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--adapter", type=Path, default=None, help="LoRA adapter dir; omit for base model")
    p.add_argument("--tag", required=True, help="name for outputs/eval/<tag>/")
    p.add_argument("--task", choices=["sharerobot", "habitat", "choice"], default="sharerobot")
    p.add_argument("--out-dir", type=Path, default=REPO_ROOT / "outputs" / "eval")
    p.add_argument("--eval-file", type=Path, default=REPO_ROOT / "data/prepared/eval.json")
    p.add_argument("--batch-size", type=int, default=8)
    # base model tends to answer in verbose grounding-JSON (~20 tokens/point), so
    # leave headroom; fine-tuned outputs are ~10 tokens/point
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--max-samples", type=int, default=None)
    return p.parse_args()


def build_messages(prompt: str, image: Image.Image) -> list[dict]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]


def main() -> None:
    args = parse_args()
    use_cuda = torch.cuda.is_available()
    dtype = torch.bfloat16 if use_cuda else torch.float32
    device = "cuda" if use_cuda else "cpu"
    print(f"tag={args.tag} adapter={args.adapter} device={device}")

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    processor.tokenizer.padding_side = "left"  # decoder-only batched generation
    model = AutoModelForImageTextToText.from_pretrained(MODEL_ID, dtype=dtype).to(device)
    if args.adapter is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    records_in = json.loads(args.eval_file.read_text())
    if args.max_samples:
        records_in = records_in[: args.max_samples]
    print(f"eval samples: {len(records_in)} (from {args.eval_file.name})")

    results = []
    t0 = time.time()
    for start in range(0, len(records_in), args.batch_size):
        chunk = records_in[start : start + args.batch_size]
        images, texts, gts = [], [], []
        for rec in chunk:
            image = Image.open(IMAGE_ROOT / rec["image"][0]).convert("RGB")
            prompt = rec["conversations"][0]["value"].replace("<image>\n", "").replace("<image>", "")
            images.append(image)
            texts.append(
                processor.apply_chat_template(
                    build_messages(prompt, image), tokenize=False, add_generation_prompt=True
                )
            )
            # choice scoring needs the accepted set, which the answer string can't carry
            if args.task == "choice":
                gts.append(rec["choice_meta"])
            else:
                gts.append(json.loads(rec["conversations"][1]["value"]))

        batch = processor(text=texts, images=images, padding=True, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(**batch, max_new_tokens=args.max_new_tokens, do_sample=False)
        new_tokens = out[:, batch["input_ids"].shape[1] :]
        decoded = processor.batch_decode(new_tokens, skip_special_tokens=True)

        for rec, gt, text in zip(chunk, gts, decoded):
            if args.task == "choice":
                parsed = parse_choice_answer(text)
                # unlike the other tasks, score even an unparseable answer: a missing
                # choice is a wrong choice, and accuracy must not be inflated by drops
                metrics = choice_metrics(parsed, gt)
                rescaled = False
            elif args.task == "habitat":
                parsed = parse_path_answer(text)
                metrics = habitat_metrics(parsed, gt) if parsed else None
                rescaled = False
            else:
                parsed = parse_waypoints(text)
                if parsed:
                    # score per-mille-style outputs (base model habit) charitably;
                    # in_range_rate still records raw convention adherence
                    scored, rescaled = normalize_prediction(parsed)
                    metrics = trajectory_metrics(scored, gt)
                else:
                    metrics, rescaled = None, False
            results.append(
                {
                    "id": rec["id"],
                    "generated": text,
                    "parsed": parsed,
                    "rescaled": rescaled,
                    "gt": gt,
                    "metrics": metrics,
                }
            )
        done = start + len(chunk)
        print(f"  {done}/{len(records_in)} ({(time.time() - t0) / done:.2f}s/sample)", flush=True)

    aggregators = {
        "choice": aggregate_choice_metrics,
        "habitat": aggregate_habitat_metrics,
        "sharerobot": aggregate_metrics,
    }
    summary = aggregators[args.task](results)
    summary["tag"] = args.tag
    summary["adapter"] = str(args.adapter) if args.adapter else None

    out_dir = args.out_dir / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "predictions.json").write_text(json.dumps(results, indent=1))
    (out_dir / "metrics.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"saved -> {out_dir}")


if __name__ == "__main__":
    main()
