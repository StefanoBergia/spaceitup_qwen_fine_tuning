"""LoRA fine-tune a Qwen3.5 vision model on prepared trajectory data.

GPU (via sbatch, see slurm/):
    uv run scripts/train.py --train-file data/prepared/train_500.json \\
        --output-dir outputs/runs/lora_500

Defaults to Qwen3.5-2B; --model-id selects another size (e.g. Qwen/Qwen3.5-0.8B).
The resolved model id is written to <output-dir>/train_config.json so a run is
self-describing.

CPU smoke test (login node, validates the whole pipeline in ~minutes):
    uv run scripts/train.py --smoke

--smoke trains 4 samples for 3 steps with a tiny LoRA on CPU and verifies that
the loss is finite and the adapter saves/reloads. Use it after any change to the
data pipeline or model setup, before submitting a GPU job.
"""

import argparse
import json
import time
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    Trainer,
    TrainingArguments,
)

from rover_vlm.training import (
    LORA_TARGETS_TEXT,
    LORA_TARGETS_VISION,
    TrajectoryCollator,
    TrajectoryDataset,
    unmatched_lora_targets,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
IMAGE_ROOT = REPO_ROOT / "data" / "sharerobot" / "trajectory"
DEFAULT_MODEL_ID = "Qwen/Qwen3.5-2B"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-id", default=DEFAULT_MODEL_ID, help="HF base model repo id")
    p.add_argument("--train-file", type=Path, default=REPO_ROOT / "data/prepared/train_500.json")
    p.add_argument("--output-dir", type=Path, default=REPO_ROOT / "outputs/runs/debug")
    p.add_argument("--epochs", type=float, default=2.0)
    p.add_argument("--max-steps", type=int, default=-1, help="override epochs if > 0")
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--train-vision", action="store_true", help="also apply LoRA to the vision tower")
    p.add_argument("--gradient-checkpointing", action="store_true")
    p.add_argument("--max-samples", type=int, default=None, help="truncate training set (debugging)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--smoke", action="store_true", help="tiny CPU run to validate the pipeline")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.smoke:
        args.max_samples = 4
        args.max_steps = 3
        args.batch_size = 2
        args.grad_accum = 1
        args.lora_r, args.lora_alpha = 4, 8
        args.output_dir = REPO_ROOT / "outputs/runs/smoke"
        print("== SMOKE TEST: 4 samples, 3 steps, tiny LoRA ==")

    use_cuda = torch.cuda.is_available()
    dtype = torch.bfloat16 if use_cuda else torch.float32
    print(f"model={args.model_id} device={'cuda' if use_cuda else 'cpu'} dtype={dtype}")

    processor = AutoProcessor.from_pretrained(args.model_id)
    model = AutoModelForImageTextToText.from_pretrained(args.model_id, dtype=dtype)
    if args.gradient_checkpointing:
        model.config.use_cache = False

    targets = LORA_TARGETS_TEXT + (LORA_TARGETS_VISION if args.train_vision else [])
    missing = unmatched_lora_targets(model, targets)
    if missing:
        raise SystemExit(
            f"LoRA targets matched no module in {args.model_id}: {missing}\n"
            "PEFT would silently skip these, training fewer parameters than intended. "
            "Check the module tree and update LORA_TARGETS_* in rover_vlm/training.py."
        )
    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=targets,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    dataset = TrajectoryDataset(args.train_file, IMAGE_ROOT, max_samples=args.max_samples)
    print(f"training samples: {len(dataset)} (from {args.train_file.name})")

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=1 if args.smoke else 10,
        save_strategy="no" if args.smoke else "epoch",
        bf16=use_cuda,
        gradient_checkpointing=args.gradient_checkpointing,
        remove_unused_columns=False,  # collator needs the raw sample dicts
        dataloader_num_workers=0 if args.smoke else 4,
        seed=args.seed,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=TrajectoryCollator(processor),
    )

    t0 = time.time()
    result = trainer.train()
    elapsed = time.time() - t0
    print(f"train done in {elapsed:.1f}s | final loss: {result.training_loss:.4f}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output_dir / "adapter")
    processor.save_pretrained(args.output_dir / "adapter")
    print(f"adapter saved -> {args.output_dir / 'adapter'}")

    # record what produced this adapter, so a run dir doesn't rely on recall
    config = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    config["train_samples"] = len(dataset)
    config["final_loss"] = result.training_loss
    (args.output_dir / "train_config.json").write_text(json.dumps(config, indent=2))

    if args.smoke:
        # verify the adapter reloads cleanly
        from peft import PeftModel

        base = AutoModelForImageTextToText.from_pretrained(args.model_id, dtype=dtype)
        PeftModel.from_pretrained(base, args.output_dir / "adapter")
        print("adapter reload: OK")
        assert torch.isfinite(torch.tensor(result.training_loss)), "non-finite loss"
        print("== SMOKE TEST PASSED ==")


if __name__ == "__main__":
    main()
