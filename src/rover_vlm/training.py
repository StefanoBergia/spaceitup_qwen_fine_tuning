"""Training utilities: dataset + collator for Qwen3.5-2B LoRA fine-tuning.

Consumes the conversation-format JSON files produced by scripts/prepare_data.py.
The collator builds batches via the Qwen3.5 processor's chat template and masks
labels so the loss is computed only on the assistant answer (the waypoint list).
"""

import json
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset

# LoRA targets for Qwen3.5's hybrid attention (verified against the actual module tree):
#   - 6 full-attention layers:    q_proj, k_proj, v_proj, o_proj
#   - 18 linear-attention layers: in_proj_qkv, out_proj  (in_proj_z/b/a are small
#     gating/decay projections — left frozen)
#   - all 24 MLPs:                gate_proj, up_proj, down_proj
LORA_TARGETS_TEXT = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "in_proj_qkv", "out_proj",
    "gate_proj", "up_proj", "down_proj",
]
# Vision tower linears (opt-in via --train-vision)
LORA_TARGETS_VISION = ["qkv", "proj", "linear_fc1", "linear_fc2"]


def unmatched_lora_targets(model, targets: list[str]) -> list[str]:
    """Target names that match no module in `model`, using PEFT's suffix semantics.

    PEFT silently ignores a target that matches nothing, producing an adapter that
    trains less than intended (or not at all) — a failure that otherwise only shows
    up as a flat loss curve hours into a GPU job. Callers should treat a non-empty
    result as fatal. Verified to be empty for both Qwen3.5-2B and Qwen3.5-0.8B.
    """
    names = [name for name, _ in model.named_modules()]
    return [t for t in targets if not any(n == t or n.endswith(f".{t}") for n in names)]


class TrajectoryDataset(Dataset):
    """Prepared conversation-format samples: image + prompt -> waypoint list string."""

    def __init__(self, json_path: Path, image_root: Path, max_samples: int | None = None):
        self.records = json.loads(Path(json_path).read_text())
        if max_samples is not None:
            self.records = self.records[:max_samples]
        self.image_root = Path(image_root)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        rec = self.records[idx]
        image = Image.open(self.image_root / rec["image"][0]).convert("RGB")
        # The "<image>" placeholder is legacy LLaVA-style; Qwen's chat template
        # inserts image tokens itself, so strip it from the text.
        prompt = rec["conversations"][0]["value"].replace("<image>\n", "").replace("<image>", "")
        answer = rec["conversations"][1]["value"]
        return {"id": rec["id"], "image": image, "prompt": prompt, "answer": answer}


class TrajectoryCollator:
    """Batch samples with the Qwen processor; mask everything but the answer in labels."""

    def __init__(self, processor):
        self.processor = processor
        self.processor.tokenizer.padding_side = "right"

    def _messages(self, sample: dict, with_answer: bool) -> list[dict]:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": sample["image"]},
                    {"type": "text", "text": sample["prompt"]},
                ],
            }
        ]
        if with_answer:
            messages.append(
                {"role": "assistant", "content": [{"type": "text", "text": sample["answer"]}]}
            )
        return messages

    def __call__(self, samples: list[dict]) -> dict:
        texts = [
            self.processor.apply_chat_template(self._messages(s, with_answer=True), tokenize=False)
            for s in samples
        ]
        batch = self.processor(
            text=texts,
            images=[s["image"] for s in samples],
            padding=True,
            return_tensors="pt",
        )

        labels = batch["input_ids"].clone()
        labels[batch["attention_mask"] == 0] = -100
        # Mask the prompt part: tokenize each sample's prompt-only rendering (which
        # includes its image tokens) and blank out that prefix.
        for i, sample in enumerate(samples):
            prompt_text = self.processor.apply_chat_template(
                self._messages(sample, with_answer=False),
                tokenize=False,
                add_generation_prompt=True,
            )
            prompt_len = len(
                self.processor(
                    text=[prompt_text], images=[sample["image"]], return_tensors="pt"
                )["input_ids"][0]
            )
            labels[i, :prompt_len] = -100
        batch["labels"] = labels
        return batch
