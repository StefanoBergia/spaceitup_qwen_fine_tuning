"""Training utilities: dataset + collator for Qwen3.5-2B LoRA fine-tuning.

Consumes the conversation-format JSON files produced by scripts/prepare_data.py.
The collator builds batches via the Qwen3.5 processor's chat template and masks
labels so the loss is computed only on the assistant response.

Reasoning traces (optional): a record may carry a top-level "reasoning" string
(attached by scripts/prepare_habitat_traces.py). When present, it is rendered inside
Qwen3.5's native <think>...</think> via the processor's `reasoning_content` field, and
the label mask is opened so the reasoning IS trained on. The mask boundary is the exact
`enable_thinking=True` generation prompt (ending at "<think>\\n"), so the trained region is
precisely what the model must produce at inference: the reasoning, then </think>, then the
answer. Records without "reasoning" render an empty <think></think> and train answer-only,
exactly as before — so plain and traced runs share one code path.
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
        # Optional reasoning trace -> Qwen's <think> block (see module docstring). None for
        # plain records, so the collator falls back to answer-only training.
        return {"id": rec["id"], "image": image, "prompt": prompt, "answer": answer,
                "reasoning": rec.get("reasoning")}


class TrajectoryCollator:
    """Batch samples with the Qwen processor; mask the prompt so loss covers only the
    assistant response. When a sample carries a "reasoning" trace, the response includes
    the <think> block and the mask boundary shifts so the reasoning is trained on too."""

    def __init__(self, processor):
        self.processor = processor
        self.processor.tokenizer.padding_side = "right"

    def _user_msg(self, sample: dict) -> dict:
        return {
            "role": "user",
            "content": [
                {"type": "image", "image": sample["image"]},
                {"type": "text", "text": sample["prompt"]},
            ],
        }

    def _assistant_msg(self, sample: dict) -> dict:
        msg = {"role": "assistant", "content": [{"type": "text", "text": sample["answer"]}]}
        if sample.get("reasoning"):
            # `reasoning_content` is the processor's official channel: it renders as
            # <think>\n{reasoning}\n</think>\n\n before the answer (verified against the
            # Qwen3.5-2B template). Putting the trace here — not in the answer text — keeps
            # the answer JSON exactly the ground-truth string.
            msg["reasoning_content"] = sample["reasoning"]
        return msg

    def _prompt_ids(self, sample: dict):
        """Token ids of the generation prompt that PRECEDES the trained region — i.e. the
        exact prefix the model is handed at inference. With reasoning, enable_thinking=True
        so the prompt ends at "<think>\\n" and the reasoning falls inside the loss; without
        it, the default empty-<think> boundary reproduces the prior answer-only behavior."""
        prompt_text = self.processor.apply_chat_template(
            [self._user_msg(sample)],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=bool(sample.get("reasoning")),
        )
        return self.processor(
            text=[prompt_text], images=[sample["image"]], return_tensors="pt"
        )["input_ids"][0]

    def __call__(self, samples: list[dict]) -> dict:
        texts = [
            self.processor.apply_chat_template(
                [self._user_msg(s), self._assistant_msg(s)], tokenize=False
            )
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
        for i, sample in enumerate(samples):
            prompt_ids = self._prompt_ids(sample)
            prompt_len = len(prompt_ids)
            # The generation prompt must be an exact token prefix of the full sequence, or
            # the mask would leak/clip the response (a silent mistrain — see the flat-loss
            # failure mode in unmatched_lora_targets). Fail loud if a template change ever
            # breaks the prefix property this collator relies on.
            if not torch.equal(batch["input_ids"][i, :prompt_len], prompt_ids):
                raise RuntimeError(
                    f"prompt/response token boundary misaligned for sample "
                    f"{sample.get('id')!r}; chat-template rendering changed."
                )
            labels[i, :prompt_len] = -100
        batch["labels"] = labels
        return batch
