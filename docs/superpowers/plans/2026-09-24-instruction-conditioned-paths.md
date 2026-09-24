# Instruction-Conditioned Paths Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the data, Cosmos labelling, training and eval pieces that test whether Qwen3.5-2B can be steered by a route instruction on shared-entry left/right Habitat pairs.

**Architecture:** A new `rover_vlm.instructions` module owns pair building, overlay rendering, the Cosmos prompts/parsers and the reject rules. `scripts/prepare_habitat_instr.py` derives pairs from the round-4 splits (`pairs` stage) and, after labelling, writes matched template/Cosmos datasets (`finalize` stage). `scripts/label_instructions.py` runs generate+verify against the existing Cosmos3-Nano endpoint. Training reuses `train_habitat.sbatch`; `evaluate.py --task habitat` gains an `instruction` block whenever records carry `alt_path`.

**Tech Stack:** Python 3.12, uv, PIL, numpy, openai client (vLLM endpoint), pytest; transformers+peft on the GPU side (unchanged).

**Spec:** `docs/superpowers/specs/2026-09-24-instruction-conditioned-paths-design.md`

## Global Constraints

- Shared 96-core login node: every compute command exports `OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4` (≤ 8 in sbatch jobs).
- Never run `scripts/train.py --smoke` (or load the 2B) on the login node — pytest is the CPU gate.
- Commands: `~/.local/bin/uv run ...` from the repo root. Tests: `~/.local/bin/uv run pytest -q`.
- Prompt: round-4 `endpoints` prompt + one instruction sentence; answer stays `{"path": [[x, y, v], ...]}`.
- Frames: original (`source == "original"`) round-4 frames only; both routes feasible, one `left` and one `right`; entry points ≤ `SHARED_ENTRY_TOL = 0.05` apart.
- Every pair has both records or neither, in every file.
- Output locations: `data/prepared_habitat_instr_v1/`, `outputs/instr_labels/`, `outputs/runs_instr_v1/<arm>/`, `outputs/eval_instr_v1/<set>/<tag>/`.
- Every new script gets a docstring saying what it does and how to run it; README gets one section.
- GPU jobs are submitted by the user; Claude writes sbatch files and reads pasted logs.

## Review Focus

1. Cosmos returns an instruction containing double quotes or newlines — the prompt must stay one clean sentence (`instruction_endpoints_prompt` sanitizes; test in Task 1).
2. One side of a frame fails to build (clipped away, not in FOV) — the whole pair is dropped, never a lone record (test in Task 2).
3. A resumed labelling run — error rows are retried, rows that ran but failed the check are kept as results (test in Task 5).
4. An unparseable prediction in eval — counts as not following and fails its pair; it is never dropped from the denominator (test in Task 7).
5. `finalize` with a label row missing for one record of a pair — the pair is dropped, no crash (test in Task 6).

---

### Task 1: Candidate choice and instruction prompt in `habitat_data`

**Files:**
- Modify: `src/rover_vlm/habitat_data.py` (`HABITAT_PROMPT_ENDPOINTS` neighbourhood, `select_correct_path`, `build_record`)
- Create: `tests/conftest.py` (synthetic two-sided sample fixture, reused by later tasks)
- Test: `tests/test_habitat_data.py`

**Interfaces:**
- Produces: `select_correct_path(fpv_paths, meta, index=None)`; `instruction_endpoints_prompt(start_wp, goal_wp, instruction) -> str`; `build_record(..., candidate=None, instruction=None)` — sidecar gains `candidate_index`, `instruction`.
- Produces (fixture): `two_sided_sample(tmp_path) -> Path` (a sample dir with a left candidate 0 = label, a right candidate 1, shared entry at (0.5, 1.0), goal (0.5, 0.586)).

- [ ] **Step 1: Write the fixture and failing tests**

`tests/conftest.py`:
```python
"""Shared pytest fixtures: a synthetic Habitat sample with a feasible left and right route."""
import json

import pytest
from PIL import Image


def write_sample(d, *, left_uv, right_uv, label=0, goal_uv=(256, 300), accepted=(0, 1)):
    d.mkdir(parents=True, exist_ok=True)
    fpv = {
        "image_size": [512, 512],
        "goal": {"uv": list(goal_uv), "hidden": False},
        "candidates": [
            {"runs": [{"hidden": False, "uv": [list(p) for p in left_uv]}]},
            {"runs": [{"hidden": False, "uv": [list(p) for p in right_uv]}]},
        ],
    }
    meta = {
        "sample_id": d.name, "scene_id": "scene_a", "flavor": "doorway",
        "label": label, "accepted": list(accepted), "correct_path_in_fov": True,
        "candidates": [{"label": "left", "feasible": 0 in accepted},
                       {"label": "right", "feasible": 1 in accepted}],
    }
    (d / "fpv_paths.json").write_text(json.dumps(fpv))
    (d / "meta.json").write_text(json.dumps(meta))
    Image.new("RGB", (64, 64), (120, 120, 120)).save(d / "fpv_enhanced.png")
    return d


@pytest.fixture
def two_sided_sample(tmp_path):
    return write_sample(
        tmp_path / "scene_a_c000",
        left_uv=[(256, 512), (140, 400), (256, 300)],
        right_uv=[(256, 512), (370, 400), (256, 300)],
    )
```

Append to `tests/test_habitat_data.py`:
```python
from rover_vlm.habitat_data import instruction_endpoints_prompt


def test_select_correct_path_index_overrides_label():
    fpv = {"candidates": [
        {"runs": [{"hidden": False, "uv": [[1.0, 2.0]]}]},
        {"runs": [{"hidden": True, "uv": [[5.0, 6.0]]}]},
    ]}
    meta = {"label": 0, "candidates": [{}, {}]}
    assert select_correct_path(fpv, meta) == [(1.0, 2.0, False)]
    assert select_correct_path(fpv, meta, index=1) == [(5.0, 6.0, True)]
    assert select_correct_path(fpv, meta, index=2) is None


def test_instruction_prompt_extends_endpoints_prompt():
    p = instruction_endpoints_prompt((0.5, 1.0, 1), (0.5, 0.6, 1), "Pass left of the sofa.")
    assert p.startswith("<image>\n")
    assert 'Drive it as this instruction says: "Pass left of the sofa."' in p
    assert "[0.5, 1.0]" in p and "[0.5, 0.6]" in p
    assert p.count("Coordinates are") == 1


def test_instruction_prompt_sanitizes_quotes_and_newlines():
    p = instruction_endpoints_prompt((0.5, 1.0, 1), (0.5, 0.6, 1), ' Go "left"\n of it. ')
    assert "\"Go 'left' of it.\"" in p
    assert "\n of" not in p


def test_build_record_with_candidate_and_instruction(two_sided_sample):
    left = build_record(two_sided_sample, framing="endpoints", instruction="Go left.")
    right = build_record(two_sided_sample, framing="endpoints", candidate=1,
                         instruction="Go right.")
    pl = json.loads(left["conversations"][1]["value"])["path"]
    pr = json.loads(right["conversations"][1]["value"])["path"]
    assert pl[0] == pr[0] and pl[-1] == pr[-1]
    assert min(x for x, _, _ in pl) < 0.5 < max(x for x, _, _ in pr)
    assert '"Go right."' in right["conversations"][0]["value"]
    assert right["habitat_meta"]["candidate_index"] == 1
    assert left["habitat_meta"]["candidate_index"] == 0
    assert right["habitat_meta"]["instruction"] == "Go right."


def test_build_record_instruction_requires_endpoints(two_sided_sample):
    with pytest.raises(ValueError):
        build_record(two_sided_sample, framing="goal", instruction="Go left.")
```

- [ ] **Step 2: Run to verify they fail**

Run: `~/.local/bin/uv run pytest -q tests/test_habitat_data.py`
Expected: FAIL (`ImportError: cannot import name 'instruction_endpoints_prompt'`).

- [ ] **Step 3: Implement**

In `src/rover_vlm/habitat_data.py`, after `HABITAT_PROMPT_ENDPOINTS`:
```python
# Instruction-conditioned framing (2026-09-24): the round-4 prompt plus one sentence naming
# the route in words. Derived from the endpoints template rather than copied, so the two
# can never drift apart -- the only difference a model sees is the instruction.
HABITAT_PROMPT_INSTRUCTION = HABITAT_PROMPT_ENDPOINTS.replace(
    " Coordinates are", ' Drive it as this instruction says: "{instr}". Coordinates are', 1
)
```
After `endpoints_prompt`:
```python
def instruction_endpoints_prompt(start_wp, goal_wp, instruction):
    """HABITAT_PROMPT_INSTRUCTION for two endpoints and a route instruction.

    The instruction arrives from Cosmos, so it is flattened to one line and its double
    quotes turned into single ones -- otherwise it would close the quoted span early.
    A trailing full stop is dropped because the template supplies the sentence's own.
    """
    text = " ".join(str(instruction).split()).replace('"', "'").rstrip(".")
    return HABITAT_PROMPT_INSTRUCTION.format(
        sx=start_wp[0], sy=start_wp[1], svis=_vis_phrase(start_wp),
        gx=goal_wp[0], gy=goal_wp[1], gvis=_vis_phrase(goal_wp), instr=text + ".",
    )
```
Note the rendered sentence reads `...instruction says: "Pass left of the sofa.". Coordinates...` — acceptable and exactly what the test asserts.

Replace `select_correct_path`:
```python
def select_correct_path(fpv_paths, meta, index=None):
    """Concatenate one candidate's runs into (u, v, hidden) points: the correct one
    (meta['label']) unless `index` names another. None if missing/misaligned."""
    label = meta.get("label") if index is None else index
    cands = fpv_paths.get("candidates", [])
    if label is None or not (0 <= label < len(cands)) or len(cands) != len(meta.get("candidates", [])):
        return None
    runs = cands[label].get("runs", [])
    return [(float(u), float(v), bool(run["hidden"])) for run in runs for (u, v) in run["uv"]]
```
In `build_record`: signature becomes
`def build_record(sample_dir, *, framing="legacy", goal_in_prompt=None, meta_extra=None, reasons=None, candidate=None, instruction=None):`
add to its docstring: "`candidate` builds the record from that candidate index instead of meta['label']; `instruction` (endpoints framing only) adds a route instruction to the prompt." Right after the `framing not in FRAMINGS` check:
```python
    if instruction is not None and framing != "endpoints":
        raise ValueError("an instruction is only defined on the endpoints framing")
```
Change `raw = select_correct_path(fpv, meta)` to `raw = select_correct_path(fpv, meta, candidate)`. Change the endpoints branch:
```python
        else:
            prompt = (endpoints_prompt(path_wps[0], goal_wp) if instruction is None
                      else instruction_endpoints_prompt(path_wps[0], goal_wp, instruction))
```
Add to `sidecar` (before the `aug` lines):
```python
        "candidate_index": meta.get("label") if candidate is None else candidate,
        "instruction": instruction,
```

- [ ] **Step 4: Run tests**

Run: `~/.local/bin/uv run pytest -q tests/test_habitat_data.py tests/test_prepare_habitat_manifest.py`
Expected: PASS. Then `~/.local/bin/uv run pytest -q` — all pass (sidecar gained two keys; nothing compares sidecars wholesale — if a test does, update it to include the new keys).

- [ ] **Step 5: Commit**
```bash
git add src/rover_vlm/habitat_data.py tests/conftest.py tests/test_habitat_data.py
git commit -m "feat: build a habitat record from any candidate, with an optional route instruction"
```

---

### Task 2: Pair building in `rover_vlm.instructions`

**Files:**
- Create: `src/rover_vlm/instructions.py`
- Test: `tests/test_instructions.py`

**Interfaces:**
- Consumes: `build_record(..., candidate, instruction)`, `instruction_endpoints_prompt`, `format_answer_goal` from Task 1; fixture `two_sided_sample`, helper `write_sample` (import from `conftest`).
- Produces: `SIDES`, `SHARED_ENTRY_TOL`, `TEMPLATE_INSTRUCTIONS`, `side_candidates(meta) -> dict | None`, `build_pair(sample_dir, *, texts=None, meta_extra=None, reasons=None) -> list[dict] | None` — returns `[left_rec, right_rec]` with ids `<sample>_L` / `<sample>_R` and `habitat_meta` keys `pair_id`, `side`, `alt_path`, `entry_gap`, `instruction_source`.

- [ ] **Step 1: Write failing tests**

`tests/test_instructions.py`:
```python
import json

from conftest import write_sample
from rover_vlm.instructions import (
    SHARED_ENTRY_TOL, TEMPLATE_INSTRUCTIONS, build_pair, side_candidates,
)


def _path(rec):
    return json.loads(rec["conversations"][1]["value"])["path"]


def test_side_candidates_prefers_label_and_needs_both_sides():
    meta = {"label": 2, "accepted": [0, 2, 1],
            "candidates": [{"label": "left", "feasible": True},
                           {"label": "right", "feasible": True},
                           {"label": "left", "feasible": True}]}
    assert side_candidates(meta) == {"left": 2, "right": 1}
    meta["accepted"] = [0, 2]
    assert side_candidates(meta) is None


def test_build_pair_two_records_same_prompt_but_instruction(two_sided_sample):
    left, right = build_pair(two_sided_sample)
    assert (left["id"], right["id"]) == ("scene_a_c000_L", "scene_a_c000_R")
    pl, pr = left["conversations"][0]["value"], right["conversations"][0]["value"]
    assert pl.replace(TEMPLATE_INSTRUCTIONS["left"], "X") == pr.replace(TEMPLATE_INSTRUCTIONS["right"], "X")
    assert left["habitat_meta"]["alt_path"] == _path(right)
    assert right["habitat_meta"]["alt_path"] == _path(left)
    assert _path(left)[0] == _path(right)[0]
    assert {left["habitat_meta"]["side"], right["habitat_meta"]["side"]} == {"left", "right"}
    assert left["habitat_meta"]["pair_id"] == "scene_a_c000"
    assert left["habitat_meta"]["instruction_source"] == "template"


def test_build_pair_snaps_small_entry_gap(tmp_path):
    d = write_sample(tmp_path / "s_c001",
                     left_uv=[(256, 512), (140, 400), (256, 300)],
                     right_uv=[(266, 512), (370, 400), (256, 300)])   # 10 px = 0.0195 apart
    left, right = build_pair(d)
    assert _path(left)[0] == _path(right)[0]
    assert 0 < left["habitat_meta"]["entry_gap"] <= SHARED_ENTRY_TOL


def test_build_pair_drops_distinct_entries(tmp_path):
    d = write_sample(tmp_path / "s_c002",
                     left_uv=[(0, 400), (140, 350), (256, 300)],
                     right_uv=[(512, 400), (370, 350), (256, 300)])
    reasons = {}
    assert build_pair(d, reasons=reasons) is None
    assert reasons == {"entry_differs": 1}


def test_build_pair_one_side_failing_drops_both(tmp_path):
    d = write_sample(tmp_path / "s_c003",
                     left_uv=[(256, 512), (140, 400), (256, 300)],
                     right_uv=[(900, 900), (950, 950)])            # clipped away entirely
    reasons = {}
    assert build_pair(d, reasons=reasons) is None
    assert reasons == {"record_failed": 1}


def test_build_pair_uses_given_texts(two_sided_sample):
    left, right = build_pair(two_sided_sample,
                             texts={"left": "Pass left of the lamp.", "right": "Pass right of the lamp."})
    assert '"Pass right of the lamp."' in right["conversations"][0]["value"]
    assert right["habitat_meta"]["instruction_source"] == "given"
```

- [ ] **Step 2: Run to verify failure**

Run: `~/.local/bin/uv run pytest -q tests/test_instructions.py`
Expected: FAIL (`ModuleNotFoundError: rover_vlm.instructions`).

- [ ] **Step 3: Implement**

`src/rover_vlm/instructions.py`:
```python
"""Instruction-conditioned paths: the route instruction is an INPUT, the path the output.

Pure and CPU-only. See docs/superpowers/specs/2026-09-24-instruction-conditioned-paths-design.md.

The experiment only means something where one image and one pair of endpoints admit two
correct routes, so the unit of data is a *pair*: a frame with a feasible route round each
side of the obstacle, both entering the frame at (almost) the same point. Each pair yields
two records that share image and endpoints and differ only in the instruction and the path;
a model that ignores the instruction can get at most one of them right.
"""

import json
import math
from pathlib import Path

from .habitat_data import build_record, format_answer_goal, instruction_endpoints_prompt

SIDES = ("left", "right")
#: Max distance between the two routes' entry points. Beyond it the entry point handed in
#: the prompt already reveals the side (57% of two-sided frames enter 0.5-1.0 apart).
SHARED_ENTRY_TOL = 0.05
TEMPLATE_INSTRUCTIONS = {"left": "Go around to the left.", "right": "Go around to the right."}


def _skip(reasons, reason):
    if reasons is not None:
        reasons[reason] = reasons.get(reason, 0) + 1
    return None


def side_candidates(meta):
    """{'left': i, 'right': j}: one feasible accepted candidate per side, or None.

    The labelled candidate wins its side, so the record for that side is exactly the
    round-4 record; otherwise the first accepted candidate of the side is used.
    """
    cands = meta.get("candidates", [])
    accepted = list(meta.get("accepted", []))
    label = meta.get("label")
    order = ([label] if label in accepted else []) + [i for i in accepted if i != label]
    out = {}
    for i in order:
        if not (0 <= i < len(cands)):
            continue
        side = cands[i].get("label")
        if side in SIDES and cands[i].get("feasible") and side not in out:
            out[side] = i
    return out if len(out) == 2 else None


def _answer_path(rec):
    return json.loads(rec["conversations"][1]["value"])["path"]


def _set_path(rec, path, instruction):
    """Rewrite a record's answer and prompt from its (possibly snapped) path."""
    rec["conversations"][1]["value"] = format_answer_goal([tuple(w) for w in path])
    rec["conversations"][0]["value"] = instruction_endpoints_prompt(path[0], path[-1], instruction)


def build_pair(sample_dir, *, texts=None, meta_extra=None, reasons=None):
    """Both records of one two-sided frame, [left, right], or None (reason counted).

    `texts` maps side -> instruction; None means the template. When the two entry points
    differ by at most SHARED_ENTRY_TOL, the right route's first waypoint is snapped onto
    the left one's, so the two prompts are identical except for the instruction.
    """
    sample_dir = Path(sample_dir)
    meta = json.loads((sample_dir / "meta.json").read_text())
    sides = side_candidates(meta)
    if sides is None:
        return _skip(reasons, "no_two_sided")
    source = "template" if texts is None else "given"
    texts = texts or TEMPLATE_INSTRUCTIONS

    recs = {}
    for side in SIDES:
        rec = build_record(sample_dir, framing="endpoints", candidate=sides[side],
                           instruction=texts[side], meta_extra=meta_extra)
        if rec is None:
            return _skip(reasons, "record_failed")
        recs[side] = rec

    paths = {s: _answer_path(r) for s, r in recs.items()}
    if math.dist(paths["left"][-1][:2], paths["right"][-1][:2]) > 1e-6:
        return _skip(reasons, "goal_differs")
    gap = math.dist(paths["left"][0][:2], paths["right"][0][:2])
    if gap > SHARED_ENTRY_TOL:
        return _skip(reasons, "entry_differs")
    if gap > 0:
        paths["right"][0] = list(paths["left"][0])
        _set_path(recs["right"], paths["right"], texts["right"])

    out = []
    for side, other in (("left", "right"), ("right", "left")):
        rec = recs[side]
        rec["id"] = f"{sample_dir.name}_{side[0].upper()}"
        rec["habitat_meta"].update({
            "pair_id": sample_dir.name,
            "side": side,
            "alt_path": paths[other],
            "entry_gap": round(gap, 4),
            "instruction_source": source,
        })
        out.append(rec)
    return out
```

- [ ] **Step 4: Run tests**

Run: `~/.local/bin/uv run pytest -q tests/test_instructions.py`
Expected: PASS.

- [ ] **Step 5: Commit**
```bash
git add src/rover_vlm/instructions.py tests/test_instructions.py
git commit -m "feat: build left/right instruction pairs from two-sided habitat frames"
```

---

### Task 3: Overlays, Cosmos prompts, parsers and reject rules

**Files:**
- Modify: `src/rover_vlm/instructions.py`
- Test: `tests/test_instructions.py`

**Interfaces:**
- Consumes: `overlay.draw_polyline`, `habitat_choice._draw_badge`, `habitat_choice._font`, `habitat_choice._point_at_fraction`, `traces.parse_trace`, `traces.answer_payload`, `traces.SCENE_INTRO`, `traces.SCRATCH_NOTE`.
- Produces: `INSTRUCTION_PROMPT_VERSION = "i1"`; `check_numbering(record_id) -> dict` (`{"target": 1|2, "other": 2|1}`); `render_generation_image(record, out_path)`; `render_check_image(record, out_path) -> int` (the expected choice); `generation_prompt() -> str`; `check_prompt(instruction) -> str`; `parse_instruction(raw) -> str | None`; `parse_choice(raw) -> int | None`; `instruction_rejects(text) -> list[str]`; `is_side_only(text) -> bool`.

- [ ] **Step 1: Write failing tests** (append to `tests/test_instructions.py`)
```python
from PIL import Image

from rover_vlm.instructions import (
    check_numbering, check_prompt, generation_prompt, instruction_rejects, is_side_only,
    parse_choice, parse_instruction, render_check_image, render_generation_image,
)


def test_check_numbering_is_deterministic_and_mixed():
    a = [check_numbering(f"s_c{i:03d}_L")["target"] for i in range(40)]
    assert a == [check_numbering(f"s_c{i:03d}_L")["target"] for i in range(40)]
    assert set(a) == {1, 2}


def test_render_images(two_sided_sample, tmp_path):
    left, _ = build_pair(two_sided_sample)
    render_generation_image(left, tmp_path / "g.jpg")
    expected = render_check_image(left, tmp_path / "c.jpg")
    assert expected == check_numbering(left["id"])["target"]
    for name in ("g.jpg", "c.jpg"):
        assert Image.open(tmp_path / name).size == (768, 768)


def test_prompts_have_no_concrete_nouns():
    text = (generation_prompt() + check_prompt("{x}")).lower()
    for noun in ("sofa", "table", "chair", "counter", "bed", "desk", "cabinet", "couch", "kitchen"):
        assert noun not in text


def test_parse_instruction_and_choice():
    raw = '<think>hm</think>\n<answer>\n{"instruction": "Pass left of the grey sofa."}\n</answer>'
    assert parse_instruction(raw) == "Pass left of the grey sofa."
    assert parse_instruction("<think>x</think><answer>Pass right of the bed.</answer>") == "Pass right of the bed."
    assert parse_instruction("<think>never closed") is None
    assert parse_choice('<answer>{"choice": 2}</answer>') == 2
    assert parse_choice("<answer>Route 1.</answer>") == 1
    assert parse_choice('<answer>{"choice": 3}</answer>') is None


def test_instruction_rejects():
    assert instruction_rejects("Pass to the left of the blue armchair, then through the door.") == []
    assert any(r.startswith("drawing") for r in instruction_rejects("Follow the magenta line past the chair on your left."))
    assert any(r.startswith("drawing") for r in instruction_rejects("Take route 2 along the wall next to the doorway."))
    assert "digits" in instruction_rejects("Drive to the point at 0.5 near the door on the left.")
    assert instruction_rejects("Go left.") == ["length:2"]
    assert instruction_rejects(None) == ["empty"]


def test_is_side_only():
    assert is_side_only("Go around the obstacle to the left.")
    assert not is_side_only("Go around the wooden table to the left.")
```

- [ ] **Step 2: Run to verify failure**

Run: `~/.local/bin/uv run pytest -q tests/test_instructions.py`
Expected: FAIL (ImportError).

- [ ] **Step 3: Implement** (append to `src/rover_vlm/instructions.py`; add imports at the top: `import hashlib`, `import re`, `from PIL import Image, ImageDraw`, `from .habitat_choice import _draw_badge, _font, _point_at_fraction`, `from .overlay import draw_polyline`, `from .traces import SCENE_INTRO, SCRATCH_NOTE, answer_payload, parse_trace`)
```python
# --- Cosmos labelling ----------------------------------------------------------------
# Bump when a prompt or overlay changes, so label files from different revisions coexist.
INSTRUCTION_PROMPT_VERSION = "i1"
RENDER_PX = 768
# Magenta is rare indoors, so the drawn route is never mistaken for an object; grey keeps
# the alternative visible without competing. The check image draws both routes alike.
TARGET_COLOR = (255, 0, 200)
OTHER_COLOR = (175, 175, 175)
CHECK_COLOR = (255, 215, 0)


def check_numbering(record_id):
    """Which badge (1 or 2) the target route gets in the check image, fixed per record id."""
    target = 1 + hashlib.sha256(record_id.encode()).digest()[0] % 2
    return {"target": target, "other": 3 - target}


def _canvas(record):
    img = Image.open(record["image"][0]).convert("RGB").resize((RENDER_PX, RENDER_PX), Image.LANCZOS)
    return img, ImageDraw.Draw(img)


def _xy(path):
    return [(x, y) for x, y, _ in path]


def render_generation_image(record, out_path):
    """Frame with the record's route bold magenta and the alternative thin grey."""
    img, draw = _canvas(record)
    W, H = img.size
    draw_polyline(draw, _xy(record["habitat_meta"]["alt_path"]), OTHER_COLOR, W, H, width=3)
    draw_polyline(draw, _xy(_answer_path(record)), TARGET_COLOR, W, H, width=7)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path, "JPEG", quality=90)


def render_check_image(record, out_path):
    """Both routes in one colour with badges 1/2 at mid-route. Returns the target's badge."""
    img, draw = _canvas(record)
    W, H = img.size
    num = check_numbering(record["id"])
    routes = {"target": _xy(_answer_path(record)), "other": _xy(record["habitat_meta"]["alt_path"])}
    for pts in routes.values():
        draw_polyline(draw, pts, CHECK_COLOR, W, H, width=5)
    font, r = _font(max(14, int(W * 0.028))), max(12, int(W * 0.022))
    for key, pts in routes.items():
        x, y = _point_at_fraction(pts, 0.5)
        _draw_badge(draw, (x * W, y * H), str(num[key]), CHECK_COLOR, font, r)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path, "JPEG", quality=90)
    return num["target"]


def _answer_block(shape):
    return (
        f"{SCRATCH_NOTE}\n\n"
        "Reply in exactly this format and nothing else:\n"
        "<think>\n(your private working — discarded)\n</think>\n\n"
        f"<answer>\n{shape}\n</answer>\n\n"
        "The <answer> block must hold exactly one JSON object and nothing else — no code "
        "fence, no sentences outside the braces."
    )


def generation_prompt():
    """Ask Cosmos for the instruction that singles out the magenta route."""
    return f"""{SCENE_INTRO}

Two possible routes to the same goal are drawn on the image. The bold magenta route is the
one the rover should drive; the thin grey route is the alternative it must not take.

Write the instruction a person standing behind the rover would say, so that a driver who
sees only the camera image, with nothing drawn on it, would take the magenta route and not
the grey one.

Rules for the instruction:
- One sentence of 6 to 25 words, as a command ("go ...", "pass ...", "keep ...").
- Refer to things that are really in the scene — furniture, walls, doorways, openings,
  floor areas. Name at least one of them and say on which side of it the route passes.
- Never mention the drawing: no lines, line colours, drawn routes, numbers, highlights or
  coordinates. The driver will not see any of them.
- Never name an object that is not in this image.

{_answer_block('{"instruction": "<the instruction>"}')}"""


def check_prompt(instruction):
    """Ask Cosmos which of two identically drawn routes an instruction describes."""
    text = " ".join(str(instruction).split()).replace('"', "'")
    return f"""{SCENE_INTRO}

Two possible routes to the same goal are drawn on the image in the same colour, each marked
with a numbered badge, 1 and 2.

Someone gave the rover this instruction: "{text}"

Which route does the instruction describe? If neither fits perfectly, pick the one it fits
better.

{_answer_block('{"choice": <1 or 2>}')}"""


def parse_instruction(raw):
    """The instruction string from a Cosmos reply, or None.

    A reply that drops the JSON envelope but leaves one short sentence in <answer> is
    salvaged: the sentence is the deliverable.
    """
    _, answer = parse_trace(raw)
    payload = answer_payload(answer)
    if payload is not None:
        text = payload.get("instruction")
        return " ".join(text.split()) if isinstance(text, str) and text.strip() else None
    if answer and "{" not in answer and len(answer.split()) <= 40:
        return " ".join(answer.strip().strip('"').split()) or None
    return None


def parse_choice(raw):
    """1 or 2 from a Cosmos check reply, or None."""
    _, answer = parse_trace(raw)
    payload = answer_payload(answer)
    if payload is not None:
        c = payload.get("choice")
        return c if c in (1, 2) else None
    found = set(re.findall(r"\b([12])\b", answer or ""))
    return int(found.pop()) if len(found) == 1 else None


# References to the overlay itself: the driver at inference sees a bare frame, so an
# instruction that leans on the drawing teaches Qwen to look for something that is not there.
_DRAWING = re.compile(
    r"\b(?:lines?|stroke|overlay|drawn|drawing|highlight\w*|marked|arrows?|badges?|magenta|"
    r"numbered|coordinates?|options?|(?:route|path) (?:1|2|one|two)|"
    r"(?:bold|thin|grey|gray) (?:route|path))\b",
    re.I,
)
_WORD_LIMITS = (6, 25)


def instruction_rejects(text):
    """Why an instruction is unusable; [] when it is fine."""
    if not text:
        return ["empty"]
    out = [f"drawing:{m.group(0).lower()}" for m in _DRAWING.finditer(text)]
    if re.search(r"\d", text):
        out.append("digits")
    n = len(text.split())
    if not (_WORD_LIMITS[0] <= n <= _WORD_LIMITS[1]):
        out.append(f"length:{n}")
    return out


_SIDE_FILLER = set(
    "go keep stay turn veer head move drive around to the left right of on your side and "
    "then a it obstacle obstacles pass take route way bear towards toward forward ahead "
    "straight past through along".split()
)


def is_side_only(text):
    """True when the instruction says nothing beyond a side — no better than the template."""
    words = re.findall(r"[a-z']+", (text or "").lower())
    return not [w for w in words if w not in _SIDE_FILLER]
```

- [ ] **Step 4: Run tests**

Run: `~/.local/bin/uv run pytest -q tests/test_instructions.py`
Expected: PASS.

- [ ] **Step 5: Commit**
```bash
git add src/rover_vlm/instructions.py tests/test_instructions.py
git commit -m "feat: cosmos overlays, prompts, parsers and reject rules for route instructions"
```

---

### Task 4: Label summary and pair finalisation (pure functions)

**Files:**
- Modify: `src/rover_vlm/instructions.py`
- Test: `tests/test_instructions.py`

**Interfaces:**
- Produces: `summarize_labels(rows) -> dict` (keys `n_records`, `n_instruction`, `n_drawing_refs`, `n_rejected`, `n_checked`, `check_accuracy`, `n_side_only`, `pairs_total`, `pairs_kept`, `pair_survival`); `gate_failures(summary, *, min_pair_survival, max_drawing_refs) -> list[str]`; `with_instruction(rec, text, source) -> dict`; `finalize_pairs(records, rows_by_id) -> (template_recs, cosmos_recs, stats)`.
- Label row shape (written by Task 5): `{"id", "pair_id", "side", "instruction", "rejects", "side_only", "check_choice", "check_expected", "check_correct", "passed"}` plus diagnostics; error rows are `{"id", "error"}`.

- [ ] **Step 1: Write failing tests** (append)
```python
from rover_vlm.instructions import finalize_pairs, gate_failures, summarize_labels, with_instruction


def _row(rid, pair, passed, rejects=(), correct=True, instruction="Pass left of the lamp by the door."):
    return {"id": rid, "pair_id": pair, "instruction": instruction, "rejects": list(rejects),
            "side_only": False, "check_correct": correct, "passed": passed}


def test_summarize_and_gate():
    rows = [_row("a_L", "a", True), _row("a_R", "a", True),
            _row("b_L", "b", True), _row("b_R", "b", False, correct=False),
            _row("c_L", "c", False, rejects=["drawing:line"]), _row("c_R", "c", True),
            {"id": "d_L", "error": "Timeout"}]
    s = summarize_labels(rows)
    assert (s["pairs_total"], s["pairs_kept"]) == (3, 1)
    assert s["n_drawing_refs"] == 1 and s["n_rejected"] == 1
    assert gate_failures(s, min_pair_survival=0.8, max_drawing_refs=0) == [
        "pair survival 0.333 < 0.8", "drawing references 1 > 0"]


def test_finalize_pairs_keeps_only_fully_passed(two_sided_sample):
    left, right = build_pair(two_sided_sample)
    rows = {left["id"]: _row(left["id"], "p", True, instruction="Pass left of the lamp."),
            right["id"]: _row(right["id"], "p", True, instruction="Pass right of the lamp.")}
    tmpl, cosmos, stats = finalize_pairs([left, right], rows)
    assert [r["id"] for r in tmpl] == [r["id"] for r in cosmos] == [left["id"], right["id"]]
    assert '"Pass right of the lamp."' in cosmos[1]["conversations"][0]["value"]
    assert cosmos[1]["habitat_meta"]["instruction_source"] == "cosmos"
    assert tmpl[1]["habitat_meta"]["instruction"] == TEMPLATE_INSTRUCTIONS["right"]
    assert stats == {"pairs_in": 1, "pairs_kept": 1}
    del rows[right["id"]]                                   # missing label row -> pair dropped
    assert finalize_pairs([left, right], rows)[2] == {"pairs_in": 1, "pairs_kept": 0}


def test_with_instruction_does_not_mutate(two_sided_sample):
    left, _ = build_pair(two_sided_sample)
    before = json.dumps(left)
    with_instruction(left, "Keep right of the plant.", "cosmos")
    assert json.dumps(left) == before
```

- [ ] **Step 2: Run to verify failure** — `~/.local/bin/uv run pytest -q tests/test_instructions.py` → FAIL (ImportError).

- [ ] **Step 3: Implement** (append; add `import copy` and `from collections import defaultdict` to the imports)
```python
# --- label bookkeeping ----------------------------------------------------------------

def _by_pair(items, key):
    groups = defaultdict(list)
    for it in items:
        groups[key(it)].append(it)
    return groups


def summarize_labels(rows):
    """The numbers the gate and the run summary are judged on."""
    ok = [r for r in rows if not r.get("error")]
    checked = [r for r in ok if r.get("check_correct") is not None]
    pair_ids = {r.get("pair_id") or r["id"].rsplit("_", 1)[0] for r in rows}
    kept = {pid for pid, rs in _by_pair(ok, lambda r: r["pair_id"]).items()
            if len(rs) == 2 and all(r.get("passed") for r in rs)}
    return {
        "n_records": len(rows),
        "n_instruction": sum(bool(r.get("instruction")) for r in ok),
        "n_drawing_refs": sum(any(x.startswith("drawing") for x in r.get("rejects", [])) for r in ok),
        "n_rejected": sum(bool(r.get("rejects")) for r in ok),
        "n_checked": len(checked),
        "check_accuracy": round(sum(bool(r["check_correct"]) for r in checked) / len(checked), 4) if checked else None,
        "n_side_only": sum(bool(r.get("side_only")) for r in ok),
        "pairs_total": len(pair_ids),
        "pairs_kept": len(kept),
        "pair_survival": round(len(kept) / len(pair_ids), 4) if pair_ids else 0.0,
    }


def gate_failures(summary, *, min_pair_survival, max_drawing_refs):
    """[] when the labelling round is good enough to scale up."""
    out = []
    if summary["pair_survival"] < min_pair_survival:
        out.append(f"pair survival {summary['pair_survival']:.3f} < {min_pair_survival}")
    if summary["n_drawing_refs"] > max_drawing_refs:
        out.append(f"drawing references {summary['n_drawing_refs']} > {max_drawing_refs}")
    return out


def with_instruction(rec, text, source):
    """A copy of a pair record whose prompt carries `text` as its instruction."""
    new = copy.deepcopy(rec)
    path = _answer_path(new)
    new["conversations"][0]["value"] = instruction_endpoints_prompt(path[0], path[-1], text)
    new["habitat_meta"]["instruction"] = text
    new["habitat_meta"]["instruction_source"] = source
    return new


def finalize_pairs(records, rows_by_id):
    """Pairs whose two records both passed labelling, as matched template/Cosmos lists.

    The two lists hold the same ids in the same order and differ only in the instruction,
    which is what makes template-vs-Cosmos a controlled comparison.
    """
    template, cosmos = [], []
    groups = _by_pair(records, lambda r: r["habitat_meta"]["pair_id"])
    for pid, recs in groups.items():
        assert len(recs) == 2, f"pair {pid} has {len(recs)} records"
        rows = [rows_by_id.get(r["id"]) for r in recs]
        if not all(row and row.get("passed") for row in rows):
            continue
        for rec, row in zip(recs, rows):
            template.append(with_instruction(rec, TEMPLATE_INSTRUCTIONS[rec["habitat_meta"]["side"]], "template"))
            cosmos.append(with_instruction(rec, row["instruction"], "cosmos"))
    return template, cosmos, {"pairs_in": len(groups), "pairs_kept": len(template) // 2}
```

- [ ] **Step 4: Run tests** — `~/.local/bin/uv run pytest -q tests/test_instructions.py` → PASS.

- [ ] **Step 5: Commit**
```bash
git add src/rover_vlm/instructions.py tests/test_instructions.py
git commit -m "feat: label summary, labelling gate and template/cosmos pair finalisation"
```

---

### Task 5: Shared labelling I/O + `label_instructions.py`

**Files:**
- Create: `src/rover_vlm/labelling_io.py`
- Modify: `scripts/label_traces.py` (use `labelling_io` for endpoint + output guard; behaviour unchanged)
- Create: `scripts/label_instructions.py`
- Test: `tests/test_labelling_io.py`

**Interfaces:**
- Produces: `resolve_endpoint(explicit, endpoint_file) -> str`; `prepare_output(out_path, *, resume, overwrite) -> tuple[set[str], str]` (done ids, file mode `"a"`/`"w"`; error rows removed so they are retried); `connect(base_url, model=None) -> (client, model_id)`; `chat_with_image(client, model, image_path, prompt, *, max_tokens, seed) -> (raw, finish_reason)`.
- Consumes: Task 3/4 functions.

- [ ] **Step 1: Write failing tests**

`tests/test_labelling_io.py`:
```python
import json

import pytest

from rover_vlm.labelling_io import prepare_output


def test_prepare_output_new_file(tmp_path):
    assert prepare_output(tmp_path / "x.jsonl", resume=False, overwrite=False) == (set(), "w")


def test_prepare_output_refuses_to_clobber(tmp_path):
    p = tmp_path / "x.jsonl"
    p.write_text('{"id": "a"}\n')
    with pytest.raises(SystemExit):
        prepare_output(p, resume=False, overwrite=False)
    assert prepare_output(p, resume=False, overwrite=True) == (set(), "w")


def test_prepare_output_resume_retries_errors_only(tmp_path):
    p = tmp_path / "x.jsonl"
    rows = [{"id": "a", "passed": False}, {"id": "b", "error": "Timeout"}, {"id": "c", "passed": True}]
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))
    done, mode = prepare_output(p, resume=True, overwrite=False)
    assert done == {"a", "c"} and mode == "a"
    assert [json.loads(l)["id"] for l in p.read_text().splitlines()] == ["a", "c"]
```

- [ ] **Step 2: Run to verify failure** — `~/.local/bin/uv run pytest -q tests/test_labelling_io.py` → FAIL.

- [ ] **Step 3: Implement `labelling_io.py`**
```python
"""Client-side plumbing shared by the Cosmos labelling scripts.

scripts/label_traces.py and scripts/label_instructions.py both talk to the endpoint that
slurm/serve_cosmos3.sbatch publishes, and both write a JSONL file that must survive being
killed and resumed. The rules for that file live here once:

* a non-empty output is never truncated unless --overwrite says so (a full run is hours);
* --resume keeps every finished row and drops error rows, so a failed request is retried
  rather than shadowed — a row that ran and merely failed a check is a result, not an error.
"""

import json
import os
from pathlib import Path


def resolve_endpoint(explicit, endpoint_file):
    if explicit:
        return explicit
    endpoint_file = Path(endpoint_file)
    if not endpoint_file.exists():
        raise SystemExit(
            f"No endpoint at {endpoint_file}.\n"
            "Start the server with 'sbatch slurm/serve_cosmos3.sbatch' and wait for it to\n"
            "report READY, or pass --endpoint http://<host>:<port>/v1 explicitly."
        )
    return endpoint_file.read_text().strip()


def prepare_output(out_path, *, resume, overwrite):
    """(ids already done, file mode) for a JSONL output about to be written."""
    out_path = Path(out_path)
    if not (out_path.exists() and out_path.stat().st_size):
        return set(), "w"
    if not resume:
        if overwrite:
            return set(), "w"
        raise SystemExit(f"{out_path} exists and is not empty. Pass --resume to continue it, "
                         "or --overwrite to replace it.")
    kept = [json.loads(line) for line in out_path.open() if line.strip()]
    failed = sum(1 for r in kept if r.get("error"))
    kept = [r for r in kept if not r.get("error")]
    tmp = out_path.with_suffix(".jsonl.tmp")
    tmp.write_text("".join(json.dumps(r) + "\n" for r in kept))
    os.replace(tmp, out_path)          # atomic: never leaves a half-written file
    done = {r["id"] for r in kept}
    print(f"resume:   {len(done)} done in {out_path.name}"
          + (f", {failed} failed rows dropped for retry" if failed else ""))
    return done, ("a" if done else "w")


def connect(base_url, model=None):
    """(openai client, served model id); a dead endpoint exits with one readable line."""
    try:
        import openai
    except ImportError:
        raise SystemExit("openai not installed — run 'uv sync' (it is a project dependency).")
    client = openai.OpenAI(api_key="EMPTY", base_url=base_url, timeout=600)
    try:
        return client, model or client.models.list().data[0].id
    except Exception as e:  # noqa: BLE001
        raise SystemExit(
            f"Cannot reach the server at {base_url} ({type(e).__name__}: {e}).\n"
            "Check the serve job is still RUNNING (squeue) and that\n"
            "outputs/cosmos3_endpoint.txt points at the node it is actually on."
        )


def chat_with_image(client, model, image_path, prompt, *, max_tokens, seed):
    """One image+text request -> (raw reply, finish_reason). Raises on transport errors."""
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": Path(image_path).resolve().as_uri()}},
            {"type": "text", "text": prompt},
        ]}],
        max_tokens=max_tokens,
        seed=seed,
    )
    return resp.choices[0].message.content or "", resp.choices[0].finish_reason
```

- [ ] **Step 4: Refactor `scripts/label_traces.py`** (behaviour-preserving)
  - Delete its local `resolve_endpoint`; add `from rover_vlm.labelling_io import connect, prepare_output, resolve_endpoint`.
  - Replace the whole `done = set()` … `elif not args.overwrite: raise SystemExit(...)` block with:
    ```python
    # A full run is hours of GPU time; silently truncating its output would be the most
    # expensive mistake this script can make.
    done, mode = prepare_output(out_path, resume=args.resume, overwrite=args.overwrite)
    records = [r for r in records if r["id"] not in done]
    ```
  - Replace the `try: import openai` … `print(f"endpoint: ...")` block with:
    ```python
    base_url = resolve_endpoint(args.endpoint, ENDPOINT_FILE)
    client, model = connect(base_url, args.model)
    print(f"endpoint: {base_url}\nmodel:    {model}")
    ```
  - Delete the later `mode = "a" if (args.resume and done) else "w"` line (mode now comes from `prepare_output`).
  - Run: `~/.local/bin/uv run pytest -q tests/test_labelling_io.py tests/test_traces.py` → PASS; `~/.local/bin/uv run scripts/label_traces.py --help` → prints usage.

- [ ] **Step 5: Write `scripts/label_instructions.py`**
```python
"""Label route instructions with Cosmos3-Nano: write one per record, then verify it.

Runs on the LOGIN NODE against the endpoint published by slurm/serve_cosmos3.sbatch.
Input is a pairs file from `scripts/prepare_habitat_instr.py pairs`; each record carries
its two overlay images (habitat_meta.gen_image / check_image).

Per record, two independent calls:
  1. generate — frame with the record's route bold magenta and the other side's grey;
     Cosmos writes the instruction that singles out the magenta one;
  2. verify   — both routes drawn alike, badges 1/2 in a per-id random order; Cosmos is
     given the instruction and must pick the route. A record passes when the instruction
     parsed, trips no reject rule and the pick is right.

    # gate: first 100 eval pairs; exits 2 if the gate fails
    uv run scripts/label_instructions.py --split data/prepared_habitat_instr_v1/pairs_eval.json \
        --limit-pairs 100 --out outputs/instr_labels/gate_i1.jsonl --concurrency 8 --gate
    # full run (resumable)
    uv run scripts/label_instructions.py --split data/prepared_habitat_instr_v1/pairs_train.json \
        --out outputs/instr_labels/train_i1.jsonl --concurrency 8 --resume

Writes one JSON line per record, flushed as it goes.
"""

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from rover_vlm.instructions import (
    INSTRUCTION_PROMPT_VERSION, check_prompt, gate_failures, generation_prompt,
    instruction_rejects, is_side_only, parse_choice, parse_instruction, summarize_labels,
)
from rover_vlm.labelling_io import chat_with_image, connect, prepare_output, resolve_endpoint

REPO_ROOT = Path(__file__).resolve().parent.parent
ENDPOINT_FILE = REPO_ROOT / "outputs" / "cosmos3_endpoint.txt"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--split", type=Path, required=True, help="pairs_*.json from prepare_habitat_instr.py")
    p.add_argument("--out", type=Path, required=True, help="output JSONL")
    p.add_argument("--limit-pairs", type=int, default=None, help="label only the first N pairs")
    p.add_argument("--endpoint", default=None)
    p.add_argument("--model", default=None)
    p.add_argument("--max-tokens", type=int, default=4096)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--concurrency", type=int, default=1)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--show", action="store_true", help="print both raw replies per record")
    p.add_argument("--gate", action="store_true", help="exit 2 unless the thresholds below hold")
    p.add_argument("--min-pair-survival", type=float, default=0.8)
    p.add_argument("--max-drawing-refs", type=int, default=0)
    args = p.parse_args()

    records = json.loads(args.split.read_text())
    if args.limit_pairs:
        keep = list(dict.fromkeys(r["habitat_meta"]["pair_id"] for r in records))[: args.limit_pairs]
        keep = set(keep)
        records = [r for r in records if r["habitat_meta"]["pair_id"] in keep]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    done, mode = prepare_output(args.out, resume=args.resume, overwrite=args.overwrite)
    todo = [r for r in records if r["id"] not in done]
    print(f"split: {args.split}  records {len(records)}  to label {len(todo)}  "
          f"prompt {INSTRUCTION_PROMPT_VERSION}")

    if todo:
        base_url = resolve_endpoint(args.endpoint, ENDPOINT_FILE)
        client, model = connect(base_url, args.model)
        print(f"endpoint: {base_url}\nmodel:    {model}")

        def label(rec):
            hm = rec["habitat_meta"]
            row = {"id": rec["id"], "pair_id": hm["pair_id"], "side": hm["side"],
                   "prompt_version": INSTRUCTION_PROMPT_VERSION}
            t0 = time.time()
            try:
                raw_gen, fin_gen = chat_with_image(client, model, hm["gen_image"], generation_prompt(),
                                                   max_tokens=args.max_tokens, seed=args.seed)
                text = parse_instruction(raw_gen)
                rejects = instruction_rejects(text)
                choice, raw_chk, fin_chk = None, None, None
                if text:
                    raw_chk, fin_chk = chat_with_image(client, model, hm["check_image"], check_prompt(text),
                                                       max_tokens=args.max_tokens, seed=args.seed)
                    choice = parse_choice(raw_chk)
            except Exception as e:  # noqa: BLE001 — one bad sample must not kill the run
                return {**row, "error": f"{type(e).__name__}: {e}"}
            correct = None if choice is None else choice == hm["check_expected"]
            row.update({
                "instruction": text, "rejects": rejects, "side_only": is_side_only(text) if text else None,
                "check_choice": choice, "check_expected": hm["check_expected"], "check_correct": correct,
                "passed": bool(text and not rejects and correct),
                "finish": [fin_gen, fin_chk], "latency_s": round(time.time() - t0, 2),
            })
            if args.show or not row["passed"]:
                row["raw_gen"], row["raw_check"] = raw_gen, raw_chk
            return row

        started = time.time()
        with args.out.open(mode) as f, ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futures = [pool.submit(label, r) for r in todo]
            for i, fut in enumerate(as_completed(futures), 1):
                row = fut.result()
                f.write(json.dumps(row) + "\n")
                f.flush()
                rate = i / max(1e-9, time.time() - started)
                status = row.get("error") or ("ok" if row["passed"] else
                          f"FAIL rejects={row['rejects']} check={row['check_choice']}/{row['check_expected']}")
                print(f"[{i}/{len(todo)}] {row['id']}  {rate * 60:.0f}/min  "
                      f"eta {(len(todo) - i) / rate / 60:.0f}m  {row.get('instruction')!r}  {status}")
                if args.show:
                    print(f"--- generate ---\n{row.get('raw_gen')}\n--- check ---\n{row.get('raw_check')}\n")

    rows = [json.loads(l) for l in args.out.read_text().splitlines() if l.strip()]
    wanted = {r["id"] for r in records}
    summary = summarize_labels([r for r in rows if r["id"] in wanted])
    print("\nsummary:\n" + json.dumps(summary, indent=2))
    errors = sum(1 for r in rows if r.get("error"))
    if errors:
        print(f"{errors} request errors — re-run with --resume to retry them")
    if args.gate:
        fails = gate_failures(summary, min_pair_survival=args.min_pair_survival,
                              max_drawing_refs=args.max_drawing_refs)
        print("GATE: " + ("PASS" if not fails else "FAIL — " + "; ".join(fails)))
        if fails:
            sys.exit(2)


if __name__ == "__main__":
    main()
```
Run: `~/.local/bin/uv run scripts/label_instructions.py --help` → usage printed.

- [ ] **Step 6: Run all tests** — `~/.local/bin/uv run pytest -q` → PASS.

- [ ] **Step 7: Commit**
```bash
git add src/rover_vlm/labelling_io.py scripts/label_traces.py scripts/label_instructions.py tests/test_labelling_io.py
git commit -m "feat: label route instructions with cosmos (generate + verify), shared labelling io"
```

---

### Task 6: `scripts/prepare_habitat_instr.py` (pairs + finalize) and a real CPU run

**Files:**
- Create: `scripts/prepare_habitat_instr.py`
- Test: `tests/test_instructions.py` (the finalize logic is already covered in Task 4; this task adds the scene-disjoint helper test)

**Interfaces:**
- Consumes: `build_pair`, `render_generation_image`, `render_check_image`, `finalize_pairs`, `TEMPLATE_INSTRUCTIONS`.
- Produces files under `data/prepared_habitat_instr_v1/`: `pairs_train.json`, `pairs_eval.json`, `pairs_meta.json`, `overlays/<id>_gen.jpg`, `overlays/<id>_check.jpg`; after finalize: `train_template.json`, `train_cosmos.json`, `eval_template.json`, `eval_cosmos.json`, `meta.json`. Pair records' `habitat_meta` gain `gen_image`, `check_image`, `check_expected`.
- Produces: `assert_scene_disjoint(train, eval_)` in `instructions.py`.

- [ ] **Step 1: Failing test** (append to `tests/test_instructions.py`)
```python
import pytest

from rover_vlm.instructions import assert_scene_disjoint


def test_assert_scene_disjoint(two_sided_sample):
    left, right = build_pair(two_sided_sample)
    assert_scene_disjoint([left], [])
    with pytest.raises(AssertionError):
        assert_scene_disjoint([left], [right])
```
Run → FAIL (ImportError).

- [ ] **Step 2: Implement** (append to `instructions.py`)
```python
def assert_scene_disjoint(train, eval_):
    shared = ({r["habitat_meta"]["scene_id"] for r in train}
              & {r["habitat_meta"]["scene_id"] for r in eval_})
    assert not shared, f"{len(shared)} scenes in both splits, e.g. {sorted(shared)[:3]}"
```
Run tests → PASS.

- [ ] **Step 3: Write the script**
```python
"""Build the instruction-conditioned dataset from the round-4 splits. CPU only.

Stage 1 — pairs (before labelling): every original round-4 frame with a feasible left and
right route that enter the frame at the same point becomes two records (template
instruction), plus the two overlay images per record that Cosmos is shown.

    uv run scripts/prepare_habitat_instr.py pairs
    uv run scripts/prepare_habitat_instr.py pairs --limit 50 --out-dir /tmp/instr_try

Stage 2 — finalize (after scripts/label_instructions.py): keep the pairs whose two
records both passed, and write matched template/Cosmos datasets (same ids, same order,
only the instruction differs).

    uv run scripts/prepare_habitat_instr.py finalize \
        --instructions outputs/instr_labels/train_i1.jsonl outputs/instr_labels/eval_i1.jsonl

The split is round 4's (data/prepared_habitat_v4), which is already scene-disjoint;
that is re-asserted here. Yaw-augmented frames are skipped: the candidate projections
belong to the unrotated camera.
"""

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path

from rover_vlm.instructions import (
    assert_scene_disjoint, build_pair, finalize_pairs, render_check_image, render_generation_image,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
SOURCE = REPO_ROOT / "data" / "prepared_habitat_v4"
OUT_DIR = REPO_ROOT / "data" / "prepared_habitat_instr_v1"
SPLITS = {"train": "train_full", "eval": "eval"}
# manifest fields carried over from round 4; everything path-specific is rebuilt per side
CARRY = ("sample_id", "label_side", "margin", "decidability", "near_symmetric",
         "n_candidates", "detour_ratio")


def build_pairs(source_recs, out_dir, reasons):
    out = []
    for i, src in enumerate(source_recs):
        hm = src["habitat_meta"]
        if hm.get("source") != "original":
            continue
        pair = build_pair(Path(src["image"][0]).parent, reasons=reasons,
                          meta_extra={k: hm.get(k) for k in CARRY})
        if pair is None:
            continue
        for rec in pair:
            gen = out_dir / "overlays" / f"{rec['id']}_gen.jpg"
            chk = out_dir / "overlays" / f"{rec['id']}_check.jpg"
            render_generation_image(rec, gen)
            rec["habitat_meta"].update({"gen_image": str(gen), "check_image": str(chk),
                                        "check_expected": render_check_image(rec, chk)})
        out.extend(pair)
        if (i + 1) % 2000 == 0:
            print(f"  {i + 1}/{len(source_recs)} scanned, {len(out) // 2} pairs")
    return out


def stage_pairs(args):
    reasons, splits = {}, {}
    for name, src_name in SPLITS.items():
        src = json.loads((args.source / f"{src_name}.json").read_text())
        if args.limit:
            src = src[: args.limit]
        splits[name] = build_pairs(src, args.out_dir, reasons)
        print(f"{name}: {len(splits[name]) // 2} pairs from {len(src)} round-4 records")
    assert_scene_disjoint(splits["train"], splits["eval"])
    args.out_dir.mkdir(parents=True, exist_ok=True)
    stats = {}
    for name, recs in splits.items():
        (args.out_dir / f"pairs_{name}.json").write_text(json.dumps(recs, indent=1))
        gaps = [r["habitat_meta"]["entry_gap"] for r in recs if r["habitat_meta"]["side"] == "left"]
        stats[name] = {
            "pairs": len(recs) // 2,
            "scenes": len({r["habitat_meta"]["scene_id"] for r in recs}),
            # which side round 4 called correct -- the side an instruction-blind model drifts to
            "label_side": dict(Counter(r["habitat_meta"].get("label_side") for r in recs
                                       if r["habitat_meta"]["side"] == "left")),
            "entry_gap_zero": sum(g == 0 for g in gaps),
            "entry_gap_median": statistics.median(gaps) if gaps else None,
        }
    meta = {"source": str(args.source), "skipped": reasons, "stats": stats}
    (args.out_dir / "pairs_meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


def stage_finalize(args):
    rows = {}
    for path in args.instructions:
        for line in Path(path).read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                if not row.get("error"):
                    rows[row["id"]] = row
    meta = {"instructions": [str(p) for p in args.instructions], "splits": {}}
    for name in SPLITS:
        pairs = json.loads((args.out_dir / f"pairs_{name}.json").read_text())
        template, cosmos, stats = finalize_pairs(pairs, rows)
        (args.out_dir / f"{name}_template.json").write_text(json.dumps(template, indent=1))
        (args.out_dir / f"{name}_cosmos.json").write_text(json.dumps(cosmos, indent=1))
        meta["splits"][name] = {**stats, "records": len(cosmos)}
        print(f"{name}: kept {stats['pairs_kept']}/{stats['pairs_in']} pairs -> {len(cosmos)} records")
    (args.out_dir / "meta.json").write_text(json.dumps(meta, indent=2))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="stage", required=True)
    a = sub.add_parser("pairs")
    a.add_argument("--source", type=Path, default=SOURCE)
    a.add_argument("--out-dir", type=Path, default=OUT_DIR)
    a.add_argument("--limit", type=int, default=None, help="cap round-4 records per split (testing)")
    b = sub.add_parser("finalize")
    b.add_argument("--out-dir", type=Path, default=OUT_DIR)
    b.add_argument("--instructions", type=Path, nargs="+", required=True, help="label JSONL files")
    args = p.parse_args()
    stage_pairs(args) if args.stage == "pairs" else stage_finalize(args)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Real run (CPU, login node)**
```bash
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
~/.local/bin/uv run scripts/prepare_habitat_instr.py pairs
```
Expected: train ≈ 973 pairs, eval ≈ 125 pairs (spec numbers; small differences are possible because this uses the resampled record starts and the label-first side choice — report them); `skipped` shows `no_two_sided`, `entry_differs`; `assert_scene_disjoint` passes. Open 3 `overlays/*_gen.jpg` and 3 `*_check.jpg` with the Read tool: the magenta route must be visibly distinct from the grey one, and check badges must sit on different routes.

- [ ] **Step 5: Commit** (data/ and outputs/ are gitignored — commit code only)
```bash
git add scripts/prepare_habitat_instr.py src/rover_vlm/instructions.py tests/test_instructions.py
git commit -m "feat: prepare_habitat_instr — shared-entry pairs, cosmos overlays, finalize"
```

---

### Task 7: Instruction-following metrics in eval

**Files:**
- Modify: `src/rover_vlm/eval.py` (append functions), `scripts/evaluate.py` (hook after `summary = aggregators[...]`)
- Test: `tests/test_habitat_eval.py`

**Interfaces:**
- Produces: `instruction_follow(pred_path, gt_path, alt_path) -> dict` (`frechet_target`, `frechet_alt`, `follows`); `aggregate_instruction_metrics(records) -> dict | None` (`n_records`, `n_pairs`, `follow_rate`, `pair_success`, `identical_output_rate`, `follow_rate_by_side`).
- Consumes: evaluate.py rows: `{"id", "generated", "parsed": {"path", "goal"} | None, "gt": {"path", "goal"}, "habitat_meta": {..., "pair_id", "side", "alt_path"}}`.

- [ ] **Step 1: Failing tests** (append to `tests/test_habitat_eval.py`)
```python
from rover_vlm.eval import aggregate_instruction_metrics, instruction_follow

_L = [[0.5, 1.0, 1], [0.3, 0.8, 1], [0.5, 0.6, 1]]
_R = [[0.5, 1.0, 1], [0.7, 0.8, 1], [0.5, 0.6, 1]]


def _rec(pid, side, pred, text=None):
    gt, alt = (_L, _R) if side == "left" else (_R, _L)
    return {"id": f"{pid}_{side[0].upper()}", "generated": text or json.dumps(pred),
            "parsed": None if pred is None else {"path": pred, "goal": pred[-1]},
            "gt": {"path": gt, "goal": gt[-1]},
            "habitat_meta": {"pair_id": pid, "side": side, "alt_path": alt}}


def test_instruction_follow():
    assert instruction_follow(_L, _L, _R)["follows"] == 1
    assert instruction_follow(_R, _L, _R)["follows"] == 0


def test_aggregate_instruction_metrics():
    recs = [_rec("a", "left", _L), _rec("a", "right", _R),          # split correctly
            _rec("b", "left", _L), _rec("b", "right", _L),          # ignores instruction
            _rec("c", "left", None, "garbage"), _rec("c", "right", _R)]  # unparseable
    m = aggregate_instruction_metrics(recs)
    assert m["n_pairs"] == 3 and m["n_records"] == 6
    assert m["pair_success"] == 1 / 3
    assert m["follow_rate"] == 4 / 6
    assert m["identical_output_rate"] == 1 / 3
    assert m["follow_rate_by_side"] == {"left": 2 / 3, "right": 2 / 3}


def test_aggregate_instruction_metrics_none_without_alt_path():
    assert aggregate_instruction_metrics([{"id": "x", "parsed": None, "habitat_meta": {}}]) is None
```
(`json` is already imported in that test file; if not, add `import json`.) Run → FAIL.

- [ ] **Step 2: Implement** (append to `src/rover_vlm/eval.py`)
```python
def instruction_follow(pred_path, gt_path, alt_path):
    """Is the prediction closer (Fréchet) to the instructed route than to the other side?"""
    xy = lambda p: np.array([q[:2] for q in p], dtype=float)
    f_t = frechet_distance(xy(pred_path), xy(gt_path))
    f_a = frechet_distance(xy(pred_path), xy(alt_path))
    return {"frechet_target": f_t, "frechet_alt": f_a, "follows": int(f_t < f_a)}


def aggregate_instruction_metrics(records):
    """Instruction-following summary for records carrying habitat_meta.alt_path, else None.

    Pair success is the headline: both records of a frame share image and endpoints and
    differ only in the instruction, so a model that ignores it returns the same path twice
    and wins at most one of the two -- 0% pair success by construction. An unparseable
    prediction counts as not following; it is never dropped from the denominator.
    """
    rows = [r for r in records if "alt_path" in (r.get("habitat_meta") or {})]
    if not rows:
        return None
    follows, by_pair, by_side = {}, {}, {}
    for r in rows:
        hm = r["habitat_meta"]
        ok = 0
        if r.get("parsed") and r["parsed"].get("path"):
            ok = instruction_follow(r["parsed"]["path"], r["gt"]["path"], hm["alt_path"])["follows"]
        follows[r["id"]] = ok
        by_pair.setdefault(hm["pair_id"], []).append(r)
        by_side.setdefault(hm["side"], []).append(ok)
    pairs = [p for p in by_pair.values() if len(p) == 2]
    return {
        "n_records": len(rows),
        "n_pairs": len(pairs),
        "follow_rate": float(np.mean(list(follows.values()))),
        "pair_success": float(np.mean([all(follows[r["id"]] for r in p) for p in pairs])) if pairs else 0.0,
        "identical_output_rate": float(np.mean(
            [p[0].get("generated", "").strip() == p[1].get("generated", "").strip() for p in pairs])) if pairs else 0.0,
        "follow_rate_by_side": {s: float(np.mean(v)) for s, v in sorted(by_side.items())},
    }
```
In `scripts/evaluate.py`: add `aggregate_instruction_metrics` to the `from rover_vlm.eval import (...)` list, and right after `summary = aggregators[args.task](results)`:
```python
    if args.task == "habitat":
        # instruction-conditioned eval sets (prepare_habitat_instr.py) carry the other
        # side's route; score whether the model followed the instruction it was given
        instr = aggregate_instruction_metrics(results)
        if instr is not None:
            summary["instruction"] = instr
```
Also add a docstring line under the existing usage block:
`Instruction-conditioned sets (data/prepared_habitat_instr_v1/eval_*.json) get an "instruction" block in metrics.json (pair success, follow rate).`

- [ ] **Step 3: Run tests** — `~/.local/bin/uv run pytest -q tests/test_habitat_eval.py` → PASS; full suite → PASS.

- [ ] **Step 4: Commit**
```bash
git add src/rover_vlm/eval.py scripts/evaluate.py tests/test_habitat_eval.py
git commit -m "feat: pair-success and follow-rate metrics for instruction-conditioned eval"
```

---

### Task 8: SLURM eval job, results report, README

**Files:**
- Create: `slurm/eval_instruction.sbatch`
- Create: `scripts/visualize_instruction_results.py`
- Modify: `README.md` (new section "Instruction-conditioned paths")
- Test: `tests/test_instructions.py` (report table helper)

**Interfaces:**
- Consumes: `outputs/eval_instr_v1/<set>/<tag>/{metrics,predictions}.json`, `overlay.embed_jpeg`, `overlay.draw_path`.
- Produces: `collect_cells(root) -> dict[(tag, set), dict]` inside the report script (tested by importing the script via `importlib`, as `tests/test_render_clip_predictions.py` does).

- [ ] **Step 1: `slurm/eval_instruction.sbatch`**
```bash
#!/bin/bash
# Evaluate every cell of the instruction-conditioned comparison, skipping finished ones.
#   sbatch slurm/eval_instruction.sbatch
# Cells: {instr_template, instr_cosmos} x {template, cosmos} eval sets, plus the round-4
# LoRA and the base model on the cosmos set. ~5 min per cell on a 3g.40gb slice.
#SBATCH --job-name=qwen-instr-eval
#SBATCH --partition=A100
#SBATCH --gres=gpu:3g.40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=3:00:00
#SBATCH --output=outputs/slurm/%x-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
mkdir -p outputs/slurm
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
UV=~/.local/bin/uv
D=data/prepared_habitat_instr_v1
OUT=outputs/eval_instr_v1
RUNS=outputs/runs_instr_v1

run() {  # run <tag> <set> [adapter]
    local tag=$1 set=$2 adapter=${3:-}
    if [ -f "$OUT/$set/$tag/metrics.json" ]; then echo "skip $tag/$set (done)"; return; fi
    if [ -n "$adapter" ] && [ ! -d "$adapter" ]; then echo "SKIP $tag/$set: no adapter at $adapter"; return; fi
    echo "=== $tag on $set ==="
    $UV run scripts/evaluate.py --task habitat --tag "$tag" --eval-file "$D/eval_$set.json" \
        --out-dir "$OUT/$set" ${adapter:+--adapter "$adapter"}
}
run instr_template template $RUNS/template/adapter
run instr_template cosmos   $RUNS/template/adapter
run instr_cosmos   template $RUNS/cosmos/adapter
run instr_cosmos   cosmos   $RUNS/cosmos/adapter
run r4_lora        cosmos   outputs/runs_v4/habitat_train_full/adapter
run base           cosmos
echo "all cells done: uv run scripts/visualize_instruction_results.py"
```

- [ ] **Step 2: Failing test for the report helper** (append to `tests/test_instructions.py`)
```python
import importlib.util
from pathlib import Path as _P


def _load_report():
    path = _P(__file__).resolve().parent.parent / "scripts" / "visualize_instruction_results.py"
    spec = importlib.util.spec_from_file_location("vir", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_collect_cells(tmp_path):
    d = tmp_path / "cosmos" / "instr_cosmos"
    d.mkdir(parents=True)
    (d / "metrics.json").write_text(json.dumps({"parse_rate": 1.0, "instruction": {"pair_success": 0.7}}))
    cells = _load_report().collect_cells(tmp_path)
    assert cells[("instr_cosmos", "cosmos")]["instruction"]["pair_success"] == 0.7
```
Run → FAIL (file missing).

- [ ] **Step 3: `scripts/visualize_instruction_results.py`**
```python
"""HTML report for the instruction-conditioned comparison. CPU only.

Reads outputs/eval_instr_v1/<set>/<tag>/{metrics,predictions}.json (written by
slurm/eval_instruction.sbatch) and writes outputs/instruction_results.html: the
model x eval-set table (pair success, follow rate, point error) and a gallery of pairs —
one frame, both instructions, both predictions.

    uv run scripts/visualize_instruction_results.py
    uv run scripts/visualize_instruction_results.py --gallery-tag instr_cosmos --gallery-set cosmos --n 12
"""

import argparse
import html
import json
from pathlib import Path

from PIL import Image, ImageDraw

from rover_vlm.overlay import GT_GREY, draw_path, embed_jpeg

REPO_ROOT = Path(__file__).resolve().parent.parent
ROOT = REPO_ROOT / "outputs" / "eval_instr_v1"
OUT = REPO_ROOT / "outputs" / "instruction_results.html"
TAGS = ["instr_template", "instr_cosmos", "r4_lora", "base"]
SETS = ["template", "cosmos"]
SIDE_COLORS = {"left": (42, 120, 214), "right": (214, 74, 42)}


def collect_cells(root):
    cells = {}
    for m in Path(root).glob("*/*/metrics.json"):
        cells[(m.parent.name, m.parent.parent.name)] = json.loads(m.read_text())
    return cells


def _fmt(v):
    return "–" if v is None else f"{v:.3f}"


def table(cells):
    head = "".join(f"<th>{s} set</th>" for s in SETS)
    rows = []
    for tag in TAGS:
        tds = []
        for s in SETS:
            c = cells.get((tag, s))
            if c is None:
                tds.append("<td>–</td>")
                continue
            ins = c.get("instruction") or {}
            tds.append(f"<td><b>{_fmt(ins.get('pair_success'))}</b> pair · "
                       f"{_fmt(ins.get('follow_rate'))} follow · "
                       f"{_fmt(c.get('mean_point_error_median'))} err · "
                       f"{_fmt(c.get('parse_rate'))} parse</td>")
        rows.append(f"<tr><th>{tag}</th>{''.join(tds)}</tr>")
    return f"<table><tr><th>model</th>{head}</tr>{''.join(rows)}</table>"


def gallery(root, tag, set_, n):
    preds = json.loads((Path(root) / set_ / tag / "predictions.json").read_text())
    pairs = {}
    for r in preds:
        pairs.setdefault(r["habitat_meta"]["pair_id"], {})[r["habitat_meta"]["side"]] = r
    evals = {r["id"]: r for r in json.loads(
        (REPO_ROOT / "data" / "prepared_habitat_instr_v1" / f"eval_{set_}.json").read_text())}
    cards = []
    for pid, sides in list(pairs.items())[:n]:
        if len(sides) != 2:
            continue
        img = Image.open(evals[sides["left"]["id"]]["image"][0]).convert("RGB").resize((512, 512))
        d = ImageDraw.Draw(img)
        for side, r in sides.items():
            draw_path(d, r["gt"]["path"], GT_GREY, 512, 512, r=0, width=3)
        lines = []
        for side, r in sides.items():
            if r.get("parsed"):
                draw_path(d, r["parsed"]["path"], SIDE_COLORS[side], 512, 512, r=4, width=3)
            instr = evals[r["id"]]["habitat_meta"]["instruction"]
            color = "#%02x%02x%02x" % SIDE_COLORS[side]
            lines.append(f'<div style="color:{color}">{side}: {html.escape(instr)}</div>')
        cards.append(f'<figure><img src="{embed_jpeg(img, 420)}"><figcaption>{"".join(lines)}</figcaption></figure>')
    return "".join(cards)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, default=ROOT)
    p.add_argument("--out", type=Path, default=OUT)
    p.add_argument("--gallery-tag", default="instr_cosmos")
    p.add_argument("--gallery-set", default="cosmos")
    p.add_argument("--n", type=int, default=12)
    args = p.parse_args()
    cells = collect_cells(args.root)
    if not cells:
        raise SystemExit(f"no metrics under {args.root} — run slurm/eval_instruction.sbatch first")
    has_gallery = (args.root / args.gallery_set / args.gallery_tag / "predictions.json").exists()
    body = table(cells) + (f"<h2>{args.gallery_tag} on the {args.gallery_set} set</h2>"
                           f"<p>grey = both routes; blue/orange = prediction for the left/right instruction</p>"
                           + gallery(args.root, args.gallery_tag, args.gallery_set, args.n)
                           if has_gallery else "")
    style = ("body{font:14px system-ui;margin:24px;background:#fafafa}table{border-collapse:collapse}"
             "td,th{border:1px solid #ccc;padding:6px 10px;text-align:left}"
             "figure{display:inline-block;margin:8px;max-width:420px}img{max-width:100%}")
    args.out.write_text(f"<title>Instruction paths</title><style>{style}</style>"
                        f"<h1>Instruction-conditioned paths</h1>"
                        f"<p>pair success: both instructions of a frame followed (instruction-blind = 0)</p>{body}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
```
Run tests → PASS.

- [ ] **Step 4: README section** — add after the round-4 relabel section:
```markdown
## Instruction-conditioned paths (2026-09-24)

The route instruction is an *input*: the round-4 prompt (entry point + goal) plus one
sentence such as "pass left of the grey sofa"; the answer is the path that obeys it. Only
frames with a feasible left AND right route entering at the same point are used, so the
instruction alone decides the side. Spec: `docs/superpowers/specs/2026-09-24-instruction-conditioned-paths-design.md`.

1. Pairs + overlays (CPU): `uv run scripts/prepare_habitat_instr.py pairs` → `data/prepared_habitat_instr_v1/pairs_{train,eval}.json`
2. Cosmos endpoint: `sbatch slurm/serve_cosmos3.sbatch`, wait for READY
3. Gate (100 eval pairs, exits 2 on failure):
   `uv run scripts/label_instructions.py --split data/prepared_habitat_instr_v1/pairs_eval.json --limit-pairs 100 --out outputs/instr_labels/gate_i1.jsonl --concurrency 8 --gate`
4. Full labels: same script on `pairs_train.json` → `train_i1.jsonl` and on `pairs_eval.json` → `eval_i1.jsonl` (`--resume` safe)
5. Finalize: `uv run scripts/prepare_habitat_instr.py finalize --instructions outputs/instr_labels/train_i1.jsonl outputs/instr_labels/eval_i1.jsonl`
6. Train both arms: `sbatch slurm/train_habitat.sbatch data/prepared_habitat_instr_v1/train_template.json outputs/runs_instr_v1/template --epochs 3` and the same with `train_cosmos.json` → `outputs/runs_instr_v1/cosmos`
7. Eval all cells: `sbatch slurm/eval_instruction.sbatch` → `outputs/eval_instr_v1/<set>/<tag>/metrics.json` (`instruction.pair_success`)
8. Report: `uv run scripts/visualize_instruction_results.py` → `outputs/instruction_results.html`
```
Also add a row to the README's report table:
`| Instruction-conditioned paths | \`uv run scripts/visualize_instruction_results.py\` | \`outputs/instruction_results.html\` |`
(The README has uncommitted edits from the real-eval work: stage only this section's hunks with `git add -p README.md`, or leave README unstaged and tell the user.)

- [ ] **Step 5: Full suite + commit**
```bash
~/.local/bin/uv run pytest -q
git add slurm/eval_instruction.sbatch scripts/visualize_instruction_results.py tests/test_instructions.py
git add -p README.md   # only the instruction-paths hunks
git commit -m "feat: eval job, results report and README for instruction-conditioned paths"
```

---

## After the plan: the GPU sequence the user runs

1. `sbatch slurm/serve_cosmos3.sbatch` → gate command (README step 3) → paste output; read 20 instructions from `outputs/instr_labels/gate_i1.jsonl` together.
2. Gate passes → full labels (README step 4) → finalize → two training jobs → eval job → report.
3. Gate fails → iterate on `generation_prompt()` (bump `INSTRUCTION_PROMPT_VERSION`), rerun the gate with a new `--out`.
