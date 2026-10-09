# General Instruction-Conditioned Model (v2) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the v2 dataset (plain + instructed + opposite records for every round-4 frame), its unattended Cosmos labelling job, resumable 1-epoch training, and the v2 eval job + report.

**Architecture:** Frame-level record builders and label bookkeeping go into `rover_vlm.instructions` (pure, tested). `scripts/prepare_habitat_instr.py` gains `general` (build what to label) and `finalize-general` (assemble train/eval files). `scripts/label_instructions.py` learns in-memory overlays, single-route rows, a record-accept gate and one retry. `scripts/train.py` gains step checkpoints and resume. Two new sbatch files and one report script.

**Tech Stack:** Python 3.12, uv, PIL, numpy, openai client → vLLM Cosmos3-Nano, transformers/peft (GPU side), pytest.

**Spec:** `docs/superpowers/specs/2026-09-25-instruction-general-model-design.md`

## Global Constraints

- Thread caps on every compute command: `export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4` (≤ 8 in sbatch).
- Never load Qwen on the login node; pytest is the CPU gate. Tests: `~/.local/bin/uv run pytest -q`.
- v1 stays reproducible: `pairs` / `finalize` stages, v1 files and v1 results untouched.
- Plain records are byte-identical to `data/prepared_habitat_v4` records (copied from them).
- Every pair has both records or neither, in every file.
- Minimum instruction length 4 words (was 6); other i2 rules unchanged; prompt version stays `i2`.
- Output locations: `data/prepared_habitat_instr_v2/`, `outputs/instr_labels_v2/`, `outputs/runs_instr_v2/general/`, `outputs/eval_instr_v2/<set>/<tag>/`.
- Only this plan's hunks are staged in `README.md` / `scripts/evaluate.py` (the user has unrelated uncommitted edits there).

## Review Focus

1. A single-route frame whose other side has no drawable candidate — the instructed record still builds, with `alt_path = None`, and the overlay draws the target alone (test in Task 2 and Task 1).
2. Eval sets containing records with `alt_path` but no `pair_id` (single-route instructed, wrong-side probe) — `aggregate_instruction_metrics` must not crash and must not form pairs from them (test in Task 5).
3. `--retry-rejected` on a file where some rows were already retried — they are not retried again (test in Task 3).
4. `--resume` for training when the output dir has no checkpoint yet — starts fresh instead of crashing (test in Task 4).
5. `finalize-general` when a pair frame has one record passed and one rejected — neither lands in train or eval, but the plain record stays (test in Task 3).

---

### Task 1: In-memory overlays, data-URL requests, 4-word minimum

**Files:**
- Modify: `src/rover_vlm/instructions.py` (`render_generation_image`, `_WORD_LIMITS`)
- Modify: `src/rover_vlm/labelling_io.py` (`chat_with_image`)
- Test: `tests/test_instructions.py`, `tests/test_labelling_io.py`

**Interfaces:**
- Produces: `render_generation_jpeg(record) -> bytes` (draws `alt_path` only when not None); `render_generation_image(record, out_path)` unchanged signature, now writes `render_generation_jpeg` output; `image_url(image) -> str` in `labelling_io` (path → `file://` URI, bytes → `data:image/jpeg;base64,...`); `chat_with_image(client, model, image, prompt, *, max_tokens, seed)` where `image` is a path or bytes.

- [ ] **Step 1: Failing tests**

Append to `tests/test_instructions.py`:
```python
from rover_vlm.instructions import render_generation_jpeg


def test_render_generation_jpeg_bytes_and_no_alt(two_sided_sample):
    left, _ = build_pair(two_sided_sample)
    data = render_generation_jpeg(left)
    assert data[:2] == b"\xff\xd8"                       # JPEG magic
    lone = json.loads(json.dumps(left))
    lone["habitat_meta"]["alt_path"] = None
    assert render_generation_jpeg(lone)[:2] == b"\xff\xd8"


def test_four_word_instructions_are_accepted():
    assert instruction_rejects("Go left around the sofa.", side="left") == []
    assert instruction_rejects("Go left sofa.", side="left") == ["length:3"]
```
Append to `tests/test_labelling_io.py`:
```python
from rover_vlm.labelling_io import image_url


def test_image_url_path_and_bytes(tmp_path):
    p = tmp_path / "x.jpg"
    p.write_bytes(b"\xff\xd8xx")
    assert image_url(p).startswith("file://")
    assert image_url(b"\xff\xd8xx") == "data:image/jpeg;base64,/9h4eA=="
```
Run: `~/.local/bin/uv run pytest -q tests/test_instructions.py tests/test_labelling_io.py` → FAIL (ImportError).

- [ ] **Step 2: Implement**

In `instructions.py` add `import io` and replace `render_generation_image` with:
```python
def render_generation_jpeg(record):
    """Cosmos's generation overlay as JPEG bytes: the record's route bold magenta and, when
    the record has one, the alternative thin grey. Rendered in memory so the ~28k v2 requests
    need no overlay files on NFS."""
    img, draw = _canvas(record)
    W, H = img.size
    alt = record["habitat_meta"].get("alt_path")
    if alt:
        draw_polyline(draw, _xy(alt), OTHER_COLOR, W, H, width=3)
    draw_polyline(draw, _xy(_answer_path(record)), TARGET_COLOR, W, H, width=7)
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=90)
    return buf.getvalue()


def render_generation_image(record, out_path):
    """render_generation_jpeg written to out_path (v1 kept its overlays on disk)."""
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_bytes(render_generation_jpeg(record))
```
Change `_WORD_LIMITS = (6, 25)` to:
```python
# v2 (2026-09-25): minimum 6 -> 4 words. 77 of v1's 91 train rejects were short, correct
# instructions such as "Go left around the sofa."; v1's finalized data is unaffected.
_WORD_LIMITS = (4, 25)
```
Update the existing assertion in `test_instruction_rejects` (`instruction_rejects("Go left.") == ["length:2"]` stays valid). In `labelling_io.py` add `import base64` and:
```python
def image_url(image):
    """A path becomes a file:// URI (the server reads NFS directly); bytes become a data URL."""
    if isinstance(image, (bytes, bytearray)):
        return "data:image/jpeg;base64," + base64.b64encode(bytes(image)).decode()
    return Path(image).resolve().as_uri()
```
and in `chat_with_image` rename the parameter `image_path` → `image` and use `{"url": image_url(image)}`; docstring: "`image` is a file path or JPEG bytes".

- [ ] **Step 3: Tests pass** — same command → PASS; full suite `~/.local/bin/uv run pytest -q` → PASS.

- [ ] **Step 4: Commit**
```bash
git add src/rover_vlm/instructions.py src/rover_vlm/labelling_io.py tests/test_instructions.py tests/test_labelling_io.py
git commit -m "feat: in-memory Cosmos overlays sent as data URLs; 4-word instruction minimum"
```

---

### Task 2: Frame-level record builders

**Files:**
- Modify: `src/rover_vlm/instructions.py`
- Test: `tests/test_instructions.py`, `tests/conftest.py` (add a 3-candidate fixture helper)

**Interfaces:**
- Consumes: `build_record`, `build_pair`, `side_candidates`, `TEMPLATE_INSTRUCTIONS`, `SIDES`.
- Produces:
  - `route_side(meta) -> "left"|"right"|"direct"|None` — side label of `meta["label"]`.
  - `other_side_candidate(meta, side) -> int|None` — best candidate labelled the opposite side (feasible first, then in index order), else any other candidate index, else None.
  - `build_frame(source_rec, *, meta_extra=None, reasons=None) -> dict` returning `{"plain": rec, "labelset": [rec, ...]}`; `plain` is a deep copy of `source_rec`; `labelset` holds the records to label, each with `habitat_meta.kind ∈ {"instructed", "opposite"}`, `side`, `alt_path` (list or None), and `pair_id` only for pair frames. `direct` frames → empty labelset.
  - `build_wrong_side(source_rec, *, meta_extra=None) -> dict|None` — for a single-route frame whose opposite-side candidate exists and is infeasible: record from that blocked candidate, id `<sample>_W`, `kind="wrong_side"`, `side` = the blocked side, `alt_path` = the feasible route; else None.

- [ ] **Step 1: Fixture helper** — in `tests/conftest.py` extend `write_sample` with optional `extra_uv=None, labels=("left","right")` so a test can make a 3-candidate sample or change labels/feasibility:
```python
def write_sample(d, *, left_uv, right_uv, label=0, goal_uv=(256, 300), accepted=(0, 1),
                 labels=("left", "right"), extra=None):
    """extra: optional list of (uv_points, side_label, feasible) appended as candidates."""
    d.mkdir(parents=True, exist_ok=True)
    cands_uv = [left_uv, right_uv] + [e[0] for e in (extra or [])]
    fpv = {
        "image_size": [512, 512],
        "goal": {"uv": list(goal_uv), "hidden": False},
        "candidates": [{"runs": [{"hidden": False, "uv": [list(p) for p in uv]}]} for uv in cands_uv],
    }
    cand_meta = [{"label": labels[0], "feasible": 0 in accepted},
                 {"label": labels[1], "feasible": 1 in accepted}]
    cand_meta += [{"label": e[1], "feasible": e[2]} for e in (extra or [])]
    meta = {
        "sample_id": d.name, "scene_id": "scene_a", "flavor": "doorway",
        "label": label, "accepted": list(accepted), "correct_path_in_fov": True,
        "candidates": cand_meta,
    }
    (d / "fpv_paths.json").write_text(json.dumps(fpv))
    (d / "meta.json").write_text(json.dumps(meta))
    Image.new("RGB", (64, 64), (120, 120, 120)).save(d / "fpv_enhanced.png")
    return d
```
(existing callers keep working: new parameters have defaults.)

- [ ] **Step 2: Failing tests** (append to `tests/test_instructions.py`)
```python
from rover_vlm.habitat_data import build_record
from rover_vlm.instructions import build_frame, build_wrong_side, other_side_candidate, route_side


def _src(d):
    """What data/prepared_habitat_v4 holds for this sample."""
    return build_record(d, framing="endpoints", meta_extra={"sample_id": d.name})


def test_build_frame_pair_frame(two_sided_sample):
    f = build_frame(_src(two_sided_sample))
    assert f["plain"] == _src(two_sided_sample)
    kinds = sorted((r["habitat_meta"]["kind"], r["habitat_meta"]["side"]) for r in f["labelset"])
    assert kinds == [("instructed", "left"), ("opposite", "right")]
    assert all(r["habitat_meta"]["pair_id"] == "scene_a_c000" for r in f["labelset"])


def test_build_frame_single_route(tmp_path):
    # right route blocked: single-route frame; its alt_path is the blocked right route
    d = write_sample(tmp_path / "s_c020", left_uv=[(256, 512), (140, 400), (256, 300)],
                     right_uv=[(256, 512), (370, 400), (256, 300)], accepted=(0,))
    f = build_frame(_src(d))
    (rec,) = f["labelset"]
    hm = rec["habitat_meta"]
    assert (rec["id"], hm["kind"], hm["side"]) == ("s_c020_L", "instructed", "left")
    assert "pair_id" not in hm and hm["alt_path"] and max(x for x, _, _ in hm["alt_path"]) > 0.6


def test_build_frame_direct_is_plain_only(tmp_path):
    d = write_sample(tmp_path / "s_c021", left_uv=[(256, 512), (256, 300)],
                     right_uv=[(256, 512), (370, 400), (256, 300)], labels=("direct", "right"))
    f = build_frame(_src(d))
    assert f["labelset"] == [] and f["plain"]["id"] == "s_c021"


def test_other_side_candidate_prefers_feasible_opposite():
    meta = {"label": 0, "candidates": [{"label": "left", "feasible": True},
                                       {"label": "right", "feasible": False},
                                       {"label": "right", "feasible": True}]}
    assert other_side_candidate(meta, "left") == 2
    meta["candidates"][2]["feasible"] = False
    assert other_side_candidate(meta, "left") == 1
    assert other_side_candidate({"label": 0, "candidates": [{"label": "left", "feasible": True}]}, "left") is None
    assert route_side(meta) == "left"


def test_single_route_without_alternative_builds_with_none(tmp_path):
    d = write_sample(tmp_path / "s_c022", left_uv=[(256, 512), (140, 400), (256, 300)],
                     right_uv=[(900, 900), (950, 950)], accepted=(0,))    # other candidate off-frame
    (rec,) = build_frame(_src(d))["labelset"]
    assert rec["habitat_meta"]["alt_path"] is None


def test_build_wrong_side(tmp_path):
    d = write_sample(tmp_path / "s_c023", left_uv=[(256, 512), (140, 400), (256, 300)],
                     right_uv=[(256, 512), (370, 400), (256, 300)], accepted=(0,))
    w = build_wrong_side(_src(d))
    hm = w["habitat_meta"]
    assert (w["id"], hm["kind"], hm["side"]) == ("s_c023_W", "wrong_side", "right")
    assert max(x for x, _, _ in json.loads(w["conversations"][1]["value"])["path"]) > 0.6
    assert max(x for x, _, _ in hm["alt_path"]) < 0.5
    assert build_wrong_side(_src(write_sample(tmp_path / "s_c024",
        left_uv=[(256, 512), (140, 400), (256, 300)], right_uv=[(256, 512), (370, 400), (256, 300)]))) is None
```
Run → FAIL (ImportError).

- [ ] **Step 3: Implement** (append to `instructions.py`; add `import copy` is already present)
```python
# --- v2: every round-4 frame -----------------------------------------------------------

def route_side(meta):
    """Side label ("left" / "right" / "direct") of the labelled candidate, or None."""
    cands, label = meta.get("candidates", []), meta.get("label")
    return cands[label].get("label") if label is not None and 0 <= label < len(cands) else None


def other_side_candidate(meta, side):
    """Best candidate on the opposite side (feasible first), else any other; None if none."""
    cands, label = meta.get("candidates", []), meta.get("label")
    other = "right" if side == "left" else "left"
    opposite = [i for i, c in enumerate(cands) if i != label and c.get("label") == other]
    opposite.sort(key=lambda i: not cands[i].get("feasible"))
    rest = [i for i in range(len(cands)) if i != label and i not in opposite]
    ranked = opposite + rest
    return ranked[0] if ranked else None


def _candidate_path(sample_dir, index):
    rec = build_record(sample_dir, framing="endpoints", candidate=index)
    return _answer_path(rec) if rec is not None else None


def _labelled(rec, side, kind, alt_path, sample_name, suffix=None):
    rec["id"] = f"{sample_name}_{suffix or side[0].upper()}"
    rec["habitat_meta"].update({"side": side, "kind": kind, "alt_path": alt_path,
                                "instruction_source": "template"})
    return rec


def build_frame(source_rec, *, meta_extra=None, reasons=None):
    """Plain copy + the records to label for one round-4 frame.

    Pair frames (build_pair accepts them) give both sides; the correct side is kind
    "instructed", the other "opposite". Single-route frames give one "instructed" record
    whose alt_path is the best other-side candidate (only for Cosmos's overlay; no pair_id,
    so no pair metric sees it). `direct` frames give the plain copy only.
    """
    sample_dir = Path(source_rec["image"][0]).parent
    meta = json.loads((sample_dir / "meta.json").read_text())
    out = {"plain": copy.deepcopy(source_rec), "labelset": []}
    side = route_side(meta)
    if side not in SIDES:
        _skip(reasons, "direct")
        return out
    pair = build_pair(sample_dir, meta_extra=meta_extra, reasons=reasons)
    if pair is not None:
        for rec in pair:
            rec["habitat_meta"]["kind"] = "instructed" if rec["habitat_meta"]["side"] == side else "opposite"
        out["labelset"] = pair
        return out
    rec = build_record(sample_dir, framing="endpoints", instruction=TEMPLATE_INSTRUCTIONS[side],
                       meta_extra=meta_extra)
    if rec is None:
        _skip(reasons, "record_failed")
        return out
    alt_idx = other_side_candidate(meta, side)
    alt = _candidate_path(sample_dir, alt_idx) if alt_idx is not None else None
    out["labelset"] = [_labelled(rec, side, "instructed", alt, sample_dir.name)]
    return out


def build_wrong_side(source_rec, *, meta_extra=None):
    """Probe record: a single-route frame instructed to take its blocked other side.

    Built from the blocked candidate, so the eval's `follows` reads as "obeyed"; alt_path is
    the feasible route. None unless the opposite-side candidate exists and is infeasible.
    """
    sample_dir = Path(source_rec["image"][0]).parent
    meta = json.loads((sample_dir / "meta.json").read_text())
    side = route_side(meta)
    if side not in SIDES or side_candidates(meta) is not None:
        return None
    idx = other_side_candidate(meta, side)
    other = "right" if side == "left" else "left"
    if idx is None or meta["candidates"][idx].get("label") != other or meta["candidates"][idx].get("feasible"):
        return None
    rec = build_record(sample_dir, framing="endpoints", candidate=idx,
                       instruction=TEMPLATE_INSTRUCTIONS[other], meta_extra=meta_extra)
    feasible = _candidate_path(sample_dir, meta["label"])
    if rec is None or feasible is None:
        return None
    return _labelled(rec, other, "wrong_side", feasible, sample_dir.name, suffix="W")
```
Note `build_record` checks `correct_path_in_fov` for every candidate; that flag concerns the label, so a blocked-side record built from a sample where it is true is fine.

- [ ] **Step 4: Tests pass**; full suite PASS.

- [ ] **Step 5: Commit**
```bash
git add src/rover_vlm/instructions.py tests/test_instructions.py tests/conftest.py
git commit -m "feat: frame-level v2 record builders (plain, instructed, opposite, wrong-side probe)"
```

---

### Task 3: v2 label bookkeeping — accept gate, retry selection, finalize

**Files:**
- Modify: `src/rover_vlm/instructions.py` (`summarize_labels`, `gate_failures`, new `retry_ids`, `finalize_general`)
- Test: `tests/test_instructions.py`

**Interfaces:**
- Produces:
  - `summarize_labels(rows)` adds `"accept_rate"` (passed / non-error rows); pair fields count only rows whose `pair_id` is not None.
  - `gate_failures(summary, *, min_pair_survival, max_drawing_refs, min_accept=None)`.
  - `retry_ids(rows) -> set[str]` — ids of non-error rows with `passed` False and no `"retried"` flag.
  - `finalize_general(plain_recs, label_recs, rows_by_id) -> dict[str, list]` with keys `train_or_all` (plain + kept labelled), `instructed` (kept `kind=="instructed"`), `pairs` (both records of fully kept pairs), `wrong_side` (kept `kind=="wrong_side"`), and `"stats"`.

- [ ] **Step 1: Failing tests** (append)
```python
from rover_vlm.instructions import finalize_general, retry_ids


def test_summarize_accept_rate_and_single_rows():
    rows = [{"id": "a_L", "pair_id": None, "passed": True, "rejects": []},
            {"id": "b_R", "pair_id": None, "passed": False, "rejects": ["length:3"]}]
    s = summarize_labels(rows)
    assert s["accept_rate"] == 0.5 and s["pairs_total"] == 0
    assert gate_failures(s, min_pair_survival=0.0, max_drawing_refs=0, min_accept=0.9) == ["accept rate 0.500 < 0.9"]


def test_retry_ids_skips_errors_passed_and_retried():
    rows = [{"id": "a", "passed": False}, {"id": "b", "passed": False, "retried": True},
            {"id": "c", "passed": True}, {"id": "d", "error": "Timeout"}]
    assert retry_ids(rows) == {"a"}


def test_finalize_general(two_sided_sample, tmp_path):
    f = build_frame(_src(two_sided_sample))
    single = build_frame(_src(write_sample(tmp_path / "s_c030", left_uv=[(256, 512), (140, 400), (256, 300)],
                                           right_uv=[(256, 512), (370, 400), (256, 300)], accepted=(0,))))
    labels = f["labelset"] + single["labelset"]
    ok = lambda r: {"id": r["id"], "passed": True, "instruction": f"Pass {r['habitat_meta']['side']} of the lamp."}
    rows = {r["id"]: ok(r) for r in labels}
    out = finalize_general([f["plain"], single["plain"]], labels, rows)
    assert len(out["train_or_all"]) == 2 + 3 and len(out["pairs"]) == 2 and len(out["instructed"]) == 2
    rows[f["labelset"][1]["id"]]["passed"] = False               # one side of the pair rejected
    out = finalize_general([f["plain"], single["plain"]], labels, rows)
    assert out["pairs"] == [] and len(out["train_or_all"]) == 2 + 1
    assert [r["id"] for r in out["instructed"]] == ["s_c030_L"]
    assert out["stats"]["pairs_dropped"] == 1
```
Run → FAIL.

- [ ] **Step 2: Implement**

In `summarize_labels` replace the `pair_ids` line and the return dict's pair fields:
```python
    pair_ids = {r["pair_id"] for r in rows if r.get("pair_id")}
    kept = {pid for pid, rs in _by_pair([r for r in ok if r.get("pair_id")], lambda r: r["pair_id"]).items()
            if len(rs) == 2 and all(r.get("passed") for r in rs)}
```
and add `"accept_rate": round(sum(bool(r.get("passed")) for r in ok) / len(ok), 4) if ok else 0.0,` to the returned dict. Update the docstring (error rows carry `pair_id`; single-route rows have `pair_id` None and count only toward `accept_rate`). The v1 test `test_summarize_and_gate` gives error row `{"id": "d_L", "error": ...}` with no pair_id — change that row to `{"id": "d_L", "pair_id": "d", "error": "Timeout"}` so it still counts as an unlabelled pair (the expectations `pairs_total == 4` stay).

`gate_failures` gains `min_accept=None`:
```python
    if min_accept is not None and summary["accept_rate"] < min_accept:
        out.append(f"accept rate {summary['accept_rate']:.3f} < {min_accept}")
```
Append:
```python
def retry_ids(rows):
    """Rows worth one more Cosmos attempt: ran, were rejected, and were not retried yet."""
    return {r["id"] for r in rows if not r.get("error") and not r.get("passed") and not r.get("retried")}


def finalize_general(plain_recs, label_recs, rows_by_id):
    """Assemble v2 files from plain copies, labelled records and their label rows.

    A labelled record is kept when its row passed; a pair frame is kept only when both of
    its records passed (otherwise both go, the plain copy stays). Kept records carry the
    Cosmos instruction in their prompt (with_instruction).
    """
    passed = {i for i, r in rows_by_id.items() if r.get("passed")}
    groups = _by_pair([r for r in label_recs if r["habitat_meta"].get("pair_id")],
                      lambda r: r["habitat_meta"]["pair_id"])
    bad_pairs = {pid for pid, rs in groups.items() if len(rs) != 2 or not all(r["id"] in passed for r in rs)}
    kept = [with_instruction(r, rows_by_id[r["id"]]["instruction"], "cosmos") for r in label_recs
            if r["id"] in passed and r["habitat_meta"].get("pair_id") not in bad_pairs]
    kind = lambda k: [r for r in kept if r["habitat_meta"]["kind"] == k]  # noqa: E731
    return {
        "train_or_all": [copy.deepcopy(p) for p in plain_recs] + [r for r in kept if r["habitat_meta"]["kind"] != "wrong_side"],
        "instructed": kind("instructed"),
        "pairs": [r for r in kept if r["habitat_meta"].get("pair_id")],
        "wrong_side": kind("wrong_side"),
        "stats": {"labelled": len(label_recs), "kept": len(kept),
                  "pairs_kept": len(groups) - len(bad_pairs), "pairs_dropped": len(bad_pairs)},
    }
```
(`bad_pairs` holds ids; a record with no pair_id has `None`, which is never in `bad_pairs`.)

- [ ] **Step 3: Tests pass**; full suite PASS.

- [ ] **Step 4: Commit**
```bash
git add src/rover_vlm/instructions.py tests/test_instructions.py
git commit -m "feat: v2 label bookkeeping — accept-rate gate, one retry, finalize_general"
```

---

### Task 4: Step checkpoints and resume in `train.py`

**Files:**
- Modify: `scripts/train.py`
- Test: `tests/test_train_args.py` (new)

**Interfaces:**
- Produces: CLI `--save-steps N` (default 0 = per-epoch saving as today) and `--resume`; pure helpers `save_kwargs(args) -> dict` and `resume_checkpoint(output_dir, resume) -> str|None` in `scripts/train.py`.

- [ ] **Step 1: Failing test** — `tests/test_train_args.py`:
```python
import importlib.util
from pathlib import Path
from types import SimpleNamespace


def _train():
    path = Path(__file__).resolve().parent.parent / "scripts" / "train.py"
    spec = importlib.util.spec_from_file_location("train_script", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_save_kwargs():
    t = _train()
    assert t.save_kwargs(SimpleNamespace(smoke=False, save_steps=0)) == {"save_strategy": "epoch"}
    assert t.save_kwargs(SimpleNamespace(smoke=False, save_steps=500)) == {
        "save_strategy": "steps", "save_steps": 500, "save_total_limit": 2}
    assert t.save_kwargs(SimpleNamespace(smoke=True, save_steps=500)) == {"save_strategy": "no"}


def test_resume_checkpoint(tmp_path):
    t = _train()
    assert t.resume_checkpoint(tmp_path, resume=True) is None          # nothing yet: start fresh
    (tmp_path / "checkpoint-500").mkdir()
    (tmp_path / "checkpoint-1000").mkdir()
    assert t.resume_checkpoint(tmp_path, resume=True).endswith("checkpoint-1000")
    assert t.resume_checkpoint(tmp_path, resume=False) is None
```
Run → FAIL (AttributeError).

- [ ] **Step 2: Implement** — in `scripts/train.py` add after the imports:
```python
def save_kwargs(args):
    """TrainingArguments save settings. --save-steps N keeps the last 2 step checkpoints, so a
    long single-epoch run killed near the end resumes instead of starting over."""
    if args.smoke:
        return {"save_strategy": "no"}
    if args.save_steps:
        return {"save_strategy": "steps", "save_steps": args.save_steps, "save_total_limit": 2}
    return {"save_strategy": "epoch"}


def resume_checkpoint(output_dir, resume):
    """Latest checkpoint-N under output_dir when --resume is set, else None (fresh start)."""
    if not resume:
        return None
    ckpts = sorted(Path(output_dir).glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))
    return str(ckpts[-1]) if ckpts else None
```
Add arguments:
```python
    p.add_argument("--save-steps", type=int, default=0,
                   help="checkpoint every N steps (keeps 2); 0 = once per epoch as before")
    p.add_argument("--resume", action="store_true",
                   help="continue from the latest checkpoint in --output-dir, if any")
```
In `TrainingArguments(...)` replace `save_strategy="no" if args.smoke else "epoch",` with `**save_kwargs(args),` and change `result = trainer.train()` to:
```python
    ckpt = resume_checkpoint(args.output_dir, args.resume)
    if ckpt:
        print(f"resuming from {ckpt}")
    result = trainer.train(resume_from_checkpoint=ckpt)
```
Add to the docstring: "Long single-epoch runs: `--save-steps 500 --resume` (re-submit the same command after a timeout)."

- [ ] **Step 3: Tests pass** (`~/.local/bin/uv run pytest -q tests/test_train_args.py`; importing train.py pulls torch/transformers but loads no model); full suite PASS.

- [ ] **Step 4: Commit**
```bash
git add scripts/train.py tests/test_train_args.py
git commit -m "feat: step checkpoints and resume in train.py"
```

---

### Task 5: Eval metrics for mixed sets + clustered bootstrap

**Files:**
- Modify: `src/rover_vlm/eval.py` (`aggregate_instruction_metrics`, new `clustered_bootstrap_ci`)
- Test: `tests/test_habitat_eval.py`

**Interfaces:**
- Produces: `aggregate_instruction_metrics(records)` — pairs formed only from records with `pair_id`; adds `pair_success_by_source` (`{"original": x, "augmented": y}` from `habitat_meta.source` of the pair's left record); returns `n_pairs = 0` and `pair_success = None` when no pairs. `clustered_bootstrap_ci(values, groups, n=4000, seed=0) -> (lo, hi)` resamples groups.

- [ ] **Step 1: Failing tests** (append)
```python
from rover_vlm.eval import clustered_bootstrap_ci


def test_aggregate_ignores_unpaired_alt_path_records():
    recs = [_irec("a", "left", _L), _irec("a", "right", _R)]
    lone = _irec("z", "left", _L)
    lone["habitat_meta"].pop("pair_id")
    recs[0]["habitat_meta"]["source"] = "augmented"
    m = aggregate_instruction_metrics(recs + [lone])
    assert m["n_pairs"] == 1 and m["n_records"] == 3 and m["pair_success"] == 1.0
    assert m["pair_success_by_source"] == {"augmented": 1.0}
    solo = aggregate_instruction_metrics([lone])
    assert solo["n_pairs"] == 0 and solo["pair_success"] is None and solo["follow_rate"] == 1.0


def test_clustered_bootstrap_resamples_groups():
    lo, hi = clustered_bootstrap_ci([1, 1, 1, 0, 0, 0], ["a", "a", "a", "b", "b", "b"], n=500)
    assert lo == 0.0 and hi == 1.0            # only two clusters: whole-cluster draws
    lo, hi = clustered_bootstrap_ci([1] * 6, list("abcdef"), n=200)
    assert lo == hi == 1.0
```
In `_irec`, pair records already carry `pair_id`. Run → FAIL.

- [ ] **Step 2: Implement** — in `aggregate_instruction_metrics` replace the loop's pair collection and the return:
```python
        follows[r["id"]] = ok
        if hm.get("pair_id"):
            by_pair.setdefault(hm["pair_id"], []).append(r)
        by_side.setdefault(hm["side"], []).append(ok)
    pairs = [p for p in by_pair.values() if len(p) == 2]
    success = [all(follows[r["id"]] for r in p) for p in pairs]
    by_source = {}
    for p, s in zip(pairs, success):
        by_source.setdefault(p[0]["habitat_meta"].get("source") or "unknown", []).append(s)
    return {
        "n_records": len(rows),
        "n_pairs": len(pairs),
        "follow_rate": float(np.mean(list(follows.values()))),
        "pair_success": float(np.mean(success)) if pairs else None,
        "pair_success_by_source": {k: float(np.mean(v)) for k, v in sorted(by_source.items())},
        "identical_output_rate": (float(np.mean(
            [p[0].get("generated", "").strip() == p[1].get("generated", "").strip() for p in pairs]))
            if pairs else None),
        "follow_rate_by_side": {s: float(np.mean(v)) for s, v in sorted(by_side.items())},
    }
```
Update its docstring: pairs come only from records with `pair_id`; single-route instructed and wrong-side probe records count toward `follow_rate` only. Append:
```python
def clustered_bootstrap_ci(values, groups, n=4000, seed=0):
    """95% bootstrap interval of mean(values), resampling whole groups (e.g. viewpoints)."""
    rng = np.random.default_rng(seed)
    by = {}
    for v, g in zip(values, groups):
        by.setdefault(g, []).append(float(v))
    keys = list(by)
    sums = np.array([sum(by[k]) for k in keys])
    counts = np.array([len(by[k]) for k in keys])
    means = []
    for _ in range(n):
        idx = rng.integers(0, len(keys), len(keys))
        means.append(sums[idx].sum() / counts[idx].sum())
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))
```
Check the v1 report still runs: `~/.local/bin/uv run scripts/visualize_instruction_results.py` (it reads `pair_success` from v1 metrics, unchanged).

- [ ] **Step 3: Tests pass**; full suite PASS.

- [ ] **Step 4: Commit**
```bash
git add src/rover_vlm/eval.py tests/test_habitat_eval.py
git commit -m "feat: instruction metrics for mixed eval sets, by-source pair success, clustered bootstrap"
```

---

### Task 6: `label_instructions.py` for v2 rows

**Files:**
- Modify: `scripts/label_instructions.py`
- Test: covered by Tasks 1/3 unit tests; `--help` smoke here.

**Interfaces:**
- Consumes: `render_generation_jpeg`, `retry_ids`, `gate_failures(min_accept=...)`, `chat_with_image(image=bytes|path)`.
- Produces: rows with `pair_id` = `habitat_meta.get("pair_id")`, `kind`, and `retried: True` on retried rows; CLI `--min-accept`, `--retry-rejected`.

- [ ] **Step 1: Edit**
  - Row: `"pair_id": hm.get("pair_id"), "kind": hm.get("kind"),`.
  - Image: `image = hm["gen_image"] if hm.get("gen_image") else render_generation_jpeg(rec)` and pass `image` to `chat_with_image`.
  - `--min-accept` (float, default None) passed to `gate_failures(..., min_accept=args.min_accept)`.
  - `--retry-rejected`: after `prepare_output` (which requires `--resume`; exit with a message otherwise), read the file, `todo_ids = retry_ids(rows)`, rewrite the file without those rows (atomic `.tmp` + replace), set `done -= todo_ids`, use `seed = args.seed + 1` for the requests, and set `row["retried"] = True` on each new row. Print `retrying N rejected rows with seed S`.
  - Docstring: add the v2 usage lines (`--split data/prepared_habitat_instr_v2/labelset_eval.json --out outputs/instr_labels_v2/eval_i2.jsonl`, `--retry-rejected --resume`, `--gate --min-accept 0.9 --min-pair-survival 0`).
- [ ] **Step 2:** `~/.local/bin/uv run scripts/label_instructions.py --help` shows both flags; full suite PASS.
- [ ] **Step 3: Commit**
```bash
git add scripts/label_instructions.py
git commit -m "feat: label_instructions — inline overlays, single-route rows, accept gate, one retry"
```

---

### Task 7: `prepare_habitat_instr.py general` + `finalize-general`, and a real CPU run

**Files:**
- Modify: `scripts/prepare_habitat_instr.py`
- Create: `configs/instructions_handwritten_v2.json` (written in Task 9; `finalize-general` skips the probe with a warning if absent)

**Interfaces:**
- Consumes: `build_frame`, `build_wrong_side`, `finalize_general`, `with_instruction`, `assert_scene_disjoint`.
- Produces under `data/prepared_habitat_instr_v2/`: `plain_{train,eval}.json`, `labelset_{train,eval}.json` (eval labelset includes the 300 wrong-side probe records), `handwritten_pairs.json` (the 30 chosen eval pair ids), `general_meta.json`; after finalize: `train_general.json`, `eval_plain.json`, `eval_instructed.json`, `eval_pairs.json`, `probe_wrong_side.json`, `probe_handwritten.json`, `meta_v2.json`.

- [ ] **Step 1: Add the stages**
```python
V2_OUT = REPO_ROOT / "data" / "prepared_habitat_instr_v2"
WRONG_SIDE_N, HANDWRITTEN_N, V2_SEED = 300, 30, 7


def stage_general(args):
    import random
    reasons, stats = {}, {}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, src_name in SPLITS.items():
        src = json.loads((args.source / f"{src_name}.json").read_text())
        if args.limit:
            src = src[: args.limit]
        plain, labelset, wrong = [], [], []
        for i, rec in enumerate(src):
            extra = {k: rec["habitat_meta"].get(k) for k in CARRY}
            f = build_frame(rec, meta_extra=extra, reasons=reasons)
            plain.append(f["plain"])
            labelset.extend(f["labelset"])
            if name == "eval" and not f["labelset"][1:]:          # single-route (or direct) frame
                w = build_wrong_side(rec, meta_extra=extra)
                if w is not None:
                    wrong.append(w)
            if (i + 1) % 2000 == 0:
                print(f"  {name}: {i + 1}/{len(src)} frames, {len(labelset)} to label", flush=True)
        if name == "eval":
            random.Random(V2_SEED).shuffle(wrong)
            labelset.extend(wrong[:WRONG_SIDE_N])
            orig_pairs = sorted({r["habitat_meta"]["pair_id"] for r in labelset
                                 if r["habitat_meta"].get("pair_id") and r["habitat_meta"].get("source") == "original"})
            random.Random(V2_SEED).shuffle(orig_pairs)
            (args.out_dir / "handwritten_pairs.json").write_text(json.dumps(sorted(orig_pairs[:HANDWRITTEN_N]), indent=1))
        (args.out_dir / f"plain_{name}.json").write_text(json.dumps(plain, indent=1))
        (args.out_dir / f"labelset_{name}.json").write_text(json.dumps(labelset, indent=1))
        kinds = Counter(r["habitat_meta"]["kind"] for r in labelset)
        pairs = {r["habitat_meta"]["pair_id"]: r["habitat_meta"].get("source") for r in labelset if r["habitat_meta"].get("pair_id")}
        stats[name] = {"plain": len(plain), "to_label": len(labelset), "kinds": dict(kinds),
                       "pairs_by_source": dict(Counter(pairs.values())),
                       "pair_viewpoints": len({r["habitat_meta"].get("parent_sample_id") or r["habitat_meta"]["pair_id"]
                                               for r in labelset if r["habitat_meta"].get("pair_id")})}
        print(f"{name}: {json.dumps(stats[name])}")
    train = json.loads((args.out_dir / "labelset_train.json").read_text())
    eval_ = json.loads((args.out_dir / "labelset_eval.json").read_text())
    assert_scene_disjoint(train, eval_)
    (args.out_dir / "general_meta.json").write_text(json.dumps({"source": str(args.source), "skipped": reasons,
                                                                "stats": stats}, indent=2))


def _rows(paths):
    rows = {}
    for path in paths:
        for line in Path(path).read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                if not row.get("error"):
                    rows[row["id"]] = row
    return rows


def stage_finalize_general(args):
    rows = _rows(args.instructions)
    meta = {"instructions": [str(p) for p in args.instructions]}
    d = args.out_dir
    for name in SPLITS:
        plain = json.loads((d / f"plain_{name}.json").read_text())
        labels = json.loads((d / f"labelset_{name}.json").read_text())
        out = finalize_general(plain, labels, rows)
        meta[name] = {**out["stats"], "records": len(out["train_or_all"])}
        if name == "train":
            (d / "train_general.json").write_text(json.dumps(out["train_or_all"], indent=1))
        else:
            for fname, recs in (("eval_plain", plain), ("eval_instructed", out["instructed"]),
                                ("eval_pairs", out["pairs"]), ("probe_wrong_side", out["wrong_side"])):
                (d / f"{fname}.json").write_text(json.dumps(recs, indent=1))
                meta[name][fname] = len(recs)
            hw_path = REPO_ROOT / "configs" / "instructions_handwritten_v2.json"
            if hw_path.exists():
                texts = json.loads(hw_path.read_text())
                chosen = set(json.loads((d / "handwritten_pairs.json").read_text()))
                hw = [with_instruction(r, texts[r["id"]], "handwritten") for r in labels
                      if r["habitat_meta"].get("pair_id") in chosen]
                missing = [r["id"] for r in labels if r["habitat_meta"].get("pair_id") in chosen and r["id"] not in texts]
                assert not missing, f"handwritten file lacks {missing[:4]}"
                (d / "probe_handwritten.json").write_text(json.dumps(hw, indent=1))
                meta[name]["probe_handwritten"] = len(hw)
            else:
                print(f"WARNING: {hw_path} missing — probe_handwritten.json not written")
        print(f"{name}: {json.dumps(meta[name])}")
    (d / "meta_v2.json").write_text(json.dumps(meta, indent=2))
```
Note for the handwritten probe: `with_instruction` on the *template* labelset records (their instruction replaced), so the handwritten probe does not depend on Cosmos passing those pairs.

Register in `main()`:
```python
    c = sub.add_parser("general", help="v2: plain + records to label for every round-4 frame")
    c.add_argument("--source", type=Path, default=SOURCE)
    c.add_argument("--out-dir", type=Path, default=V2_OUT)
    c.add_argument("--limit", type=int, default=None)
    e = sub.add_parser("finalize-general", help="v2: assemble train/eval files after labelling")
    e.add_argument("--out-dir", type=Path, default=V2_OUT)
    e.add_argument("--instructions", type=Path, nargs="+", required=True)
```
and dispatch `{"pairs": stage_pairs, "finalize": stage_finalize, "general": stage_general, "finalize-general": stage_finalize_general}[args.stage](args)`. Imports: add `build_frame, build_wrong_side, finalize_general, with_instruction`. Docstring: add the v2 usage block.

Note: `not f["labelset"][1:]` is true for single-route (1 record) and `direct` (0) frames only; `build_wrong_side` itself returns None for `direct` and pair frames.

- [ ] **Step 2: Quick check** — `~/.local/bin/uv run scripts/prepare_habitat_instr.py general --limit 300 --out-dir /tmp/claude-1013/v2_try` runs and prints stats with non-zero `instructed`, `opposite`, `wrong_side`.

- [ ] **Step 3: Real run** — `~/.local/bin/uv run scripts/prepare_habitat_instr.py general`.
Expected (spec): train plain 23,770, to_label 25,985 (instructed 23,722, opposite 2,263); eval plain 2,591, to_label 2,866 + 300 wrong-side; eval pairs 102 original + 180 augmented from ~104 viewpoints. Report any difference.

- [ ] **Step 4: Commit** (code only)
```bash
git add scripts/prepare_habitat_instr.py
git commit -m "feat: prepare_habitat_instr general + finalize-general (v2 dataset)"
```

---

### Task 8: Unattended labelling job and v2 eval job

**Files:**
- Create: `slurm/label_instructions.sbatch`, `slurm/eval_instruction_v2.sbatch`

- [ ] **Step 1: `slurm/label_instructions.sbatch`**
```bash
#!/bin/bash
# Label the v2 instruction set with Cosmos3-Nano, unattended: serve + client in one job.
#     sbatch slurm/label_instructions.sbatch
# Stage 0 gate (GATE_N single-route eval records; accept >= GATE_MIN_ACCEPT, drawing refs 0)
# aborts before the full run on failure. Then eval + train labelsets, then one retry of the
# rejected rows with seed+1. Safe to re-submit: every call uses --resume.
# After it: uv run scripts/prepare_habitat_instr.py finalize-general --instructions \
#     outputs/instr_labels_v2/train_i2.jsonl outputs/instr_labels_v2/eval_i2.jsonl
#SBATCH --job-name=cosmos3-label-instr
#SBATCH --partition=A100
#SBATCH --gres=gpu:3g.40gb:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=14:00:00
#SBATCH --output=outputs/slurm/%x-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
mkdir -p outputs/slurm outputs/instr_labels_v2
source slurm/cosmos3_serve_lib.sh
D="${DATA:-data/prepared_habitat_instr_v2}"
OUT=outputs/instr_labels_v2
CONCURRENCY="${CONCURRENCY:-8}"
GATE_N="${GATE_N:-200}"
GATE_MIN_ACCEPT="${GATE_MIN_ACCEPT:-0.9}"
UV="${UV:-$HOME/.local/bin/uv}"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
unset VIRTUAL_ENV
cosmos3_start_server
trap 'rm -f "$ENDPOINT_FILE"; kill $SERVER_PID 2>/dev/null || true' EXIT
cosmos3_wait_healthy
if [ "$GATE_N" -gt 0 ]; then
    "$UV" run python - "$D/labelset_eval.json" "$OUT/gate_split.json" "$GATE_N" <<'EOF'
import json, sys
recs = [r for r in json.load(open(sys.argv[1])) if r["habitat_meta"]["kind"] == "instructed"
        and not r["habitat_meta"].get("pair_id")]
json.dump(recs[: int(sys.argv[3])], open(sys.argv[2], "w"))
EOF
    "$UV" run scripts/label_instructions.py --split "$OUT/gate_split.json" --out "$OUT/gate_i2.jsonl" \
        --endpoint "$BASE/v1" --concurrency "$CONCURRENCY" --resume \
        --gate --min-accept "$GATE_MIN_ACCEPT" --min-pair-survival 0
    echo "gate PASSED"
fi
for s in eval train; do
    "$UV" run scripts/label_instructions.py --split "$D/labelset_$s.json" --out "$OUT/${s}_i2.jsonl" \
        --endpoint "$BASE/v1" --concurrency "$CONCURRENCY" --resume
    "$UV" run scripts/label_instructions.py --split "$D/labelset_$s.json" --out "$OUT/${s}_i2.jsonl" \
        --endpoint "$BASE/v1" --concurrency "$CONCURRENCY" --resume --retry-rejected
done
echo "=============== done ==============="
wc -l "$OUT"/*_i2.jsonl
```
- [ ] **Step 2: `slurm/eval_instruction_v2.sbatch`**
```bash
#!/bin/bash
# v2 eval cells (skips finished ones):
#   general on eval_plain / eval_instructed / eval_pairs / probe_wrong_side / probe_handwritten,
#   v1 Cosmos adapter on eval_pairs.
#     sbatch slurm/eval_instruction_v2.sbatch
#SBATCH --job-name=qwen-instr-eval-v2
#SBATCH --partition=A100
#SBATCH --gres=gpu:3g.40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=outputs/slurm/%x-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
mkdir -p outputs/slurm
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
UV=~/.local/bin/uv
D=data/prepared_habitat_instr_v2
OUT=outputs/eval_instr_v2
GEN=outputs/runs_instr_v2/general/adapter
run() {  # run <tag> <set> <adapter>
    local tag=$1 set=$2 adapter=$3
    if [ -f "$OUT/$set/$tag/metrics.json" ]; then echo "skip $tag/$set (done)"; return; fi
    if [ ! -f "$D/$set.json" ]; then echo "SKIP $tag/$set: no $D/$set.json"; return; fi
    if [ ! -d "$adapter" ]; then echo "SKIP $tag/$set: no adapter at $adapter"; return; fi
    echo "=== $tag on $set ==="
    $UV run scripts/evaluate.py --task habitat --tag "$tag" --eval-file "$D/$set.json" \
        --out-dir "$OUT/$set" --adapter "$adapter"
}
run general eval_plain         $GEN
run general eval_instructed    $GEN
run general eval_pairs         $GEN
run v1_cosmos eval_pairs       outputs/runs_instr_v1/cosmos/adapter
run general probe_wrong_side   $GEN
run general probe_handwritten  $GEN
echo "all cells done: uv run scripts/visualize_instruction_general.py"
```
- [ ] **Step 3:** `bash -n` both files → OK.
- [ ] **Step 4: Commit**
```bash
git add slurm/label_instructions.sbatch slurm/eval_instruction_v2.sbatch
git commit -m "feat: unattended v2 labelling job and v2 eval job"
```

---

### Task 9: Hand-written probe instructions

**Files:**
- Create: `configs/instructions_handwritten_v2.json` — `{record_id: instruction}` for both sides of the 30 pairs in `data/prepared_habitat_instr_v2/handwritten_pairs.json`.

- [ ] **Step 1:** Render each chosen frame with both routes and their side labels (scratchpad contact sheet via `render_generation_jpeg`-style drawing, both routes drawn, sides annotated). Do **not** open `outputs/instr_labels_v2/` for these ids.
- [ ] **Step 2:** Write 60 instructions (both sides of 30 pairs) in varied operator phrasing: terse ("left of the couch"), sequential ("go past the chair, then bear left"), colloquial ("hug the wall on the left"), longer landmark chains. Each must contain its own side word and never the opposite one, so the probe measures phrasing rather than contradiction — this rules out forms like "keep the table on your right" for a left route.
- [ ] **Step 3:** Validate: `instruction_rejects(text, side=...)` returns `[]` for all 60 (quick python check), ids cover both records of every chosen pair.
- [ ] **Step 4: Commit**
```bash
git add configs/instructions_handwritten_v2.json
git commit -m "data: 30 hand-written instruction pairs for the v2 phrasing probe"
```

---

### Task 10: Report page + README

**Files:**
- Create: `scripts/visualize_instruction_general.py`
- Modify: `README.md` (new section "General instruction model (v2)", staged hunk-only)
- Test: `tests/test_instructions.py` (`cells_table` helper)

**Interfaces:**
- Consumes: v1 report helpers via `importlib` (`chart_svg`, `pair_figure`, `pair_outcomes`, `CSS`, `JS`, `_pct`) from `scripts/visualize_instruction_results.py`; `clustered_bootstrap_ci`.
- Produces: `outputs/instruction_general.html`; pure `cells_table(root) -> list[dict]` (one row per finished cell: tag, set, parse rate, median error, win vs to_goal, pair success, follow rate).

- [ ] **Step 1: Failing test** (append)
```python
def _load_general_report():
    path = _P(__file__).resolve().parent.parent / "scripts" / "visualize_instruction_general.py"
    spec = importlib.util.spec_from_file_location("vig", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_cells_table(tmp_path):
    d = tmp_path / "eval_pairs" / "general"
    d.mkdir(parents=True)
    (d / "metrics.json").write_text(json.dumps({"parse_rate": 1.0, "mean_point_error_median": 0.02,
                                                "win_vs_to_goal": 0.7, "instruction": {"pair_success": 0.8, "follow_rate": 0.9}}))
    (row,) = _load_general_report().cells_table(tmp_path)
    assert (row["tag"], row["set"], row["pair_success"]) == ("general", "eval_pairs", 0.8)
```
Run → FAIL.

- [ ] **Step 2: Write the script.** Sections, in order, each fed by the eval outputs (no hard-coded results):
  1. Title **"One Model, Both Prompts"**, lede, three headline tiles: plain-prompt median error vs round 4 (0.011), pair success on `eval_pairs` with clustered CI, wrong-side obey rate.
  2. Pass/fail table for cells 1–3 against the spec criteria (≤ 0.013 and ≥ 69% win; not worse than plain; ≥ 75% pair success and CI overlap with `v1_cosmos`).
  3. Pair success chart (`chart_svg`) for `general` and `v1_cosmos` on `eval_pairs`, split original / augmented, intervals from `clustered_bootstrap_ci` grouped by `parent_sample_id` (else pair id).
  4. Gallery (`pair_figure`, with waypoint occlusion markers) of 4 `eval_pairs` examples + 2 hand-written probe pairs + 1 wrong-side probe frame.
  5. Full cells table (`cells_table`), the dataset counts from `general_meta.json` / `meta_v2.json`, limits, reproduce commands.
  Round-4 reference numbers are read from `outputs/eval_habitat_v4/habitat_train_full/metrics.json`.
- [ ] **Step 3:** Test passes; full suite PASS.
- [ ] **Step 4: README** — section listing the v2 steps (general → sbatch labelling → finalize-general → train with `--epochs 1 --save-steps 500 --resume` and `--time=14:00:00` → eval_instruction_v2 → report) and a report-table row; stage only these hunks.
- [ ] **Step 5: Commit**
```bash
git add scripts/visualize_instruction_general.py tests/test_instructions.py README.md   # README via hunk-only staging
git commit -m "feat: v2 report page and README section"
```

---

## After the plan: GPU sequence

1. `sbatch slurm/label_instructions.sbatch` (~4–6 h) → `prepare_habitat_instr.py finalize-general --instructions outputs/instr_labels_v2/train_i2.jsonl outputs/instr_labels_v2/eval_i2.jsonl`
2. `sbatch --time=14:00:00 slurm/train_habitat.sbatch data/prepared_habitat_instr_v2/train_general.json outputs/runs_instr_v2/general --epochs 1 --save-steps 500 --resume`
3. `sbatch slurm/eval_instruction_v2.sbatch` → `uv run scripts/visualize_instruction_general.py` → publish.
