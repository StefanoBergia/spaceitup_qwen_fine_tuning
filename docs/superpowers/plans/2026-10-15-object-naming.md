# Object Naming Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure four object-naming variants on a sampled, blind-audited set, and — once one reaches ≥ 95% — rename every object d2 uses with it and substitute the new names into the existing route texts.

**Architecture:** A revised naming prompt (n2) is added next to today's (n1) and selected by a flag. A CPU stage samples objects and records today's names; one full-A100 job names the sample with three models; an audit page shows each object with its distinct candidate names; a scorer turns the marks into per-variant accuracy. After the audit, the same job renames all objects with the winner, and a CPU stage rebuilds the d2 name files and substitutes names into the desc files (failures go to compose).

**Tech Stack:** Python 3.12, numpy, Pillow, pytest, bash + SLURM, vLLM 0.21 (`.venv-cosmos3`); `uv run` for everything.

**Spec:** `docs/superpowers/specs/2026-10-15-object-naming-design.md`

## Global Constraints

- Branch `feat/turn-events`. Never `git add -A`, never `git stash`. Stage only the files a task names.
- `README.md` holds the user's uncommitted "Prompt framings" table row and `scripts/evaluate.py` is the user's: never stage either. README edits are committed index-only (Task 6 shows how).
- `uv run python` / `uv run pytest` only; `export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4` before tests.
- Never load a model on the login node; ask the user before every `sbatch`; at most 3 SLURM jobs.
- d1 files stay read-only: `outputs/instr_labels_v3/*_d1.jsonl`, `data/prepared_habitat_instr_v2/`, `data/prepared_habitat_instr_v3/`.
- Temp files in the session scratchpad.
- Exact values (spec): sample = 100 random objects, strata 45 wall-kind / 45 object / 10 opening (`kind_hint`), seed 0, plus the user-flagged objects (group `flagged`); variants V0 (today's names), V1 `google/gemma-3-12b-it`, V2 `Qwen/Qwen3.6-35B-A3B-FP8`, V3 `nvidia/Cosmos-Reason2-32B`, V1–V3 with prompt `n2`; GPU `--gres=gpu:a100:1`; gates: best variant ≥ 0.95 on the random 100, G3′ ≥ 90%, steering Gate 1′ ≥ 90%.
- `naming_prompt(need)` with no version stays byte-for-byte today's prompt (n1).
- The artifact page must stay under 16 MB: the names page embeds each naming image at 768 px, JPEG quality 75.

## Review Focus

1. **A variant that fails to start or names nothing** (a model that does not load, all rows rejected). Expected: the job skips that model and still runs the others; scoring treats a missing name as wrong and still prints the other variants. Tests in Tasks 3 and 4.
2. **Two objects whose names swap or overlap** ("sofa"→"bed" while another object "bed"→"table"; "table" inside "dining table"). Expected: substitution never double-replaces and never edits a name inside a longer one. Test in Task 5.
3. **Re-running `rename-texts`** after it already ran (it overwrites the d2 name files). Expected: it refuses unless `--force`, because its backups would be overwritten with already-renamed files. Test in Task 5.
4. **Marks saved as "Done" with nothing ticked, or names ticked without pressing Done.** Expected: the first counts as "none is right" for every variant; the second counts its ticks (the page saves each tick at once). Tests in Task 4.
5. **A flagged object id that is in no objects file** (typo, steering-only object). Expected: `naming-sample` stops with the missing ids listed, never silently drops them. Test in Task 2.

---

## File map

| File | Change |
|---|---|
| `src/rover_vlm/descriptions.py` | `naming_prompt(need, version="n1")` gains n2; `label_object(..., name_prompt="n1")` |
| `scripts/label_descriptions.py` | `--name-prompt n1\|n2` |
| `scripts/prepare_habitat_instr.py` | stages `naming-sample`, `rename-texts`; helpers `naming_sample_rows`, `substitute_names`, `rename_desc_rows` |
| `slurm/naming_variants.sbatch` (new) | names a set of objects with V1/V2/V3 on the full A100 |
| `scripts/audit_instruction_labels.py` + `.html` | `--mode names` page, `score_names`, `--score-marks` |
| `tests/test_naming_fix.py` (new) | Tasks 1–5 |
| `tests/test_slurm_thread_caps.py` | adds the new sbatch |
| `README.md` | run book (index-only) |

---

### Task 1: Prompt n2 and the `--name-prompt` flag

**Files:**
- Modify: `src/rover_vlm/descriptions.py` (`naming_prompt`, `label_object`)
- Modify: `scripts/label_descriptions.py` (argument + `label_object` call)
- Test: `tests/test_naming_fix.py` (create)

**Interfaces:**
- Produces: `naming_prompt(need, version="n1") -> str`; `label_object(ask, obj, seed, image=None, name_prompt="n1") -> dict` (rows gain `"name_prompt"`); CLI `--name-prompt {n1,n2}` (default n1).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_naming_fix.py`:

```python
"""Object naming fix (spec 2026-10-15): prompt n2, the naming sample, the variants job, the names audit, renaming."""
import json
from pathlib import Path

import pytest

from rover_vlm.descriptions import label_object, naming_prompt

REPO = Path(__file__).resolve().parent.parent


def _norm(p):
    return " ".join(p.split())


def _reply(payload):
    return f"<answer>{json.dumps(payload)}</answer>"


@pytest.mark.parametrize("need", ["opening", "wall", "object", "anything"])
def test_n1_is_the_default_and_unchanged(need):
    assert naming_prompt(need) == naming_prompt(need, "n1")


def test_n2_wording():
    p = _norm(naming_prompt("wall", "n2"))
    assert "Name the single thing that fills most of the outline" in p
    assert "Look at what is inside the outline, not at what is next to it or behind it." in p
    assert "If the outline is mostly part of the building, say which part" in p
    assert "Name furniture or an appliance only if the outline is clearly on it" in p
    assert "such as a sofa or a counter" not in p
    assert "something long that the rover drives alongside or stops in front of" in p
    assert '"wall" (part of the building: a wall, a column or pillar, a door, a partition, a railing)' in p
    assert "Never use the words left or right" in p and "not a small part on it" in p
    assert "The rover drives through it." in _norm(naming_prompt("opening", "n2"))
    assert "something the rover drives past" in _norm(naming_prompt("object", "n2"))
    assert "something the rover drives past" in _norm(naming_prompt("anything", "n2"))
    with pytest.raises(ValueError):
        naming_prompt("object", "n9")


def test_label_object_uses_and_records_the_prompt_version():
    seen = []
    obj = {"object_id": "f:b1", "kind_hint": "wall", "image": None, "polygon": [], "crop_box": None}

    def ask(img, prompt, seed):
        seen.append(prompt)
        return _reply({"name": "white wall", "kind": "wall"}), "stop"
    row = label_object(ask, obj, seed=0, image=b"x", name_prompt="n2")
    assert row["passed"] and row["name_prompt"] == "n2" and seen[0] == naming_prompt("wall", "n2")
    assert label_object(ask, obj, seed=0, image=b"x")["name_prompt"] == "n1"


def test_label_descriptions_has_the_flag():
    text = (REPO / "scripts" / "label_descriptions.py").read_text()
    assert '"--name-prompt"' in text and 'choices=["n1", "n2"]' in text
    assert "name_prompt=args.name_prompt" in text
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_naming_fix.py -q`
Expected: FAIL (`naming_prompt()` takes 1 positional argument; `label_object()` got an unexpected keyword argument).

- [ ] **Step 3: Implement**

In `src/rover_vlm/descriptions.py`, rename today's `naming_prompt` body into `_naming_prompt_n1(need)` (unchanged text) and add:

```python
def naming_prompt(need, version="n1"):
    """The naming prompt: n1 (A5, today's) or n2 (naming fix, spec 2026-10-15: name what fills the outline, building parts
    first, no furniture example for long blobs)."""
    if version == "n1":
        return _naming_prompt_n1(need)
    if version == "n2":
        return _naming_prompt_n2(need)
    raise ValueError(f"unknown naming prompt version {version!r}")


_N2_WHAT = {"opening": "an opening (a doorway, an arch, a gap between walls). The rover drives through it.",
            "wall": "something long that the rover drives alongside or stops in front of."}


def _naming_prompt_n2(need):
    what = _N2_WHAT.get(need, "something the rover drives past.")
    return f"""{SCENE_INTRO}

The picture has two panels. Left: the camera view with one region outlined and everything
else dimmed. Right: a close-up of that region. The outlined region is {what}

1. Name the single thing that fills most of the outline, in 1 to 4 words, the way a person would
   say it. Look at what is inside the outline, not at what is next to it or behind it.
   If the outline is mostly part of the building, say which part ("white wall", "brick wall",
   "stone column", "glass partition", "open doorway", "wooden door", "stair railing").
   Name furniture or an appliance only if the outline is clearly on it ("grey sofa",
   "kitchen island", "refrigerator"). Never use the words left or right, and never describe the
   picture itself (outline, panel, close-up, dimming). Name the whole thing the outline covers,
   not a small part on it (a handle, a switch, a socket, a leg).
2. Say what kind of thing it is: "wall" (part of the building: a wall, a column or pillar, a door,
   a partition, a railing), "object" (furniture, an appliance, any other object) or "opening"
   (a doorway, an arch, a gap between walls).

{_answer_block('{"name": "<name>", "kind": "object|wall|opening"}')}"""
```

Change `label_object`:

```python
def label_object(ask, obj, seed, image=None, name_prompt="n1"):
    """Name one object: up to two calls (seed, seed + 1) with naming prompt `name_prompt`. Never raises on a bad reply."""
    need = obj["kind_hint"]
    image = image if image is not None else naming_image(obj["image"], obj["polygon"], obj["crop_box"] or obj["bbox"])
    prompt = naming_prompt(need, name_prompt)
    row = {"id": obj["object_id"], "need": need, "prompt_version": PROMPT_VERSION, "name_prompt": name_prompt}
    rejects, raws, fin, reply = ["parse"], [], None, None
    for attempt, s in enumerate((seed, seed + 1), 1):
        raw, fin = ask(image, prompt, s)
        ...  # rest unchanged
```

(`SCENE_INTRO` and `_answer_block` already exist in `descriptions.py`; n1's body moves unchanged into
`_naming_prompt_n1`. The example names in n2 all pass `name_rejects` today — checked on 2026-10-15: "white wall",
"brick wall", "stone column", "glass partition", "open doorway", "wooden door", "stair railing" -> `[]` for need/kind
wall, opening/opening.)

In `scripts/label_descriptions.py` add next to `--seed`:

```python
    p.add_argument("--name-prompt", choices=["n1", "n2"], default="n1",
                   help="stage name: naming prompt (n2 = naming fix, spec 2026-10-15)")
```

and change the name-stage call (line ~145, `label_object(ask_image, it, args.seed) if args.stage == "name"`) to
`label_object(ask_image, it, args.seed, name_prompt=args.name_prompt)`. Add one docstring line under the steering
example: `#   --name-prompt n2: the naming-fix prompt (spec 2026-10-15); default n1`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_naming_fix.py tests/test_descriptions.py -q`
Expected: PASS (the existing n1 tests are unchanged).

- [ ] **Step 5: Commit**

```bash
git add src/rover_vlm/descriptions.py scripts/label_descriptions.py tests/test_naming_fix.py
git commit -m "feat: naming prompt n2 and --name-prompt (naming fix)"
```

---

### Task 2: The naming sample (`naming-sample` stage)

**Files:**
- Modify: `scripts/prepare_habitat_instr.py` (constants, `naming_sample_rows`, `stage_naming_sample`, CLI)
- Test: `tests/test_naming_fix.py`

**Interfaces:**
- Produces: `NAMING_AB = V4_OUT / "naming_ab"`; `NAMING_STRATA = (("wall", 45), ("object", 45), ("opening", 10))`;
  `naming_sample_rows(objects, strata, seed, flagged_rows) -> list[dict]` (each row the object row plus `"group"`: `"random"` / `"flagged"`); files `<out>/objects_sample.jsonl`, `<out>/names_V0.jsonl` (rows `{"id", "name", "passed", "namer"}`), `<out>/flagged_ids.txt`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_naming_fix.py`:

```python
import importlib.util  # noqa: E402
import sys  # noqa: E402
from argparse import Namespace  # noqa: E402

_SPEC = importlib.util.spec_from_file_location("prepare_habitat_instr", REPO / "scripts" / "prepare_habitat_instr.py")
prep = importlib.util.module_from_spec(_SPEC)
sys.modules["prepare_habitat_instr"] = prep
_SPEC.loader.exec_module(prep)


def _obj(i, kind):
    return {"object_id": f"s{i}:b1", "kind_hint": kind, "image": f"/x/{i}.png", "polygon": [[0, 0], [1, 0], [1, 1]],
            "bbox": [0, 0, 1, 1], "crop_box": [0, 0, 1, 1], "area": 0.5, "depth_fallback": False}


def test_naming_sample_is_stratified_reproducible_and_adds_flagged():
    objs = [_obj(i, "wall") for i in range(60)] + [_obj(100 + i, "object") for i in range(60)] + \
           [_obj(200 + i, "opening") for i in range(5)]
    flagged = [dict(_obj(999, "object"))]
    rows = prep.naming_sample_rows(objs, prep.NAMING_STRATA, 0, flagged)
    rnd = [r for r in rows if r["group"] == "random"]
    assert [sum(r["kind_hint"] == k for r in rnd) for k in ("wall", "object", "opening")] == [45, 45, 5]  # 5 openings exist
    assert [r["object_id"] for r in rows if r["group"] == "flagged"] == ["s999:b1"]
    assert rows == prep.naming_sample_rows(objs, prep.NAMING_STRATA, 0, flagged)
    assert rows != prep.naming_sample_rows(objs, prep.NAMING_STRATA, 1, flagged)


def _write(p, rows):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))


def _stage_args(tmp_path, flagged_ids):
    ev, lab = tmp_path / "v4", tmp_path / "labels"
    _write(ev / "objects_train.jsonl", [_obj(i, "wall") for i in range(50)] + [_obj(100 + i, "object") for i in range(50)]
           + [_obj(200 + i, "opening") for i in range(12)])
    _write(ev / "objects_eval.jsonl", [_obj(500, "object")])
    _write(ev / "steer" / "objects_eval.jsonl", [_obj(600, "wall")])
    _write(lab / "names_train_gemma_d2.jsonl", [{"id": "s0:b1", "name": "white wall", "passed": True},
                                                 {"id": "s100:b1", "name": None, "passed": False}])
    _write(lab / "names_train_d2.jsonl", [{"id": "s100:b1", "name": "grey sofa", "passed": True}])
    _write(lab / "names_eval_gemma_d2.jsonl", [{"id": "s500:b1", "name": "shelf", "passed": True}])
    _write(lab / "names_gate_gemma_d2.jsonl", [{"id": "s500:b1", "name": "bookshelf", "passed": True}])  # gate wins
    _write(lab / "names_eval_d2.jsonl", [])                     # names_gate_d2.jsonl missing: tolerated
    _write(lab / "names_steer_gemma_d2.jsonl", [{"id": "s600:b1", "name": "glass door", "passed": True}])
    fl = tmp_path / "flagged.txt"
    fl.write_text("\n".join(flagged_ids) + "\n")
    return Namespace(events_dir=ev, labels=lab, out_dir=tmp_path / "ab", n=100, seed=0, flagged=fl)


def test_naming_sample_stage_writes_sample_names_v0_and_flagged(tmp_path):
    args = _stage_args(tmp_path, ["s500:b1", "s600:b1", "s100:b1"])
    prep.stage_naming_sample(args)
    rows = [json.loads(x) for x in (args.out_dir / "objects_sample.jsonl").read_text().splitlines()]
    v0 = {r["id"]: r for r in map(json.loads, (args.out_dir / "names_V0.jsonl").read_text().splitlines())}
    rnd = [r["object_id"] for r in rows if r["group"] == "random"]
    assert len(rnd) == 100 and "s100:b1" not in rnd                 # a flagged object is never also a random one
    assert [r["object_id"] for r in rows if r["group"] == "flagged"] == ["s500:b1", "s600:b1", "s100:b1"]
    assert v0["s500:b1"]["name"] == "bookshelf" and v0["s600:b1"]["name"] == "glass door"
    assert v0["s100:b1"] == {"id": "s100:b1", "name": "grey sofa", "passed": True, "namer": "cosmos"}
    assert (args.out_dir / "flagged_ids.txt").read_text().split() == ["s500:b1", "s600:b1", "s100:b1"]


def test_naming_sample_stops_on_unknown_flagged_ids(tmp_path):            # Review Focus 5
    with pytest.raises(SystemExit, match="nope:b9"):
        prep.stage_naming_sample(_stage_args(tmp_path, ["s500:b1", "nope:b9"]))
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_naming_fix.py -q -k naming_sample`
Expected: FAIL (`module 'prepare_habitat_instr' has no attribute 'naming_sample_rows'`).

- [ ] **Step 3: Implement**

In `scripts/prepare_habitat_instr.py`, after `stage_d2_seed`, add:

```python
NAMING_AB = V4_OUT / "naming_ab"
NAMING_STRATA = (("wall", 45), ("object", 45), ("opening", 10))   # naming fix (spec 2026-10-15): 100 random objects


def naming_sample_rows(objects, strata, seed, flagged_rows):
    """The naming comparison sample: per (kind_hint, n) stratum, n random objects (all of them if fewer), seeded,
    never a flagged one (known failures would bias the random accuracy); then the flagged rows. Each row is the
    object row plus "group": "random" | "flagged"."""
    import random
    rng = random.Random(seed)
    flagged = {o["object_id"] for o in flagged_rows}
    out = []
    for kind, n in strata:
        pool = sorted((o for o in objects if o["kind_hint"] == kind and o["object_id"] not in flagged),
                      key=lambda o: o["object_id"])
        out += [{**o, "group": "random"} for o in rng.sample(pool, min(n, len(pool)))]
    return out + [{**o, "group": "flagged"} for o in flagged_rows]


def _rows_if(path):
    return _jsonl(path) if Path(path).exists() else []


def _passed_names(*paths):
    """{object id: name} over the passed rows of the files that exist (later files win)."""
    return {r["id"]: r["name"] for p in paths for r in _rows_if(p) if r.get("passed")}


def stage_naming_sample(args):
    """Naming fix step 1 (CPU): objects_sample.jsonl (random train objects + flagged ids), names_V0.jsonl (the name the
    routes use today: Gemma, else Cosmos; eval objects: the gate's names over the eval files', as the G3 texts used
    them; steering objects: steer Gemma), flagged_ids.txt. objects_train.jsonl holds exactly the objects of described
    train routes (29,556 = the route objects, checked 2026-10-15), so it is sampled as is."""
    train = _jsonl(args.events_dir / "objects_train.jsonl")
    lookup = {o["object_id"]: o for p in (args.events_dir / "objects_train.jsonl", args.events_dir / "objects_eval.jsonl",
                                          args.events_dir / "steer" / "objects_eval.jsonl") if p.exists() for o in _jsonl(p)}
    flagged_ids = [x.strip() for x in args.flagged.read_text().splitlines() if x.strip()] if args.flagged else []
    missing = [i for i in flagged_ids if i not in lookup]
    if missing:
        raise SystemExit(f"flagged ids in no objects file: {missing}")
    strata = tuple((k, round(n * args.n / 100)) for k, n in NAMING_STRATA)
    rows = naming_sample_rows(train, strata, args.seed, [lookup[i] for i in flagged_ids])
    L = args.labels
    primary = _passed_names(L / "names_train_gemma_d2.jsonl", L / "names_eval_gemma_d2.jsonl",
                            L / "names_gate_gemma_d2.jsonl", L / "names_steer_gemma_d2.jsonl")
    fallback = _passed_names(L / "names_train_d2.jsonl", L / "names_eval_d2.jsonl", L / "names_gate_d2.jsonl")
    v0 = []
    for r in rows:
        oid = r["object_id"]
        name, namer = (primary[oid], "gemma") if oid in primary else (fallback[oid], "cosmos") if oid in fallback else (None, None)
        v0.append({"id": oid, "name": name, "passed": name is not None, "namer": namer})
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    _write_jsonl(out / "objects_sample.jsonl", rows)
    _write_jsonl(out / "names_V0.jsonl", v0)
    (out / "flagged_ids.txt").write_text("".join(i + "\n" for i in flagged_ids))
    print(json.dumps({"random": sum(r["group"] == "random" for r in rows), "flagged": len(flagged_ids),
                      "by_kind": dict(Counter(r["kind_hint"] for r in rows if r["group"] == "random")),
                      "v0_named": sum(r["passed"] for r in v0)}, indent=2))
```

CLI, after the `d2-seed` parser:

```python
    ns = sub.add_parser("naming-sample", help="naming fix: sample objects + today's names for the naming audit (CPU)")
    ns.add_argument("--events-dir", type=Path, default=V4_OUT)
    ns.add_argument("--labels", type=Path, default=D1_LABELS)
    ns.add_argument("--out-dir", type=Path, default=NAMING_AB)
    ns.add_argument("--n", type=int, default=100)
    ns.add_argument("--seed", type=int, default=0)
    ns.add_argument("--flagged", type=Path, default=None, help="object ids the user marked misnamed, one per line")
```

and `"naming-sample": stage_naming_sample` in the dispatch dict.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_naming_fix.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/prepare_habitat_instr.py tests/test_naming_fix.py
git commit -m "feat: naming-sample stage (naming fix comparison sample + today's names)"
```

---

### Task 3: The variants job (`slurm/naming_variants.sbatch`)

**Files:**
- Create: `slurm/naming_variants.sbatch`
- Modify: `tests/test_slurm_thread_caps.py` (add the file to the parametrize list)
- Test: `tests/test_naming_fix.py`

**Interfaces:**
- Consumes: `label_descriptions.py --stage name --all-objects --objects F --out F --name-prompt n2` (Task 1); `naming_ab/objects_sample.jsonl` (Task 2).
- Produces: env `VARIANTS` (default `"V1 V2 V3"`), `SETS` (default `"sample"`; also `train eval steer`), `NAME_PROMPT` (default `n2`); outputs `naming_ab/names_V<k>.jsonl` (sample) and `outputs/instr_labels_v3/names_{train,eval,steer}_V<k>_n2.jsonl`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_naming_fix.py`:

```python
def test_naming_variants_sbatch():
    text = (REPO / "slurm" / "naming_variants.sbatch").read_text()
    assert "#SBATCH --gres=gpu:a100:1" in text and "#SBATCH --partition=A100" in text
    for model in ("google/gemma-3-12b-it", "Qwen/Qwen3.6-35B-A3B-FP8", "nvidia/Cosmos-Reason2-32B"):
        assert model in text
    assert 'VARIANTS="${VARIANTS:-V1 V2 V3}"' in text and 'SETS="${SETS:-sample}"' in text
    assert 'NAME_PROMPT="${NAME_PROMPT:-n2}"' in text and '--name-prompt "$NAME_PROMPT"' in text
    assert "( cosmos3_wait_healthy )" in text            # a model that fails to start is skipped (Review Focus 1)
    assert "names_${v}.jsonl" in text and 'names_${s}_${v}_${NAME_PROMPT}.jsonl' in text
```

and in `tests/test_slurm_thread_caps.py` add `"naming_variants.sbatch"` to the `test_thread_caps` parametrize list.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_naming_fix.py tests/test_slurm_thread_caps.py -q`
Expected: FAIL (file not found).

- [ ] **Step 3: Implement**

Create `slurm/naming_variants.sbatch`:

```bash
#!/bin/bash
#SBATCH --job-name=naming-variants
#SBATCH --partition=A100
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=outputs/slurm/%x-%j.out
# Naming fix (spec docs/superpowers/specs/2026-10-15-object-naming-design.md): name a set of objects with the naming
# variants, one model at a time on the full A100 (V2 is 35 GB FP8, V3 63 GB bf16 -- neither fits a 3g.40gb slice).
#
#     sbatch slurm/naming_variants.sbatch                                   # comparison: V1 V2 V3 on naming_ab sample
#     VARIANTS=V2 SETS="train eval steer" sbatch slurm/naming_variants.sbatch  # rollout with the winner (V2 here)
#
# A model that fails to start is logged and skipped; the others still run. Every call passes --resume, so a
# re-submit continues where a TIMEOUT stopped it.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
mkdir -p outputs/slurm outputs/instr_labels_v3
source slurm/cosmos3_serve_lib.sh
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
unset VIRTUAL_ENV
UV="${UV:-$HOME/.local/bin/uv}"
D="${DATA:-data/prepared_habitat_instr_v4}"
OUT=outputs/instr_labels_v3
VARIANTS="${VARIANTS:-V1 V2 V3}"
SETS="${SETS:-sample}"
NAME_PROMPT="${NAME_PROMPT:-n2}"
CONCURRENCY="${CONCURRENCY:-8}"

model_of() {
    case "$1" in
        V1) echo google/gemma-3-12b-it ;;
        V2) echo Qwen/Qwen3.6-35B-A3B-FP8 ;;
        V3) echo nvidia/Cosmos-Reason2-32B ;;
        *) echo "unknown variant $1" >&2; return 1 ;;
    esac
}
objects_of() {   # objects_of <set>
    case "$1" in
        sample) echo "$D/naming_ab/objects_sample.jsonl" ;;
        train) echo "$D/objects_train.jsonl" ;;
        eval) echo "$D/objects_eval.jsonl" ;;
        steer) echo "$D/steer/objects_eval.jsonl" ;;
        *) echo "unknown set $1" >&2; return 1 ;;
    esac
}
out_of() {   # out_of <set> <variant>
    local s=$1 v=$2
    if [ "$s" = sample ]; then echo "$D/naming_ab/names_${v}.jsonl"; else echo "$OUT/names_${s}_${v}_${NAME_PROMPT}.jsonl"; fi
}

trap 'cosmos3_stop_server' EXIT
echo "VARIANTS=$VARIANTS SETS=$SETS NAME_PROMPT=$NAME_PROMPT"
for v in $VARIANTS; do
    MODEL="$(model_of "$v")"
    echo; echo "=============== $v: $MODEL ==============="
    cosmos3_stop_server
    cosmos3_start_server
    if ! ( cosmos3_wait_healthy ); then
        echo "SKIP $v: $MODEL did not start"
        cosmos3_stop_server
        continue
    fi
    for s in $SETS; do
        objs="$(objects_of "$s")"
        [ -f "$objs" ] || { echo "missing $objs"; exit 1; }
        args=(--stage name --all-objects --objects "$objs" --out "$(out_of "$s" "$v")" --name-prompt "$NAME_PROMPT"
              --endpoint "$BASE/v1" --concurrency "$CONCURRENCY")
        echo; echo "--- $v $s (pass 1) ---"
        "$UV" run scripts/label_descriptions.py "${args[@]}" --resume
        echo; echo "--- $v $s (pass 2, retries errors) ---"
        "$UV" run scripts/label_descriptions.py "${args[@]}" --resume
    done
    cosmos3_stop_server
done
echo "naming variants done"
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_naming_fix.py tests/test_slurm_thread_caps.py -q && bash -n slurm/naming_variants.sbatch`
Expected: PASS, and `bash -n` prints nothing.

- [ ] **Step 5: Commit**

```bash
git add slurm/naming_variants.sbatch tests/test_naming_fix.py tests/test_slurm_thread_caps.py
git commit -m "feat: naming_variants.sbatch — V1/V2/V3 naming on the full A100 (naming fix)"
```

---

### Task 4: The names audit page and its scoring

**Files:**
- Modify: `scripts/audit_instruction_labels.py` (`VERDICTS["names"]`, `names_items`, `score_names`, `--mode names`, `--sample-dir`, `--score-marks`)
- Modify: `scripts/audit_instruction_labels.html` (names panel)
- Test: `tests/test_naming_fix.py`

**Interfaces:**
- Consumes: `naming_ab/objects_sample.jsonl`, `naming_ab/names_V*.jsonl` (Tasks 2–3); `descriptions.naming_image`.
- Produces: `names_items(sample_rows, variant_names, seed) -> list[dict]` (`{"id", "img", "names", "group", "kind_hint", "instruction", ...}`); `score_names(sample_rows, variant_names, marks) -> dict` (`{"random": {V: {"n", "correct", "accuracy", "by_kind": {...}}}, "flagged": {V: {"n", "correct"}}, "best": V, "best_accuracy": float, "gate": bool, "unmarked": int}`); `_read_marks(dir) -> {object id: mark}` (the spec's `"none": true` is the verdict `"none"` here: the template's verdict buttons give it a key and a tally for free); a mark is `{"oid": object id, "verdict": "done"|"none" (absent while only names are ticked), "right": [names], "note": str}`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_naming_fix.py`:

```python
_ASPEC = importlib.util.spec_from_file_location("audit_instruction_labels", REPO / "scripts" / "audit_instruction_labels.py")
audit = importlib.util.module_from_spec(_ASPEC)
_ASPEC.loader.exec_module(audit)

SAMPLE = [{"object_id": "a", "kind_hint": "wall", "group": "random"}, {"object_id": "b", "kind_hint": "object", "group": "random"},
          {"object_id": "c", "kind_hint": "object", "group": "random"}, {"object_id": "f", "kind_hint": "wall", "group": "flagged"}]
VNAMES = {"V0": {"a": "grey sofa", "b": "Coffee table", "c": "chair", "f": "bookshelf"},
          "V1": {"a": "white wall", "b": "coffee table", "c": "armchair", "f": "steel column"},
          "V2": {"a": "white wall", "b": None, "c": "armchair", "f": "bookshelf"}}


def test_names_items_list_distinct_names_shuffled(monkeypatch):
    monkeypatch.setattr(audit, "_naming_jpeg", lambda row: b"jpg")
    items = audit.names_items(SAMPLE, VNAMES, seed=0)
    by = {it["id"]: it for it in items}
    assert sorted(by["a"]["names"]) == ["grey sofa", "white wall"]
    assert sorted(n.lower() for n in by["b"]["names"]) == ["coffee table"]            # case-insensitive distinct
    assert by["f"]["group"] == "flagged" and by["a"]["img"].startswith("data:image/jpeg;base64,")


def test_score_names():
    marks = {"a": {"verdict": "done", "right": ["white wall"]}, "b": {"verdict": "done", "right": ["coffee table"]},
             "c": {"verdict": "done", "right": []},                                      # done, nothing ticked = none
             "f": {"verdict": "done", "right": ["steel column"]}}
    s = audit.score_names(SAMPLE, VNAMES, marks)
    assert s["random"]["V0"]["correct"] == 1 and s["random"]["V1"]["correct"] == 2
    assert s["random"]["V2"]["correct"] == 1                       # no name for "b" counts as wrong (Review Focus 1)
    assert s["random"]["V1"]["accuracy"] == pytest.approx(2 / 3)
    assert s["random"]["V1"]["by_kind"] == {"wall": {"n": 1, "correct": 1}, "object": {"n": 2, "correct": 1}}
    assert s["flagged"]["V1"] == {"n": 1, "correct": 1} and s["flagged"]["V0"] == {"n": 1, "correct": 0}
    assert (s["best"], s["gate"], s["unmarked"]) == ("V1", False, 0)


def test_score_names_none_and_unmarked():                          # Review Focus 4
    s = audit.score_names(SAMPLE, VNAMES, {"a": {"verdict": "none", "right": ["white wall"]}})
    assert s["random"]["V1"]["correct"] == 0 and s["unmarked"] == 3
    # names ticked but Done never pressed: the ticks count; a note-only mark is unmarked
    s = audit.score_names(SAMPLE, VNAMES, {"a": {"right": ["White Wall"]}, "b": {"note": "?"}})
    assert s["random"]["V1"]["correct"] == 1 and s["random"]["V1"]["n"] == 1 and s["unmarked"] == 3


def test_read_marks_takes_the_object_id_from_the_mark(tmp_path):
    d = tmp_path / "marks"
    d.mkdir()
    (d / "x.json").write_text(json.dumps({"oid": "00001-UVdNNRcVyV1_c000:b1", "verdict": "done", "right": ["sofa"]}))
    (d / "a.json").write_text(json.dumps({"verdict": "none"}))
    assert audit._read_marks(tmp_path) == {"00001-UVdNNRcVyV1_c000:b1": {"oid": "00001-UVdNNRcVyV1_c000:b1",
                                                                          "verdict": "done", "right": ["sofa"]},
                                           "a": {"verdict": "none"}}


def test_template_has_a_names_panel():
    html = audit.TEMPLATE.read_text()
    assert 'DATA.mode === "names"' in html and 'id="names"' in html and "right:" in html
    assert {v["v"] for v in audit.VERDICTS["names"]} == {"done", "none"}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_naming_fix.py -q -k "names_items or score_names or names_panel"`
Expected: FAIL (`module 'audit_instruction_labels' has no attribute 'names_items'`).

- [ ] **Step 3: Implement**

In `scripts/audit_instruction_labels.py`:

```python
from rover_vlm.descriptions import naming_image   # add to the existing descriptions import

VERDICTS["names"] = [
    {"v": "done", "label": "Done", "help": "Save the names you ticked as right and go to the next object"},
    {"v": "none", "label": "None is right", "help": "No candidate name fits the outlined thing (type the right one in the note)"},
]
NAMES_PX, NAMES_Q = 768, 75     # page budget: ~120 two-panel images under 16 MB


def _naming_jpeg(row):
    """The two-panel naming image the namer saw, re-encoded small for the page."""
    raw = naming_image(row["image"], row["polygon"], row["crop_box"] or row["bbox"])
    img = Image.open(io.BytesIO(raw)).convert("RGB").resize((NAMES_PX, NAMES_PX), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=NAMES_Q)
    return buf.getvalue()


def _key(name):
    return " ".join((name or "").lower().split())


def names_items(sample_rows, variant_names, seed=0):
    """One audit item per sampled object: the naming image and the distinct names (case-insensitive) the variants gave,
    shuffled per object with no variant labels."""
    rng = random.Random(seed)
    items = []
    for row in sample_rows:
        oid, names = row["object_id"], {}
        for v in sorted(variant_names):
            n = variant_names[v].get(oid)
            if n:
                names.setdefault(_key(n), n)
        shown = list(names.values())
        rng.shuffle(shown)
        items.append({"id": oid, "img": "data:image/jpeg;base64," + base64.b64encode(_naming_jpeg(row)).decode(),
                      "names": shown, "group": row["group"], "kind_hint": row["kind_hint"],
                      "instruction": "Which names fit the outlined thing?", "side": "", "kind": f"{row['kind_hint']} · {row['group']}",
                      "has_alt": False, "target": "object", "reason": None})
    return items


def score_names(sample_rows, variant_names, marks):
    """Per-variant accuracy from the names-audit marks (spec 2026-10-15): a variant is right on an object when its name
    (case-insensitive) is among the ticked names; "none", or "done" with nothing ticked, makes every variant wrong; a
    variant with no name is wrong. An object is marked once it has a verdict or a ticked name (a note alone is not a
    mark). Random objects give the accuracy (by kind_hint) and the gate; flagged ones a count."""
    out = {"random": {}, "flagged": {}}
    marked = [r for r in sample_rows if (marks.get(r["object_id"]) or {}).get("verdict")
              or (marks.get(r["object_id"]) or {}).get("right")]
    for v in sorted(variant_names):
        for group in ("random", "flagged"):
            rows = [r for r in marked if r["group"] == group]
            ok = []
            for r in rows:
                m = marks[r["object_id"]]
                right = set() if m.get("verdict") == "none" else {_key(x) for x in m.get("right") or []}
                ok.append(bool(variant_names[v].get(r["object_id"])) and _key(variant_names[v].get(r["object_id"])) in right)
            cell = {"n": len(rows), "correct": sum(ok)}
            if group == "random":
                by = {}
                for r, k in zip(rows, ok):
                    b = by.setdefault(r["kind_hint"], {"n": 0, "correct": 0})
                    b["n"] += 1
                    b["correct"] += int(k)
                cell.update(accuracy=cell["correct"] / cell["n"] if cell["n"] else 0.0, by_kind=by)
            out[group][v] = cell
    best = max(out["random"], key=lambda v: out["random"][v]["accuracy"])
    out.update(best=best, best_accuracy=out["random"][best]["accuracy"],
               gate=out["random"][best]["accuracy"] >= 0.95, unmarked=len(sample_rows) - len(marked))
    return out


def _variant_names(sample_dir):
    """{variant: {object_id: name or None}} from <sample_dir>/names_V*.jsonl (passed rows only)."""
    out = {}
    for p in sorted(Path(sample_dir).glob("names_V*.jsonl")):
        out[p.stem[len("names_"):]] = {r["id"]: (r.get("name") if r.get("passed") else None) for r in jsonl(p)}
    return out


def _read_marks(marks_dir):
    """Marks saved by the Artifact database read: <dir>/marks/<doc id>.json, each the document itself (as the G3 marks
    were saved on 2026-10-14). The page writes the object id into the mark ("oid"), so a doc id the save mangled
    (object ids hold ':') still maps back; older marks fall back to the file stem."""
    out = {}
    for p in sorted(Path(marks_dir).rglob("*.json")):
        d = json.loads(p.read_text())
        out[d.get("oid", p.stem)] = d
    return out
```

CLI: add `"names"` to `--mode` choices; `ap.add_argument("--sample-dir", type=Path, default=REPO_ROOT / "data" / "prepared_habitat_instr_v4" / "naming_ab")`; `ap.add_argument("--score-marks", type=Path, default=None, help="dir of names-audit mark files: print per-variant accuracy and the gate")`. At the top of the mode dispatch in `main()`:

```python
    if args.score_marks:
        rows = jsonl(args.sample_dir / "objects_sample.jsonl")
        print(json.dumps(score_names(rows, _variant_names(args.sample_dir), _read_marks(args.score_marks)), indent=2))
        return
    if args.mode == "names":
        rows = jsonl(args.sample_dir / "objects_sample.jsonl")
        items = names_items(rows, _variant_names(args.sample_dir), args.seed)
        payload = {"items": items, "population": {"random": sum(r["group"] == "random" for r in rows),
                                                  "flagged": sum(r["group"] == "flagged" for r in rows)},
                   "seed": args.seed, "mode": "names", "verdicts": VERDICTS["names"], "source": str(args.sample_dir)}
        html = TEMPLATE.read_text().replace("/*__DATA__*/null", json.dumps(payload))
        html = html.replace("<title>Instruction Label Audit</title>", "<title>Object Names Audit</title>", 1)
        out = args.out or REPO_ROOT / "outputs" / "names_audit.html"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(html)
        print(f"{len(items)} objects -> {out} ({out.stat().st_size / 1e6:.1f} MB)")
        return
```

In `scripts/audit_instruction_labels.html`:
1. Markup: just before `<div class="verdicts" id="verdicts"></div>` add `<div class="namepick" id="names" hidden></div>`; CSS:
   `.namepick { display: flex; flex-wrap: wrap; gap: 8px; } .namepick button { font: inherit; font-size: 16px; color: var(--ink); background: var(--surface); border: 1px solid var(--line); border-radius: 999px; padding: 6px 14px; cursor: pointer; } .namepick button[aria-pressed="true"] { background: var(--ok-soft); border-color: var(--ok); }`
2. `tone`: treat `"done"` as ok, and a mark with no verdict yet (names ticked, or a note only) as unsure instead of
   bad: `const tone = (v) => !v ? "unsure" : (v === "ok" || v === "none_ok" || v === "done") ? "ok" : v === "unsure" ? "unsure" : "bad";`
3. A mode block after the `steer` block (a tick saves at once — the user may move on with the arrow keys without
   pressing Done; every names mark carries the object id as `oid`, see `_read_marks`):
```js
let pick = new Set();
if (DATA.mode === "names") {
  document.title = "Object Names Audit";
  $("h1").textContent = "Object names audit";
  $("sub").textContent = "Each object is shown as the namer saw it: the frame with the region outlined, and a close-up. Tick every name that fits the outlined thing (letters a, b, c… toggle), then Done (1); if none fits, None is right (2) and type the right name in the note.";
  document.querySelector(".legend").hidden = true;
  $("names").hidden = false;
  document.querySelector(".stripcap span").textContent = "All objects: click to jump. Keys: a, b, c… tick a name, 1 Done, 2 None is right, ← → move.";
}
function renderNames(it, m) {
  pick = new Set(m.right || []);
  const box = $("names"); box.innerHTML = "";
  (it.names || []).forEach((n, i) => {
    const b = document.createElement("button");
    b.textContent = `${String.fromCharCode(97 + i)}  ${n}`; b.dataset.name = n;
    b.setAttribute("aria-pressed", String(pick.has(n)));
    b.onclick = () => {
      pick.has(n) ? pick.delete(n) : pick.add(n);
      b.setAttribute("aria-pressed", String(pick.has(n)));
      marks[it.id] = { ...(marks[it.id] || {}), oid: it.id, right: [...pick] };
      persist(it.id); tally(); strip();
    };
    box.appendChild(b);
  });
}
```
4. In `render()`, before the verdict-button line: `if (DATA.mode === "names") { $("instr").textContent = "Which of these names fit the outlined thing?"; $("side").textContent = ""; $("kind").textContent = it.kind; renderNames(it, m); }`
5. In `mark(v)`: `marks[id] = { ...(marks[id] || {}), verdict: v, ...(DATA.mode === "names" ? { oid: id, right: v === "done" ? [...pick] : [] } : {}) };`
6. In the keydown handler, before the number lookup: `if (DATA.mode === "names" && /^[a-z]$/.test(e.key)) { const b = $("names").children[e.key.charCodeAt(0) - 97]; if (b) { e.preventDefault(); b.click(); } return; }`
7. In the `onSnapshot` loop keep `right` and `oid`: `const { verdict, note, right, oid } = d.data(); marks[d.id] = { verdict, note, right, oid };`

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_naming_fix.py tests/test_d2_stages.py -q`
Expected: PASS (the existing audit-template tests still pass).

- [ ] **Step 5: Commit**

```bash
git add scripts/audit_instruction_labels.py scripts/audit_instruction_labels.html tests/test_naming_fix.py
git commit -m "feat: names audit page (blind distinct names per object) and per-variant scoring (naming fix)"
```

---

### Task 5: Rename the d2 objects and substitute names into the texts (`rename-texts`)

**Files:**
- Modify: `scripts/prepare_habitat_instr.py` (`substitute_names`, `rename_desc_rows`, `stage_rename_texts`, CLI)
- Test: `tests/test_naming_fix.py`

**Interfaces:**
- Consumes: winner name files `outputs/instr_labels_v3/names_{train,eval,steer}_<V>_n2.jsonl` (Task 3 rollout); d2 name and desc files, and the gate's name files `names_gate{_gemma,}_d2.jsonl` (read only: `desc_gate_mix_d2.jsonl` was written with them); `descriptions.choose_names`, `description_rejects`, `d2_rejects`, `route_objects`; `_rows_if`, `_passed_names` (Task 2).
- Produces: `substitute_names(text, old, new) -> str`; `rename_desc_rows(desc_rows, events_by_id, old_names, new_primary, new_fallback) -> (kept_rows, Counter of drop reasons)`; stage `rename-texts --variant V<k> [--force]` which (a) stops before touching anything if a winner file is missing, (b) backs up every file it rewrites to `<stem>.pre_n2.jsonl`, (c) rewrites `names_{split}_gemma_d2.jsonl` (primary := the winner's rows) and `names_{split}_d2.jsonl` (fallback := today's names, one per object: Gemma over Cosmos; for eval also the gate's and the steering names), and `names_steer_gemma_d2.jsonl` (:= the winner's steering rows), (d) rewrites `desc_{train,eval}_mix_d2.jsonl` and `desc_gate_mix_d2.jsonl` with renamed texts (dropped rows are composed by the full job's `--resume`), (e) writes `<events-dir>/rename_meta.json` (per desc file: rows, kept, renamed, kept_on_old_names, dropped_to_compose).

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_naming_fix.py`:

```python
PASS_A = {"type": "pass", "side": "left", "objects": ["f:b1"], "arc_m": 1.0}
BETWEEN = {"type": "between", "side": None, "objects": ["f:b1", "f:b2"], "arc_m": 1.0}
NEAR2 = {"type": "near", "relation": "in front of", "amount": None, "distance_m": None, "object": "f:b2"}


def _ev(rid, events, stop=NEAR2):
    return {"id": rid, "reason": None, "events": events, "stop": stop}


def test_substitute_names_swaps_and_overlaps():                    # Review Focus 2
    text = "Go between the sofa and the dining table, then stop in front of the bed."
    old = {"f:b1": "sofa", "f:b2": "dining table", "f:b3": "bed"}
    new = {"f:b1": "bed", "f:b2": "desk", "f:b3": "table"}
    assert prep.substitute_names(text, old, new) == "Go between the bed and the desk, then stop in front of the table."
    assert prep.substitute_names("Keep the Sofa on your left.", {"f:b1": "sofa"}, {"f:b1": "grey wall"}) == \
        "Keep the Grey wall on your left."                          # a capitalised mention stays capitalised
    assert prep.substitute_names("Pass the sofabed.", {"f:b1": "sofa"}, {"f:b1": "bed"}) == "Pass the sofabed."
    assert prep.substitute_names("Pass the table.", {"f:b1": "table", "f:b2": None}, {"f:b1": "desk"}) == "Pass the desk."


def test_rename_desc_rows_keeps_passing_texts_and_drops_clashes():
    ev = {"a": _ev("a", [PASS_A]), "b": _ev("b", [BETWEEN], stop={**NEAR2, "object": None, "type": "after_last",
                                                                  "relation": "just after"}), "c": _ev("c", [PASS_A])}
    desc = [{"id": "a", "description": "Go left of the sofa, then stop in front of the fridge.", "passed": True,
             "namer": "primary", "instruction_source": "cosmos"},
            {"id": "b", "description": "Go between the sofa and the fridge, then stop just after it.", "passed": True,
             "instruction_source": "cosmos"},
            {"id": "c", "description": None, "passed": False, "rejects": ["unnamed_object"]},     # recomposed
            {"id": "gone", "description": "Go left of the sofa.", "passed": True}]               # route not described now
    old = {"primary": {"f:b1": "sofa", "f:b2": "fridge"}, "fallback": {}}
    new_primary = {"f:b1": "white wall", "f:b2": "refrigerator"}
    kept, dropped = prep.rename_desc_rows(desc, ev, old, new_primary, {})
    assert [r["id"] for r in kept] == ["a", "b"] and dropped == {"not_passed": 1, "no_route": 1}
    assert kept[0]["description"] == "Go left of the white wall, then stop in front of the refrigerator."
    assert kept[0]["renamed"] == 2 and kept[0]["namer"] == "primary"
    clash = {"f:b1": "white cabinet", "f:b2": "tall cabinet"}           # two objects, one head noun
    kept, dropped = prep.rename_desc_rows(desc[:2], ev, old, clash, {})
    assert kept == [] and dropped == {"name:clash": 2}
    kept, _ = prep.rename_desc_rows(desc[:2], ev, old, clash, {"f:b1": "sofa", "f:b2": "fridge"})  # old names serve
    assert [(r["description"], r["namer"], r["renamed"]) for r in kept] == [(desc[0]["description"], "fallback", 0),
                                                                            (desc[1]["description"], "fallback", 0)]


def test_rename_texts_refuses_a_second_run(tmp_path):              # Review Focus 3
    lab, v4 = tmp_path / "labels", tmp_path / "v4"
    for s in ("train", "eval"):
        _write(lab / f"names_{s}_gemma_d2.jsonl", [{"id": "f:b1", "name": "sofa", "passed": True}])
        _write(lab / f"names_{s}_d2.jsonl", [])
        _write(lab / f"names_{s}_V2_n2.jsonl", [{"id": "f:b1", "name": "white wall", "passed": True}])
        _write(lab / f"desc_{s}_mix_d2.jsonl", [{"id": f"{s}a", "description": "Go left of the sofa, then stop in front of it.",
                                                  "passed": True}])
        _write(v4 / f"events_{s}.jsonl", [_ev(f"{s}a", [PASS_A], stop={**NEAR2, "object": "f:b1"})])
    _write(lab / "names_steer_gemma_d2.jsonl", [])
    _write(lab / "desc_gate_mix_d2.jsonl", [])
    args = Namespace(labels=lab, events_dir=v4, variant="V2", force=False)
    with pytest.raises(SystemExit, match="names_steer_V2_n2"):        # the rollout job has not run: nothing touched
        prep.stage_rename_texts(args)
    assert not (lab / "names_train_gemma_d2.pre_n2.jsonl").exists()
    _write(lab / "names_steer_V2_n2.jsonl", [])
    prep.stage_rename_texts(args)
    got = [json.loads(x) for x in (lab / "desc_train_mix_d2.jsonl").read_text().splitlines()]
    assert got[0]["description"] == "Go left of the white wall, then stop in front of it."
    assert (lab / "names_train_gemma_d2.pre_n2.jsonl").exists()
    fb = {r["id"]: r for r in map(json.loads, (lab / "names_train_d2.jsonl").read_text().splitlines())}
    assert fb["f:b1"]["name"] == "sofa" and fb["f:b1"]["passed"]                 # today's name is the fallback
    with pytest.raises(SystemExit, match="already"):
        prep.stage_rename_texts(args)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_naming_fix.py -q -k "substitute or rename"`
Expected: FAIL (`module 'prepare_habitat_instr' has no attribute 'substitute_names'`).

- [ ] **Step 3: Implement**

In `scripts/prepare_habitat_instr.py` add (and `import re` at the top; add `choose_names, description_rejects` to the
`rover_vlm.descriptions` import — `d2_rejects` and `route_objects` are already imported; `_rows_if` and
`_passed_names` come from Task 2):

```python
def substitute_names(text, old, new):
    """Each object's old name replaced by its new one (case-insensitive, whole words; a capitalised mention stays
    capitalised). Longest old names first and through placeholders, so a name inside a longer one is never edited and
    swapped names never double-replace. Objects with no old or no new name are left alone."""
    pairs = [(old[o], new[o]) for o in old if old.get(o) and new.get(o) and old[o].lower() != new[o].lower()]
    pairs.sort(key=lambda t: -len(t[0]))
    for i, (a, _) in enumerate(pairs):
        text = re.sub(r"(?<![\w-])" + re.escape(a) + r"(?![\w-])",
                      lambda m, i=i: f"\x00{i}{'C' if m.group(0)[0].isupper() else 'l'}\x00", text, flags=re.I)
    for i, (_, b) in enumerate(pairs):
        text = text.replace(f"\x00{i}l\x00", b).replace(f"\x00{i}C\x00", b[:1].upper() + b[1:])
    return text


def rename_desc_rows(desc_rows, events_by_id, old_names, new_primary, new_fallback):
    """(kept rows, drop reasons): every passed desc row's text with its objects' names replaced by the names the A5
    rule (choose_names) picks under the new name dicts; a row whose route is no longer described, a row that had not
    passed, a route the new names cannot serve (unnamed / clash) or a new text that breaks a rule is dropped -- the
    full labelling job (--resume) composes it. old_names = {"primary": {...}, "fallback": {...}}: the names the texts
    were written with (a row's "namer" says which). Dry run 2026-10-15 with every object renamed to a fresh name:
    17,040 of 17,056 passed train texts kept."""
    kept, dropped = [], Counter()
    for d in desc_rows:
        ev = events_by_id.get(d["id"])
        if ev is None or ev.get("reason") is not None:
            dropped["no_route"] += 1
            continue
        if not d.get("passed") or not d.get("description"):
            dropped["not_passed"] += 1
            continue
        row = {"events": ev["events"], "stop": ev["stop"]}
        chosen, namer, reject = choose_names(row, new_primary, new_fallback)
        if chosen is None:
            dropped[reject] += 1
            continue
        before = old_names["fallback"] if d.get("namer") == "fallback" else old_names["primary"]
        ids = list(dict.fromkeys(route_objects(row)))
        old = {o: before.get(o) for o in ids}
        new_text = substitute_names(d["description"], old, {o: chosen[o] for o in ids})
        if description_rejects(new_text, ev["events"], ev["stop"], chosen) or d2_rejects(new_text, ev):
            dropped["rejects"] += 1
            continue
        changed = sum(1 for o in ids if old.get(o) and old[o].lower() != chosen[o].lower())
        kept.append({**d, "description": new_text, "namer": namer, "renamed": changed})
    return kept, dropped


def _passed_rows(*paths):
    """{object id: name row} over the passed rows of the files that exist (later files win)."""
    return {r["id"]: r for p in paths for r in _rows_if(p) if r.get("passed")}


def stage_rename_texts(args):
    """Naming fix rollout (CPU). The label job and the steering stages read fixed file names, so the winner takes the
    "gemma" (primary) files and today's names become the fallback files:
      names_<split>_gemma_d2.jsonl := the winner's rows (names_<split>_<V>_n2.jsonl), tagged "namer_variant": V;
      names_<split>_d2.jsonl       := today's passed names, one per object -- Cosmos, overridden by Gemma (eval: also
                                      the gate's name files and the steering Gemma names), tagged "namer_variant": "old";
      names_steer_gemma_d2.jsonl   := the winner's steering rows;
      desc_{train,eval}_mix_d2.jsonl and desc_gate_mix_d2.jsonl := their rows renamed by rename_desc_rows (each file
                                      read with the names it was written with: the gate file with names_gate_*).
    Every rewritten file is first copied to <stem>.pre_n2.jsonl; a second run refuses (the copies would be lost) unless
    --force, which redoes it from the copies. After this stage never re-run d2-seed (it would overwrite these files)."""
    L, v = args.labels, args.variant
    winners = {s: L / f"names_{s}_{v}_n2.jsonl" for s in (*SPLITS, "steer")}
    missing = [str(p) for p in winners.values() if not p.exists()]
    if missing:
        raise SystemExit(f"winner names missing: {missing} — run VARIANTS={v} SETS=\"train eval steer\" "
                         "sbatch slurm/naming_variants.sbatch first")
    files = [L / f"names_{s}_gemma_d2.jsonl" for s in SPLITS] + [L / f"names_{s}_d2.jsonl" for s in SPLITS] + \
            [L / "names_steer_gemma_d2.jsonl"] + [desc_path(L, s, "_mix", "d2") for s in SPLITS] + \
            [L / "desc_gate_mix_d2.jsonl"]
    pre = lambda p: p.with_name(p.stem + ".pre_n2.jsonl")          # noqa: E731
    if any(pre(p).exists() for p in files) and not args.force:
        raise SystemExit("rename-texts already ran (backups *.pre_n2.jsonl exist) — pass --force only to redo it from them")
    for p in files:
        if not pre(p).exists():
            pre(p).write_text(p.read_text() if p.exists() else "")
    meta = {}
    for s in SPLITS:
        old = {"primary": {i: r["name"] for i, r in _passed_rows(pre(L / f"names_{s}_gemma_d2.jsonl")).items()},
               "fallback": {i: r["name"] for i, r in _passed_rows(pre(L / f"names_{s}_d2.jsonl")).items()}}
        old_gate = {"primary": _passed_names(L / "names_gate_gemma_d2.jsonl"), "fallback": _passed_names(L / "names_gate_d2.jsonl")}
        extra_cosmos = [L / "names_gate_d2.jsonl"] if s == "eval" else []
        extra_gemma = [L / "names_gate_gemma_d2.jsonl", pre(L / "names_steer_gemma_d2.jsonl")] if s == "eval" else []
        fallback = _passed_rows(pre(L / f"names_{s}_d2.jsonl"), *extra_cosmos, pre(L / f"names_{s}_gemma_d2.jsonl"), *extra_gemma)
        winner = _jsonl(winners[s])
        _write_jsonl(L / f"names_{s}_gemma_d2.jsonl", [{**r, "namer_variant": v} for r in winner])
        _write_jsonl(L / f"names_{s}_d2.jsonl", [{**r, "namer_variant": "old"} for r in fallback.values()])
        new_p = {r["id"]: r["name"] for r in winner if r.get("passed")}
        new_f = {i: r["name"] for i, r in fallback.items()}
        events = {r["id"]: r for r in _jsonl(args.events_dir / f"events_{s}.jsonl")}
        meta[s] = {"winner_rows": len(winner), "winner_named": len(new_p), "fallback_names": len(new_f)}
        for dp, names_before in [(desc_path(L, s, "_mix", "d2"), old)] + \
                                ([(L / "desc_gate_mix_d2.jsonl", old_gate)] if s == "eval" else []):
            rows = _jsonl(pre(dp))
            kept, dropped = rename_desc_rows(rows, events, names_before, new_p, new_f)
            _write_jsonl(dp, kept)
            meta[s][dp.name] = {"rows": len(rows), "kept": len(kept), "renamed": sum(r["renamed"] > 0 for r in kept),
                                "kept_on_old_names": sum(r["namer"] == "fallback" for r in kept),
                                "dropped_to_compose": dict(dropped)}
    _write_jsonl(L / "names_steer_gemma_d2.jsonl", [{**r, "namer_variant": v} for r in _jsonl(winners["steer"])])
    (args.events_dir / "rename_meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))
```

`kept_on_old_names` counts the routes the winner's names cannot serve (mostly two wall-kind objects both named
"… wall": a head-noun clash; 1,536 of 21,007 train routes have two or more wall-kind objects), which keep today's names.

CLI:

```python
    rt = sub.add_parser("rename-texts", help="naming fix: winner names into the d2 name files + texts (CPU)")
    rt.add_argument("--labels", type=Path, default=D1_LABELS)
    rt.add_argument("--events-dir", type=Path, default=V4_OUT)
    rt.add_argument("--variant", required=True, help="the winning variant, e.g. V2 (names_<split>_<V>_n2.jsonl)")
    rt.add_argument("--force", action="store_true", help="redo from the *.pre_n2.jsonl backups")
```

and `"rename-texts": stage_rename_texts` in the dispatch dict. Also add to the module docstring:

```
Naming fix (2026-10-15, docs/superpowers/specs/2026-10-15-object-naming-design.md): sample, compare naming variants on
an audit, then rename with the winner and substitute names into the texts. After rename-texts, never re-run d2-seed
(it would overwrite the renamed files).

    uv run scripts/prepare_habitat_instr.py naming-sample --flagged data/prepared_habitat_instr_v4/naming_ab/flagged_ids.txt
    sbatch slurm/naming_variants.sbatch                                     # GPU, ask first
    uv run scripts/audit_instruction_labels.py --mode names --out outputs/names_audit.html
    VARIANTS=V2 SETS="train eval steer" sbatch slurm/naming_variants.sbatch  # GPU, ask first (winner V2 here)
    uv run scripts/prepare_habitat_instr.py rename-texts --variant V2
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_naming_fix.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/prepare_habitat_instr.py tests/test_naming_fix.py
git commit -m "feat: rename-texts — winner names into the d2 name files, names substituted into texts (naming fix)"
```

---

### Task 6: README run book, full suite

**Files:**
- Modify: `README.md` (index-only)

- [ ] **Step 1: Edit the run book (working tree)**

In `README.md`, at the end of "## Turn events and d2 (2026-10-13)" (just before "## Visualizations"), add:

```markdown
**Naming fix (2026-10-15, spec `docs/superpowers/specs/2026-10-15-object-naming-design.md`).** Object names cost
points in every label check (G3 12/100, steering Gate 1 5/50). Four naming variants are compared on 100 random objects
(+ the objects the user flagged) in a blind audit; the best (>= 95%) renames every d2 object, and the names are
substituted into the existing texts (failures are composed by the full job). Never re-run `d2-seed` after `rename-texts`.

    uv run scripts/prepare_habitat_instr.py naming-sample --flagged $D/naming_ab/flagged_ids.txt
    sbatch slurm/naming_variants.sbatch                                # GPU (full A100), ask first: V1 Gemma, V2 Qwen3.6-35B-A3B, V3 Cosmos-Reason2-32B
    uv run scripts/audit_instruction_labels.py --mode names --out outputs/names_audit.html   # publish with marks; gate: best >= 95%
    uv run scripts/audit_instruction_labels.py --score-marks <marks dir>                     # per-variant accuracy
    VARIANTS=<winner> SETS="train eval steer" sbatch slurm/naming_variants.sbatch            # GPU, ask first
    uv run scripts/prepare_habitat_instr.py rename-texts --variant <winner>
    uv run scripts/prepare_habitat_instr.py steer --frames data/prepared_habitat_instr_v3/eval_desc_entry.json && \
        uv run scripts/prepare_habitat_instr.py steer-select
    # then G3' (label audit, as above) and steering Gate 1' (>= 90% each) before STAGE=steer and STAGE=full
```

- [ ] **Step 2: Stage only this edit (index-only)**

```bash
S=/tmp/claude-1013/-nfs-home-bergia-Projects-SPACEITUP-spaceitup-qwen-fine-tuning/c4398192-246a-4944-9676-c66bb717e27d/scratchpad   # the session scratchpad
git show HEAD:README.md > $S/README.idx.md
# apply exactly the Step 1 edit to $S/README.idx.md (Edit tool on the copy), then:
BLOB=$(git hash-object -w $S/README.idx.md)
git update-index --cacheinfo 100644,$BLOB,README.md
git diff --cached --stat                                   # README.md only
git diff --cached README.md | grep -c "Prompt framings"    # must print 0
git diff README.md | grep "Prompt framings"                # the user's row is still unstaged
```

- [ ] **Step 3: Full suite**

Run: `uv run pytest -q`
Expected: all pass (1199 before this plan + the new tests).

- [ ] **Step 4: Commit**

```bash
git commit -m "docs: README naming fix run book"
```

---

### Task 7: Run — the comparison, up to the names audit (controller), then STOP

All on the login node with `export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2`; `D=data/prepared_habitat_instr_v4`.

- [ ] **Step 1: Build `flagged_ids.txt` from the user's notes** (scratchpad script; each note mapped to one object id; ambiguous notes are left out and listed in the report):

| audit | item | note | object |
|---|---|---|---|
| G3 | 00803-k1cupFYWXJ6_c000_y0_L | round table, not a sofa | the "ornate sofa" (wall event, also the ending object) |
| G3 | 00804-BHXhpBwSMLh_c015_y1_L | the sideboard is actually a wall | the ending object |
| G3 | 00823-7MXmsvcQjpJ_c000_L | object 2 is a cabinet | 2nd object in drawing order |
| G3 | 00829-QaLdnwvtxbs_c001_y1_L | a wall, not a dark cabinet | the ending object |
| G3 | 00839-zt1RVoi7PcG_c012_y0_L | a bed, not a sofa | the "gray sofa" (wall event = ending object) |
| G3 | 00857-L53DsRRk4Ch_c003_y0_R | a wall, not a refrigerator | the "white refrigerator" |
| G3 | 00861-GLAQ4DNUx5U_c013_y0_R | object 2 is a chair, not a bookcase | 2nd object |
| G3 | 00861-GLAQ4DNUx5U_c020_y0_R | object 1 chairs/dining table, object 2 a wall | 1st and 2nd objects |
| G3 | 00891-cvZr5TUy5C5_c010_y0_L | brown wall, not wooden partition | the ending object |
| G3 | 00891-cvZr5TUy5C5_c026_y0_R | object 2 is a wall, not a sofa | 2nd object |
| G3 | 00897-LEFTm3JecaC_c008_R | a wall (or go through the door) | the ending object |
| steer | steer_stop_00821-eF36g7L6Z9M_c032 | storage unit is a door | the "coat storage unit" |
| steer | steer_stop_00842-hkr2MGpHD6B_c002 | bookshelf is a steel column | the "built-in bookshelf" |
| steer | steer_stop_00861-GLAQ4DNUx5U_c013 | object 1 is a dining table | object 1 on the page (A's stop object) |
| steer | steer_stop_00885-dVW2D7TDctW_c002 | no long sofa (a table), no glass door (a pillar) | "long sofa" and "glass door" |
| steer | steer_turn_00894-HY1NcmCgn3n_c003 | object 2 is a cabin | object 2 on the page (the turn object) |
| steer | steer_turn_00873-bxsVRursffK_c004 (marked true) | the refrigerator is actually a wall | the "refrigerator" |

Write `$D/naming_ab/flagged_ids.txt` (one id per line).

- [ ] **Step 2:** `uv run scripts/prepare_habitat_instr.py naming-sample --flagged $D/naming_ab/flagged_ids.txt` — report the counts.
- [ ] **Step 3: Ask the user**, then `sbatch slurm/naming_variants.sbatch` (full A100, ~45 min; may queue). Wait for it (background `squeue` loop). Report per-variant accept rates from the log.
- [ ] **Step 4:** `uv run scripts/audit_instruction_labels.py --mode names --out outputs/names_audit.html`; check the size (< 16 MB) and view two items; publish as an Artifact with `capabilities: {db: {}}`; read the empty `marks` collection once; record the URL in memory. **STOP** for the user's marks.
- [ ] **Step 5 (after the marks):** save the marks with the Artifact database read (`out_dir` in the scratchpad), run `uv run scripts/audit_instruction_labels.py --score-marks <dir>`, and report per-variant accuracy (random by kind; flagged fixed). Best >= 0.95 -> ask the user to confirm the winner, then the rollout (Task 8). Below 0.95 -> STOP and report (the next lever per the spec: a verification pass).

### Task 8: Rollout with the winner (controller; after the user confirms), then STOP

- [ ] **Step 1: Ask the user**, then `VARIANTS=<winner> SETS="train eval steer" sbatch slurm/naming_variants.sbatch`; wait; report accept rates.
- [ ] **Step 2:** `uv run scripts/prepare_habitat_instr.py rename-texts --variant <winner>`; report `rename_meta.json` per file: renamed, kept_on_old_names, dropped to compose (and the extra compose hours at 14 routes/min). If `kept_on_old_names` is large (the head-noun clash of two "… wall" names on one route), say so to the user before going on — it is the known limit of the name-clash rule, not a bug.
- [ ] **Step 3:** with `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1`: `uv run scripts/prepare_habitat_instr.py steer --frames data/prepared_habitat_instr_v3/eval_desc_entry.json && uv run scripts/prepare_habitat_instr.py steer-select` (Gate 0 must still pass: >= the pair minimum and >= 35 turn pairs). If `steer_meta.json` reports `unnamed_objects_added_for_naming` > 0, those objects are in `steer/objects_eval.jsonl` but have no winner name: ask, run `VARIANTS=<winner> SETS=steer sbatch slurm/naming_variants.sbatch` (`--resume` names only them), copy the new rows into `names_steer_gemma_d2.jsonl` (append the rows whose id is not there yet), and re-run `steer` + `steer-select`.
- [ ] **Step 4:** G3′: `uv run scripts/audit_instruction_labels.py --mode d1 --round d2 --events $D/events_eval.jsonl --objects $D/objects_eval.jsonl --desc outputs/instr_labels_v3/desc_gate_mix_d2.jsonl outputs/instr_labels_v3/desc_eval_mix_d2.jsonl --out outputs/instr_d2_audit_n2.html`; steering Gate 1′: `uv run scripts/audit_instruction_labels.py --mode steer --n 50 --n-turn 20 --events $D/steer/events_steer_eval.jsonl --out outputs/steer_audit_d2_n2.html`; publish both with marks. **STOP** for the user's marks (>= 90% each), then — asking before each — `STAGE=steer` and `STAGE=full`.
