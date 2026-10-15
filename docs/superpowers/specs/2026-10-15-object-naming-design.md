# Object naming — measure naming variants, rename, substitute into texts — design

Date: 2026-10-15. Branch: `feat/turn-events` (d2 round, before its full labelling job). Builds on
`2026-09-30-route-descriptions-design.md` (names report-only in the label job; Addendum A5: Gemma 3 12B names, Cosmos3-Nano
fallback, compose falls back to Cosmos names) and `2026-10-13-turn-events-d2-design.md` (Addendum T2: endings).

## Why

The d2 label checks keep losing points to object names, not to geometry or wording:

- **G3 label audit** (100 eval routes, https://claude.ai/artifact/1Z3ZCbkSxF3uKkKjGVS3Uw): 85/100; 12 wrong names.
- **Steering Gate 1** (50 pairs, https://claude.ai/artifact/Sa8RH4UeoiBYvZTGGxtTYM): 42/50; 5 of the 8 misses are names.
- d1's own label audit with the same namer was 84/100 (11 wrong names); A5's mixed namer was estimated at 88.7%.

The user's notes sort the errors into three groups, about equally common:
- walls named as furniture ("it's a wall, not a refrigerator / dark cabinet", "the sideboard is actually a wall");
- furniture named as other furniture (a bed / round table called a sofa, a chair called a bookcase, a table called a
  long sofa, a statue called a cabin);
- structural parts named as furniture (a door called a storage unit, a steel column called a bookshelf, a pillar called
  a glass door).

Contributing facts: 52% of the objects d2 routes use are "wall-kind" (any blob >= 1.5 m, so also sofas, islands,
counters); the naming prompt for them offers "a wall, a partition, a door in a wall, or a long piece of furniture such as
a sofa or a counter", and only 27% of wall-kind "behind" objects get a wall-type head noun.

## Decisions (user, 2026-10-15)

1. Fix naming before the steering compose job and the full labelling job.
2. Scope: all names (walls, furniture confusions, structural parts), **measured first**.
3. Target: compare a few naming variants on the same audited objects and take the best one, **provided it reaches >= 95%
   correct names** on a random sample.
4. Approach A: one comparison sample, 3–4 variants, one blind audit page; then rename everything with the winner and
   substitute the new names into the existing texts.

## Success criteria

- **Naming audit**: the best variant names >= 95% of the 100 random sampled objects correctly (user's marks).
- **G3′** (the usual label audit, 100 renamed eval texts, before the full job): >= 90% correct.
- **Steering Gate 1′** (50 pairs, renamed): >= 90% true.

## 1. The comparison sample and the audit

**Sample** (CPU, seed 0; stage `prepare_habitat_instr.py naming-sample`): objects the described d2 train routes use
(`data/prepared_habitat_instr_v4/events_train.jsonl` + `objects_train.jsonl`):
- **100 random**: 45 wall-kind, 45 object, 10 opening (`kind_hint`), each a normal naming row (image, polygon, crop box —
  the same two-panel image the namer sees);
- **the flagged objects** (~20) from the user's G1e / G3 / steering notes, listed in `flagged_ids.txt`
  (built by hand from the notes: "object 2 is a cabinet" -> the route's second object in drawing order; a note about the
  ending object -> the stop object). Flagged objects may come from eval or steering object files; they are scored
  separately (known failures would bias the random accuracy).

Output dir: `data/prepared_habitat_instr_v4/naming_ab/` — `objects_sample.jsonl` (random + flagged, a `group` field
`random` / `flagged`), `flagged_ids.txt`, and `names_V0.jsonl` (the name each object's routes use today, by the A5 rule:
Gemma name when usable, else Cosmos).

**Audit page** (`audit_instruction_labels.py --mode names`; marks in the artifact db, collection `marks`, doc id = object
id): one item per object — the two-panel naming image, and the **distinct** names the variants produced as toggle buttons
(shuffled per object, no variant labels), a "none is right" toggle and a note box for the right name. A mark is
`{"right": [names], "none": bool, "note": str}`. About 120 items.

**Scoring** (`audit_instruction_labels.py --score-marks DIR --sample naming_ab/`): for each variant, the share of the 100
random objects whose variant name is in `right` (an object with "none" counts as wrong for every variant; a variant with no
name for an object counts as wrong), by `kind_hint`; for the flagged objects, per variant, how many are now right. Prints
the gate line (best random accuracy >= 0.95).

## 2. The variants

| | namer | prompt | GPU |
|---|---|---|---|
| V0 | today's names (Gemma 3 12B, Cosmos3-Nano fallback) | n1 | none (on disk) |
| V1 | `google/gemma-3-12b-it` | n2 | full A100 job |
| V2 | `Qwen/Qwen3.6-35B-A3B-FP8` (MoE, ~3B active) | n2 | full A100 job |
| V3 | `nvidia/Cosmos-Reason2-32B` (reasoning; `parse_trace` strips the thinking) | n2 | full A100 job |

All three are cached and served by the installed vLLM 0.21 (`Qwen3_5MoeForConditionalGeneration`,
`Qwen3VLForConditionalGeneration`, `Gemma3ForConditionalGeneration` are registered). V2 (35 GB FP8) and V3 (63 GB bf16)
need thor's full A100 (`--gres=gpu:a100:1`).

**Prompt n2** (same two-panel image, same JSON answer `{"name", "kind"}`, same name rejects):
- "Name the single thing that fills most of the outline, in 1 to 4 words, the way a person would say it. Look at what is
  inside the outline, not at what is next to it or behind it."
- "If the outline is mostly part of the building, say which part ('white wall', 'brick wall', 'stone column',
  'glass partition', 'open doorway', 'wooden door', 'stair railing'). Name furniture or an appliance only if the
  outline is clearly on it ('grey sofa', 'kitchen island', 'refrigerator')."
- The rules kept from n1: never left / right, never describe the picture, name the whole thing, not a small part on it.
- Kind: "wall" (part of the building: a wall, a column or pillar, a door, a partition, a railing), "object" (furniture,
  an appliance, any other object), "opening" (a doorway, an arch, a gap between walls).
- Per-need hint: opening -> "The rover drives through it."; wall -> "something long that the rover drives alongside or
  stops in front of" (the "such as a sofa or a counter" example is dropped); object -> "something the rover drives past".

No verification pass ("is this really a {name}?") in this round; it is the next lever if no variant reaches 95%.

## 3. Code and the comparison job

- `descriptions.naming_prompt(need, version="n1")`: `"n2"` adds the wording above; n1 is unchanged byte for byte.
  `label_object(ask, obj, seed, image=None, name_prompt="n1")` passes it; every name row records `"name_prompt"`.
- `scripts/label_descriptions.py --name-prompt n1|n2` (stage name; default n1).
- `scripts/prepare_habitat_instr.py naming-sample [--n 100] [--seed 0] [--flagged FILE]` -> section 1's files.
- `slurm/naming_variants.sbatch` (`--gres=gpu:a100:1`, ~45 min): for each of V1, V2, V3 — serve the model with the
  existing helpers (`cosmos3_start_server` / `cosmos3_stop_server`, `MODEL`, `MAX_MODEL_LEN` per model), run
  `label_descriptions.py --stage name --all-objects --objects naming_ab/objects_sample.jsonl --name-prompt n2
  --out naming_ab/names_V<k>.jsonl --resume`, stop the server. A model that fails to load is logged and skipped; the
  others still run.
- `scripts/audit_instruction_labels.py --mode names` (page) and `--score-marks` (scoring); the HTML template gains a
  `names` panel (toggle buttons per candidate name, "none", note).

## 4. Rollout once a variant wins

1. **Rename** every object d2 uses — train 29,556, eval 3,033 (objects of described routes) and the 1,064 steering
   objects — with the winner (n2) into `names_{split}_<tag>_d2.jsonl` and `names_steer_<tag>_d2.jsonl`. The winner is
   the primary namer; the old A5 names (Gemma, then Cosmos) are the fallback for objects whose new name is rejected —
   the existing `choose_names` primary / fallback rule. ~2 h for V1/V2, longer for V3.
2. **Substitute names into the existing texts** (CPU stage `prepare_habitat_instr.py rename-texts`): in every reused,
   rewritten and gate-composed desc row, each object whose name changed has its old name replaced by the new one
   (case-insensitive, whole words; 99.9% of name mentions are verbatim today — 23,343 of 23,355 in the train desc file).
   The new text must pass `description_rejects` + `d2_rejects` with the new names and the name-clash rule; a pass keeps
   the row (tagged `"renamed": k`), a failure moves the route to the compose bucket (`events_eval_changed.jsonl` for
   eval). The ~3.8k train routes still to compose get the new names in the full job directly.
3. **Steering**: `steer` with the new names (CPU), `steer-select` (templates regenerate).
4. **Checks**: G3′ label audit (100 renamed eval texts; the user's stop before the full job) and steering Gate 1′
   (50 pairs); then — asking before each — `STAGE=steer` compose and `STAGE=full`.

d1 files (`*_d1.jsonl`, v2, v3) stay read-only.

## Run order

| # | Step | Where | Gate |
|---|---|---|---|
| 1 | `naming-sample` | CPU | — |
| 2 | `sbatch slurm/naming_variants.sbatch` | GPU (full A100), ask first | — |
| 3 | `audit_instruction_labels.py --mode names` -> publish with marks | human | best variant >= 95% (random 100) |
| 4 | rename all objects with the winner | GPU, ask first | report accept rate |
| 5 | `rename-texts` (train, eval), steering `steer` + `steer-select` | CPU | counts |
| 6 | G3′ label audit, steering Gate 1′ | human | >= 90% each |
| 7 | `STAGE=steer`, `finalize-steer`; `STAGE=full` | GPU, ask first | as before |

## Testing (test-first, CPU)

`naming_prompt` n2 wording and n1 unchanged; `--name-prompt` reaches `label_object` and the row records it; the sample
is stratified 45/45/10, reproducible, and includes the flagged ids; the names page lists the distinct names per object;
scoring on hand-made marks (none, a variant with no name, several variants with the same name, flagged vs random);
`rename-texts` on hand-made rows (a verbatim rename, a rename that creates a name clash -> compose, a case difference, a
name inside another word not replaced); the sbatch's model list and gres (static check of the script text).

## Files touched

`src/rover_vlm/descriptions.py` (naming_prompt n2, label_object), `scripts/label_descriptions.py` (`--name-prompt`),
`scripts/prepare_habitat_instr.py` (`naming-sample`, `rename-texts`), `scripts/audit_instruction_labels.py` +
`.html` (`--mode names`, `--score-marks`), `slurm/naming_variants.sbatch` (new), README run book (index-only commit),
dated pointers in the two specs above.

## Out of scope

Which blob gets named (outline geometry, ghost "empty space" outlines, missing steps), the `no_behind_anchor` drop rule,
the off-height-floor scene 00831, any change to compose. If the audit shows names failing because the outline itself
covers the wrong thing, that is reported, not fixed here.
