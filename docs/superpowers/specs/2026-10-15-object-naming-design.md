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

## Addendum C (2026-10-19): choose among the candidate names

### Why

The names audit (https://claude.ai/artifact/NUcQg8M3DmTCEpUKmDZ629, 117 of 119 objects marked) failed the 95% gate:

| | random 100 (98 marked) | wall / object / opening | flagged fixed |
|---|---|---|---|
| V0 today | 45 (0.459) | 27/45, 17/45, 1/8 | 0/19 |
| V1 Gemma n2 | 55 (0.561) | 31/45, 23/45, 1/8 | 4/19 |
| V2 Qwen3.6 n2 | 62 (0.633) | 34/45, 27/45, 1/8 | 10/19 |
| V3 Cosmos-Reason2-32B n2 | 49 (0.500) | 24/45, 24/45, 1/8 | 9/19 |

- n2 fixed the targeted error (building part named as furniture: V0 18 → V2 2 of the 90 non-opening random objects); V2's
  remaining misses are other wrong objects (20) and blanks (6, thinking cut at `--max-tokens 4096`).
- The four variants disagree a lot (V1 = V2 = V3 on 15 of 119 objects), and a right name is among the candidates far more
  often than any one variant gives it: on the 90 non-opening random objects a right name is among all four candidates
  for 81 (90%); V0+V1+V2 77, V1+V2+V3 78, V1+V2 74.
- Openings fail for every variant (1/8): the user's notes call 7 of 8 "empty space" — outline geometry, out of scope.
- 2 marks were lost: object ids with ',' (`gap1@-8.0,-1.3`) are not valid artifact-db doc ids.

### Decisions (user, 2026-10-19)

1. A second model **chooses** among the candidate names, or says none fits (option 2 of the result report).
2. **Target: as close to 90% as possible** (the ceiling of choosing among the four), on the random sample.
3. **"None fits" → today's name (V0) stays** — the existing rename-texts fallback.
4. **GPU time is not a constraint**: all four candidates for every object.
5. **A fresh 50-object check** of the winning chooser's final names, by the user, before renaming everything.

### C1. Candidates for every d2 object

V1, V2 and V3 name every object d2 uses — train 29,556, eval 3,033, steering 1,064 — with prompt n2
(`VARIANTS="V1 V2 V3" SETS="train eval steer" sbatch slurm/naming_variants.sbatch`, outputs
`outputs/instr_labels_v3/names_{train,eval,steer}_V<k>_n2.jsonl`). Changes to the job: `--max-tokens 8192` (env
`MAX_TOKENS`, default 8192; V2 lost 7 of 119 to the 4096 cut) and a higher `CONCURRENCY` default (32; the builder
measures throughput on the first minutes of the run and may set it differently). V0 is today's name per object, by the
same precedence as `names_V0.jsonl` (gate names last), for every object in the three sets.

Estimated cost at the comparison job's settings (8 concurrent requests): V1 ~2.5 h, V2 ~30 h, V3 ~30 h (25 s per object
each); higher concurrency should cut V2/V3 severalfold. The job resumes after a TIMEOUT (`--resume`), so `--time` may stay
moderate and the job is re-submitted until complete; `rename-texts` already refuses incomplete candidate files.

### C2. The chooser

- **Input**: the same two-panel naming image the namers saw; the object's distinct candidate names (V0–V3, case-
  insensitive, only names that passed the name rejects), shuffled with a seed stable per object, lettered a, b, c, d;
  plus "none of these fits". No model labels on the names.
- **Prompt (c1)**: the n2 guidance (name the single thing that fills most of the outline; look inside it, not next to it
  or behind it; a building part when the outline is mostly part of the building; furniture or an appliance only if the
  outline is clearly on it), the per-need hint of n2, then: pick the one name that best names what fills the outline, or
  "none" when no name fits. Answer `{"choice": "<letter>|none"}` in the usual `<answer>` block.
- **Output row** per object: `{"id", "need", "name": the object's final name, "passed": a final name exists,
  "choice": letter|"none"|null, "name_source": the variants that proposed the pick, or "V0_fallback", "candidates": [the
  names shown, in shown order], "chooser": model id, "choose_prompt": "c1"}`; a row records `"single": true` when there
  was only one distinct candidate. The final name is the pick, or today's name (V0) when the chooser said none or failed.
- **One distinct candidate** (all variants agree; 12 of 119 objects): no call; that name is the pick. **No candidate**:
  no call; the final name is V0 (null if V0 has none either).
- **Parse failure**: one retry with seed + 1 (as `label_object`); two failures → `choice: null`, final name V0,
  counted and printed, never silent.
- **Fallback is per object (decision 3)**: "none" or a failure gives *that object* today's name, written into the
  chooser's own row (`name` = V0, `passed: true`, `name_source: "V0_fallback"`). It must not reach the per-route rule:
  `choose_names` (used by `rename-texts`, compose in `STAGE=full`, and steering) switches a whole route to the fallback
  names when any one object lacks a primary name, which would throw away the chooser's picks for the route's other
  objects (7,936 of 21,007 described train routes have two or more objects). The per-route switch still happens on a
  head-noun clash (route rules are out of scope); `rename-texts` reports it as `kept_on_old_names`. Only an object
  with neither a pick nor a V0 name has `passed: false`.

### C3. Choosing the chooser (automatic, no new marks)

One GPU job runs three choosers on the 119 audited objects with prompt c1: **C1 = Qwen3.6-35B-A3B-FP8, C2 =
Cosmos-Reason2-32B, C3 = Gemma 3 12B** (`--max-tokens 8192`), outputs `naming_ab/choose_C<k>.jsonl`.

Scoring against the existing marks (`audit_instruction_labels.py --score-choosers`):
- **Final-name accuracy** (the gate number): the name the object would end up with — the pick, or V0 when the chooser
  said none or failed — is among the user's ticked names; an object marked "none is right" (or Done with nothing
  ticked) counts as wrong for every chooser (no candidate was right; the fallback name is wrong too). Reported on the 98
  marked random objects, by kind, with the 90-object non-opening figure and the ceiling (81/90) beside it; flagged
  objects as a count.
- Also reported: how often each chooser said none, failed to parse, and picked each source variant.
- **Winner**: the highest final-name accuracy on the random objects (ties → the faster model). Prompt changes after
  seeing the scores are allowed at most once (c2), on the same three or fewer choosers, to limit tuning to these 98
  objects; the fresh check below is the guard.

### C4. The fresh 50-object check (user)

- **Sample**: 50 random train objects not among the 119, **population-weighted** (plain random over `objects_train.jsonl`,
  seed 1 — not the 45/45/10 strata), so the check estimates what training sees; openings appear at their ~1% share.
- **Needs**: the four candidates for those objects (from C1's train run), then the winning chooser on them.
- **Page**: the names audit page (`--mode names`) with one candidate per object — the final name (pick, or V0 on
  "none") — tick it if right, "none is right" otherwise, note for the right name.
- **Report**: accuracy with a 95% interval next to the C3 estimate. No hard gate: the user decides whether to go on.
- **Page fix first**: artifact-db doc ids may hold only letters, digits and `_ - . ~ : @ +`; the page stores each mark
  under a sanitized doc id (other characters → `_`) and keeps the real object id in the mark's `oid`, which
  `--score-marks` already prefers.

### C5. Rollout

Once the user accepts the check: the winning chooser runs on train, eval and steering (all objects); its rows become the
primary names via `rename-texts` (the chooser output written as, or mapped to, the winner-file shape `rename-texts`
reads: `{id, name, passed, need, ...}` with rows for every object, "none" rows already carrying V0 per C2), today's
names the fallback — the rest as in section 4: `steer` + `steer-select`, G3′ and steering Gate 1′ (≥ 90% each), then — asking
before each — `STAGE=steer` and `STAGE=full`.

### Goals / non-goals

- Goal: final names as close to 90% right as possible on the random sample, confirmed on 50 fresh objects.
- Non-goals: writing new names (the chooser only picks); fixing outlines (openings that are "empty space", outlines over
  several things) — reported, not fixed; changing compose or the route rules.
- Anti-goal: a score that only holds on the 98 tuned objects — hence one prompt revision at most, and the fresh check.

### Left to the builder

The exact c1 wording beyond the points above; where the choose stage lives (`label_descriptions.py --stage choose`
or its own script); how chooser rows map to the winner-file name `rename-texts` reads; the concurrency value; how the
50-object sample is built (a `naming-sample` option or a new stage) — all with tests first, as before.
