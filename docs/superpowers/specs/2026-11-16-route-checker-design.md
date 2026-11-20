# Route checker: drop hard frames from the d3 training texts

Status: user chose "full job now, checker later" and "train only, go ahead" (2026-11-16). Branch feat/stop-phrases.

## Why

The user's audit marks show that wrong texts keep landing on the same frames: across 16 audit pages (1,304
judgments), a frame marked wrong once is wrong again 43% of the time vs 14% when it was right (3.1x, within-page
shuffle p < .001), and 2.4x even when the second audit is a different task (p = .03). Only the frame matters
(scene level 1.3x). The user's notes name what makes them hard: pillars and columns that are not separate objects,
open floor they would call "middle of the room", big furniture or windows with no outline, wooden walls taken for
furniture, a few scenes ambiguous everywhere (00831, 00879, 00891).

Cheap per-route metadata does not find these frames (AUC 0.42-0.56 on 1,006 marked routes: object count, walls,
name disagreement among the C1 candidates, unnamed objects, object area, route length, depth fallback). Only the
stop kind carries signal, and dropping whole kinds would undo d3. What makes a frame hard is visible in the image,
so the check has to look at the image.

Goal: a cleaner d3 **training** set. Estimated d3 text accuracy is ~90% overall (reused d2 texts ~94%, rewritten
stops ~75%, composed ~77%); a perfect filter would remove ~10% of train routes, a realistic one ~16-23%.

## Goals, non-goals, anti-goals

- Goal: a per-route "probably wrong" score for every d3 train text, validated against the user's marks before it
  removes anything; a cutoff the user picks from a size-vs-accuracy curve.
- Goal: the same score on the eval texts, used only to report eval results on the checker-passed subset as an
  extra number.
- Non-goal: filtering eval (user: train only). d2 vs d3 eval comparisons stay on the full eval set.
- Non-goal: relabelling or fixing texts; dropped routes are just left out at finalize.
- Anti-goal: a checker that re-asks geometry. Round i1 (2026-09-25) showed Cosmos-Nano at chance on left/right of
  drawn routes (59-63%); sides come exactly from the metadata. The checker never judges sides.
- Anti-goal: a checker that is just the stop kind in disguise (drops "behind"/"next to" wholesale). It must add
  signal within stop kinds.
- Anti-goal: a score tuned on the marks it is reported on (one prompt revision at most, see Validation).

## Approaches

1. **One question, "is this instruction true for this route?"** Simplest, one call. But it asks the model to do the
   whole judgment, geometry included, which is what failed in i1, and a yes/no gives no hint of which part is wrong.
2. **Targeted questions about what the map cannot see** (chosen). Ask, per route, only what the user's notes say goes
   wrong: does each outlined object fit its name and is it a thing rather than part of a wall; does the route pass a
   big obstacle (pillar, column, furniture) the text leaves out; is the stop phrase a natural way to say where the
   route ends (or would a person say "middle of the room"); and one overall yes/no with a confidence. The answers are
   combined into one score on the marks. More output to parse, but each answer can be checked against the matching
   kind of user mark (name / stop / missing obstacle), and a weak question can be dropped.
3. **Two-model agreement** (Qwen3.6 + Cosmos-Reason2 or Gemma, flag where they disagree). Doubles the GPU time and
   still needs approach 1 or 2 underneath; kept as a follow-up if approach 2 is weak.

Choice: approach 2 (my call; the user handed over the how).

## Design

**Model and serving.** `Qwen/Qwen3.6-35B-A3B-FP8` (a vision-language model; C1 name chooser, 92% agreement with the
user on names) served with vLLM on the full A100, the way `slurm/naming_variants.sbatch` serves it
(`slurm/cosmos3_serve_lib.sh`). A new stage `--stage check` in `scripts/label_descriptions.py` reusing
`labelling_io.ask_image`; a new `slurm/check_routes.sbatch` for the job (the naming job is names-only).

**What the model sees.** Exactly what the user saw on the d2/d3 label audits: `events_item(..., dashed=True)` in
`scripts/audit_instruction_labels.py` (the frame, the route solid in view and dashed out of sight, a ring where it
leaves the view, numbered outlines, the stop object marked) plus the text. The prompt states that sides are already
correct and must not be judged.

**Questions (strict JSON reply).** For each numbered object the text names: `name_ok` (yes/no/unsure) and
`is_object` (a thing vs part of a wall/floor). Then `missing_obstacle` (yes/no: the route passes close to a big
obstacle the text does not mention), `stop_ok` (yes/no) with `better_stop` (free text, for reading only), and
`overall_true` (yes/no) with `confidence` 0-100. Unparseable replies are retried once, then recorded as errors.

**Score.** A small logistic combination of the answers (any name/object "no", missing obstacle, stop "no",
overall "no", confidence), fitted on the calibration marks with place-grouped cross-validation; the plain
`overall_true` answer is reported alongside as the no-fitting baseline.

**Calibration set (CPU, no rendering).** The four saved text-audit pages, items with the exact image and text the
user judged: `outputs/instr_d2_audit.html` (d2 G3), `outputs/instr_d2_audit_n2.html` (d2 G3'),
`outputs/instr_d3_audit.html` (d3 G3), `outputs/instr_d3_audit_a2.html` (d3 G3') = ~347 judged texts, ~61 wrong.
Marks (now in the scratchpad) are copied to `outputs/audit_marks/<page>/` first. The two G1 stop pages
(`outputs/stops_audit.html`, `outputs/stops_audit_a2_plus.html`, ~224 items, template texts with "object N") are
scored as a secondary check of `stop_ok` only.

**Validation (the checker's own gate, before it touches data).** Prompt v1 is fixed before any reply is read; one
revision allowed. Report: AUC of the score vs the user's marks (overall, per page, per stop kind, per mark type);
the curve "routes removed vs accuracy of the kept routes"; agreement of `name_ok` with name marks and `stop_ok` with
stop marks. Proposed bar: AUC >= 0.75 overall and > 0.6 within the large stop kinds; otherwise report and stop.

**Production run.** After the full job 97236 and `finalize-descriptions` inputs exist: render and check every d3
train text (~20,037) and every eval text (~2,084) in one GPU job (rendering on the node's CPUs inside the job;
resumable, like the other label stages). Output `outputs/instr_labels_v3/check_{train,eval}_d3.jsonl`.

**Filtering.** The user picks the cutoff from the curve. `finalize-descriptions` gets an option to leave out the
train route ids above the cutoff (eval untouched); eval reports gain a "checker-passed subset" line.

## Left to the plan / the builder

Exact prompt wording; JSON schema details; how object numbers map to names in the prompt (the page numbering);
concurrency and image size for vLLM; how the logistic fit is stored and applied; the finalize option's name.

## Run order (every sbatch: ask first)

1. CPU: copy marks to `outputs/audit_marks/`, build the calibration set, tests for parsing and scoring.
2. GPU job 1 (ask): check the ~570 calibration items. CPU: validation report (artifact) -> user decision.
3. GPU job 2 (ask): check all train + eval d3 texts (after job 97236).
4. User picks the cutoff; finalize-descriptions with the train drop list; then d3 training.
