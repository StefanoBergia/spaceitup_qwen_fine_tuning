#!/bin/bash
# Label d1 route descriptions with Cosmos3-Nano, unattended: serve + both stages in one job.
#
#     sbatch slurm/label_descriptions.sbatch              # gate: 300 eval routes, then stop for the audit
#     STAGE=full sbatch slurm/label_descriptions.sbatch   # eval + train (resumes after a TIMEOUT)
#
# Per split: name pass 1, name pass 2 (--resume retries request errors), compose pass 1,
# compose pass 2 (--resume retries). The name passes are report-only (read their summary in
# the log); in the gate run only compose pass 2 passes --gate, so one transient HTTP error
# cannot fail it and a weak name pass never stops compose from running.
#
# Needs data/prepared_habitat_instr_v2/{events,objects}_{eval,train}.jsonl
# (uv run scripts/prepare_habitat_instr.py events). Spec:
# docs/superpowers/specs/2026-09-30-route-descriptions-design.md. Safe to re-submit: every
# call passes --resume. After a prompt fix, re-run the gate from scratch (OVERWRITE applies to
# the first pass of each stage only; the retry passes always resume):
#     OVERWRITE=1 sbatch slurm/label_descriptions.sbatch
#
# NAMER=cosmos|gemma (default cosmos = the commands above, unchanged). NAMER=gemma serves
# google/gemma-3-12b-it for the name passes, stops it, then serves Cosmos3-Nano for the
# compose passes; outputs get a _gemma suffix (names_gate_gemma_d1.jsonl, desc_gate_gemma_d1.jsonl)
# so the Cosmos files are never touched. Combinable with STAGE=full (and OVERWRITE=1 for gate re-runs only):
#     NAMER=gemma sbatch slurm/label_descriptions.sbatch
#     NAMER=gemma STAGE=full sbatch slurm/label_descriptions.sbatch   # re-submit the same line after a TIMEOUT; never OVERWRITE=1 on a full run
#
# NAMER=mix (Addendum A5): Gemma names every split -> names_<tag>_gemma_d1.jsonl; ONE switch to
# Cosmos; Cosmos names the same splits -> names_<tag>_d1.jsonl; Cosmos composes with the Gemma
# names, falling back to the Cosmos names where Gemma's are unusable (name clash) ->
# desc_<tag>_mix_d1.jsonl:
#     NAMER=mix STAGE=full sbatch slurm/label_descriptions.sbatch   # re-submit the same line after a TIMEOUT; never OVERWRITE=1 on a full run
# Afterwards, on the login node:
#     uv run scripts/prepare_habitat_instr.py finalize-descriptions            # cosmos / gemma
#     uv run scripts/prepare_habitat_instr.py finalize-descriptions --desc-suffix _mix
#SBATCH --job-name=cosmos3-label-desc
#SBATCH --partition=A100
#SBATCH --gres=gpu:3g.40gb:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=14:00:00
#SBATCH --output=outputs/slurm/%x-%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=telegram:422627024
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
mkdir -p outputs/slurm outputs/instr_labels_v3
source slurm/cosmos3_serve_lib.sh
D="${DATA:-data/prepared_habitat_instr_v2}"
OUT=outputs/instr_labels_v3
CONCURRENCY="${CONCURRENCY:-8}"
UV="${UV:-$HOME/.local/bin/uv}"
STAGE="${STAGE:-gate}"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
unset VIRTUAL_ENV
for s in eval train; do
    for f in events objects; do
        [ -f "$D/${f}_$s.jsonl" ] || { echo "missing $D/${f}_$s.jsonl — run prepare_habitat_instr.py events"; exit 1; }
    done
done

NAMER="${NAMER:-cosmos}"
COSMOS_MODEL="$MODEL"
GEMMA_MODEL="${GEMMA_MODEL:-google/gemma-3-12b-it}"
case "$NAMER" in
    cosmos) SFX="" ;;
    gemma) SFX="_gemma" ;;   # Gemma outputs never overwrite the Cosmos files
    mix) SFX="_mix" ;;       # desc files only; the names-file suffix is set per phase below
    *) echo "NAMER must be cosmos, gemma or mix, got '$NAMER'"; exit 1 ;;
esac
NAME_SFX="$SFX"            # suffix of the names file written by name_stage / read by compose
FALLBACK=()                # mix: extra compose args naming the Cosmos fallback names file
NAME_EXTRA=()              # mix: extra name-stage args (Cosmos names only what Gemma cannot serve)
echo "NAMER=$NAMER file suffix='${SFX}' STAGE=$STAGE"

serve() {  # serve <model>: stop whatever runs, start this model, wait until healthy (sets BASE)
    cosmos3_stop_server
    MODEL=$1
    cosmos3_start_server
    cosmos3_wait_healthy
}
# Whichever server is running (or the half-started one) is stopped on exit, incl. a failed gate.
trap 'cosmos3_stop_server' EXIT

parse_opts() {  # parse_opts [--limit N] [--gate] -> sets first, gate, limit for the caller
    first=--resume gate=() limit=()
    [ -n "${OVERWRITE:-}" ] && first=--overwrite   # only the first call of each stage overwrites
    while [ $# -gt 0 ]; do
        case "$1" in
            --gate) gate=(--gate) ;;
            --limit) limit=(--limit "$2"); shift ;;
        esac
        shift
    done
}

# Each stage runs twice: pass 1 may leave request-error rows, pass 2 (--resume) retries
# them. Names are report-only (their summary is printed, never gated), so a weak name
# pass still lets compose run and the audit sees both; only compose pass 2 gates.
name_stage() {  # name_stage <split> <tag> [--limit N]
    local s=$1 tag=$2; shift 2
    local first gate limit; parse_opts "$@"
    local name_args=(--stage name --events "$D/events_$s.jsonl" --objects "$D/objects_$s.jsonl"
                     --out "$OUT/names_${tag}${NAME_SFX}_d1.jsonl" --endpoint "$BASE/v1" --concurrency "$CONCURRENCY" "${NAME_EXTRA[@]}")
    echo; echo "=============== name: $tag$NAME_SFX (pass 1) ==============="
    "$UV" run scripts/label_descriptions.py "${name_args[@]}" "$first" "${limit[@]}"
    echo; echo "=============== name: $tag$NAME_SFX (pass 2, retries errors; summary is report-only) ==============="
    "$UV" run scripts/label_descriptions.py "${name_args[@]}" --resume "${limit[@]}"
}

compose_stage() {  # compose_stage <split> <tag> [--limit N] [--gate]
    local s=$1 tag=$2; shift 2
    local first gate limit; parse_opts "$@"
    local comp_args=(--stage compose --events "$D/events_$s.jsonl" --names "$OUT/names_${tag}${NAME_SFX}_d1.jsonl" "${FALLBACK[@]}"
                     --out "$OUT/desc_${tag}${SFX}_d1.jsonl" --endpoint "$BASE/v1" --concurrency "$CONCURRENCY")
    echo; echo "=============== compose: $tag$SFX (pass 1) ==============="
    "$UV" run scripts/label_descriptions.py "${comp_args[@]}" "$first" "${limit[@]}"
    echo; echo "=============== compose: $tag$SFX (pass 2, retries errors${gate[*]:+, gated}) ==============="
    "$UV" run scripts/label_descriptions.py "${comp_args[@]}" --resume "${limit[@]}" "${gate[@]}"
}

# Order: NAMER=cosmos -> one Cosmos server, per split
# name+compose (today's order). NAMER=gemma -> Gemma names every split, then one switch to
# Cosmos which composes every split. NAMER=mix -> Gemma names every split, ONE switch to Cosmos,
# which names only the objects of routes Gemma cannot serve (--only-unusable-under), then composes
# eval, names train, composes train (eval descriptions finish first).
# A failed compose gate exits 2 under set -e.
mix_cosmos_phase() {  # mix_cosmos_phase <split> <tag> [opts]: Cosmos names ONLY the objects of routes
    local tag=$2      # the Gemma names cannot serve (clash / unnamed); locals are seen by name_stage
    local NAME_SFX="" NAME_EXTRA=(--only-unusable-under "$OUT/names_${tag}_gemma_d1.jsonl")
    name_stage "$@"
}
mix_compose() {  # mix_compose <split> <tag> [opts]: compose with Gemma names + Cosmos fallback names
    local tag=$2
    local NAME_SFX="_gemma" FALLBACK=(--fallback-names "$OUT/names_${tag}_d1.jsonl")
    compose_stage "$@"
}

if [ "$NAMER" = "cosmos" ]; then
    serve "$COSMOS_MODEL"
else
    serve "$GEMMA_MODEL"
fi
[ "$NAMER" = "mix" ] && NAME_SFX="_gemma"   # the Gemma name passes

if [ "$STAGE" = "gate" ]; then
    case "$NAMER" in
        gemma) name_stage eval gate --limit 300
               serve "$COSMOS_MODEL"
               compose_stage eval gate --limit 300 --gate ;;
        mix)   name_stage eval gate --limit 300
               serve "$COSMOS_MODEL"
               mix_cosmos_phase eval gate --limit 300
               mix_compose eval gate --limit 300 --gate ;;
        *)     name_stage eval gate --limit 300
               compose_stage eval gate --limit 300 --gate ;;
    esac
    echo "gate labelled — audit it, then: $([ "$NAMER" != cosmos ] && echo "NAMER=$NAMER ")STAGE=full sbatch slurm/label_descriptions.sbatch"
    exit 0
fi
case "$NAMER" in
    gemma) name_stage eval eval
           name_stage train train
           serve "$COSMOS_MODEL"
           compose_stage eval eval
           compose_stage train train ;;
    mix)   name_stage eval eval
           name_stage train train
           serve "$COSMOS_MODEL"
           mix_cosmos_phase eval eval
           mix_compose eval eval
           mix_cosmos_phase train train
           mix_compose train train ;;
    *)     name_stage eval eval
           compose_stage eval eval
           name_stage train train
           compose_stage train train ;;
esac
echo; echo "=============== done ==============="
wc -l "$OUT"/*_d1.jsonl
