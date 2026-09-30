#!/usr/bin/env bash
#
# Stage-gated full-pipeline orchestrator. Run with CWD at the repo root
# (every pipeline script uses CWD-relative dataset/, output/, cache/,
# final_results/ paths). Python *must* be the project venv interpreter.
#
#   ./scripts/run_pipeline.sh <stage> | --full
#
# Stages run one at a time by default; --full chains all six. Three stages
# are hours-long and disk is tight, so stage-gated is the default until
# --full has been exercised once. `merge` always chains into `validate`
# as a non-skippable hard gate.
#
# Embedding sidecars need sentence-transformers + model weights, which are
# pinned but not installed on this box. ENABLE_EMBEDDINGS=true therefore
# fails fast with an explicit message until sidecars are actually
# generated (no sidecar-generation stage exists yet — see 2.2 notes).
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CODE_DIR="$ROOT_DIR/code/business_entity_resolution/src"
VENV_PY="$CODE_DIR/.venv/bin/python"
LOG_FILE="$ROOT_DIR/output/run_log.txt"
mkdir -p "$ROOT_DIR/output"

cd "$ROOT_DIR"

ENABLE_EMBEDDINGS="${ENABLE_EMBEDDINGS:-false}"

log() {
  echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] $1" | tee -a "$LOG_FILE"
}

require_path() {
  local path="$1" reason="$2"
  if [[ ! -e "$path" ]]; then
    echo "ERROR: missing required input: $path" >&2
    echo "  ($reason)" >&2
    exit 1
  fi
}

require_embeddings_ready() {
  if [[ "$ENABLE_EMBEDDINGS" == "true" ]]; then
    echo "ERROR: ENABLE_EMBEDDINGS=true is not runnable on this box:" >&2
    echo "  sentence-transformers is pinned but not installed, and no train/test" >&2
    echo "  embedding sidecars exist yet. Generate sidecars elsewhere first," >&2
    echo "  then pass their paths via --embed-sidecar explicitly." >&2
    exit 1
  fi
}

stage_fit_indices() {
  log "STAGE fit_indices: start"
  for split in train test; do
    for source in 2 3; do
      require_path "$ROOT_DIR/dataset/$split/${split}_source${source}.tsv" \
        "raw $split source$source file must exist"
      "$VENV_PY" "$CODE_DIR/blocking/fit_target_index.py" \
        --split "$split" --target_source "$source"
    done
  done
  log "STAGE fit_indices: done"
}

stage_build_train_candidates() {
  log "STAGE build_train_candidates: start"
  for source in 2 3; do
    require_path "$ROOT_DIR/cache/train/s${source}_word_vec.joblib" \
      "run fit_indices first (train/S$source)"
    for chunk in 0 1 2 3; do
      "$VENV_PY" "$CODE_DIR/training/build_train_candidates.py" \
        --target_source "$source" --chunk "$chunk" --total_chunks 4
    done
  done
  log "STAGE build_train_candidates: done"
}

stage_train() {
  log "STAGE train: start (ENABLE_EMBEDDINGS=$ENABLE_EMBEDDINGS)"
  require_embeddings_ready
  require_path "$ROOT_DIR/final_results" "run build_train_candidates first"
  "$VENV_PY" "$CODE_DIR/training/train_model.py"
  log "STAGE train: done -> xgb_model_v3.json + calibrator_v3.pkl + threshold_v3.json"
}

stage_generate_test_candidates() {
  log "STAGE generate_test_candidates: start"
  for source in 2 3; do
    require_path "$ROOT_DIR/cache/test/s${source}_word_vec.joblib" \
      "run fit_indices first (test/S$source)"
    for chunk in 0 1 2 3; do
      "$VENV_PY" "$CODE_DIR/blocking/generate_candidates.py" \
        --target_source "$source" --chunk "$chunk" --total_chunks 4
    done
  done
  log "STAGE generate_test_candidates: done"
}

stage_predict() {
  log "STAGE predict: start"
  require_embeddings_ready
  require_path "$ROOT_DIR/calibrator_v3.pkl" "run train first"
  for source in 2 3; do
    for chunk in 0 1 2 3; do
      "$VENV_PY" "$CODE_DIR/inference/predict.py" \
        --target_source "$source" --chunk "$chunk"
    done
  done
  log "STAGE predict: done"
}

stage_merge() {
  log "STAGE merge: start"
  "$VENV_PY" "$CODE_DIR/submission/merge_submission.py"
  log "STAGE merge: done"

  log "STAGE validate: start (hard gate, non-skippable)"
  "$VENV_PY" "$ROOT_DIR/utils/validate_submission.py" \
    --matching "$ROOT_DIR/output/matching_results.tsv" \
    --candidate "$ROOT_DIR/output/candidate_pairs.tsv" \
    --test-dir "$ROOT_DIR/dataset/test"
  log "STAGE validate: PASS"
}

STAGES=(fit_indices build_train_candidates train generate_test_candidates predict merge)

run_stage() {
  case "$1" in
    fit_indices) stage_fit_indices ;;
    build_train_candidates) stage_build_train_candidates ;;
    train) stage_train ;;
    generate_test_candidates) stage_generate_test_candidates ;;
    predict) stage_predict ;;
    merge) stage_merge ;;
    *) echo "Unknown stage: $1" >&2; echo "Valid: ${STAGES[*]}" >&2; exit 1 ;;
  esac
}

if [[ "${1:-}" == "--full" ]]; then
  log "FULL RUN: chaining all stages, ENABLE_EMBEDDINGS=$ENABLE_EMBEDDINGS"
  for s in "${STAGES[@]}"; do run_stage "$s"; done
  log "FULL RUN: complete"
elif [[ -n "${1:-}" ]]; then
  run_stage "$1"
else
  echo "Usage: $0 <stage> | --full" >&2
  echo "Stages (in order): ${STAGES[*]}" >&2
  echo "Env: ENABLE_EMBEDDINGS=true|false (default false)" >&2
  exit 1
fi
