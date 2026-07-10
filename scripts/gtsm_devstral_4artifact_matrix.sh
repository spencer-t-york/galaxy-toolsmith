#!/usr/bin/env bash 
set -Eeuo pipefail

MODE="${1:-run}"
case "$MODE" in
  run|resume|status) ;;
  *)
    echo "Usage: $0 {run|resume|status}" >&2
    exit 2
    ;;
esac

MATRIX_ROOT="${MATRIX_ROOT:-.gtsm-cache/runs/devstral-matrix}"
CURRENT_FILE="$MATRIX_ROOT/current"

if [[ -z "${RUN_TAG:-}" ]]; then
  if [[ "$MODE" == "run" ]]; then
    RUN_TAG="$(date -u +%Y%m%dT%H%M%SZ)"
  elif [[ -f "$CURRENT_FILE" ]]; then
    RUN_TAG="$(cat "$CURRENT_FILE")"
  else
    echo "RUN_TAG is not set and no current devstral matrix run exists." >&2
    exit 2
  fi
fi

if [[ ! "$RUN_TAG" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "RUN_TAG must contain only letters, numbers, '.', '_', and '-'" >&2
  exit 2
fi

RUN_ROOT="$MATRIX_ROOT/$RUN_TAG"
STATE_ENV="$RUN_ROOT/state.env"
if [[ "$MODE" != "run" && ! -f "$STATE_ENV" ]]; then
  echo "No state found for RUN_TAG=$RUN_TAG at $STATE_ENV." >&2
  exit 2
fi

: "${DRY_RUN:=0}"
: "${GTSM:=gtsm}"
: "${CORPUS_JSONL:=.gtsm-cache/datasets/tools-iuc-corpus.jsonl}"
: "${DATASET_MANIFEST:=config/dataset.manifest.json}"
: "${GPU_DEVICES:=0,1,2,3}"
: "${NUM_PROCESSES:=4}"
: "${STATUS_INTERVAL_SECONDS:=30}"
: "${LOG_TAIL_LINES:=40}"
: "${PYTORCH_CUDA_ALLOC_CONF:=expandable_segments:True}"

# Training inputs
: "${FT_PROFILE:=agentic-devstral-24b}"
: "${FT_DISTRIBUTED_STRATEGY:=fsdp}"
: "${FT_MAX_SEQ_LENGTH:=8192}"
: "${FT_BATCH_SIZE:=1}"
: "${FT_GRAD_ACCUM:=2}"
: "${FT_VARIANT:=tools-iuc-devstral-24b-full-$RUN_TAG}"

# Base profile/variant for base artifact bootstrap.
: "${BASE_PROFILE:=agentic-devstral-24b}"
: "${BASE_VARIANT:=${BASE_PROFILE}-base}"

# Quantization and benchmark matrix settings
: "${EXPORT_QUANTIZATION:=q4_k_m}"
: "${SOURCE_CONTEXT_MODES:=none,snippets,all-filtered}"
: "${BENCHMARK_LIMIT:=100}"
: "${MAX_TOKENS:=4096}"
: "${MAX_PROMPT_HELP_CHARS:=4000}"
: "${BENCHMARK_GPU_TOPOLOGY:=per-process}"
: "${BENCHMARK_OFFLOAD_POLICY:=allow}"
: "${BENCHMARK_GPU_MEMORY_RESERVE_GIB:=2.0}"
: "${BENCHMARK_RECORD_TIMEOUT_SECONDS:=0}"
: "${BENCHMARK_RESUME_EXISTING:=1}"

# Ollama bridge for quantized GGUF benchmarking.
: "${CREATE_OLLAMA_MODELS:=1}"
: "${BASE_OLLAMA_MODEL:=gtsm-${RUN_TAG}-devstral-base-${EXPORT_QUANTIZATION}}"
: "${FT_OLLAMA_MODEL:=gtsm-${RUN_TAG}-devstral-ft-${EXPORT_QUANTIZATION}}"

LOG_DIR="$RUN_ROOT/logs"
MARKER_DIR="$RUN_ROOT/markers"
BENCHMARK_DIR="$RUN_ROOT/benchmarks"
EXPORT_DIR="$RUN_ROOT/exports"
RUN_LOG="$RUN_ROOT/run.log"
SUMMARY_MD="$RUN_ROOT/summary.md"
TRAIN_ROOT=".gtsm-cache/runs/training"
MODEL_VARIANTS_DIR=".gtsm-cache/models/variants"

mkdir -p "$RUN_ROOT" "$LOG_DIR" "$MARKER_DIR" "$BENCHMARK_DIR" "$EXPORT_DIR"

if [[ -f "$STATE_ENV" ]]; then
  # shellcheck disable=SC1090
  source "$STATE_ENV"
fi

STEPS=(
  preflight
  finetune_devstral
  export_quantized_base
  export_quantized_finetuned
  benchmark_matrix
  summarize_matrix
)

is_true() {
  case "${1:-}" in
    1|true|TRUE|yes|YES|on|ON) return 0 ;;
    *) return 1 ;;
  esac
}

is_dry_run() {
  is_true "$DRY_RUN"
}

timestamp() {
  date -u +%Y-%m-%dT%H:%M:%SZ
}

log() {
  printf '[%s] %s\n' "$(timestamp)" "$*" | tee -a "$RUN_LOG"
}

die() {
  log "ERROR: $*"
  exit 1
}

command_string() {
  local rendered=""
  local arg
  for arg in "$@"; do
    printf -v rendered '%s%q ' "$rendered" "$arg"
  done
  printf '%s' "${rendered% }"
}

run_cmd_allow_failure() {
  local log_file="$1"
  shift
  mkdir -p "$(dirname "$log_file")"
  log "+ $(command_string "$@")"
  if is_dry_run; then
    return 0
  fi
  set +e
  "$@" 2>&1 | tee -a "$RUN_LOG" "$log_file"
  local rc="${PIPESTATUS[0]}"
  set -e
  return "$rc"
}

run_cmd() {
  local log_file="$1"
  shift
  run_cmd_allow_failure "$log_file" "$@" || die "Command failed: $(command_string "$@")"
}

run_json_cmd() {
  local output_json="$1"
  local log_file="$2"
  shift 2
  mkdir -p "$(dirname "$output_json")" "$(dirname "$log_file")"
  log "+ $(command_string "$@") > $output_json"
  if is_dry_run; then
    printf '{}\n' > "$output_json"
    return 0
  fi
  local stdout_tmp="$output_json.stdout.tmp"
  local stderr_tmp="$output_json.stderr.tmp"
  set +e
  "$@" > "$stdout_tmp" 2> "$stderr_tmp"
  local rc="$?"
  set -e
  cat "$stdout_tmp" | tee -a "$RUN_LOG" "$log_file"
  cat "$stderr_tmp" | tee -a "$RUN_LOG" "$log_file" >&2
  if [[ "$rc" -eq 0 ]]; then
    cp "$stdout_tmp" "$output_json"
  fi
  rm -f "$stdout_tmp" "$stderr_tmp"
  return "$rc"
}

marker_path() {
  printf '%s/%s.ok' "$MARKER_DIR" "$1"
}

mark_done() {
  local step="$1"
  if ! is_dry_run; then
    printf '%s\n' "$(timestamp)" > "$(marker_path "$step")"
  fi
}

step_done() {
  [[ -f "$(marker_path "$1")" ]]
}

append_summary() {
  mkdir -p "$(dirname "$SUMMARY_MD")"
  printf '%s\n' "$*" >> "$SUMMARY_MD"
}

run_step() {
  local step="$1"
  local function_name="$2"
  local description="$3"
  if step_done "$step"; then
    log "Skipping $step: marker exists."
    return 0
  fi
  log "Starting $step: $description"
  "$function_name"
  mark_done "$step"
  log "Completed $step."
}

split_csv() {
  local value="$1"
  local IFS=,
  read -r -a SPLIT_CSV_RESULT <<< "$value"
}

save_state() {
  {
    printf 'RUN_TAG=%q\n' "$RUN_TAG"
    printf 'GTSM=%q\n' "$GTSM"
    printf 'CORPUS_JSONL=%q\n' "$CORPUS_JSONL"
    printf 'DATASET_MANIFEST=%q\n' "$DATASET_MANIFEST"
    printf 'GPU_DEVICES=%q\n' "$GPU_DEVICES"
    printf 'NUM_PROCESSES=%q\n' "$NUM_PROCESSES"
    printf 'STATUS_INTERVAL_SECONDS=%q\n' "$STATUS_INTERVAL_SECONDS"
    printf 'LOG_TAIL_LINES=%q\n' "$LOG_TAIL_LINES"
    printf 'PYTORCH_CUDA_ALLOC_CONF=%q\n' "$PYTORCH_CUDA_ALLOC_CONF"
    printf 'FT_PROFILE=%q\n' "$FT_PROFILE"
    printf 'FT_DISTRIBUTED_STRATEGY=%q\n' "$FT_DISTRIBUTED_STRATEGY"
    printf 'FT_MAX_SEQ_LENGTH=%q\n' "$FT_MAX_SEQ_LENGTH"
    printf 'FT_BATCH_SIZE=%q\n' "$FT_BATCH_SIZE"
    printf 'FT_GRAD_ACCUM=%q\n' "$FT_GRAD_ACCUM"
    printf 'FT_VARIANT=%q\n' "$FT_VARIANT"
    printf 'BASE_PROFILE=%q\n' "$BASE_PROFILE"
    printf 'BASE_VARIANT=%q\n' "$BASE_VARIANT"
    printf 'EXPORT_QUANTIZATION=%q\n' "$EXPORT_QUANTIZATION"
    printf 'SOURCE_CONTEXT_MODES=%q\n' "$SOURCE_CONTEXT_MODES"
    printf 'BENCHMARK_LIMIT=%q\n' "$BENCHMARK_LIMIT"
    printf 'MAX_TOKENS=%q\n' "$MAX_TOKENS"
    printf 'MAX_PROMPT_HELP_CHARS=%q\n' "$MAX_PROMPT_HELP_CHARS"
    printf 'BENCHMARK_GPU_TOPOLOGY=%q\n' "$BENCHMARK_GPU_TOPOLOGY"
    printf 'BENCHMARK_OFFLOAD_POLICY=%q\n' "$BENCHMARK_OFFLOAD_POLICY"
    printf 'BENCHMARK_GPU_MEMORY_RESERVE_GIB=%q\n' "$BENCHMARK_GPU_MEMORY_RESERVE_GIB"
    printf 'BENCHMARK_RECORD_TIMEOUT_SECONDS=%q\n' "$BENCHMARK_RECORD_TIMEOUT_SECONDS"
    printf 'BENCHMARK_RESUME_EXISTING=%q\n' "$BENCHMARK_RESUME_EXISTING"
    printf 'CREATE_OLLAMA_MODELS=%q\n' "$CREATE_OLLAMA_MODELS"
    printf 'BASE_OLLAMA_MODEL=%q\n' "$BASE_OLLAMA_MODEL"
    printf 'FT_OLLAMA_MODEL=%q\n' "$FT_OLLAMA_MODEL"
    printf 'ACTIVE_FT_VARIANT=%q\n' "${ACTIVE_FT_VARIANT:-}"
  } > "$STATE_ENV"
  printf '%s\n' "$RUN_TAG" > "$CURRENT_FILE"
}

require_variant_manifest() {
  local variant_id="$1"
  [[ -n "$variant_id" ]] || die "Variant id is empty."
  [[ -f "$MODEL_VARIANTS_DIR/$variant_id.manifest.json" ]] || die "Variant manifest not found: $MODEL_VARIANTS_DIR/$variant_id.manifest.json"
}

bootstrap_base_variant_manifest() {
  local variant_manifest="$MODEL_VARIANTS_DIR/$BASE_VARIANT.manifest.json"
  if [[ -f "$variant_manifest" ]]; then
    log "Using existing base variant manifest: $variant_manifest"
    return 0
  fi

  mkdir -p "$MODEL_VARIANTS_DIR"
  log "Base variant manifest missing; bootstrapping BASE_VARIANT=$BASE_VARIANT from BASE_PROFILE=$BASE_PROFILE"

  local profiles_json="$RUN_ROOT/base.train_profiles.json"
  run_json_cmd \
    "$profiles_json" \
    "$LOG_DIR/base-profile.log" \
    "$GTSM" list-train-profiles

  if is_dry_run; then
    cat > "$variant_manifest" <<JSON
{
  "variant_id": "$BASE_VARIANT",
  "base_model": "dry-run",
  "quantization": "none",
  "training_dataset_id": "",
  "provider": "local",
  "skills_profile": "default",
  "backend": "auto",
  "training_method": "lora",
  "effective_training_method": "lora",
  "artifact_kind": "unknown",
  "artifact_dir": ""
}
JSON
    log "Dry run: wrote placeholder base variant manifest at $variant_manifest"
    return 0
  fi

  local profile_json="$RUN_ROOT/base.profile.selected.json"
  jq --arg profile "$BASE_PROFILE" '.profiles[] | select(.name == $profile)' "$profiles_json" > "$profile_json"

  if [[ ! -s "$profile_json" ]]; then
    die "BASE_PROFILE not found in configured profiles: $BASE_PROFILE"
  fi

  local base_model
  base_model="$(jq -r '.base_model // empty' "$profile_json")"
  [[ -n "$base_model" ]] || die "BASE_PROFILE=$BASE_PROFILE does not define base_model."

  jq -n \
    --arg variant_id "$BASE_VARIANT" \
    --arg base_model "$base_model" \
    --arg quantization "none" \
    --arg provider "$(jq -r '.provider // "local"' "$profile_json")" \
    --arg skills_profile "$(jq -r '.skills_profile // "default"' "$profile_json")" \
    --arg backend "$(jq -r '.backend // "auto"' "$profile_json")" \
    --arg training_method "$(jq -r '.training_method // "lora"' "$profile_json")" \
    '{
      variant_id: $variant_id,
      schema_version: "0.1.0",
      created_at: (now | todateiso8601),
      base_model: $base_model,
      quantization: $quantization,
      training_dataset_id: "",
      provider: $provider,
      skills_profile: $skills_profile,
      backend: $backend,
      training_method: $training_method,
      effective_training_method: $training_method,
      artifact_kind: "unknown",
      artifact_dir: "",
      export_quantizations: [],
      ollama_model_name: "",
      requested_ollama_model_name: "",
      ollama_modelfile_path: ""
    }' > "$variant_manifest"

  log "Bootstrapped base variant manifest: $variant_manifest"
}

preflight() {
  save_state
  : > "$SUMMARY_MD"
  append_summary "# GTSM Devstral 4-Artifact Matrix Run $RUN_TAG"
  append_summary ""
  append_summary "- Started: $(timestamp)"
  append_summary "- Base profile: $BASE_PROFILE"
  append_summary "- Base variant: $BASE_VARIANT"
  append_summary "- Finetuned variant: $FT_VARIANT"
  append_summary "- Quantization: $EXPORT_QUANTIZATION"
  append_summary "- Source context modes: $SOURCE_CONTEXT_MODES"
  append_summary "- Corpus: $CORPUS_JSONL"

  command -v "$GTSM" >/dev/null || die "gtsm command not found: $GTSM"
  command -v jq >/dev/null || die "jq is required."
  command -v nvidia-smi >/dev/null || die "nvidia-smi is required."

  [[ -n "$BASE_PROFILE" ]] || die "BASE_PROFILE is required."
  [[ -n "$BASE_VARIANT" ]] || die "BASE_VARIANT is required."
  bootstrap_base_variant_manifest
  require_variant_manifest "$BASE_VARIANT"

  [[ -f "$CORPUS_JSONL" ]] || die "Corpus JSONL not found: $CORPUS_JSONL"
  [[ -f "$DATASET_MANIFEST" ]] || die "Dataset manifest not found: $DATASET_MANIFEST"

  run_cmd "$LOG_DIR/preflight.log" "$GTSM" benchmark-summary --help
  run_cmd "$LOG_DIR/preflight.log" "$GTSM" export-model --help
  run_cmd "$LOG_DIR/preflight.log" "$GTSM" export-ollama-model --help
  run_cmd "$LOG_DIR/preflight.log" "$GTSM" runtime-detect
  run_cmd "$LOG_DIR/preflight.log" nvidia-smi
}

finetune_devstral() {
  local run_id="train-$RUN_TAG-devstral24b"
  local status_log="$LOG_DIR/finetune.status.jsonl"

  if ! is_dry_run && [[ -f "$MODEL_VARIANTS_DIR/$FT_VARIANT.manifest.json" ]]; then
    ACTIVE_FT_VARIANT="$FT_VARIANT"
    save_state
    log "Using existing finetuned variant: $ACTIVE_FT_VARIANT"
    append_summary "- Reused finetuned variant: $ACTIVE_FT_VARIANT"
    return 0
  fi

  export CUDA_VISIBLE_DEVICES="$GPU_DEVICES"
  export PYTORCH_CUDA_ALLOC_CONF
  run_cmd "$LOG_DIR/finetune.log" \
    "$GTSM" train \
      --profile "$FT_PROFILE" \
      --dataset-manifest "$DATASET_MANIFEST" \
      --corpus-jsonl "$CORPUS_JSONL" \
      --variant-id "$FT_VARIANT" \
      --backend axolotl \
      --num-processes "$NUM_PROCESSES" \
      --distributed-strategy "$FT_DISTRIBUTED_STRATEGY" \
      --max-seq-length "$FT_MAX_SEQ_LENGTH" \
      --per-device-batch-size "$FT_BATCH_SIZE" \
      --gradient-accumulation-steps "$FT_GRAD_ACCUM" \
      --status-log "$status_log" \
      --status-interval-seconds "$STATUS_INTERVAL_SECONDS" \
      --stream-logs \
      --log-tail-lines "$LOG_TAIL_LINES" \
      --internal-run-id "$run_id"

  ACTIVE_FT_VARIANT="$FT_VARIANT"
  require_variant_manifest "$ACTIVE_FT_VARIANT"
  save_state
  append_summary "- Finetuned variant: $ACTIVE_FT_VARIANT"
}

export_quantized_variant() {
  local role="$1"
  local variant_id="$2"
  local ollama_model="$3"
  local log_prefix="$4"
  local out_json="$EXPORT_DIR/$log_prefix.export.json"

  require_variant_manifest "$variant_id"

  run_json_cmd \
    "$out_json" \
    "$LOG_DIR/$log_prefix.export.log" \
    "$GTSM" export-model \
      --variant-id "$variant_id" \
      --format gguf \
      --quantizations "$EXPORT_QUANTIZATION"

  if is_true "$CREATE_OLLAMA_MODELS"; then
    run_json_cmd \
      "$EXPORT_DIR/$log_prefix.ollama.json" \
      "$LOG_DIR/$log_prefix.ollama.log" \
      "$GTSM" export-ollama-model \
        --variant-id "$variant_id" \
        --model-name "$ollama_model" \
        --from-quantization "$EXPORT_QUANTIZATION" \
        --create
  fi

  append_summary "- Quantized $role variant: $variant_id ($EXPORT_QUANTIZATION)"
  if is_true "$CREATE_OLLAMA_MODELS"; then
    append_summary "- Ollama model ($role quant): $ollama_model"
  fi
}

export_quantized_base() {
  export_quantized_variant "base" "$BASE_VARIANT" "$BASE_OLLAMA_MODEL" "base"
}

export_quantized_finetuned() {
  local ft_variant="${ACTIVE_FT_VARIANT:-$FT_VARIANT}"
  export_quantized_variant "finetuned" "$ft_variant" "$FT_OLLAMA_MODEL" "finetuned"
}

benchmark_cell() {
  local role="$1"
  local provider="$2"
  local model_variant_label="$3"
  local model_name="$4"
  local source_mode="$5"

  local out_dir="$BENCHMARK_DIR/$role/$source_mode"
  local log_file="$LOG_DIR/bench-$role-$source_mode.log"
  local status_log="$LOG_DIR/bench-$role-$source_mode.status.jsonl"
  mkdir -p "$out_dir"

  if ! is_dry_run && [[ -f "$out_dir/benchmark.summary.json" ]]; then
    log "Skipping benchmark cell (already exists): role=$role mode=$source_mode"
    return 0
  fi

  export CUDA_VISIBLE_DEVICES="$GPU_DEVICES"
  export PYTORCH_CUDA_ALLOC_CONF

  local cmd=(
    "$GTSM" benchmark-generate
      --corpus-jsonl "$CORPUS_JSONL"
      --limit "$BENCHMARK_LIMIT"
      --provider "$provider"
      --model-variant "$model_variant_label"
      --temperature 0
      --max-tokens "$MAX_TOKENS"
      --max-prompt-help-chars "$MAX_PROMPT_HELP_CHARS"
      --repair-invalid-xml
      --num-processes "$NUM_PROCESSES"
      --gpu-devices "$GPU_DEVICES"
      --min-items-per-process 1
      --local-gpu-topology "$BENCHMARK_GPU_TOPOLOGY"
      --local-offload-policy "$BENCHMARK_OFFLOAD_POLICY"
      --local-gpu-memory-reserve-gib "$BENCHMARK_GPU_MEMORY_RESERVE_GIB"
      --record-timeout-seconds "$BENCHMARK_RECORD_TIMEOUT_SECONDS"
      --source-context-mode "$source_mode"
      --checkpoint-records "$out_dir/checkpoint.records.jsonl"
      --wrappers-dir "$out_dir/wrappers"
      --generation-records "$out_dir/generation.records.json"
      --evaluation-report "$out_dir/evaluation.summary.json"
      --benchmark-summary "$out_dir/benchmark.summary.json"
      --status-log "$status_log"
  )

  if [[ -n "$model_name" ]]; then
    cmd+=(--model "$model_name")
  fi
  if is_true "$BENCHMARK_RESUME_EXISTING"; then
    cmd+=(--resume-existing)
  fi

  run_cmd "$log_file" "${cmd[@]}"
}

benchmark_matrix() {
  local ft_variant="${ACTIVE_FT_VARIANT:-$FT_VARIANT}"
  split_csv "$SOURCE_CONTEXT_MODES"
  local mode

  for mode in "${SPLIT_CSV_RESULT[@]}"; do
    mode="${mode//[[:space:]]/}"
    [[ -n "$mode" ]] || continue

    benchmark_cell "base" "local" "$BASE_VARIANT" "" "$mode"
    benchmark_cell "finetuned" "local" "$ft_variant" "" "$mode"

    if is_true "$CREATE_OLLAMA_MODELS"; then
      benchmark_cell "quant-base" "ollama" "${BASE_VARIANT}-${EXPORT_QUANTIZATION}" "$BASE_OLLAMA_MODEL" "$mode"
      benchmark_cell "quant-finetuned" "ollama" "${ft_variant}-${EXPORT_QUANTIZATION}" "$FT_OLLAMA_MODEL" "$mode"
    else
      die "CREATE_OLLAMA_MODELS=0 is not currently supported for quantized benchmarking."
    fi
  done
}

summarize_matrix() {
  append_summary ""
  append_summary "## Benchmark Matrix"
  local roles=(base finetuned quant-base quant-finetuned)
  split_csv "$SOURCE_CONTEXT_MODES"
  local role
  local mode
  for role in "${roles[@]}"; do
    for mode in "${SPLIT_CSV_RESULT[@]}"; do
      mode="${mode//[[:space:]]/}"
      [[ -n "$mode" ]] || continue
      local summary="$BENCHMARK_DIR/$role/$mode/benchmark.summary.json"
      if [[ -f "$summary" ]]; then
        append_summary ""
        append_summary "### $role @ $mode"
        "$GTSM" benchmark-summary --summary "$summary" | tee -a "$RUN_LOG" "$SUMMARY_MD"
      else
        append_summary "- Missing summary for $role @ $mode: $summary"
      fi
    done
  done
}

print_status() {
  echo "run_tag=$RUN_TAG"
  echo "run_root=$RUN_ROOT"
  echo "dry_run=$DRY_RUN"
  echo
  if [[ -f "$STATE_ENV" ]]; then
    echo "state:"
    sed -n '1,200p' "$STATE_ENV"
    echo
  fi
  echo "steps:"
  local step
  local next_step=""
  for step in "${STEPS[@]}"; do
    if step_done "$step"; then
      printf '  [x] %s\n' "$step"
    else
      printf '  [ ] %s\n' "$step"
      [[ -z "$next_step" ]] && next_step="$step"
    fi
  done
  echo
  echo "next_step=${next_step:-complete}"
}

if [[ "$MODE" == "status" ]]; then
  print_status
  exit 0
fi

save_state

run_step preflight preflight "validate inputs and runtime"
run_step finetune_devstral finetune_devstral "finetune base Devstral"
run_step export_quantized_base export_quantized_base "export base quantized artifact"
run_step export_quantized_finetuned export_quantized_finetuned "export finetuned quantized artifact"
run_step benchmark_matrix benchmark_matrix "run 4-artifact x context benchmark matrix"
run_step summarize_matrix summarize_matrix "summarize matrix benchmark outputs"

append_summary ""
append_summary "- Finished: $(timestamp)"
log "Devstral 4-artifact matrix workflow complete."
