# Environment
```bash

```

# Corpus Extraction
```bash
gtsm init-workspace
gtsm sync-tools-iuc --ref main
gtsm sync-galaxy-skills --ref main
gtsm sync-galaxy-xsd --ref dev
gtsm extract-corpus \
  --max-workers 8 \
  --source-workers 8 \
  --no-fetch-docs \
  --resolve-containers \
  --execute-containers \
  --container-runtime auto \
  --container-prepare-workers 2 \
  --container-probe-workers 4 \
  --container-help-probe-mode exploratory \
  --container-cache-dir .gtsm-cache/containers \
  --container-sif-exec-mode auto \
  --source-download-timeout-seconds 60 \
  --status-log .gtsm-cache/logs/extract-corpus.status.jsonl \
  --retry-manifest .gtsm-cache/datasets/tools-iuc-corpus.retry-manifest.json \
  --bioconda-checkout-sources \
  --bioconda-ref master
```
```bash
export GTSM_MODEL_SOURCE_REGISTRY="https://huggingface.co"
export GTSM_MODEL_CACHE_ROOT="$PWD/.gtsm-cache/models/hf-cache"
export GTSM_MODEL_REVISION=""
export GTSM_MODEL_LOCAL_FILES_ONLY=false
```
```bash
export GTSM_SERVER_AUTH_TOKEN="$(openssl rand -hex 32)"
printf "%s\n" "$GTSM_SERVER_AUTH_TOKEN" > .gtsm-cache/server.tokens
chmod 600 .gtsm-cache/server.tokens
```

# Context Ladder Training
## Baseline Qwen 2.5 Coder Instruct 7B
### Command
First, install the baseline Qwen model to `.gtsm-cache/models/base/` and create the manifest file:
```bash
cd /data/home/yorks7/git/galaxy-toolsmith
set -euo pipefail

python -m venv .venv-hf-download
source .venv-hf-download/bin/activate
python -m pip install -U pip huggingface_hub

python - <<'PY'
import json
from datetime import datetime, timezone
from pathlib import Path
from huggingface_hub import snapshot_download

repo_id = "Qwen/Qwen2.5-Coder-7B-Instruct"
variant_id = "qwen25-coder-7b-base-local"
local_dir = Path(".gtsm-cache/models/base/qwen25-coder-7b-instruct")
local_dir.mkdir(parents=True, exist_ok=True)

snapshot_download(
    repo_id=repo_id,
    local_dir=str(local_dir),
    local_dir_use_symlinks=True,
)

manifest = {
    "variant_id": variant_id,
    "schema_version": "0.1.0",
    "created_at": datetime.now(timezone.utc).isoformat(),
    "base_model": repo_id,
    "quantization": "none",
    "training_dataset_id": "",
    "provider": "local",
    "skills_profile": "default",
    "backend": "axolotl",
    "training_method": "full",
    "effective_training_method": "full",
    "artifact_kind": "hf_full_model",
    "artifact_dir": str(local_dir.resolve()),
    "export_quantizations": [],
    "ollama_model_name": "",
    "requested_ollama_model_name": "",
    "ollama_modelfile_path": "",
}

out = Path(f".gtsm-cache/models/variants/{variant_id}.manifest.json")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

print("Downloaded to", local_dir.resolve())
print("Wrote", out.resolve())
PY

deactivate
```
### Output
`.gtsm-cache/models/variants/qwen25-coder-7b-base-local.manifest.json`:
```json
{
  "variant_id": "qwen25-coder-7b-base-local",
  "schema_version": "0.1.0",
  "created_at": "2026-08-13T18:40:33.282871+00:00",
  "base_model": "Qwen/Qwen2.5-Coder-7B-Instruct",
  "quantization": "none",
  "training_dataset_id": "",
  "provider": "local",
  "skills_profile": "default",
  "backend": "axolotl",
  "training_method": "full",
  "effective_training_method": "full",
  "artifact_kind": "hf_full_model",
  "artifact_dir": "/data/home/yorks7/git/galaxy-toolsmith/.gtsm-cache/models/base/qwen25-coder-7b-instruct",
  "export_quantizations": [],
  "ollama_model_name": "",
  "requested_ollama_model_name": "",
  "ollama_modelfile_path": ""
}
```

## Baseline DeepSeek R1 Distill Qwen 32B
### Command
Next, install the baseline DeepSeek model to `.gtsm-cache/models/base/` and create the manifest file:
```bash
cd /data/home/yorks7/git/galaxy-toolsmith
set -euo pipefail

python -m venv .venv-hf-download
source .venv-hf-download/bin/activate
python -m pip install -U pip huggingface_hub

python - <<'PY'
import json
from datetime import datetime, timezone
from pathlib import Path
from huggingface_hub import snapshot_download

repo_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
variant_id = "deepseek-r1-distill-qwen-32b-base-local"
local_dir = Path(".gtsm-cache/models/base/deepseek-r1-distill-qwen-32b")
local_dir.mkdir(parents=True, exist_ok=True)

snapshot_download(
    repo_id=repo_id,
    local_dir=str(local_dir),
    local_dir_use_symlinks=True,
)

manifest = {
    "variant_id": variant_id,
    "schema_version": "0.1.0",
    "created_at": datetime.now(timezone.utc).isoformat(),
    "base_model": repo_id,
    "quantization": "none",
    "training_dataset_id": "",
    "provider": "local",
    "skills_profile": "default",
    "backend": "axolotl",
    "training_method": "full",
    "effective_training_method": "full",
    "artifact_kind": "hf_full_model",
    "artifact_dir": str(local_dir.resolve()),
    "export_quantizations": [],
    "ollama_model_name": "",
    "requested_ollama_model_name": "",
    "ollama_modelfile_path": "",
}

out = Path(f".gtsm-cache/models/variants/{variant_id}.manifest.json")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

print("Downloaded to", local_dir.resolve())
print("Wrote", out.resolve())
PY

deactivate
```

### Output
`.gtsm-cache/models/variants/deepseek-r1-distill-qwen-32b-base-local.manifest.json`:
```json
{
  "variant_id": "deepseek-r1-distill-qwen-32b-base-local",
  "schema_version": "0.1.0",
  "created_at": "2026-08-13T13:01:46.301691+00:00",
  "base_model": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
  "quantization": "none",
  "training_dataset_id": "",
  "provider": "local",
  "skills_profile": "default",
  "backend": "axolotl",
  "training_method": "full",
  "effective_training_method": "full",
  "artifact_kind": "hf_full_model",
  "artifact_dir": "/data/home/yorks7/git/galaxy-toolsmith/.gtsm-cache/models/base/deepseek-r1-distill-qwen-32b",
  "export_quantizations": [],
  "ollama_model_name": "",
  "requested_ollama_model_name": "",
  "ollama_modelfile_path": ""
}
```

## Qwen 2.5 Coder (DDP / 1GPU)
```bash
RUN_TAG=qwen-lora-sharding-sidecars-fixtures-20260727 \
PROFILE=proto-qwen25-7b \
ARTIFACT_FORMAT=mixed \
SOURCE_MODES=all-raw,all-filtered \
SOURCE_MODE_PREFERENCE=all-raw,all-filtered \
TEST_CONTEXT_MODE=fixtures \
TEST_CONTEXT_MAX_CHARS=4000 \
TEST_CONTEXT_MAX_FILES=6 \
TEST_CONTEXT_MAX_FILE_BYTES=64KB \
CONTEXT_LADDER=16k,12k,8k,4k,2k \
DISTRIBUTED_STRATEGIES=ddp \
GPU_DEVICES=1 \
GPU_REQUIRE_NO_COMPUTE_APPS=0 \
NUM_PROCESSES=1 \
PROBE_MAX_STEPS=3 \
TRAIN_GRAD_ACCUM=1 \
POST_EXPORT_ENV_DIR=.conda/gtsm-unsloth-export \
POST_EXPORT_QUANTIZATIONS=q4_k_m \
POST_OLLAMA_CREATE=1 \
scripts/gtsm_context_ladder_train.sh launch
```
**Results:** `qwen-lora-sharding-sidecars-fixtures-20260727/full-12288-all-raw-ddp.log`

## Qwen 2.5 Coder (ZeRO3 / 2GPUs)
```bash
RUN_TAG=qwen-lora2-sharding-sidecars-fixtures-20260727 \
PROFILE=proto-qwen25-7b \
ARTIFACT_FORMAT=mixed \
SOURCE_MODES=all-raw,all-filtered \
SOURCE_MODE_PREFERENCE=all-raw,all-filtered \
TEST_CONTEXT_MODE=fixtures \
TEST_CONTEXT_MAX_CHARS=4000 \
TEST_CONTEXT_MAX_FILES=6 \
TEST_CONTEXT_MAX_FILE_BYTES=64KB \
CONTEXT_LADDER=64k,32k,16k,12k,8k,4k,2k \
DISTRIBUTED_STRATEGIES=deepspeed-zero3 \
GPU_DEVICES=2,3 \
GPU_REQUIRE_NO_COMPUTE_APPS=0 \
NUM_PROCESSES=2 \
PROBE_MAX_STEPS=3 \
TRAIN_GRAD_ACCUM=1 \
POST_EXPORT_ENV_DIR=.conda/gtsm-unsloth-export \
POST_EXPORT_QUANTIZATIONS=q4_k_m \
POST_OLLAMA_CREATE=1 \
scripts/gtsm_context_ladder_train.sh launch
```
**Results:** `qwen-lora2-sharding-sidecars-fixtures-20260727/full-4096-all-raw-deepspeed-zero3.log`

## Qwen 2.5 Coder (FSDP / 4GPUs)
```bash
RUN_TAG=qwen-lora-sharding-4-fsdp-sidecars-fixtures-20260805 \
PROFILE=proto-qwen25-7b \
ARTIFACT_FORMAT=mixed \
SOURCE_MODES=all-raw,all-filtered \
SOURCE_MODE_PREFERENCE=all-raw,all-filtered \
TEST_CONTEXT_MODE=fixtures \
TEST_CONTEXT_MAX_CHARS=4000 \
TEST_CONTEXT_MAX_FILES=6 \
TEST_CONTEXT_MAX_FILE_BYTES=64KB \
CONTEXT_LADDER=64k,32k,16k,12k,8k,4k,2k \
DISTRIBUTED_STRATEGIES=fsdp \
GPU_DEVICES=0,1,2,3 \
GPU_REQUIRE_NO_COMPUTE_APPS=0 \
NUM_PROCESSES=4 \
PROBE_MAX_STEPS=3 \
TRAIN_GRAD_ACCUM=1 \
POST_EXPORT_ENV_DIR=.conda/gtsm-unsloth-export \
POST_EXPORT_QUANTIZATIONS=q4_k_m \
POST_OLLAMA_CREATE=1 \
scripts/gtsm_context_ladder_train.sh launch
```
**Results:** `qwen-lora-sharding-4-fsdp-sidecars-fixtures-20260805/full-16384-all-raw-fsdp.log`

## DeepSeek R1 (FSDP / 4GPUs)
Command
```bash
RUN_TAG=devstral-lora-sharding-4-fsdp-sidecars-fixtures-20260810 \
PROFILE=deepseek-r1-distill-qwen-32b \
ARTIFACT_FORMAT=mixed \
SOURCE_MODES=all-raw,all-filtered \
SOURCE_MODE_PREFERENCE=all-raw,all-filtered \
TEST_CONTEXT_MODE=fixtures \
TEST_CONTEXT_MAX_CHARS=4000 \
TEST_CONTEXT_MAX_FILES=6 \
TEST_CONTEXT_MAX_FILE_BYTES=64KB \
CONTEXT_LADDER=64k,32k,16k,12k,8k,4k,2k \
DISTRIBUTED_STRATEGIES=deepspeed-zero3,fsdp \
GPU_DEVICES=0,1,2,3 \
GPU_REQUIRE_NO_COMPUTE_APPS=0 \
NUM_PROCESSES=4 \
PROBE_MAX_STEPS=3 \
TRAIN_GRAD_ACCUM=1 \
POST_EXPORT_ENV_DIR=.conda/gtsm-unsloth-export \
POST_EXPORT_QUANTIZATIONS=q4_k_m \
POST_OLLAMA_CREATE=1 \
scripts/gtsm_context_ladder_train.sh launch
```
**Results:** `devstral-lora-sharding-4-fsdp-sidecars-fixtures-20260810/full-8192-all-raw-fsdp.log`


# Benchmarking Process
## Qwen 2.5 Coder Instruct (7B)
### Base Model
Command
```bash
gtsm benchmark-generate \
  --corpus-jsonl .gtsm-cache/datasets/tools-iuc-corpus.jsonl \
  --limit 50 \
  --temperature 0 \
  --model-variant qwen25-coder-7b-base-local \
  --gpu-devices 0 \
  --num-processes 1 \
  --max-workers 1 \
  --wrappers-dir .gtsm-cache/runs/benchmark/baseline.wrappers \
  --generation-records .gtsm-cache/runs/benchmark/baseline.generation.records.json \
  --evaluation-report .gtsm-cache/runs/benchmark/baseline.evaluation.summary.json \
  --benchmark-summary .gtsm-cache/runs/benchmark/baseline.summary.json \
  --checkpoint-records .gtsm-cache/runs/benchmark/baseline.checkpoint.jsonl \
  --status-log .gtsm-cache/logs/benchmark.baseline.status.jsonl \
  --resume-existing
```
Output: `baseline.summary.json`

### DDP / 1 GPU
Command
```bash
gtsm benchmark-generate \
  --corpus-jsonl .gtsm-cache/datasets/tools-iuc-corpus.jsonl \
  --limit 50 \
  --temperature 0 \
  --model-variant tools-iuc-devstral-24b-mixed-all-raw-12288-ddp-qwen-lora-sharding-sidecars-fixtures-20260727 \
  --gpu-devices 1 \
  --num-processes 1 \
  --max-workers 1 \
  --wrappers-dir .gtsm-cache/runs/benchmark/ddp0727.wrappers \
  --generation-records .gtsm-cache/runs/benchmark/ddp0727.generation.records.json \
  --evaluation-report .gtsm-cache/runs/benchmark/ddp0727.evaluation.summary.json \
  --benchmark-summary .gtsm-cache/runs/benchmark/ddp0727.summary.json \
  --checkpoint-records .gtsm-cache/runs/benchmark/ddp0727.checkpoint.jsonl \
  --status-log .gtsm-cache/logs/benchmark.ddp0727.status.jsonl \
  --resume-existing
```
Output: `ddp0727.summary.json`

### ZeRO-3 / 2 GPUs
Command
```bash
gtsm benchmark-generate \
  --corpus-jsonl .gtsm-cache/datasets/tools-iuc-corpus.jsonl \
  --limit 50 \
  --temperature 0 \
  --model-variant tools-iuc-devstral-24b-mixed-all-raw-4096-deepspeed-zero3-qwen-lora2-sharding-sidecars-fixtures-20260727 \
  --gpu-devices 2,3 \
  --num-processes 2 \
  --max-workers 1 \
  --wrappers-dir .gtsm-cache/runs/benchmark/zero30727.wrappers \
  --generation-records .gtsm-cache/runs/benchmark/zero30727.generation.records.json \
  --evaluation-report .gtsm-cache/runs/benchmark/zero30727.evaluation.summary.json \
  --benchmark-summary .gtsm-cache/runs/benchmark/zero30727.summary.json \
  --checkpoint-records .gtsm-cache/runs/benchmark/zero30727.checkpoint.jsonl \
  --status-log .gtsm-cache/logs/benchmark.zero30727.status.jsonl \
  --resume-existing
```
Output: `zero30727.summary.json`

### FSDP / 4 GPUs
Command
```bash
gtsm benchmark-generate \
  --corpus-jsonl .gtsm-cache/datasets/tools-iuc-corpus.jsonl \
  --limit 50 \
  --temperature 0 \
  --model-variant tools-iuc-devstral-24b-mixed-all-raw-16384-fsdp-qwen-lora-sharding-4-fsdp-sidecars-fixtures-20260805 \
  --gpu-devices 0,1,2,3 \
  --num-processes 4 \
  --max-workers 1 \
  --wrappers-dir .gtsm-cache/runs/benchmark/fsdp0805.wrappers \
  --generation-records .gtsm-cache/runs/benchmark/fsdp0805.generation.records.json \
  --evaluation-report .gtsm-cache/runs/benchmark/fsdp0805.evaluation.summary.json \
  --benchmark-summary .gtsm-cache/runs/benchmark/fsdp0805.summary.json \
  --checkpoint-records .gtsm-cache/runs/benchmark/fsdp0805.checkpoint.jsonl \
  --status-log .gtsm-cache/logs/benchmark.fsdp0805.status.jsonl \
  --resume-existing
```
Output: `fsdp0805.summary.json`

## DeepSeek R1 Distill Qwen (32B)
### Base Model
Command
```bash
gtsm benchmark-generate \
  --corpus-jsonl .gtsm-cache/datasets/tools-iuc-corpus.jsonl \
  --limit 50 \
  --temperature 0 \
  --model-variant deepseek-r1-distill-qwen-32b-base-local \
  --gpu-devices 0,1,2,3 \
  --num-processes 4 \
  --max-workers 1 \
  --local-gpu-topology model-parallel \
  --local-offload-policy fail \
  --wrappers-dir .gtsm-cache/runs/benchmark/baseline-deepseek.wrappers \
  --generation-records .gtsm-cache/runs/benchmark/baseline-deepseek.generation.records.json \
  --evaluation-report .gtsm-cache/runs/benchmark/baseline-deepseek.evaluation.summary.json \
  --benchmark-summary .gtsm-cache/runs/benchmark/baseline-deepseek.summary.json \
  --checkpoint-records .gtsm-cache/runs/benchmark/baseline-deepseek.checkpoint.jsonl \
  --status-log .gtsm-cache/logs/benchmark.baseline-deepseek.status.jsonl \
  --resume-existing
```
Output: `baseline-deepseek.summary.json`

### FSDP / 4GPUs
Command
```bash
gtsm benchmark-generate \
  --corpus-jsonl .gtsm-cache/datasets/tools-iuc-corpus.jsonl \
  --limit 50 \
  --temperature 0 \
  --model-variant tools-iuc-devstral-24b-mixed-all-raw-8192-fsdp-devstral-lora-sharding-4-fsdp-sidecars-fixtures-20260810 \
  --gpu-devices 0,1,2,3 \
  --num-processes 4 \
  --max-workers 1 \
  --local-gpu-topology model-parallel \
  --local-offload-policy fail \
  --wrappers-dir .gtsm-cache/runs/benchmark/fsdp0811.wrappers \
  --generation-records .gtsm-cache/runs/benchmark/fsdp0811.generation.records.json \
  --evaluation-report .gtsm-cache/runs/benchmark/fsdp0811.evaluation.summary.json \
  --benchmark-summary .gtsm-cache/runs/benchmark/fsdp0811.summary.json \
  --checkpoint-records .gtsm-cache/runs/benchmark/fsdp0811.checkpoint.jsonl \
  --status-log .gtsm-cache/logs/benchmark.fsdp0811.status.jsonl \
  --resume-existing
```
Output: `fsdp0811.summary.json`


# Promotion Process
## DDP
### Qwen 2.5 Coder (DDP / 1GPU) vs Qwen Base
```json
 gtsm promote-candidate \
  --candidate-summary .gtsm-cache/runs/benchmark/ddp0727.summary.json \
  --baseline-summary .gtsm-cache/runs/benchmark/baseline.summary.json \
  --policy staging
{
  "created_at": "2026-08-10T16:16:17.711509+00:00",
  "promote": false,
  "candidate_summary_path": "/data/home/yorks7/git/galaxy-toolsmith/.gtsm-cache/runs/benchmark/ddp0727.summary.json",
  "candidate_eval_path": "/data/home/yorks7/git/galaxy-toolsmith/.gtsm-cache/runs/benchmark/ddp0727.evaluation.summary.json",
  "baseline_summary_path": "/data/home/yorks7/git/galaxy-toolsmith/.gtsm-cache/runs/benchmark/baseline.summary.json",
  "baseline_eval_path": "/data/home/yorks7/git/galaxy-toolsmith/.gtsm-cache/runs/benchmark/baseline.evaluation.summary.json",
  "metrics": {
    "candidate": {
      "attempted": 50,
      "succeeded": 50,
      "failed": 0,
      "generation_success_rate": 1.0,
      "total_wrappers": 50,
      "xml_well_formed_count": 50,
      "xml_well_formed_rate": 1.0,
      "wrappers_with_unknown_datatypes": 18,
      "unknown_datatype_rate": 0.36,
      "xsd_status": "not_configured",
      "planemo_status": "not_run",
      "planemo_test_status": "not_run"
    },
    "baseline": {
      "attempted": 50,
      "succeeded": 50,
      "failed": 0,
      "generation_success_rate": 1.0,
      "total_wrappers": 50,
      "xml_well_formed_count": 50,
      "xml_well_formed_rate": 1.0,
      "wrappers_with_unknown_datatypes": 9,
      "unknown_datatype_rate": 0.18,
      "xsd_status": "not_configured",
      "planemo_status": "not_run",
      "planemo_test_status": "not_run"
    }
  },
  "policy": {
    "min_generation_success_rate": 0.95,
    "min_xml_well_formed_rate": 0.95,
    "max_unknown_datatype_rate": 0.1,
    "require_xsd_pass": false,
    "require_planemo_pass": false,
    "require_planemo_test_pass": false,
    "baseline_tolerance": 0.02
  },
  "reasons": [
    "Unknown datatype rate 0.360 exceeds maximum 0.100.",
    "Unknown datatype rate worsened versus baseline beyond tolerance."
  ]
}
```

### Qwen 2.5 Coder (DDP / 1GPU) vs Qwen 2.5 Coder (ZeRO-3 / 2GPUs)
```bash
gtsm promote-candidate \
  --candidate-summary .gtsm-cache/runs/benchmark/ddp0727.summary.json \
  --baseline-summary .gtsm-cache/runs/benchmark/zero30727.summary.json \
  --policy staging
{
  "created_at": "2026-08-10T16:17:14.200559+00:00",
  "promote": false,
  "candidate_summary_path": "/data/home/yorks7/git/galaxy-toolsmith/.gtsm-cache/runs/benchmark/ddp0727.summary.json",
  "candidate_eval_path": "/data/home/yorks7/git/galaxy-toolsmith/.gtsm-cache/runs/benchmark/ddp0727.evaluation.summary.json",
  "baseline_summary_path": "/data/home/yorks7/git/galaxy-toolsmith/.gtsm-cache/runs/benchmark/zero30727.summary.json",
  "baseline_eval_path": "/data/home/yorks7/git/galaxy-toolsmith/.gtsm-cache/runs/benchmark/zero30727.evaluation.summary.json",
  "metrics": {
    "candidate": {
      "attempted": 50,
      "succeeded": 50,
      "failed": 0,
      "generation_success_rate": 1.0,
      "total_wrappers": 50,
      "xml_well_formed_count": 50,
      "xml_well_formed_rate": 1.0,
      "wrappers_with_unknown_datatypes": 18,
      "unknown_datatype_rate": 0.36,
      "xsd_status": "not_configured",
      "planemo_status": "not_run",
      "planemo_test_status": "not_run"
    },
    "baseline": {
      "attempted": 50,
      "succeeded": 39,
      "failed": 11,
      "generation_success_rate": 0.78,
      "total_wrappers": 39,
      "xml_well_formed_count": 39,
      "xml_well_formed_rate": 1.0,
      "wrappers_with_unknown_datatypes": 6,
      "unknown_datatype_rate": 0.15384615384615385,
      "xsd_status": "not_configured",
      "planemo_status": "not_run",
      "planemo_test_status": "not_run"
    }
  },
  "policy": {
    "min_generation_success_rate": 0.95,
    "min_xml_well_formed_rate": 0.95,
    "max_unknown_datatype_rate": 0.1,
    "require_xsd_pass": false,
    "require_planemo_pass": false,
    "require_planemo_test_pass": false,
    "baseline_tolerance": 0.02
  },
  "reasons": [
    "Unknown datatype rate 0.360 exceeds maximum 0.100.",
    "Unknown datatype rate worsened versus baseline beyond tolerance."
  ]
}
```

### Qwen 2.5 Coder (DDP / 1GPU) vs Qwen 2.5 Coder (FSDP / 4GPUs)
```bash
 gtsm promote-candidate \
  --candidate-summary .gtsm-cache/runs/benchmark/ddp0727.summary.json \
  --baseline-summary .gtsm-cache/runs/benchmark/fsdp0805.summary.json \
  --policy staging
{
  "created_at": "2026-08-10T16:26:20.907161+00:00",
  "promote": false,
  "candidate_summary_path": "/data/home/yorks7/git/galaxy-toolsmith/.gtsm-cache/runs/benchmark/ddp0727.summary.json",
  "candidate_eval_path": "/data/home/yorks7/git/galaxy-toolsmith/.gtsm-cache/runs/benchmark/ddp0727.evaluation.summary.json",
  "baseline_summary_path": "/data/home/yorks7/git/galaxy-toolsmith/.gtsm-cache/runs/benchmark/fsdp0805.summary.json",
  "baseline_eval_path": "/data/home/yorks7/git/galaxy-toolsmith/.gtsm-cache/runs/benchmark/fsdp0805.evaluation.summary.json",
  "metrics": {
    "candidate": {
      "attempted": 50,
      "succeeded": 50,
      "failed": 0,
      "generation_success_rate": 1.0,
      "total_wrappers": 50,
      "xml_well_formed_count": 50,
      "xml_well_formed_rate": 1.0,
      "wrappers_with_unknown_datatypes": 18,
      "unknown_datatype_rate": 0.36,
      "xsd_status": "not_configured",
      "planemo_status": "not_run",
      "planemo_test_status": "not_run"
    },
    "baseline": {
      "attempted": 50,
      "succeeded": 46,
      "failed": 4,
      "generation_success_rate": 0.92,
      "total_wrappers": 46,
      "xml_well_formed_count": 46,
      "xml_well_formed_rate": 1.0,
      "wrappers_with_unknown_datatypes": 13,
      "unknown_datatype_rate": 0.2826086956521739,
      "xsd_status": "not_configured",
      "planemo_status": "not_run",
      "planemo_test_status": "not_run"
    }
  },
  "policy": {
    "min_generation_success_rate": 0.95,
    "min_xml_well_formed_rate": 0.95,
    "max_unknown_datatype_rate": 0.1,
    "require_xsd_pass": false,
    "require_planemo_pass": false,
    "require_planemo_test_pass": false,
    "baseline_tolerance": 0.02
  },
  "reasons": [
    "Unknown datatype rate 0.360 exceeds maximum 0.100.",
    "Unknown datatype rate worsened versus baseline beyond tolerance."
  ]
}
```

## ZeRO-3
### Qwen 2.5 Coder (ZeRO-3 / 2GPUs) vs Qwen Base
```json
 gtsm promote-candidate \
  --candidate-summary .gtsm-cache/runs/benchmark/zero30727.summary.json \
  --baseline-summary .gtsm-cache/runs/benchmark/baseline.summary.json \
  --policy staging
{
  "created_at": "2026-08-10T16:19:49.303147+00:00",
  "promote": false,
  "candidate_summary_path": "/data/home/yorks7/git/galaxy-toolsmith/.gtsm-cache/runs/benchmark/zero30727.summary.json",
  "candidate_eval_path": "/data/home/yorks7/git/galaxy-toolsmith/.gtsm-cache/runs/benchmark/zero30727.evaluation.summary.json",
  "baseline_summary_path": "/data/home/yorks7/git/galaxy-toolsmith/.gtsm-cache/runs/benchmark/baseline.summary.json",
  "baseline_eval_path": "/data/home/yorks7/git/galaxy-toolsmith/.gtsm-cache/runs/benchmark/baseline.evaluation.summary.json",
  "metrics": {
    "candidate": {
      "attempted": 50,
      "succeeded": 39,
      "failed": 11,
      "generation_success_rate": 0.78,
      "total_wrappers": 39,
      "xml_well_formed_count": 39,
      "xml_well_formed_rate": 1.0,
      "wrappers_with_unknown_datatypes": 6,
      "unknown_datatype_rate": 0.15384615384615385,
      "xsd_status": "not_configured",
      "planemo_status": "not_run",
      "planemo_test_status": "not_run"
    },
    "baseline": {
      "attempted": 50,
      "succeeded": 50,
      "failed": 0,
      "generation_success_rate": 1.0,
      "total_wrappers": 50,
      "xml_well_formed_count": 50,
      "xml_well_formed_rate": 1.0,
      "wrappers_with_unknown_datatypes": 9,
      "unknown_datatype_rate": 0.18,
      "xsd_status": "not_configured",
      "planemo_status": "not_run",
      "planemo_test_status": "not_run"
    }
  },
  "policy": {
    "min_generation_success_rate": 0.95,
    "min_xml_well_formed_rate": 0.95,
    "max_unknown_datatype_rate": 0.1,
    "require_xsd_pass": false,
    "require_planemo_pass": false,
    "require_planemo_test_pass": false,
    "baseline_tolerance": 0.02
  },
  "reasons": [
    "Generation success rate 0.780 is below minimum 0.950.",
    "Unknown datatype rate 0.154 exceeds maximum 0.100.",
    "Generation success rate regressed versus baseline beyond tolerance."
  ]
}
```

### Qwen 2.5 Coder (ZeRO-3 / 2GPUs) vs Qwen 2.5 Coder (FSDP / 4GPUs)
```json
 gtsm promote-candidate \
  --candidate-summary .gtsm-cache/runs/benchmark/zero30727.summary.json \
  --baseline-summary .gtsm-cache/runs/benchmark/fsdp0805.summary.json \
  --policy staging
{
  "created_at": "2026-08-10T16:23:36.571744+00:00",
  "promote": false,
  "candidate_summary_path": "/data/home/yorks7/git/galaxy-toolsmith/.gtsm-cache/runs/benchmark/zero30727.summary.json",
  "candidate_eval_path": "/data/home/yorks7/git/galaxy-toolsmith/.gtsm-cache/runs/benchmark/zero30727.evaluation.summary.json",
  "baseline_summary_path": "/data/home/yorks7/git/galaxy-toolsmith/.gtsm-cache/runs/benchmark/fsdp0805.summary.json",
  "baseline_eval_path": "/data/home/yorks7/git/galaxy-toolsmith/.gtsm-cache/runs/benchmark/fsdp0805.evaluation.summary.json",
  "metrics": {
    "candidate": {
      "attempted": 50,
      "succeeded": 39,
      "failed": 11,
      "generation_success_rate": 0.78,
      "total_wrappers": 39,
      "xml_well_formed_count": 39,
      "xml_well_formed_rate": 1.0,
      "wrappers_with_unknown_datatypes": 6,
      "unknown_datatype_rate": 0.15384615384615385,
      "xsd_status": "not_configured",
      "planemo_status": "not_run",
      "planemo_test_status": "not_run"
    },
    "baseline": {
      "attempted": 50,
      "succeeded": 46,
      "failed": 4,
      "generation_success_rate": 0.92,
      "total_wrappers": 46,
      "xml_well_formed_count": 46,
      "xml_well_formed_rate": 1.0,
      "wrappers_with_unknown_datatypes": 13,
      "unknown_datatype_rate": 0.2826086956521739,
      "xsd_status": "not_configured",
      "planemo_status": "not_run",
      "planemo_test_status": "not_run"
    }
  },
  "policy": {
    "min_generation_success_rate": 0.95,
    "min_xml_well_formed_rate": 0.95,
    "max_unknown_datatype_rate": 0.1,
    "require_xsd_pass": false,
    "require_planemo_pass": false,
    "require_planemo_test_pass": false,
    "baseline_tolerance": 0.02
  },
  "reasons": [
    "Generation success rate 0.780 is below minimum 0.950.",
    "Unknown datatype rate 0.154 exceeds maximum 0.100.",
    "Generation success rate regressed versus baseline beyond tolerance."
  ]
}
```



## FSDP (Qwen)
### Qwen 2.5 Coder (FSDP / 4GPUs) vs Qwen Base
```bash
gtsm promote-candidate \
  --candidate-summary .gtsm-cache/runs/benchmark/fsdp0805.summary.json \
  --baseline-summary .gtsm-cache/runs/benchmark/baseline.summary.json \
  --policy staging
{
  "created_at": "2026-08-10T16:26:51.452323+00:00",
  "promote": false,
  "candidate_summary_path": "/data/home/yorks7/git/galaxy-toolsmith/.gtsm-cache/runs/benchmark/fsdp0805.summary.json",
  "candidate_eval_path": "/data/home/yorks7/git/galaxy-toolsmith/.gtsm-cache/runs/benchmark/fsdp0805.evaluation.summary.json",
  "baseline_summary_path": "/data/home/yorks7/git/galaxy-toolsmith/.gtsm-cache/runs/benchmark/baseline.summary.json",
  "baseline_eval_path": "/data/home/yorks7/git/galaxy-toolsmith/.gtsm-cache/runs/benchmark/baseline.evaluation.summary.json",
  "metrics": {
    "candidate": {
      "attempted": 50,
      "succeeded": 46,
      "failed": 4,
      "generation_success_rate": 0.92,
      "total_wrappers": 46,
      "xml_well_formed_count": 46,
      "xml_well_formed_rate": 1.0,
      "wrappers_with_unknown_datatypes": 13,
      "unknown_datatype_rate": 0.2826086956521739,
      "xsd_status": "not_configured",
      "planemo_status": "not_run",
      "planemo_test_status": "not_run"
    },
    "baseline": {
      "attempted": 50,
      "succeeded": 50,
      "failed": 0,
      "generation_success_rate": 1.0,
      "total_wrappers": 50,
      "xml_well_formed_count": 50,
      "xml_well_formed_rate": 1.0,
      "wrappers_with_unknown_datatypes": 9,
      "unknown_datatype_rate": 0.18,
      "xsd_status": "not_configured",
      "planemo_status": "not_run",
      "planemo_test_status": "not_run"
    }
  },
  "policy": {
    "min_generation_success_rate": 0.95,
    "min_xml_well_formed_rate": 0.95,
    "max_unknown_datatype_rate": 0.1,
    "require_xsd_pass": false,
    "require_planemo_pass": false,
    "require_planemo_test_pass": false,
    "baseline_tolerance": 0.02
  },
  "reasons": [
    "Generation success rate 0.920 is below minimum 0.950.",
    "Unknown datatype rate 0.283 exceeds maximum 0.100.",
    "Generation success rate regressed versus baseline beyond tolerance.",
    "Unknown datatype rate worsened versus baseline beyond tolerance."
  ]
}
```





## FSDP (DeepSeek R1)
### DeepSeek R1 (FSDP / 4GPUs) vs Qwen Base
Command
```bash
gtsm promote-candidate \
  --candidate-summary .gtsm-cache/runs/benchmark/fsdp0811.summary.json \
  --baseline-summary .gtsm-cache/runs/benchmark/baseline.summary.json \
  --policy staging
```
Output
```json
{
  "created_at": "2026-08-13T15:58:37.542074+00:00",
  "promote": false,
  "candidate_summary_path": "/data/home/yorks7/git/galaxy-toolsmith/.gtsm-cache/runs/benchmark/fsdp0811.summary.json",
  "candidate_eval_path": "/data/home/yorks7/git/galaxy-toolsmith/.gtsm-cache/runs/benchmark/fsdp0811.evaluation.summary.json",
  "baseline_summary_path": "/data/home/yorks7/git/galaxy-toolsmith/.gtsm-cache/runs/benchmark/baseline.summary.json",
  "baseline_eval_path": "/data/home/yorks7/git/galaxy-toolsmith/.gtsm-cache/runs/benchmark/baseline.evaluation.summary.json",
  "metrics": {
    "candidate": {
      "attempted": 50,
      "succeeded": 25,
      "failed": 25,
      "generation_success_rate": 0.5,
      "total_wrappers": 25,
      "xml_well_formed_count": 25,
      "xml_well_formed_rate": 1.0,
      "wrappers_with_unknown_datatypes": 3,
      "unknown_datatype_rate": 0.12,
      "xsd_status": "not_configured",
      "planemo_status": "not_run",
      "planemo_test_status": "not_run"
    },
    "baseline": {
      "attempted": 50,
      "succeeded": 50,
      "failed": 0,
      "generation_success_rate": 1.0,
      "total_wrappers": 50,
      "xml_well_formed_count": 50,
      "xml_well_formed_rate": 1.0,
      "wrappers_with_unknown_datatypes": 9,
      "unknown_datatype_rate": 0.18,
      "xsd_status": "not_configured",
      "planemo_status": "not_run",
      "planemo_test_status": "not_run"
    }
  },
  "policy": {
    "min_generation_success_rate": 0.95,
    "min_xml_well_formed_rate": 0.95,
    "max_unknown_datatype_rate": 0.1,
    "require_xsd_pass": false,
    "require_planemo_pass": false,
    "require_planemo_test_pass": false,
    "baseline_tolerance": 0.02
  },
  "reasons": [
    "Generation success rate 0.500 is below minimum 0.950.",
    "Unknown datatype rate 0.120 exceeds maximum 0.100.",
    "Generation success rate regressed versus baseline beyond tolerance."
  ]
}
```

### DeepSeek R1 (FSDP / 4GPUs) vs DeepSeek Base
Command
```bash
gtsm promote-candidate \
  --candidate-summary .gtsm-cache/runs/benchmark/fsdp0811.summary.json \
  --baseline-summary .gtsm-cache/runs/benchmark/baseline-deepseek.summary.json \
  --policy staging
```
Output
```json
{
  "created_at": "2026-08-13T16:02:55.068562+00:00",
  "promote": false,
  "candidate_summary_path": "/data/home/yorks7/git/galaxy-toolsmith/.gtsm-cache/runs/benchmark/fsdp0811.summary.json",
  "candidate_eval_path": "/data/home/yorks7/git/galaxy-toolsmith/.gtsm-cache/runs/benchmark/fsdp0811.evaluation.summary.json",
  "baseline_summary_path": "/data/home/yorks7/git/galaxy-toolsmith/.gtsm-cache/runs/benchmark/baseline-deepseek.summary.json",
  "baseline_eval_path": "/data/home/yorks7/git/galaxy-toolsmith/.gtsm-cache/runs/benchmark/baseline-deepseek.evaluation.summary.json",
  "metrics": {
    "candidate": {
      "attempted": 50,
      "succeeded": 25,
      "failed": 25,
      "generation_success_rate": 0.5,
      "total_wrappers": 25,
      "xml_well_formed_count": 25,
      "xml_well_formed_rate": 1.0,
      "wrappers_with_unknown_datatypes": 3,
      "unknown_datatype_rate": 0.12,
      "xsd_status": "not_configured",
      "planemo_status": "not_run",
      "planemo_test_status": "not_run"
    },
    "baseline": {
      "attempted": 50,
      "succeeded": 50,
      "failed": 0,
      "generation_success_rate": 1.0,
      "total_wrappers": 50,
      "xml_well_formed_count": 50,
      "xml_well_formed_rate": 1.0,
      "wrappers_with_unknown_datatypes": 14,
      "unknown_datatype_rate": 0.28,
      "xsd_status": "not_configured",
      "planemo_status": "not_run",
      "planemo_test_status": "not_run"
    }
  },
  "policy": {
    "min_generation_success_rate": 0.95,
    "min_xml_well_formed_rate": 0.95,
    "max_unknown_datatype_rate": 0.1,
    "require_xsd_pass": false,
    "require_planemo_pass": false,
    "require_planemo_test_pass": false,
    "baseline_tolerance": 0.02
  },
  "reasons": [
    "Generation success rate 0.500 is below minimum 0.950.",
    "Unknown datatype rate 0.120 exceeds maximum 0.100.",
    "Generation success rate regressed versus baseline beyond tolerance."
  ]
}
```