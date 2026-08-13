# GTSM context ladder run qwen-lora-sharding-4-fsdp-sidecars-fixtures-20260805

- Status: post-export-failed
- Profile: proto-qwen25-7b
- Artifact format: mixed
- Context ladder: 64k,32k,16k,12k,8k,4k,2k
- Source modes: all-raw,all-filtered
- Test context: fixtures, max chars 4000, max files 6, max file bytes 64KB
- Strategies: fsdp
- External GGUF export env: .conda/gtsm-unsloth-export
- External llama.cpp dir: .gtsm-cache/llama.cpp
- External export quantizations: q4_k_m
- Approximate estimate: .gtsm-cache/runs/context-ladder/qwen-lora-sharding-4-fsdp-sidecars-fixtures-20260805/estimate.approx.json
- Exact estimate: .gtsm-cache/runs/context-ladder/qwen-lora-sharding-4-fsdp-sidecars-fixtures-20260805/estimate.exact.json
- Candidates: .gtsm-cache/runs/context-ladder/qwen-lora-sharding-4-fsdp-sidecars-fixtures-20260805/candidates.tsv
- Probe report: .gtsm-cache/runs/context-ladder/qwen-lora-sharding-4-fsdp-sidecars-fixtures-20260805/probes.tsv
- Training report: .gtsm-cache/runs/context-ladder/qwen-lora-sharding-4-fsdp-sidecars-fixtures-20260805/full-training.tsv

## Selection
- SELECTED_MAX_SEQ_LENGTH=16384
- SELECTED_SOURCE_MODE=all-raw
- SELECTED_SOURCE_MAX_CHARS=36000
- SELECTED_SOURCE_MAX_FILES=128
- SELECTED_DISTRIBUTED_STRATEGY=fsdp
- SELECTED_VARIANT_ID=tools-iuc-devstral-24b-mixed-all-raw-16384-fsdp-qwen-lora-sharding-4-fsdp-sidecars-fixtures-20260805
- SELECTED_RUN_ID=train-qwen-lora-sharding-4-fsdp-sidecars-fixtures-20260805-full-16384-all-raw-fsdp
