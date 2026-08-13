# GTSM context ladder run qwen-lora-sharding-sidecars-fixtures-20260727

- Status: post-export-failed
- Profile: proto-qwen25-7b
- Artifact format: mixed
- Context ladder: 16k,12k,8k,4k,2k
- Source modes: all-raw,all-filtered
- Test context: fixtures, max chars 4000, max files 6, max file bytes 64KB
- Strategies: ddp
- External GGUF export env: .conda/gtsm-unsloth-export
- External llama.cpp dir: .gtsm-cache/llama.cpp
- External export quantizations: q4_k_m
- Approximate estimate: .gtsm-cache/runs/context-ladder/qwen-lora-sharding-sidecars-fixtures-20260727/estimate.approx.json
- Exact estimate: .gtsm-cache/runs/context-ladder/qwen-lora-sharding-sidecars-fixtures-20260727/estimate.exact.json
- Candidates: .gtsm-cache/runs/context-ladder/qwen-lora-sharding-sidecars-fixtures-20260727/candidates.tsv
- Probe report: .gtsm-cache/runs/context-ladder/qwen-lora-sharding-sidecars-fixtures-20260727/probes.tsv
- Training report: .gtsm-cache/runs/context-ladder/qwen-lora-sharding-sidecars-fixtures-20260727/full-training.tsv

## Selection
- SELECTED_MAX_SEQ_LENGTH=8192
- SELECTED_SOURCE_MODE=all-filtered
- SELECTED_SOURCE_MAX_CHARS=12000
- SELECTED_SOURCE_MAX_FILES=64
- SELECTED_DISTRIBUTED_STRATEGY=ddp
- SELECTED_VARIANT_ID=tools-iuc-devstral-24b-mixed-all-filtered-8192-ddp-qwen-lora-sharding-sidecars-fixtures-20260727
- SELECTED_RUN_ID=train-qwen-lora-sharding-sidecars-fixtures-20260727-full-8192-all-filtered-ddp
