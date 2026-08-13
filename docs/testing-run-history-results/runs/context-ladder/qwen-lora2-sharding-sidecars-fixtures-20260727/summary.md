# GTSM context ladder run qwen-lora2-sharding-sidecars-fixtures-20260727

- Status: post-export-failed
- Profile: proto-qwen25-7b
- Artifact format: mixed
- Context ladder: 64k,32k,16k,12k,8k,4k,2k
- Source modes: all-raw,all-filtered
- Test context: fixtures, max chars 4000, max files 6, max file bytes 64KB
- Strategies: deepspeed-zero3
- External GGUF export env: .conda/gtsm-unsloth-export
- External llama.cpp dir: .gtsm-cache/llama.cpp
- External export quantizations: q4_k_m
- Approximate estimate: .gtsm-cache/runs/context-ladder/qwen-lora2-sharding-sidecars-fixtures-20260727/estimate.approx.json
- Exact estimate: .gtsm-cache/runs/context-ladder/qwen-lora2-sharding-sidecars-fixtures-20260727/estimate.exact.json
- Candidates: .gtsm-cache/runs/context-ladder/qwen-lora2-sharding-sidecars-fixtures-20260727/candidates.tsv
- Probe report: .gtsm-cache/runs/context-ladder/qwen-lora2-sharding-sidecars-fixtures-20260727/probes.tsv
- Training report: .gtsm-cache/runs/context-ladder/qwen-lora2-sharding-sidecars-fixtures-20260727/full-training.tsv

## Selection
- SELECTED_MAX_SEQ_LENGTH=4096
- SELECTED_SOURCE_MODE=all-raw
- SELECTED_SOURCE_MAX_CHARS=6000
- SELECTED_SOURCE_MAX_FILES=32
- SELECTED_DISTRIBUTED_STRATEGY=deepspeed-zero3
- SELECTED_VARIANT_ID=tools-iuc-devstral-24b-mixed-all-raw-4096-deepspeed-zero3-qwen-lora2-sharding-sidecars-fixtures-20260727
- SELECTED_RUN_ID=train-qwen-lora2-sharding-sidecars-fixtures-20260727-full-4096-all-raw-deepspeed-zero3
