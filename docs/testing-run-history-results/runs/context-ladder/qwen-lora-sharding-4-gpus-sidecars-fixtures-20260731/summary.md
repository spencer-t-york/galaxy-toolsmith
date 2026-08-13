# GTSM context ladder run qwen-lora-sharding-4-gpus-sidecars-fixtures-20260731

- Status: estimated
- Profile: proto-qwen25-7b
- Artifact format: mixed
- Context ladder: 64k,32k,16k,12k,8k,4k,2k
- Source modes: all-raw,all-filtered
- Test context: fixtures, max chars 4000, max files 6, max file bytes 64KB
- Strategies: deepspeed-zero3
- External GGUF export env: .conda/gtsm-unsloth-export
- External llama.cpp dir: .gtsm-cache/llama.cpp
- External export quantizations: q4_k_m
- Approximate estimate: .gtsm-cache/runs/context-ladder/qwen-lora-sharding-4-gpus-sidecars-fixtures-20260731/estimate.approx.json
- Exact estimate: .gtsm-cache/runs/context-ladder/qwen-lora-sharding-4-gpus-sidecars-fixtures-20260731/estimate.exact.json
- Candidates: .gtsm-cache/runs/context-ladder/qwen-lora-sharding-4-gpus-sidecars-fixtures-20260731/candidates.tsv
- Probe report: .gtsm-cache/runs/context-ladder/qwen-lora-sharding-4-gpus-sidecars-fixtures-20260731/probes.tsv
- Training report: .gtsm-cache/runs/context-ladder/qwen-lora-sharding-4-gpus-sidecars-fixtures-20260731/full-training.tsv
