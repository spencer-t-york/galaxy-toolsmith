# GTSM context ladder run qwen-sharding-sidecars-fixtures-20260724

- Status: estimated
- Profile: full-qwen25-7b
- Artifact format: mixed
- Context ladder: 32k,16k,12k,8k,4k,2k
- Source modes: all-raw,all-filtered
- Test context: fixtures, max chars 4000, max files 6, max file bytes 64KB
- Strategies: fsdp,deepspeed-zero3
- External GGUF export env: .conda/gtsm-unsloth-export
- External llama.cpp dir: .gtsm-cache/llama.cpp
- External export quantizations: q4_k_m
- Approximate estimate: .gtsm-cache/runs/context-ladder/qwen-sharding-sidecars-fixtures-20260724/estimate.approx.json
- Exact estimate: .gtsm-cache/runs/context-ladder/qwen-sharding-sidecars-fixtures-20260724/estimate.exact.json
- Candidates: .gtsm-cache/runs/context-ladder/qwen-sharding-sidecars-fixtures-20260724/candidates.tsv
- Probe report: .gtsm-cache/runs/context-ladder/qwen-sharding-sidecars-fixtures-20260724/probes.tsv
- Training report: .gtsm-cache/runs/context-ladder/qwen-sharding-sidecars-fixtures-20260724/full-training.tsv
