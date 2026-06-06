# MiMo V2.5 JANG_2L runtime note - 2026-06-06

Current local bundle:

- `/Users/eric/.mlxstudio/models/JANGQ-AI/MiMo-V2.5-JANG_2L`
- Source/intake path used by vMLX audit: `erics-m5-max2.local:/Users/eric/.mlxstudio/models/JANGQ-AI/MiMo-V2.5-JANG_2L`
- Structural manifest verification passed in vMLX artifact `build/current-mimo-jang2l-local-structural-verify-20260606.json`.

Runtime facts:

- MiMo V2 uses asymmetric full/SWA attention: full layers use 4 KV heads, SWA layers use 8 KV heads, `sliding_window=128`, and SWA attention sink bias is present.
- vMLX now routes `cache_subtype="mimo_v2_asymmetric_swa"` into the mixed full/SWA KV cache contract. That fixes cache classification, not model quality.
- Direct `mlx_lm` generation still corrupts above the SWA-window prompt threshold, so the remaining blocker is not only vMLX server scheduling.

Sink A/B probe:

- Script: `jang-tools/scripts/mimo_v2/sink_ab_probe.py`
- Artifact: `/Users/eric/mlx/vllm-mlx-finite-launch-guard/build/current-mimo-v2-jang2l-sink-above-swa-probe-20260606.json`
- Prompt length: `303` tokens.
- Native MLX `sinks=` path: failed with punctuation output.
- Manual source-equivalent sink softmax: failed with repetition/CJK output.
- Sink disabled: failed with punctuation output.

Conclusion:

- Do not classify the remaining MiMo corruption as only a native MLX `sinks=` kernel bug.
- Do not disable sinks as a fake fix.
- Next proof should compare the local quantized JANG_2L path against a known-good source or higher-quality MiMo profile, and then trace first divergence through attention vs routed expert quantized forward.
