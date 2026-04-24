# Kimi K2.6 Expert-Pruning Plan (private)

Target: `moonshotai/Kimi-K2.6` (1T total params, 32B active via top-8 of 384 routed experts + 1 shared). KimiK25 VL wrapper over DeepseekV3 text backbone. MLA attention, 61 layers (first dense, 60 MoE), 256K context.

Storage on HF: **595 GB** across 64 shards. Experts are **already INT4-packed** (8 int4s per int32, BF16 group_size=32 scales) via `compressed-tensors`. Non-MoE tensors (attention, norms, embeddings, first-dense-MLP) stored as BF16.

## Pipeline

```
corpus (build_calib.py) -> routing profile (profile.py) -> score/plan (score.py)
        -> prune + absorb-merge (prune.py) -> bench gate (bench.py)
```

## Goal & gates

Iterate 30% → 40% → 50% (→ 60% if gates still pass). Stop at first ratio where any domain gate fails; optionally heal-finetune and retry.

Gate thresholds (rel delta vs unpruned baseline):
- coding (HumanEval pass@1): ≥ -5%
- tool (BFCL AST-match): ≥ -5%
- agentic (SWE-bench_Lite 10-sample patch similarity): ≥ -10%
- pentest (custom 50-MCQ): ≥ -10%
- general (MMLU 500): ≥ -3%
- chinese (C-Eval 200): ≥ -5%

## Calibration corpus (5M tokens pilot, 20M for production)

Domains (tokens, mix %):
- coding 22% — Magicoder OSS + CodeFeedback + Evol-Instruct-Code + CodeAlpaca + python_code_instructions
- cybersec 18% — CyberNative DPO + ShellCommands + Lily-Cybersecurity + pentesting-eval
- agentic 18% — xlam-function-calling + glaive-FC-v2 + AgentInstruct + SWE-bench_oracle
- general 20% — tulu-3 SFT + OpenHermes-2.5 + ultrachat + arxiv-summarization
- systems 8% — sql-create-context + dolphin-coder
- chinese 8% — COIG-CQIA (ruozhiba) + ShareGPT-Chinese-English-90k
- longctx 6% — deepmind/pg19

## Routing profile

Streams tokens through MLX-loaded model (deepseek_v3 shim on text backbone, vision tower skipped). Hooks each MoE layer's router forward to capture:
- freq[L, e] — selection count (normalized to fraction)
- weighted_freq[L, e] — Σ post-renorm gate score when selected (mass contribution)
- coact[L, e, f] — co-selection count for merge-partner identification
- (optional) output_energy[L, e] — ‖expert(x)‖ for stronger importance signal

Output: `routing_profile.safetensors` + sidecar JSON with token counts, layer names, topk.

## Importance scoring

`score[e] = α*norm(weighted_freq) + β*norm(freq) + γ*norm(energy)`
Defaults α=1.0, β=0.2, γ=0.1. All normalized to each term's own max.

Adaptive per-layer prune ratio: ratio scaled by router entropy — high-entropy (uniform routing) layers get lower effective prune ratio since dropping more hurts more there.

## Absorb-merge (drop → merged convex blend)

For each dropped expert `e`, find kept expert `k` with max `coact[e, k]`. Blend:
```
W_k_new = w_keep * W_k + w_drop * W_e    with damping d, w_drop = d/(1+d), w_keep = 1/(1+d)
```
Default d=0.5 (dropped contributes 1/3 weight). Operates in dequantized BF16/F32, then re-quantizes to INT4 with fresh group-wise scale.

Multiple drops merging into the same kept target are applied sequentially in coact-descending order; rescaling happens once at the end.

## Output model

Same directory layout as source, same `.safetensors` shard count (64). Difference:
- Dropped expert tensor keys removed
- Kept experts renumbered 0..n_keep-1
- Router weight + bias rewritten with only kept rows
- config.json text_config.n_routed_experts updated (or per-layer sidecar)

Estimated sizes (rough):
- 30% prune: ~420 GB (drops 115 experts/layer)
- 40% prune: ~360 GB
- 50% prune: ~300 GB

Subsequent JANGTQ conversion would bring these to ~100 / 85 / 70 GB at JANGTQ_2L. Potentially shippable on a 128 GB MacBook Pro at 50% + JANGTQ_2L.

## Known gaps / TODO when work resumes

1. `build_pentest_mcq.py` — need to author the 50-question OSCP/eJPT-style MCQ set from public sources (HTB/THM writeups, OWASP, offsec-certs). Manual curation. Stored in `assets/pentest_mcq.jsonl`.
2. Profile module's `_load_model_mlx` shim: assumes mlx_lm's deepseek_v3.py handles the kimi_k2 text backbone. Likely needs small adaptation — rope scaling factor, first_k_dense_replace respect, sigmoid+bias router are all DSV3 stock. MLA is too. Should mostly work.
3. `output_energy` capture requires running expert forwards, not just router. Not wired up in v1 profile (set to zeros). Add via a second hook on `switch_mlp` after initial router-only run validates.
4. Router rewrite assumes `gate.weight` is BF16 and accepts direct row-slice. Confirmed from shard headers; no quantization on router.
5. INT4 unpacker assumes "8 int4s per int32, lsb-first". Verified shape-wise from shard30 (`weight_packed [2048, 896]` unpacks to `[2048, 7168]` — 7168/896=8). Sign convention (signed 4-bit, -8..7) is the compressed-tensors default but worth verifying with a round-trip test on one expert before running full prune.
6. Eval harness uses `mlx_lm.generate`; tokenizer is Kimi's tiktoken (via trust_remote_code). Should just work once tokenizer files land.
