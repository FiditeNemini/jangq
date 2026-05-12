# M5 JANGTQ TensorOps Production Plan

Date: 2026-05-12
Scope: `jang-tools` TurboQuant/JANGTQ kernels used by vMLX Python and Swift.

This is the committable handoff for the M5/JANGTQ speed lane. The longer
working note is also present at:

`research/M5_JANGTQ_TENSOROPS_PRODUCTION_PLAN_2026_05_12.md`

That `research/` path is ignored by this repo, so commit this root file or
force-add the research file if the long note should be preserved.

## Confirmed

- MLX custom kernels can call Metal/M5 `mpp::tensor_ops::matmul2d`.
- True JANGTQ cannot be replaced by `mx.quantized_matmul` directly, because
  JANGTQ is Hadamard/sign rotation + Lloyd-Max codebook + per-row norm, not
  MLX affine scales/biases.
- Direct codebook TensorOps matmul is working without dense materialization.
- Current synthetic proof shows plain TQ matmul speedups around 1.4x-2.7x.
- Grouped routed gather wins once there are enough same-expert rows:
  about 1.6x at 256 dispatches, 2.18x at 1024 dispatches, and 2.51x for a
  2048x2048 grouped case.
- The current proof hooks do **not** speed up the full expert cluster. A fresh
  synthetic cluster probe shows relative parity around `5e-4`, but speedups
  below `1.0x` because the proof path loses fusion and still pays grouping
  overhead. This confirms the production work must be fused/grouped, not just
  flipping `JANGTQ_MPP_NAX=1`.
- A first grouped fused gate/up/SwiGLU implementation is now wired for sorted
  prefill rows, with single-entry tile metadata reuse shared by gate/up and
  down. The cache now preserves sorted `uint32` index object identity across
  fused gate/up and down gather, preventing duplicate CPU tile-metadata builds
  in the normal sorted expert-cluster path.
- Current `auto` probes show real full-cluster synthetic wins, but they are
  geometry-dependent:
  - square 2048x2048, top-k 8, 2048-token synthetic cluster: `3.35x`
  - Qwen3.6-35B-A3B-like 2048->512, top-k 8 component: `2.41x`
  - ZAYA1-VL-like 2048->4096, top-k 1 component: `2.48x`
  - MiniMax-like 3072->1536, top-k 8, 2-bit component: `1.89x`
- A precomputed-tile upper-bound probe crosses `3x-4x` on high-dispatch
  top-k 8 synthetic cases, but the normal wrapper path is lower until tile
  metadata moves fully out of the CPU-sync path.

## Not Confirmed

- 3-4x whole-model prompt-processing speedup.
- decode token/s speedup.
- safe global default enablement of `JANGTQ_MPP_NAX=1`.
- real-model long-prompt server/app speedup. The current proof is kernel and
  synthetic component proof plus source-server/API gates. Installed-app GUI
  proof is still separate.

Do not claim 3-4x until a real long-prompt model gate proves it.

## Amdahl Bound

Total speedup:

```text
S_total = 1 / ((1 - f) + f / S_kernel)
```

where `f` is the fraction of PP time in the optimized region.

| Optimized-region speedup | 3x total requires | 4x total requires |
| ---: | ---: | ---: |
| 2x | impossible | impossible |
| 3x | 100% | impossible |
| 4x | 88.9% | 100% |
| 6x | 80.0% | 90.0% |
| 8x | 76.2% | 85.7% |

Therefore a 3-4x whole-model PP target requires optimizing the whole expert
prefill cluster, not only one TQ matmul.

Repeatable calculator:

```sh
cd /Users/eric/jang/jang-tools
python3 jangtq_speed_ceiling.py \
  --optimized-speedup 2 \
  --optimized-speedup 4 \
  --optimized-speedup 6 \
  --optimized-speedup 8 \
  --fraction 0.8 \
  --fraction 0.9
```

Current output proves that 3x requires either about 80% of PP time sped up by
6x or about 76.2% sped up by 8x. 4x requires about 90% of PP time sped up by
6x or about 85.7% sped up by 8x.

## Files In The Current Local Prototype

Commit these together if accepting the prototype. Several are ignored by
`.gitignore`, so use `git add -f` or fix the ignore rules.

- `jang_tools/turboquant/mpp_nax_kernel.py`
- `jang_tools/turboquant/mpp_dense_kernel.py`
- `tests/test_turboquant_mpp_nax_kernel.py`
- `tests/test_turboquant_mpp_dense_kernel.py`
- `examples/turboquant/mpp_dense_probe.py`
- `examples/turboquant/mpp_prefill_mlp_probe.py`
- `jangtq_expert_cluster_probe.py`
- `jangtq_live_nax_probe.py`
- `m5_jangtq_tensorops_component_probe_2026_05_12.json`
- `qwen36_jangtq4_nax_live_2k_2026_05_12.json`
- `qwen36_jangtq4_nax_live_8k_2026_05_12.json`

Tracked hook files currently dirty:

- `jang_tools/turboquant/tq_kernel.py`
- `jang_tools/turboquant/gather_tq_kernel.py`
- `jang_tools/turboquant/fused_gate_up_kernel.py`

## Production Method

1. Add `JANGTQ_MPP_NAX=0|1|auto`, not a blind boolean.
2. `auto` should require:
   - `mpp_nax_tensorops_available()`
   - bits in `{2,3,4,8}`
   - prefill-shaped batch/dispatch count
   - sorted routed rows for MoE gather/down
3. Keep current shader kernels for small decode and fragmented routed traffic.
4. Replace CPU-built grouped metadata with a production path:
   - GPU metadata builder, or
   - fixed sorted tiles with boundary repair, or
   - padded per-expert buckets.
5. Build a fused grouped gate/up/SwiGLU NAX kernel:
   - one A tile load
   - gate and up TensorOps in one launch
   - in-kernel limited SwiGLU
   - write only activated `x_act`
   - first implementation is present for sorted prefill rows
6. Collapse the expert-prefill cluster if aiming for 3-4x:
   - gate/up Hadamard
   - gate/up projections
   - SwiGLU
   - down Hadamard
   - down projection
   - sort/scatter/unsort
   - intermediate allocation

## Required Gates

Synthetic correctness:

```sh
cd /Users/eric/jang/jang-tools
/Users/eric/mlx/vllm-mlx/.venv/bin/python -m pytest -q \
  tests/test_turboquant_mpp_nax_kernel.py \
  tests/test_turboquant_mpp_dense_kernel.py \
  tests/test_turboquant_kernel_source_contract.py \
  tests/test_turboquant_codebook.py \
  tests/test_turboquant_retiled_layout.py
```

Current result after grouped fused gate/up and `auto` gates: `36 passed`.

Synthetic timing gates:

- plain TQ direct NAX
- grouped routed down
- fused grouped gate/up/SwiGLU
- small decode cases proving `auto` avoids slower NAX paths
- full expert cluster using `jangtq_expert_cluster_probe.py`

Fresh current-hook cluster probe:

```sh
cd /Users/eric/jang/jang-tools
/Users/eric/mlx/vllm-mlx/.venv/bin/python jangtq_expert_cluster_probe.py \
  --repeat 2 \
  --case 32,2,1024,1024,8,4 \
  --case 128,2,1024,1024,8,4 \
  --case 512,2,1024,1024,8,4 \
  --case 128,2,2048,2048,8,4
```

Current proof-hook result:

| Dispatches | Shape | Baseline | Current NAX hook | Speedup | rel L2 |
| ---: | --- | ---: | ---: | ---: | ---: |
| 256 | 1024x1024 | 1.839 ms | 0.840 ms | 2.19x | 0.0 |
| 1024 | 1024x1024 | 2.756 ms | 1.483 ms | 1.86x | 0.000512 |
| 256 | 2048x2048 | 4.112 ms | 2.276 ms | 1.81x | 0.000525 |
| 1024 | 2048x2048 | 13.307 ms | 5.946 ms | 2.24x | 0.000524 |

High-dispatch synthetic cluster probes:

| Dispatches | Shape | Path | Speedup | rel L2 |
| ---: | --- | --- | ---: | ---: |
| 16384 | 1024x1024, top-k 8 | precomputed tile upper bound | 3.61x | 0.000513 |
| 32768 | 1024x1024, top-k 8 | precomputed tile upper bound | 4.37x | 0.000512 |
| 16384 | 2048x2048, top-k 8 | normal wrapper `auto` | 3.35x | 0.000523 |

Model-like sorted expert-MLP component probe:

```sh
cd /Users/eric/jang/jang-tools
/Users/eric/mlx/vllm-mlx/.venv/bin/python \
  examples/turboquant/mpp_prefill_mlp_probe.py \
  --repeat 2 \
  --json-out m5_jangtq_tensorops_component_probe_2026_05_12.json \
  --case qwen36a3b,2048,8,256,2048,512,4 \
  --case zaya1vl,2048,1,16,2048,4096,4 \
  --case minimax,2048,8,256,3072,1536,2
```

| Geometry | Current MLP | Grouped MPP/NAX MLP | Speedup | Max abs err |
| --- | ---: | ---: | ---: | ---: |
| Qwen3.6 A3B-like, 2048->512, top-k 8, 256 experts | 40.030 ms | 16.616 ms | 2.41x | 0.0115 |
| ZAYA1-VL-like, 2048->4096, top-k 1, 16 experts | 34.640 ms | 13.972 ms | 2.48x | 0.0100 |
| MiniMax-like, 3072->1536, top-k 8, 256 experts, 2-bit | 188.684 ms | 99.761 ms | 1.89x | 0.0039 |

Live model gate, real Qwen3.6-35B-A3B-JANGTQ4:

```sh
cd /Users/eric/jang/jang-tools
PYTHONPATH=/Users/eric/jang/jang-tools \
  /Users/eric/mlx/vllm-mlx/.venv/bin/python \
  jangtq_live_nax_probe.py \
  /Volumes/EricsLLMDrive/jangq-ai/Qwen3.6-35B-A3B-JANGTQ4 \
  --prompt-tokens 8192 \
  --max-tokens 8 \
  --json-out qwen36_jangtq4_nax_live_8k_2026_05_12.json
```

| Prompt | Mode | TTFT | Approx PP tok/s | Decode tok/s after first | Output |
| ---: | --- | ---: | ---: | ---: | --- |
| 299 tokens | off | 0.300 s | 998 | 108.6 | same as auto |
| 299 tokens | auto | 0.237 s | 1262 | 108.7 | same as off |
| 2084 tokens | off | 1.941 s | 1074 | 106.8 | same as auto |
| 2084 tokens | auto | 1.099 s | 1896 | 106.6 | same as off |
| 8244 tokens | off | 7.833 s | 1052 | 101.3 | same as auto |
| 8244 tokens | auto | 4.452 s | 1852 | 101.9 | same as off |

This is a real model proof that the current `auto` path improves Qwen
long-prompt prefill by about `1.76x-1.77x` without changing decode speed or
the generated prefix. It is not a 3-4x whole-model proof.

Interpretation: grouped fused gate/up is now a real full-cluster win on the
synthetic path, and the first real Qwen long-prompt gate improves PP from
about `1050-1075` tok/s to `1850-1900` tok/s. It can cross `3x` in
high-dispatch synthetic cases, but the model-like geometries and live Qwen
gate do not yet prove a universal `3-4x` whole-model win. The remaining
production work is to remove CPU metadata/sync, fuse or co-schedule down more
tightly, reduce Hadamard rotation overhead for wide 2-bit models, and run
live gates across ZAYA, MiniMax, Hy3, and vMLX-bundled app startup.

Real model gates:

- Qwen3.6 35B-A3B JANGTQ4, 2048 and 8192 prompt tokens
- ZAYA JANGTQ_K long prompt
- Hy3 JANGTQ2 long prompt
- MiniMax Small JANGTQ long prompt
- ZAYA1-VL JANGTQ4 through vMLX MLLM server

Quality gates:

- deterministic greedy math
- multi-turn recall
- no end-of-generation loops
- Thinking on/off where supported
- native top-k unchanged from baseline
- exact same output path with `JANGTQ_MPP_NAX=0` and `JANGTQ_MPP_NAX=auto`
  except for expected small floating-point reduction drift.

vMLX integration gates:

- bundled app includes all new kernel files
- `/health` reports JANGTQ NAX mode and fallback reason
- default remains safe when TensorOps is unavailable
- user/runtime can disable the feature

## Packaging Hygiene Follow-Up

The first prototype left the new helper modules hidden by the parent repo
ignore rules. That is fixed locally now.

Confirmed:

- `/Users/eric/jang/.gitignore` previously ignored
  `jang-tools/jang_tools/turboquant/` and broad `test_*` files.
- `mpp_nax_kernel.py`, `mpp_dense_kernel.py`, and their tests are now explicitly
  unignored so a normal `git status` surfaces them.

Fresh verification after the ignore-rule fix:

```sh
cd /Users/eric/jang/jang-tools
/Users/eric/mlx/vllm-mlx/.venv/bin/python -m pytest -q \
  tests/test_turboquant_mpp_nax_kernel.py \
  tests/test_turboquant_mpp_dense_kernel.py \
  tests/test_turboquant_kernel_source_contract.py \
  tests/test_turboquant_cache.py
# 47 passed

/Users/eric/mlx/vllm-mlx/.venv/bin/python -B -m py_compile \
  jang_tools/turboquant/mpp_nax_kernel.py \
  jang_tools/turboquant/mpp_dense_kernel.py \
  jang_tools/turboquant/tq_kernel.py \
  jang_tools/turboquant/gather_tq_kernel.py \
  jang_tools/turboquant/fused_gate_up_kernel.py

cd /Users/eric/jang
git diff --check
```

Release boundary:

- Commit the JANG helper modules/tests and call-site changes together.
- Then bundle vMLX against that exact JANG revision.
- Do not ship a vMLX UI/CLI flag without the matching packaged JANG runtime.

## Codex Live Integration Gates

Package build:

```sh
cd /Users/eric/jang/jang-tools
/Users/eric/mlx/vllm-mlx/.venv/bin/python -m build \
  --wheel --sdist --outdir /tmp/jang-mpp-package-check
```

Archive check:

- Wheel contains `jang_tools/turboquant/mpp_nax_kernel.py`.
- Wheel contains `jang_tools/turboquant/mpp_dense_kernel.py`.
- Sdist contains both runtime modules and both tests.
- Installed-wheel import from `/tmp/jang-wheel-import-check` loaded
  `mpp_nax_kernel.py` from the wheel target and reported TensorOps available.

Live model probes:

| Model | Prompt | Mode | Approx PP tok/s | Decode tok/s after first | Output |
| --- | ---: | --- | ---: | ---: | --- |
| Qwen3.6-35B-A3B-JANGTQ4 | 2084 | off | 1075.0 | 106.8 | same prefix |
| Qwen3.6-35B-A3B-JANGTQ4 | 2084 | auto | 1902.3 | 106.7 | same prefix |
| Qwen3.6-35B-A3B-JANGTQ4 | 8244 | off | 1056.6 | 101.9 | same prefix |
| Qwen3.6-35B-A3B-JANGTQ4 | 8244 | auto | 1858.5 | 101.8 | same prefix |
| ZAYA1-8B-JANGTQ_K | 2092 | off | 2394.9 | 71.9 | same prefix |
| ZAYA1-8B-JANGTQ_K | 2092 | auto | 4292.3 | 72.4 | same prefix |
| MiniMax-M2.7-Small-JANGTQ | 2126 | off | 161.8 | 40.5 | same prefix |
| MiniMax-M2.7-Small-JANGTQ | 2126 | auto | 233.9 | 34.2 | same prefix |
| Hy3-preview-JANGTQ2 | 2099 | off | 111.0 | 17.1 | `READY CERULEAN 45...` |
| Hy3-preview-JANGTQ2 | 2099 | auto | 202.9 | 15.8 | `READY CERULEAN 45...` |

Interpretation:

- Qwen and ZAYA show a clean prefill win with no output drift in the tested
  rows and no decode regression.
- Hy3 JANGTQ2 shows a prefill win and the same correct visible prefix. The
  probe kept sampling past the model's EOS, so the post-EOS continuation is
  not treated as a loop/coherency failure; app/API generation must stop at EOS.
- MiniMax Small shows a real prefill win and output parity, but its short-run
  decode was slower under `auto`; keep it as a follow-up before broad default
  claims.
- The generic live probe cannot test `zaya1_vl` because it uses the text
  `load_jangtq_model` harness. VLM proof must use vMLX or the VLM loader.

vMLX API gate:

- Source vMLX served Qwen3.6-35B-A3B-JANGTQ4 with
  `--jangtq-mpp-nax auto`.
- `/health` reported `turboquant_codebook_mpp_nax`, active MPP/NAX, and
  trained top-k `8`.
- Responses API produced `READY\n45`, then a second turn produced `CERULEAN`
  with `cached_tokens=24`, `cache_detail=paged+ssm`.
- Chat Completions, Ollama `/api/chat`, and Anthropic `/v1/messages` each
  returned the requested sentinel output.

vMLX ZAYA1-VL gate:

- Source vMLX served `/Users/eric/models/JANGQ/ZAYA1-VL-8B-JANGTQ4` with
  `--is-mllm --jangtq-mpp-nax auto`.
- `/health` reported `kernel_type=turboquant_codebook_mpp_nax`,
  `jangtq_mpp_nax.active=true`, model family `zaya1_vl`,
  `cache_subtype=zaya_cca`, and `is_mllm=True`.
- First Responses turn stored typed paged ZAYA CCA cache.
- Second Responses turn with `previous_response_id` returned `CERULEAN` with
  `cached_tokens=25` and `cache_detail=paged+zaya_cca`.
- Log confirmed native JANGTQ VLM fast path, 40 fused `SwitchGLU` instances,
  paged cache, block disk write-through, and ZAYA CCA records.
