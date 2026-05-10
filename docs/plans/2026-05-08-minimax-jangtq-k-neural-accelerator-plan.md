# MiniMax JANGTQ_K M5 Neural Accelerator Plan

> **For Claude / Codex workers:** this is a research-to-implementation handoff.
> Do not start by rewriting MiniMax or re-quantizing the model. Build a fail-closed
> prefill-only TensorOps path first, with the existing JANGTQ path as fallback.

**Goal:** accelerate MiniMax-M2.7-JANGTQ_K prompt processing on M5 Max using GPU
Neural Accelerators while preserving current decode quality and bundle
compatibility.

**Architecture:** add a sidecar-backed TensorOps prefill primitive for MiniMax
JANGTQ_K routed expert MLPs. Existing `.tq_packed/.tq_norms/.tq_bits` remain the
source of truth. The new path converts codebook values to int8 tiles inside the
kernel or from a lightweight sidecar, quantizes activations online, and computes
prefill GEMMs with Metal 4 TensorOps. Decode stays on the current P15/P17 path.

**Primary repo surfaces:**

- `/Users/eric/jang/jang-tools/jang_tools/convert_minimax_jangtq.py`
- `/Users/eric/jang/jang-tools/jang_tools/build_jangtq_sidecar.py`
- `/Users/eric/jang/jang-tools/jang_tools/load_jangtq.py`
- `/Users/eric/jang/jang-runtime/Sources/JANGCoreMetal/JANGTQMatmul.metal`
- `/Users/eric/jang/swift-stage/DSV4/JANGTQKernels.swift`
- likely implementation target for production Swift path:
  `/Users/eric/vmlx/swift` after Claude confirms current branch state

---

## Phase 0 — verify the target

- [ ] Confirm local machine and OS:

```bash
system_profiler SPHardwareDataType SPDisplaysDataType | sed -n '1,90p'
sw_vers
python3 - <<'PY'
import mlx.core as mx
print("mlx", getattr(mx, "__version__", "unknown"))
PY
```

Expected: Apple M5 Max, Metal 4. If macOS / MLX is too old for TensorOps, stop
and document the required upgrade instead of writing fallback-only code.

- [ ] Reconfirm MiniMax K bundle metadata:

```bash
jq '{profile,mxtq_bits,quantization,routed_expert_layout}' \
  /Users/eric/models/JANGQ/MiniMax-M2.7-JANGTQ_K/jang_config.json
jq '{model_type,hidden_size,intermediate_size,num_hidden_layers,num_local_experts,num_experts_per_tok,quantization}' \
  /Users/eric/models/JANGQ/MiniMax-M2.7-JANGTQ_K/config.json
```

Expected: `JANGTQ_K`, 2/2/4 routed bits, prestacked, hidden 3072,
intermediate 1536, 62 layers, 256 experts, top-8.

## Phase 1 — establish baselines

- [ ] Run current JANGTQ_K prefill/decode benchmark with long prompt and short
  generation. Record pp/s, TTFT, tok/s, peak memory, and model path.

Suggested prompts:

- 4096-token synthetic prompt for Apple/Cider comparability.
- 8192-token prompt to expose prefill scaling.
- 128-token decode to isolate steady decode.

- [ ] Run the same benchmark on existing JANGTQ2 if present, only to compare
  quality/speed tradeoff, not as a target replacement.

## Phase 2 — inspect Cider integration shape

- [ ] Clone or inspect `Mininglamp-AI/cider` locally.
- [ ] Identify these files and map them to the JANG equivalent:
  - C++ primitive subclass and nanobind registration.
  - W4A8 Metal GEMM kernel.
  - activation quantization kernel.
  - Python wrapper returning lazy `mx.array`.
  - M5 feature gate.
- [ ] Write a one-page implementation note under `docs/research/` with exact
  file names and ABI assumptions.

Do not copy code blindly. Use it to confirm MLX primitive shape, TensorOps
headers, build flags, and feature gates.

## Phase 3 — sidecar builder

- [ ] Add a new builder command, likely:

```bash
python3 -m jang_tools.build_jangtq_na_sidecar \
  /Users/eric/models/JANGQ/MiniMax-M2.7-JANGTQ_K
```

- [ ] Emit `jangtq_na_runtime.safetensors` with:
  - `codebook_int8.<in_features>.<bits>`;
  - `codebook_scale.<in_features>.<bits>`;
  - `na_tile_shape`;
  - `na_format_version`;
  - optional per-projection `<base>.tq_int8_weight_scales` if profiling shows
    recomputing `norms * codebook_scale` in kernel is costly.

- [ ] Validate that sidecar tensors match all `.tq_bits` values in the bundle.

## Phase 4 — Python MLX custom primitive spike

- [ ] Build a micro primitive that accepts one projection:

```text
x_prefill:       [M, in_features]
packed:          [experts, out_features, packed_cols]
norms:           [experts, out_features]
indices:         [M, top_k]
codebook_int8:   [2^bits]
codebook_scale:  scalar
```

Output:

```text
y: [M, top_k, out_features]
```

- [ ] First support `down_proj` 4-bit, because that validates the hardest K
  profile case.
- [ ] Then support 2-bit gate/up.
- [ ] Compare against existing `gather_tq_matmul` / `fused_gate_up_swiglu`
  with strict numerical tolerances on a fixed tensor sample.

Acceptance for spike:

- M=128 and M=1024 outperform existing JANGTQ prefill projection path.
- M=1 is allowed to lose and should route to existing decode path.
- output max relative error stays within a documented bound that does not
  change MiniMax canonical prompt outputs.

## Phase 5 — integrate prefill dispatch

- [ ] In `load_jangtq.py`, route only prefill (`seq_len > 1`) through the NA
  primitive when all gates pass.
- [ ] Keep decode (`seq_len == 1`) on existing P15/P17 path.
- [ ] Add env toggles:
  - `JANGTQ_NA_PREFILL=1`
  - `JANGTQ_NA_PREFILL=0`
  - `JANGTQ_NA_STRICT=1` to error instead of fallback.

## Phase 6 — Swift / vMLX path

Only after Python proves speed:

- [ ] Port the primitive shape into Swift / vMLX with the same fail-closed gates.
- [ ] Keep the sidecar format identical.
- [ ] Add a `vmlxctl bench-direct` mode that reports:
  - TensorOps available: yes/no;
  - sidecar loaded: yes/no;
  - prefill path: existing vs NA;
  - decode path: existing.

## Phase 7 — validation

Required measurements:

- [ ] Long prompt TTFT and pp/s with NA on/off.
- [ ] 128-token decode tok/s with NA on/off.
- [ ] canonical MiniMax coherence prompts, byte saved.
- [ ] small MMLU/GSM8K sample if numerical differences exceed tolerance.
- [ ] peak memory and sidecar size.

Ship gate:

- prefill gain >= 1.3x on 4096-token prompt;
- decode regression <= 3%;
- no canonical prompt regression;
- strict fallback works on non-M5 or missing sidecar.

## Notes for Claude

The proposed Approach A is the correct first build. Make these refinements:

- call it GPU Neural Accelerators / TensorOps, not ANE;
- treat 3-4x as TTFT/prefill upside, not decode upside;
- keep scale math precise: if codebook is mapped to signed int8, dequant scale
  includes `/127` unless the kernel handles it elsewhere;
- do not re-upload or re-quantize MiniMax for v1;
- do not remove current JANGTQ_K decode kernels;
- fail closed and measure first.

