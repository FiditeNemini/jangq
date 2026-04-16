# GLM 5.1 JANGTQ end-to-end runbook (M3 Ultra)

**Audience:** Eric, on Mac Studio M3 Ultra (256 GB).
**Why this runbook exists:** GLM 5.1 cannot run on the MacBook M4 Max
(128 GB) because the smallest existing JANGTQ artifact is 191 GB.
Every GLM 5.1 step happens on the M3 Ultra. This runbook is the
hands-free executable path.

**Prerequisite:** ssh access to Mac Studio. The user's note
`feedback_never_run_on_macstudio` says Claude shouldn't autonomously
push work there — these are commands for Eric to run by hand or
explicitly authorize.

---

## 0. Setup verification (5 min)

Open a terminal on the Mac Studio. Verify the artifacts and the
patched runtime are present.

```bash
# Existing JANGTQ artifact
ls -la /Volumes/EricsLLMDrive/GLM-5.1-JANGTQ_1L/ | head
# Should show: config.json, jang_config.json, model-*-of-00204.safetensors

cat /Volumes/EricsLLMDrive/GLM-5.1-JANGTQ_1L/jang_config.json
# Should show: weight_format=mxtq, profile=JANGTQ_1L, mxtq_seed=42,
# mxtq_bits={attention:8, shared_expert:8, routed_expert:2}

# Existing standard JANG artifact (for A/B baseline)
ls -la /Volumes/EricsLLMDrive/GLM-5.1-JANG_1L/ | head

# The MLA SDPA fp32 patch (canonical fix for the L==1 absorb bug)
ls -la /Users/eric/jang/research/deepseek_v32_patched.py
```

If any are missing, stop here — the artifacts predate this runbook.

---

## 1. Apply the deepseek_v32 SDPA fp32 patch (one-time, 1 min)

The bug: at decode L==1, `mx.fast.scaled_dot_product_attention`
accumulates the dim-512 contraction in bf16, drifting after ~50
tokens into "1.1.1.1..." attractor loops. Without this fix GLM 5.1
JANGTQ_1L decodes garbage past short answers.

The fix is in `/Users/eric/jang/research/deepseek_v32_patched.py`.
Apply it to the active mlx_lm install (whichever Python env you use
for inference):

```bash
# Find the installed mlx_lm location
python3 -c "import mlx_lm; print(mlx_lm.__file__)"
# e.g. /opt/homebrew/lib/python3.14/site-packages/mlx_lm/__init__.py

MLXLM_DIR=$(dirname $(python3 -c "import mlx_lm; print(mlx_lm.__file__)"))

# Backup the original
cp -p $MLXLM_DIR/models/deepseek_v32.py $MLXLM_DIR/models/deepseek_v32.py.bak

# Drop in the patched version
cp /Users/eric/jang/research/deepseek_v32_patched.py $MLXLM_DIR/models/deepseek_v32.py

# Verify the patch is the float32-cast-on-L==1 variant (not the older
# always-prefill workaround). Look for "astype(mx.float32)" inside
# DeepseekV32Attention.__call__ on the L==1 branch.
grep -n "astype.*float32" $MLXLM_DIR/models/deepseek_v32.py
```

If the patch isn't present, GLM 5.1 will load fine but produce
incoherent text past ~20 tokens. **Do not skip this step.**

---

## 2. Python decode smoke (~5 min after model load)

Verify the JANGTQ_1L artifact decodes coherent text via the JANGTQ
loader.

```bash
cd /Users/eric/jang
python3 - <<'PY'
import time
from jang_tools.load_jangtq import load_jangtq_model
from mlx_lm import generate

t0 = time.time()
m, tok = load_jangtq_model("/Volumes/EricsLLMDrive/GLM-5.1-JANGTQ_1L")
print(f"load wall: {time.time() - t0:.1f}s")

# Three prompts: factual recall, reasoning, multi-step
for p in [
    "The capital of France is",
    "If I have 3 apples and eat 1, how many remain?",
    "Explain photosynthesis in one paragraph.",
]:
    t0 = time.time()
    out = generate(m, tok, p, max_tokens=64, verbose=False)
    dt = time.time() - t0
    print(f"\nprompt: {p!r}")
    print(f"resp  : {out[:240]!r}")
    print(f"wall  : {dt:.2f}s  ~{64/dt:.2f} tok/s")
PY
```

**Pass criteria** (per `GLM-5.1-RUNTIME-AUDIT.md:42-47`):
- Coherent answers on factual + reasoning prompts (Paris; 2 apples)
- ≥7/10 prompts coherent on a longer eval set
- Decode rate ≥ 5 tok/s (JANGTQ_1L is 233 GB equivalent — slow but
  must not crash or thrash)

**Failure modes**:
- "1.1.1.1..." attractor → patch wasn't applied or wrong variant
- Out-of-memory → check `vm_stat`; may need to close other apps
- Garbage tokens → run the `JANG_1L` baseline below to compare

---

## 3. Standard JANG (affine) baseline for A/B (~5 min)

Same prompts, on the affine `JANG_1L` artifact. This is the speed +
quality reference JANGTQ_1L should beat (or at least match) on
coherence.

```bash
python3 - <<'PY'
import time
from mlx_lm import load, generate
m, tok = load("/Volumes/EricsLLMDrive/GLM-5.1-JANG_1L")
for p in [
    "The capital of France is",
    "If I have 3 apples and eat 1, how many remain?",
    "Explain photosynthesis in one paragraph.",
]:
    t0 = time.time()
    out = generate(m, tok, p, max_tokens=64, verbose=False)
    dt = time.time() - t0
    print(f"\nprompt: {p!r}")
    print(f"resp  : {out[:240]!r}")
    print(f"wall  : {dt:.2f}s  ~{64/dt:.2f} tok/s")
PY
```

Compare:
- Coherence diff (subjective; if JANGTQ_1L outputs match JANG_1L on
  the same prompts, the codebook quant is doing its job)
- Speed diff (JANG_1L should win — it's affine 2-bit, simpler kernel
  than codebook 2-bit)

---

## 4. Swift smoke via vmlxctl (~3 min after build, ~2 min for load)

The Swift JANGTQ runtime has `GLM4MoEJANGTQModel` ready, but **GLM 5.1
uses `model_type: "glm_moe_dsa"`, NOT `"glm4_moe"`**. The factory
won't auto-route. Two options:

### 4a. Ad-hoc factory route via env (preferred for first run)

There isn't an env override yet — would require a code patch. Skip
this for the first runtime smoke; do option 4b instead.

### 4b. Patch GLM-5.1's `config.json` to declare `glm4_moe` (not for production)

**Do this in a copy, not the original artifact.** The MLA attention
in GLM 5.1 is *different* from GLM 4 MoE's GQA — the Swift
`GLM4MoEAttention` will instantiate with the wrong shape. This will
NOT work end-to-end without an MLA attention block in Swift.

**Conclusion:** the Swift JANGTQ runtime cannot run GLM 5.1 today
without a `GlmMoeDsaJANGTQ.swift` model that ports MLA. Scope estimate
~800 LOC (q_a_proj/q_b_proj/kv_a_proj/kv_b_proj/lora layers + the
absorb branch). Tracked as future work.

**Today on M3 Ultra: GLM 5.1 lives on the Python path (steps 2 + 3).**

---

## 5. MMLU validation (~30-60 min if running 50-q subset)

Per `JANGTQ-PLAN.md:43-52`, the open question for GLM-5.1-JANGTQ_1L
is whether the codebook quant beats affine 2-bit on quality. Run an
MMLU 50-question subset on both artifacts and compare scores.

```bash
# Use the existing MMLU harness if present
ls /Users/eric/jang/jang-tools/jang_tools/eval_*.py /Users/eric/jang/research/run_mmlu*.py 2>/dev/null

# Else: a minimal eval loop
python3 - <<'PY'
# 50-q subset MMLU evaluation
# Loads both models in sequence (releases first before second),
# runs the same 50 questions, prints accuracies.
# (~30 GB working set per model; do them sequentially.)
import json, time
from mlx_lm import generate

questions = [
    # First 50 MMLU questions or use a stable seed from datasets lib
    # Format: {"prompt": "...", "answer": "A"}
]
# ... see /Users/eric/jang/research/run_mmlu*.py if present, else build one
PY
```

Pass criteria for shipping JANGTQ_1L:
- MMLU score within 2 pp of JANG_1L (affine 2-bit baseline)
- No infinite loops on any prompt

---

## 6. If nothing decodes coherently — diagnostic flow

1. **First check the patch** (step 1 above) — most common cause of
   "loads OK, garbage out" on GLM 5.1.
2. **Check token counts** — if generate returns ≤2 tokens then quits,
   the multi-EOS isn't being respected. Verify
   `eos_token_id: [154820, 154827, 154829]` is the `eos_token_id`
   list in `config.json`, not a single int.
3. **Compare logits at L==1** between affine and JANGTQ paths on the
   same prompt — if they diverge, the codebook decode kernel has
   numerical issues.
4. **Reset and re-load** — MLX caches some state; `pkill -f python3`
   between runs to force a clean state.

---

## 7. What success looks like

After step 5 you should be able to write a one-line note like:

> GLM-5.1-JANGTQ_1L runs at X.X tok/s on M3 Ultra, MMLU-50 = Y%
> (baseline JANG_1L = Z tok/s, MMLU-50 = W%). Pass.

Then the only remaining GLM 5.1 work is the Swift `GlmMoeDsaJANGTQ`
port for vmlx-swift parity, which is its own ~1-week project.

---

## File map

- Existing JANGTQ artifact: `/Volumes/EricsLLMDrive/GLM-5.1-JANGTQ_1L/` (191 GB)
- Existing JANG_1L baseline: `/Volumes/EricsLLMDrive/GLM-5.1-JANG_1L/` (233 GB)
- Patched runtime file: `/Users/eric/jang/research/deepseek_v32_patched.py`
- JANGTQ Python loader: `/Users/eric/jang/jang-tools/jang_tools/load_jangtq.py`
- Reference (MiniMax): `/Users/eric/jang/research/JANGTQ-REFERENCE.md`
- Quality baseline: `/Users/eric/jang/research/GLM-5.1-RUNTIME-AUDIT.md`
- Swift runtime (GLM 4 MoE only — NOT GLM 5.1 yet): `/Users/eric/vmlx/swift/Sources/vMLXLLM/Models/GLM4MoEJANGTQ.swift`
