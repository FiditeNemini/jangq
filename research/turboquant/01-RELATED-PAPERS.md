# Related Papers — KV Cache Quantization
**Author:** Jinho Jang (eric@jangq.ai)
**Date:** 2026-03-24

---

## 1. TurboQuant (arXiv:2504.19874, ICLR 2026 — Google DeepMind)

### Core Insight
Random rotation turns a fixed vector x on the unit sphere into coordinates that behave as i.i.d. samples from a Beta distribution, converging to N(0, 1/d) in high dimensions. Near-independence allows optimal *scalar* quantizers per coordinate, achieving near-optimal *vector* quantization. Data-oblivious — no training, no Hessian, no calibration.

### Algorithm 1: TurboQuant_mse (MSE-Optimized, for VALUES)

1. Generate random rotation Π ∈ ℝ^(d×d) via QR decomposition of random Gaussian matrix
2. Construct codebook by solving continuous 1D k-means:
   ```
   C(f_X, b) = min_{-1≤c₁≤...≤c_{2^b}≤1} Σᵢ ∫ |x-cᵢ|² · f_X(x) dx
   ```
   where f_X(x) = Γ(d/2)/(√π·Γ((d-1)/2)) · (1-x²)^((d-3)/2)

**Quant_mse(x):**
- y = Π·x (rotate)
- idx_j = argmin_k |y_j - c_k| for each coordinate j
- Output: idx (codebook indices)

**DeQuant_mse(idx):**
- ỹ_j = c_{idx_j} (lookup)
- x̃ = Π^T·ỹ (inverse rotation)
- Output: x̃

### Algorithm 2: TurboQuant_prod (Inner-Product Optimized, for KEYS)

1. Instantiate TurboQuant_mse with bit-width (b-1)
2. Generate random projection S ∈ ℝ^(d×d), S_{ij} ~ N(0,1)

**Quant_prod(x):**
- idx = Quant_mse(x) using (b-1) bits
- r = x - DeQuant_mse(idx) — residual
- qjl = sign(S·r) — 1-bit QJL encoding
- Output: (idx, qjl, ‖r‖₂)

**DeQuant_prod(idx, qjl, γ):**
- x̃_mse = DeQuant_mse(idx)
- x̃_qjl = √(π/2)/d · γ · S^T · qjl
- Output: x̃_mse + x̃_qjl

### Key vs Value Treatment (CRITICAL)
- **Keys → TurboQuant_prod:** Inner-product optimized, UNBIASED. Because attention computes q^T·k
- **Values → TurboQuant_mse:** MSE-optimized. Because output is weighted sum Σ αᵢvᵢ where MSE matters

### Mathematical Results

**Lemma 1 (Coordinate Distribution):**
For x uniform on S^(d-1), each coordinate follows:
f_X(x) = Γ(d/2)/(√π·Γ((d-1)/2)) · (1-x²)^((d-3)/2)
Converges to N(0, 1/d) as d→∞.

**Theorem 1 (MSE Distortion):**
D_mse ≤ √3·π/2 · (1/4^b) ≈ 2.72/4^b

| b (bits) | D_mse bound |
|----------|-------------|
| 1 | 0.36 |
| 2 | 0.117 |
| 3 | 0.03 |
| 4 | 0.009 |

**Theorem 2 (Inner-Product Distortion):**
- Unbiased: E[⟨y, DeQuant_prod(Quant_prod(x))⟩] = ⟨y, x⟩
- D_prod ≤ √3·π²·‖y‖²/d · (1/4^b)

| b (bits) | D_prod (normalized) |
|----------|-------------------|
| 1 | 1.57/d |
| 2 | 0.56/d |
| 3 | 0.18/d |
| 4 | 0.047/d |

**Lemma 4 (QJL):**
- Q_qjl(x) = sign(S·x)
- Unbiased: E[⟨y, Q_qjl^(-1)(Q_qjl(x))⟩] = ⟨y, x⟩
- Variance: ≤ π/(2d)·‖y‖²

**Bias of 1-bit MSE for inner products:**
E[⟨y, Q_mse^(-1)(Q_mse(x))⟩] = (2/π)·⟨y, x⟩
Multiplicative bias = 2/π ≈ 0.637. This is WHY the inner-product variant uses (b-1) bits MSE + 1 bit QJL.

**Information-theoretic bounds:** Gap to optimality ≈ 2.7×

### Optimal Codebook (Gaussian approx for high d)
- b=1: {±√(2/(πd))}
- b=2: {±0.453/√d, ±1.51/√d}

### Experimental Results
- Quality-neutral at 3.5 bits/channel
- Marginal degradation at 2.5 bits/channel
- 6×+ KV memory reduction
- 4-bit: up to 8× speedup over 32-bit keys on H100
- Tested: Llama-3.1-8B, Gemma, Mistral on LongBench, NIAH, ZeroSCROLLS, RULER, L-Eval
- No public implementation (Google Research internal)

---

## 2. KIVI (arXiv:2402.02750, ICML 2024)

### Core: Tuning-Free Asymmetric 2-bit

**Key insight — ASYMMETRIC quantization:**
- **Keys: per-CHANNEL** — some channels have persistent large magnitudes (outliers). Per-channel error: 4.55% vs per-token 13.67%
- **Values: per-TOKEN** — no channel outlier pattern. 84.3% attention sparsity helps

**Streaming:** Group size G=32, residual R≤128 tokens in FP16. When residual fills, quantize+append.

### Results (2-bit)
| Model | FP16 GSM8K | KIVI-2bit |
|-------|-----------|-----------|
| Llama-2-7B | 13.50% | 12.74% |
| Llama-2-13B | 22.67% | 20.77% |
| Mistral-7B | 38.36% | 36.01% |

- 2.6× less peak memory
- 2.35-3.47× throughput
- LongBench averages within 1% of FP16

---

## 3. QJL (arXiv:2406.03482, AAAI 2025)

### Core: 1-bit JL Transform for Keys

**Quantize:** QJL(S, k) = sign(S·k) ∈ {-1,+1}^m
**Estimate:** Prod_QJL(q, k) = √(π/2)/m · ‖k‖₂ · ⟨S·q, QJL(S,k)⟩

The query gets full JL transform (not quantized), only key is 1-bit. Unbiased estimator.

### Implementation
- General layers: 256-bit key quantization
- First 15 layers: 512-bit (higher precision)
- Values: 2-bit standard
- 8 outlier coordinates handled separately

### Results
- 5×+ KV memory reduction
- LongBench NarrativeQA: 21.83 vs 20.79 baseline
- Zero overhead from quantization constants (unlike KIVI)

---

## 4. GEAR (arXiv:2403.05527)

### Core: Three-Component Decomposition
```
X ≈ D̂ + L + S
```
- D̂: Ultra-low precision backbone (98% of entries)
- L: Low-rank residual (SVD, r=4 prefill / r=2 decode)
- S: Sparse outlier correction (top/bottom 2%)

### Results (2-bit)
| Model | FP16 | KIVI | GEAR |
|-------|------|------|------|
| LLaMA3-8B GSM8k | 54.21% | 30.17% | **54.59%** |
| LLaMA3-8B AQuA | 38.19% | 25.36% | **38.19%** |

14.95% average improvement over KIVI at 2-bit. 2.39× memory reduction.

---

## 5. QuIP (arXiv:2307.13304, NeurIPS 2023)

### Core: Random Rotation for Weight Quantization (same principle as TurboQuant)

**Incoherence Processing:**
W → U·W·V^T (before quantization)
Ŵ → U^T·Ŵ·V (after dequantization)

Using random orthogonal U, V. Makes weights "incoherent" — evenly spread magnitudes.

**Results:**
- 2-bit Llama-2-70B WikiText2: 6.33 (QuIP) vs 123.9 (OPTQ) — catastrophic failure without rotation
- Without incoherence processing: ppl 41,000+ → random rotation is ESSENTIAL

**TurboQuant connection:** Same mathematical principle — random rotation → near-uniform coordinates → better quantization. QuIP applies to weights, TurboQuant applies to KV cache.

---

## 6. QuIP# (arXiv:2402.04396, ICML 2024)

### Improvements Over QuIP
1. **Randomized Hadamard Transform (RHT):** O(n·log(n)) vs O(n·√n) for Kronecker. Uses {-1,+1} entries — no FP multiply
2. **E8 Lattice Codebook:** 2^16 codewords in 8D, 1 KiB, optimal sphere packing
3. **Fine-tuning:** Sign vector optimization

### Results (2-bit)
| Model | QuIP# | QuIP | OPTQ |
|-------|-------|------|------|
| Llama-2-70B WikiText2 | **3.91** | 5.90 | 123.9 |

---

## 7. PolarQuant (arXiv:2502.02617)

Maps coordinate pairs to polar form recursively. Angles factorize independently for Gaussian inputs.
- LongBench: 48.37 (PolarQuant) vs 48.63 (FP16) vs 46.70 (KIVI)
- NIAH: 0.991 vs 0.984 (KIVI)

---

## 8. KVLinC (arXiv:2510.05373)

Hadamard rotation for values + learnable linear correction adapters.
- **Key finding:** Hadamard rotation HURTS keys (increases scaling factors) but HELPS values
- 2-bit Llama-3.1-8B WikiText PPL: 7.1 (KVLinC) vs 7.8 (KIVI)

---

## 9. KVSplit (Apple Silicon, llama.cpp)

- K8V4 (8-bit keys, 4-bit values): 59% memory reduction, +0.86% PPL, +5.7% speed
- K4V8: 7× MORE degradation than K8V4 despite same total bits

---

## Universal Finding Across ALL Papers

**Keys are MORE sensitive to quantization than values.** Every paper confirms:
- KIVI: Keys need per-channel, values per-token
- QJL: Keys get expensive QJL treatment, values get simple quant
- TurboQuant: Keys use inner-product optimized (unbiased), values use MSE
- KVLinC: Hadamard helps values but hurts keys
- KVSplit: K8V4 dramatically better than K4V8
- GEAR: Key sensitivity higher than value

### Implementation Repositories
- KIVI: https://github.com/jy-yuan/KIVI
- QJL: https://github.com/amirzandieh/QJL
- GEAR: https://github.com/opengear-project/GEAR
- QuIP#: https://github.com/Cornell-RelaxML/quip-sharp
- KVSplit: https://github.com/dipampaul17/KVSplit
- TurboQuant: No public repo (Google Research internal)
