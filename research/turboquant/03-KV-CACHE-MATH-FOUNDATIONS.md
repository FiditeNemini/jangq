# KV Cache Mathematical Foundations
**Author:** Jinho Jang (eric@jangq.ai)
**Date:** 2026-03-24

---

## 1. Standard Attention — How KV Cache Works

### The Attention Equation

```
Attention(Q, K, V) = softmax(Q·K^T / √d_k) · V
```

Where:
- Q ∈ ℝ^(n×d_k) — queries (current tokens)
- K ∈ ℝ^(m×d_k) — keys (all tokens including cached)
- V ∈ ℝ^(m×d_v) — values (all tokens including cached)
- d_k = head dimension (typically 128)
- n = number of new query tokens
- m = total sequence length (cached + new)

### Why We Cache K and V

During autoregressive generation, each new token needs to attend to ALL previous tokens:

**Token 1:** Q₁·K₁^T → softmax → ·V₁
**Token 2:** Q₂·[K₁;K₂]^T → softmax → ·[V₁;V₂]
**Token t:** Qₜ·[K₁;...;Kₜ]^T → softmax → ·[V₁;...;Vₜ]

Without caching, generating token t requires recomputing K and V for ALL previous tokens.
With caching, we only compute Kₜ and Vₜ, then append to cached [K₁;...;Kₜ₋₁].

### Cache Size Formula

```
Cache_bytes = B × n_layers × n_kv_heads × seq_len × head_dim × 2 (K+V) × dtype_bytes
```

For float16 (2 bytes per element):
```
Cache_bytes = 4 × B × n_layers × n_kv_heads × seq_len × head_dim
```

### Cache Size Examples at 4096 Tokens

| Model | Layers | KV Heads | Head Dim | Cache (4K, fp16) |
|-------|--------|----------|----------|-----------------|
| Qwen3-MoE-30B | 48 | 4 (GQA) | 128 | 403 MB |
| Llama-3.1-70B | 80 | 8 (GQA) | 128 | 1.34 GB |
| Mistral-Small-4 | 56 | 128 (MLA) | 192+128 | 18.4 GB |
| DeepSeek-V3-236B | 61 | 128 (MLA) | 192+128 | 20.0 GB |

### Why Cache Grows Linearly

Each new token adds one row to K and V per layer per head:
```
Δ_cache = n_layers × n_kv_heads × head_dim × 2 × dtype_bytes  per token
```

At 128K context with MLA (Mistral 4): 128K/4K × 18.4 GB = **589 GB** — impossible without compression.

---

## 2. Why Quantization Errors Matter Differently for K vs V

### Key Quantization → Attention Score Error

The attention scores are: s = Q·K^T/√d_k

If K̂ = K + ε_K (quantization error), then:
```
ŝ = Q·K̂^T/√d_k = Q·(K+ε_K)^T/√d_k = s + Q·ε_K^T/√d_k
```

The error propagates through softmax:
```
α̂ᵢ = exp(ŝᵢ) / Σⱼ exp(ŝⱼ)
```

Softmax is EXPONENTIALLY sensitive to score perturbations. A small error in s causes large changes in attention weights α. This is why keys are more sensitive.

### Value Quantization → Output Error

The attention output is: o = Σᵢ αᵢ·vᵢ

If V̂ = V + ε_V, then:
```
ô = Σᵢ αᵢ·(vᵢ + ε_Vᵢ) = o + Σᵢ αᵢ·ε_Vᵢ
```

The error is a WEIGHTED AVERAGE of value errors, damped by attention weights (which sum to 1 and are typically sparse ~84%). This is why values tolerate more quantization.

### Implication for TurboQuant
- **Keys need inner-product preservation** → TurboQuant_prod (unbiased estimator)
- **Values need MSE minimization** → TurboQuant_mse (lower reconstruction error)
- Asymmetric treatment is ESSENTIAL (confirmed by KIVI, QJL, KVSplit, KVLinC)

---

## 3. The Random Rotation Principle

### Why Rotate Before Quantizing?

Neural network activations have highly non-uniform coordinate distributions:
- Some channels have very large magnitudes (outliers)
- Some channels are concentrated near zero
- The distribution varies dramatically across channels

Uniform quantization assigns equal resolution to all channels, wasting bits on small-magnitude channels and under-resolving large ones.

### The Solution: Random Rotation

For x on the unit sphere S^(d-1), apply random orthogonal rotation Π:
```
y = Π·x
```

**Property:** Each coordinate yⱼ follows:
```
f_Y(y) = Γ(d/2) / (√π · Γ((d-1)/2)) · (1-y²)^((d-3)/2)
```

This is proportional to Beta((d-1)/2, (d-1)/2) rescaled to [-1, 1].

**In high dimensions (d ≥ 128):** This concentrates around 0 with std ≈ 1/√d, approaching N(0, 1/d).

### Why This Helps

After rotation:
1. **All coordinates have the same distribution** — no outlier channels
2. **Coordinates are nearly independent** — scalar quantization is near-optimal
3. **The distribution is KNOWN** — we can precompute optimal quantizer boundaries
4. **Data-oblivious** — rotation matrix generated once, works for any input

### Comparison: Without Rotation

Without rotation (standard affine quantization):
- Must estimate min/max per group (data-dependent)
- Outlier channels waste bits
- Non-uniform distribution → suboptimal quantizer boundaries
- Group-size tradeoff: small groups = more overhead, large groups = more error

---

## 4. Optimal Scalar Quantization After Rotation

### The 1D k-means Problem

Given the coordinate distribution f_X(x), find 2^b centroids {c₁,...,c_{2^b}} minimizing:
```
C(f_X, b) = min Σᵢ ∫_{Bᵢ} |x - cᵢ|² · f_X(x) dx
```

where Bᵢ is the Voronoi region of cᵢ (all x closer to cᵢ than any other centroid).

### Solution: Lloyd-Max Quantizer

Boundaries at midpoints: bᵢ = (cᵢ + cᵢ₊₁)/2
Centroids: cᵢ = E[X | X ∈ Bᵢ] = ∫_{bᵢ₋₁}^{bᵢ} x·f_X(x)dx / ∫_{bᵢ₋₁}^{bᵢ} f_X(x)dx

For the Beta/Gaussian distribution after rotation:

**b=1 (2 centroids):** {±√(2/(πd))}
**b=2 (4 centroids):** {±0.453/√d, ±1.51/√d}
**b=3 (8 centroids):** precomputed from numerical optimization
**b=4 (16 centroids):** precomputed from numerical optimization

### Distortion Rate

**Theorem:** D_mse(b) ≤ √3·π/(2·4^b)

This is within 2.7× of the information-theoretic lower bound D*(b) ≥ 1/4^b.

### Per-Bit Improvement

Each additional bit reduces distortion by 4×:
```
D_mse(b+1) / D_mse(b) = 1/4 = 0.25
```

This is OPTIMAL — matches the Shannon rate for fixed-rate quantization.

---

## 5. QJL — Correcting Inner Product Bias

### The Problem

The 1-bit MSE quantizer introduces multiplicative bias:
```
E[⟨y, Q_mse^(-1)(Q_mse(x))⟩] = (2/π) · ⟨y, x⟩ ≈ 0.637 · ⟨y, x⟩
```

For attention, this means: E[Q·K̂^T] = 0.637 · Q·K^T — ALL attention scores are systematically shrunk by 36.3%. Softmax amplifies this error.

### QJL Definition

```
Q_qjl(x) = sign(S·x)  ∈ {-1, +1}^d
```
where S ∈ ℝ^(d×d), S_{ij} ~ N(0,1)

**Dequantization:**
```
Q_qjl^(-1)(z) = √(π/2)/d · S^T · z
```

### Properties
- **Unbiased:** E[⟨y, Q_qjl^(-1)(Q_qjl(x))⟩] = ⟨y, x⟩
- **Variance:** Var[...] ≤ π/(2d) · ‖y‖²
- **Cost:** 1 bit per coordinate (sign bit only)
- **No overhead:** No scales, biases, or codebook stored

### Two-Stage Pipeline

Use (b-1) bits for MSE quantizer + 1 bit for QJL on residual:
```
r = x - DeQuant_mse(Quant_mse(x))   ← residual from MSE stage
q = sign(S·r)                          ← 1-bit QJL
γ = ‖r‖₂                               ← residual norm (stored as scalar)
```

**Combined estimator:**
```
x̂ = DeQuant_mse(idx) + √(π/2)/d · γ · S^T · q
```

**Total storage per vector:**
- (b-1) bits/coord for MSE codebook indices
- 1 bit/coord for QJL sign
- 1 scalar (γ) for residual norm
- Total: b bits/coord + 1 scalar ≈ b bits/coord

---

## 6. Memory Storage Analysis

### What TurboQuant Stores Per Token Per Layer

**For Keys (TurboQuant_prod, b bits total):**
- MSE indices: (b-1) bits × head_dim per KV head
- QJL signs: 1 bit × head_dim per KV head
- Residual norm: 1 × fp16 per KV head
- Rotation matrix Π: shared across all tokens (stored once)
- Projection matrix S: shared across all tokens (stored once)

**For Values (TurboQuant_mse, b bits total):**
- MSE indices: b bits × head_dim per KV head
- Rotation matrix: shared (same Π or different)

### Comparison at 4-bit, head_dim=128, n_kv_heads=4

| Method | Bytes/token/layer | vs FP16 |
|--------|-------------------|---------|
| FP16 (baseline) | 4×128×2×2 = 2048 | 1.0× |
| MLX q4 (affine) | ~384 + scales/bias | ~5× |
| TurboQuant 4-bit K + 4-bit V | ~512 + norms | ~4× |
| TurboQuant 3-bit K + 3-bit V | ~384 + norms | ~5.3× |
| KIVI 2-bit | ~256 + residual | ~6× |

### Shared Matrices (One-Time Cost)

Rotation Π ∈ ℝ^(d×d): d² × 4 bytes (fp32)
- head_dim=128: 128² × 4 = 64 KB per layer (shared across all tokens)
- For 48 layers: 3 MB total — negligible

Projection S ∈ ℝ^(d×d) (for QJL, keys only): same as Π
- Can use structured matrices (Hadamard) to avoid storage entirely: O(d·log d) compute, O(1) storage

---

## 7. Computational Cost Analysis

### Per-Token Operations

**Quantize (on key/value insertion):**
1. Rotate: y = Π·x — matrix-vector multiply: O(d²) per head
2. Quantize: nearest codebook lookup: O(d·2^b) per head
3. QJL (keys only): sign(S·r) — matrix-vector + sign: O(d²) per head
4. Norm: ‖r‖₂ — vector norm: O(d) per head

**Dequantize (on attention computation):**
1. Lookup centroids: O(d) per head per cached token
2. Inverse rotate: x̃ = Π^T·ỹ — O(d²) per head per cached token
3. QJL reconstruct (keys): √(π/2)/d·γ·S^T·q — O(d²) per head per cached token
4. Add: x̃_total = x̃_mse + x̃_qjl — O(d) per head

### The Critical Cost: Dequantization

For each attention step, we dequantize ALL cached tokens:
- m cached tokens × O(d²) per token = O(m·d²)
- Standard attention: O(m·d) for Q·K^T

**TurboQuant dequant is d× slower than raw attention per token.**

### Mitigation Strategies

1. **Structured rotation (Hadamard):** O(d·log d) instead of O(d²). QuIP# proved this works.
2. **Fused kernel:** Combine rotation + codebook lookup in one Metal pass
3. **Keep rotated form:** Store quantized values in rotated space, apply Π^T to query instead
   - Q̃ = Π·Q, then Q̃·K̂^T ≈ Q·K^T (rotation preserves inner products!)
   - Avoids per-token inverse rotation entirely
4. **mx.quantized_matmul integration:** If TurboQuant output matches mx.quantize format, existing Metal kernel works

### Strategy 3 is KEY: Rotate Query, Not Cache

Since Π is orthogonal: ⟨Π·q, Π·k⟩ = ⟨q, k⟩ (inner products preserved)

Instead of:
```
for each cached k: k_derotated = Π^T · dequant(k_stored)  ← O(m·d²)
scores = Q · K_derotated^T                                  ← O(m·d)
```

Do:
```
Q_rotated = Π · Q                                           ← O(n·d²), n=1 for decode
scores = Q_rotated · K_quantized_rotated^T                  ← O(m·d)
```

Cost goes from O(m·d²) to O(n·d²) where n=1 during generation. **Constant cost per step regardless of context length.**

---

## 8. Error Propagation Through Attention

### Attention Score Error (Key Quantization)

True scores: s = Q·K^T/√d
Quantized: ŝ = Q·K̂^T/√d = s + Q·ε^T/√d

For TurboQuant_prod (unbiased):
- E[ŝ] = s (no systematic bias)
- Var[ŝᵢⱼ] ≤ √3·π²·‖qᵢ‖²/(d·4^b·√d)

Softmax error bound (for small perturbations δ = ŝ - s):
```
|α̂ᵢ - αᵢ| ≤ αᵢ · |δᵢ - Σⱼ αⱼ·δⱼ| ≤ αᵢ · 2·max|δ|
```

At 3 bits: max|δ| ≈ √(0.18/d) — for d=128: max|δ| ≈ 0.037
Attention weight error: ≤ 7.5% relative per token.

### Output Error (Value Quantization)

True output: o = Σᵢ αᵢ·vᵢ
Quantized: ô = Σᵢ αᵢ·v̂ᵢ

For TurboQuant_mse:
```
‖ô - o‖² = ‖Σᵢ αᵢ·εᵢ‖² ≤ (Σᵢ αᵢ²) · max‖εᵢ‖²
```

With 84% attention sparsity: Σᵢ αᵢ² ≈ 0.1 (effective number of attended tokens ≈ 10)
At 3 bits: max‖ε‖² ≈ 0.03
Output error: √(0.1 × 0.03) ≈ 0.055 — 5.5% relative error.

---

## 9. Hadamard Transform as Efficient Rotation

### Definition

The Hadamard matrix H_n ∈ ℝ^(n×n) for n = 2^k:
```
H_1 = [1]
H_2 = [1  1; 1 -1] / √2
H_{2n} = [H_n  H_n; H_n  -H_n] / √2
```

### Properties
- Orthogonal: H·H^T = I
- Fast transform: O(n·log n) — same as FFT
- Entries: {±1/√n} — no floating-point multiply needed (just add/subtract)
- Randomized: multiply by random ±1 diagonal first: D·H where D = diag(±1)

### Randomized Hadamard Transform (RHT)
```
y = H · D · x  where D = diag(s₁,...,sₙ), sᵢ ~ Uniform{-1,+1}
```

This achieves the same incoherence guarantee as random orthogonal rotation:
- With prob ≥ 1-δ: max|yⱼ| ≤ √(2·log(2n/δ)/n)
- O(n·log n) compute vs O(n²) for general rotation

### MLX Availability

MLX supports Hadamard natively? Need to verify. If not:
- Can implement as recursive butterfly: log₂(n) passes of n/2 butterfly operations
- Each butterfly: a' = a+b, b' = a-b (no multiply)
- Metal kernel: highly parallelizable

For head_dim=128: log₂(128) = 7 passes of 64 butterflies = 448 operations (vs 16384 for general matmul)

---

## 10. Codebook Precomputation

### Steps to Build Optimal Codebook

For given d (head dimension) and b (bits):

1. **Compute f_X(x):** The marginal distribution on [-1, 1]
   ```
   f_X(x) = Γ(d/2) / (√π·Γ((d-1)/2)) · (1-x²)^((d-3)/2)
   ```

2. **Run Lloyd-Max algorithm:** Iterative optimization
   - Initialize centroids uniformly in [-1, 1]
   - Repeat until convergence:
     - Boundaries: bᵢ = (cᵢ + cᵢ₊₁)/2
     - Centroids: cᵢ = E[X | bᵢ₋₁ ≤ X < bᵢ]

3. **Store codebook:** Only 2^b values (e.g., 16 values for 4-bit). Tiny.

### Codebook Examples

**d=128, b=1 (2 centroids):**
c = {-0.070, +0.070} ≈ {±√(2/(π·128))}

**d=128, b=2 (4 centroids):**
c ≈ {-0.134, -0.040, +0.040, +0.134}

**d=128, b=3 (8 centroids):**
c ≈ {-0.179, -0.107, -0.054, -0.018, +0.018, +0.054, +0.107, +0.179}

**d=128, b=4 (16 centroids):** Precomputed numerically.

These codebooks are FIXED for a given (d, b) pair. Computed once, used for all models.
