# TurboQuant Integration Research Plan

## Goal
Integrate Google DeepMind's TurboQuant KV cache compression into JANG for MLX/Apple Silicon.
JANG compresses **weights** (model at rest). TurboQuant compresses **KV cache** (grows during generation).
Together: the only quantization system that optimally compresses BOTH.

## What TurboQuant Does
- ICLR 2026 paper from Google DeepMind
- Compresses KV cache to 3-4 bits at inference time (no training needed)
- Random rotation → Beta distribution on coordinates → optimal scalar quantizers
- Two-stage: MSE quantizer + 1-bit QJL on residual → unbiased inner product
- Proven on Gemma and Mistral with zero accuracy loss at 3-bit
- 8x speedup on H100 at 4-bit (vs 32-bit unquantized keys)

## Research Topics Needed

### 1. KV Cache Mathematics
- How KV cache works in standard attention (Q @ K^T → softmax → @ V)
- Cache size formula: batch × n_heads × seq_len × head_dim × 2 (K+V) × dtype_bytes
- Why cache grows linearly with sequence length
- Memory bottleneck analysis at different context lengths (4K, 32K, 128K, 256K)

### 2. KV Cache in Different Architectures
- **Standard GQA** (Qwen, Llama): num_kv_heads × head_dim per token
- **MLA** (Mistral 4, DeepSeek V2/V3): compressed latent (kv_lora_rank) + rope component
  - Cache stores DECOMPRESSED K/V (DeepSeek V2 naive) or COMPRESSED latent (V3 absorbed)
  - TurboQuant on latent vs decompressed — different compression targets
- **MoE models**: KV cache same as dense (MoE only affects MLP, not attention)
  - But MoE routing sensitivity means cache precision matters more
- **Mamba/SSM** (Nemotron Super, Qwen3.5 hybrid): NO KV cache for SSM layers
  - Only attention layers have KV cache
  - Hybrid models: cache only for full-attention layers (25% of layers in Qwen3.5)
- **Sliding window attention**: Cache only needs last W tokens, not full context

### 3. MLX Current KV Cache Implementation
- `mlx_lm/models/cache.py`: KVCache class, RotatingKVCache
- `kv_bits` parameter in generate: q4, q8 quantization options
- How MLX stores cache (mx.array in GPU memory)
- Current quantization: simple affine per-group (naive)
- Performance impact of current kv_bits on different models

### 4. TurboQuant Algorithm Deep Dive
- Random rotation matrices (Hadamard transform variant?)
- Beta distribution concentration property in high dimensions
- Scalar quantizer per coordinate (which quantizer? uniform? Lloyd-Max?)
- QJL (Quantized Johnson-Lindenstrauss) transform for residual
- Unbiased inner product estimation — why this matters for attention
- Error bounds: distortion rate vs bits per coordinate

### 5. Metal/MLX Implementation Considerations
- Random rotation: matrix multiply (supported in MLX)
- Per-coordinate scalar quantize: vectorized operation (supported)
- QJL 1-bit: binary operations + random signs (supported)
- Memory layout: how to store rotated+quantized cache efficiently
- Kernel fusion opportunities: rotate + quantize in one pass
- Comparison with current MLX kv quantization kernels

### 6. Integration Architecture
- Where in the JANG pipeline does TurboQuant fit?
  - NOT in the converter (that's weight quantization)
  - In the LOADER or INFERENCE path (runtime KV cache compression)
  - Could be a separate module: `jang_tools/kv_compress.py`
- API design: `load_jang_model(path, kv_bits=3, kv_method="turboquant")`
- Compatibility with existing models (should work with ANY JANG model)
- Performance targets: what speedup do we expect on Apple Silicon?

### 7. Architecture-Specific Integration
- **Standard attention**: straightforward — compress K and V separately
- **MLA (Mistral 4)**: compress the latent representation? Or decompressed K/V?
  - If compressing latent: smaller target, higher compression ratio
  - If compressing decompressed: standard TurboQuant applies
- **Hybrid SSM+attention**: only compress attention layers' cache
- **MoE**: no special handling needed (MoE is in MLP, not attention)

### 8. Benchmarking Plan
- Measure KV cache memory at different context lengths
- Compare: no compression vs MLX q4/q8 vs TurboQuant 3-bit/4-bit
- Accuracy: perplexity and MMLU at different cache compression levels
- Speed: generation tok/s at long contexts (8K, 32K, 128K)
- Models to test: Qwen3.5-27B, Mistral Small 4, Nemotron Cascade 2

## Key Papers and Resources
- TurboQuant paper: https://arxiv.org/abs/2504.19874
- Google blog: https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/
- ICLR 2026 review: https://openreview.net/pdf/6593f484501e295cdbe7efcbc46d7f20fc7e741f.pdf
- MLX KV cache: mlx_lm/models/cache.py
- KIVI paper (KV cache quantization baseline)
- Gear paper (grouped quantization for KV cache)

## Status
- [ ] Read full TurboQuant paper
- [ ] Analyze MLX KV cache implementation
- [ ] Write KV cache math foundations doc
- [ ] Write architecture-specific analysis
- [ ] Prototype random rotation in MLX
- [ ] Benchmark current MLX kv_bits performance
- [ ] Design integration API
- [ ] Implement prototype
- [ ] Benchmark and compare
