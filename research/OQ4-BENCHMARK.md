# oQ4 vs JANG Benchmark — Qwen3.5-35B-A3B

## Models

| Model | Source | Size | Bits | VL | API |
|-------|--------|------|------|----|-----|
| deepsweet/Qwen3.5-35B-A3B-MLX-oQ4 | oMLX oQ | 17 GB | ~4 (mixed mxfp4/affine) | No | localhost:8009 |
| JANGQ-AI/Qwen3.5-35B-A3B-JANG_4K | JANG | ~19 GB | ~4 (CRITICAL=6, COMPRESS=4) | Yes | TBD |
| JANGQ-AI/Qwen3.5-35B-A3B-JANG_2S | JANG | ~12 GB | ~2.5 (CRITICAL=6, COMPRESS=2) | Yes | TBD |

## oQ4 Analysis
- 227 per-layer quantization overrides in config.json (known speed regression bug)
- group_size=32 (smaller than standard 64)
- Mixed formats: mxfp4 (U8 scales) + affine (BF16 scales)
- No VLM support
- No reasoning/thinking mode documented

## Tests to Run
- [ ] MMLU (200 questions, 10 subjects)
- [ ] Speed (tok/s generation + prefill)
- [ ] Coherency (code, math, reasoning)
- [ ] Compare at matching size points

## Results

(pending)
