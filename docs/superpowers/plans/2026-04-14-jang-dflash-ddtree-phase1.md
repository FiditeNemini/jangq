# JANG-DFlash + DDTree Phase 1 Implementation Plan

> **Design reference:** `docs/superpowers/specs/2026-04-14-jang-dflash-ddtree-design.md`
> **Goal:** Beat 50 tok/s baseline on MiniMax-JANG_2L. Stretch 200+ tok/s on M3 Ultra.

**Architecture recap:** 5-layer block-diffusion drafter (JangDFlashDrafter) consumes hidden states from 5 evenly-spaced target layers via KV injection, emits B=16 parallel candidate distributions in one forward, top-k=4 per slot → lattice beam m=60 → prefix-trie → EAGLE-2 tree-attention verification on MiniMax target.

**Tech stack:** PyTorch for drafter training on 5090. MLX-Swift for inference on Apple Silicon. JANG loader handles drafter weights. vmlx-swift git tree is at `/Users/eric/vmlx/swift/`.

---

## Task 1: Python drafter scaffold (PyTorch, trains on 5090)

**Files:**
- Create: `/Users/eric/jang/jang-tools/jang_tools/dflash/__init__.py`
- Create: `/Users/eric/jang/jang-tools/jang_tools/dflash/drafter.py`
- Create: `/Users/eric/jang/jang-tools/jang_tools/dflash/config.py`
- Test: `/Users/eric/jang/jang-tools/tests/test_dflash_drafter.py`

- [ ] **Step 1: Write the failing test for drafter forward shape**

```python
# tests/test_dflash_drafter.py
import torch
from jang_tools.dflash import JangDFlashConfig, JangDFlashDrafter

def test_drafter_forward_shapes():
    cfg = JangDFlashConfig(
        vocab_size=200064, hidden_dim=1536, num_layers=5,
        num_heads=12, num_kv_heads=4, ffn_dim=4096, block_size=16,
        mask_token_id=200064, tap_dim=15360,  # 5 * 3072
    )
    drafter = JangDFlashDrafter(cfg)
    block = torch.full((2, 16), cfg.mask_token_id, dtype=torch.long)
    block[:, 0] = 42  # anchor
    h_ctx = torch.randn(2, 16, cfg.hidden_dim, dtype=torch.bfloat16)
    logits = drafter(block, h_ctx_kv=h_ctx)
    assert logits.shape == (2, 16, cfg.vocab_size)
    assert logits.dtype == torch.bfloat16
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/eric/jang && python -m pytest jang-tools/tests/test_dflash_drafter.py -v`
Expected: FAIL with ImportError

- [ ] **Step 3: Write `config.py`**

```python
# jang_tools/dflash/config.py
from dataclasses import dataclass

@dataclass
class JangDFlashConfig:
    vocab_size: int = 200064
    hidden_dim: int = 1536
    num_layers: int = 5
    num_heads: int = 12
    num_kv_heads: int = 4
    ffn_dim: int = 4096
    block_size: int = 16
    mask_token_id: int = 200064
    tap_dim: int = 15360          # 5 tap layers × 3072 target hidden
    head_dim: int = 128
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-6
    loss_gamma: float = 7.0
```

- [ ] **Step 4: Write `drafter.py` — JangDFlashDrafter**

Full model with `JangDFlashAttention`, `JangDFlashFFN`, `JangDFlashBlock`, and `JangDFlashDrafter` classes. Key method signature:
`def forward(self, block_ids, h_taps=None, h_ctx_kv=None)`. Pass exactly one of the two h_* arguments. When `h_taps` is provided, run `fusion_mlp` first; when `h_ctx_kv` is provided, skip fusion.

The attention layer takes two K/V sources: the block's own token-derived K/V (via `wk`, `wv`) and the injected context's K/V (via `wk_ctx`, `wv_ctx`). Concatenate them along the sequence axis with ctx first. RoPE only on block positions. Block-side mask is causal within block and all-ones against context.

- [ ] **Step 5: Run test to verify it passes**

```bash
python -m pytest jang-tools/tests/test_dflash_drafter.py -v
```

- [ ] **Step 6: Commit**

```bash
cd /Users/eric/jang
git add jang-tools/jang_tools/dflash/ jang-tools/tests/test_dflash_drafter.py
git commit -m "jang-dflash: scaffold PyTorch drafter with KV injection"
```

---

## Task 2: Weighted masked CE loss

**Files:**
- Modify: `jang-tools/jang_tools/dflash/drafter.py` (add `dflash_loss`)
- Test: `jang-tools/tests/test_dflash_loss.py`

- [ ] **Step 1: Test**

```python
# tests/test_dflash_loss.py
import torch
from jang_tools.dflash import JangDFlashConfig
from jang_tools.dflash.drafter import dflash_loss

def test_loss_decay_weights():
    cfg = JangDFlashConfig(block_size=16, loss_gamma=7.0, vocab_size=100)
    logits = torch.randn(2, 16, 100)
    targets = torch.randint(0, 100, (2, 16))
    loss = dflash_loss(logits, targets, cfg)
    assert torch.isfinite(loss)

def test_loss_zero_when_perfect():
    cfg = JangDFlashConfig(block_size=16, loss_gamma=7.0, vocab_size=100)
    targets = torch.randint(0, 100, (2, 16))
    logits = torch.full((2, 16, 100), -10.0)
    logits.scatter_(-1, targets.unsqueeze(-1), 10.0)
    loss = dflash_loss(logits, targets, cfg)
    assert loss.item() < 1e-3
```

- [ ] **Step 2: Implement `dflash_loss`**

Append to `jang_tools/dflash/drafter.py`:

```python
def dflash_loss(logits, targets, cfg):
    """
    Weighted masked cross-entropy (DFlash Eq. 4).
    logits:  [B, L, V], targets: [B, L]
    Position 0 is the anchor (always clean), skipped in the loss.
    w_k = exp(-(k-1)/gamma) for k in 1..L-1.
    """
    import torch.nn.functional as F
    B, L, V = logits.shape
    assert L == cfg.block_size
    pred = logits[:, 1:, :].reshape(-1, V)
    tgt = targets[:, 1:].reshape(-1)
    ks = torch.arange(1, L, device=logits.device, dtype=logits.dtype)
    w = torch.exp(-(ks - 1) / cfg.loss_gamma).unsqueeze(0).expand(B, -1).reshape(-1)
    per_tok = F.cross_entropy(pred, tgt, reduction="none")
    return (per_tok * w).sum() / w.sum()
```

- [ ] **Step 3: Test, commit**

```bash
python -m pytest jang-tools/tests/test_dflash_loss.py -v
git add jang-tools/jang_tools/dflash/drafter.py jang-tools/tests/test_dflash_loss.py
git commit -m "jang-dflash: add weighted masked CE loss (DFlash Eq. 4)"
```

---

## Task 3: Hidden-tap distillation data generator

**Files:**
- Create: `jang-tools/jang_tools/dflash/distill_data.py`

- [ ] **Step 1: Write the generator that wraps mlx_lm with a layer-tap hook**

It iterates prompts, calls `mlx_lm.stream_generate`, captures intermediate layer outputs by patching `model.model.layers[i]` with a wrapper, records `(h_taps, tokens)` per prompt, writes one safetensors file per prompt (`h_taps` as fp16, `tokens` as int32).

Tap layer indices hardcoded to `[10, 22, 34, 46, 58]` (5 evenly spaced through MiniMax's 62 layers).

- [ ] **Step 2: Smoke-run 3 prompts**

```bash
printf "What is 2+2?\nExplain photosynthesis.\nName three planets.\n" > /tmp/prompts.txt
cd /Users/eric/jang && python -m jang_tools.dflash.distill_data \
  --model /Users/eric/models/MiniMax-M2.7-JANG_2L \
  --prompts /tmp/prompts.txt \
  --out /tmp/dflash-distill-smoke \
  --max-tokens 32 --limit 3
ls /tmp/dflash-distill-smoke/ | wc -l
```

Expected: 3 files. Fix the tap hook if it reports 0.

- [ ] **Step 3: Commit**

```bash
git add jang-tools/jang_tools/dflash/distill_data.py
git commit -m "jang-dflash: distillation data generator with layer tap"
```

---

## Task 4: Training loop (runs on 5090)

**Files:**
- Create: `jang-tools/jang_tools/dflash/train.py`

- [ ] **Step 1: Trainer skeleton**

```python
# jang_tools/dflash/train.py
import argparse, glob
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from safetensors import safe_open
from .config import JangDFlashConfig
from .drafter import JangDFlashDrafter, dflash_loss

class DistillDataset(Dataset):
    def __init__(self, root):
        self.files = sorted(glob.glob(f"{root}/*.safetensors"))
    def __len__(self): return len(self.files)
    def __getitem__(self, idx):
        with safe_open(self.files[idx], framework="pt") as sf:
            h_taps = sf.get_tensor("h_taps").to(torch.bfloat16)
            tokens = sf.get_tensor("tokens").long()
        # h_taps: [5, T, 3072] → [T, 5*3072]
        h_taps = h_taps.permute(1, 0, 2).reshape(h_taps.shape[1], -1)
        return h_taps, tokens

def collate(batch, cfg):
    B = cfg.block_size
    h_ctx_list, in_list, tgt_list = [], [], []
    for h_taps, tokens in batch:
        T = tokens.shape[0]
        if T < B: continue
        anchor = torch.randint(0, T - B + 1, (1,)).item()
        block_tokens = tokens[anchor:anchor+B].clone()
        block_input = block_tokens.clone()
        block_input[1:] = cfg.mask_token_id
        h_ctx_list.append(h_taps[anchor:anchor+B])
        in_list.append(block_input)
        tgt_list.append(block_tokens)
    return (
        torch.stack(h_ctx_list),
        torch.stack(in_list),
        torch.stack(tgt_list),
    )

def train(args):
    cfg = JangDFlashConfig()
    drafter = JangDFlashDrafter(cfg).to("cuda").to(torch.bfloat16)
    opt = torch.optim.AdamW(drafter.parameters(), lr=args.lr, betas=(0.9, 0.95))
    ds = DistillDataset(args.data)
    loader = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=4,
                        collate_fn=lambda b: collate(b, cfg))
    step = 0
    for epoch in range(1000):
        for h_ctx, block_in, tgt in loader:
            h_ctx = h_ctx.to("cuda", torch.bfloat16)
            block_in = block_in.to("cuda"); tgt = tgt.to("cuda")
            logits = drafter(block_in, h_taps=h_ctx)
            loss = dflash_loss(logits, tgt, cfg)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(drafter.parameters(), 1.0)
            opt.step()
            step += 1
            if step % 10 == 0:
                print(f"step={step} loss={loss.item():.4f}")
            if step >= args.max_steps: break
        if step >= args.max_steps: break
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    torch.save(drafter.state_dict(), out / "drafter.pt")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--max-steps", type=int, default=2000)
    train(p.parse_args())
```

- [ ] **Step 2: Smoke run 5 steps on smoke data**

```bash
cd /Users/eric/jang && python -m jang_tools.dflash.train \
  --data /tmp/dflash-distill-smoke \
  --out /tmp/dflash-drafter-smoke \
  --max-steps 5 --batch 1
```

Expected: 5 loss lines, no NaN, writes `drafter.pt`.

- [ ] **Step 3: Commit**

```bash
git add jang-tools/jang_tools/dflash/train.py
git commit -m "jang-dflash: PyTorch distillation trainer"
```

---

## Task 5: Checkpoint converter PT → MLX safetensors

**Files:**
- Create: `jang-tools/jang_tools/dflash/convert_to_mlx.py`

- [ ] **Step 1: Write converter**

Walks a PT state dict, casts to fp16 numpy, writes safetensors preserving key paths so they match Swift module keys.

- [ ] **Step 2: Run on smoke checkpoint**

- [ ] **Step 3: Commit**

---

## Task 6: Swift drafter module

**Files:**
- Create: `/Users/eric/vmlx/swift/Sources/vMLXLMCommon/DFlash/JangDFlashDrafter.swift`
- Create: `/Users/eric/vmlx/swift/Sources/vMLXLMCommon/DFlash/JangDFlashConfig.swift`

- [ ] **Step 1: `JangDFlashConfig.swift`**

Codable struct mirroring the Python config fields.

- [ ] **Step 2: `JangDFlashDrafter.swift`**

Mirrors the PyTorch drafter exactly:
- `JangDFlashAttention` (wq/wk/wv/wo + wk_ctx/wv_ctx + q_norm/k_norm, concat ctx & block K/V, SDPA with block-side causal + all-ones-to-ctx mask)
- `JangDFlashFFN` (SwiGLU: w1/w3 gated, w2 down)
- `JangDFlashBlock` (pre-norm residual)
- `JangDFlashDrafter` (embed, fusion_mlp, layers, norm, lm_head)

Forward `callAsFunction(_ blockIDs: MLXArray, hTaps: MLXArray) -> MLXArray`. Fusion MLP inside the forward runs once per call.

- [ ] **Step 3: Build clean**

```bash
cd /Users/eric/vmlx/swift && swift build --target vMLXLMCommon 2>&1 | tail -30
```

- [ ] **Step 4: Commit in the vmlx tree**

```bash
cd /Users/eric/vmlx && git add swift/Sources/vMLXLMCommon/DFlash/
git commit -m "dflash: scaffold JangDFlashDrafter Swift module"
```

---

## Task 7: Hidden tap + tree-attention mask in MiniMax.swift

**Files:**
- Modify: `/Users/eric/vmlx/swift/Sources/vMLXLLM/Models/MiniMax.swift`

- [ ] **Step 1: Extend `MiniMaxModelInner.callAsFunction`**

Add optional `tapLayers: Set<Int>?`. When non-nil, collect `hiddens[i]` after each matching layer into a `[Int: MLXArray]` dictionary and return it alongside the logits.

- [ ] **Step 2: Add tree-mask parameter**

Add optional `treeMask: MLXArray?` threaded down to every attention layer. When non-nil, replace the default causal mask with `treeMask` (broadcasted to `[B, heads, N, N]` additive bias).

- [ ] **Step 3: Build + run existing MiniMax tests**

```bash
cd /Users/eric/vmlx/swift && swift build --target vMLXLLM
```

- [ ] **Step 4: Commit**

```bash
cd /Users/eric/vmlx && git add swift/Sources/vMLXLLM/Models/MiniMax.swift
git commit -m "minimax: optional layer-hidden-tap and tree-attention mask"
```

---

## Task 8: DDTreeBuilder (Swift CPU)

**Files:**
- Create: `/Users/eric/vmlx/swift/Sources/vMLXLMCommon/DFlash/DDTreeBuilder.swift`
- Test: `/Users/eric/vmlx/swift/Tests/vMLXLMCommonTests/DDTreeBuilderTests.swift`

- [ ] **Step 1: Write failing tests**

`testBeamTopMBasic` — 2 slots, k=2 each, verify top-3 paths by joint prob.
`testTreeMaskShape` — 2 paths sharing prefix, verify flat tokens deduped + mask shape + ancestry.

- [ ] **Step 2: Implement `DDTreeBuilder`**

Two methods:
- `beamTopMLattice(vals: [[Float]], ids: [[Int]], m: Int) -> [Path]` — log-space beam search over slots, keep top-m by joint log-prob
- `flatten(paths: [Path]) -> ([Int] flat, [[Bool]] mask, [Int] leafIdx)` — prefix trie, DFS flat order, mask[i][j]=true iff j is ancestor-or-self of i

`Path` struct holds `tokens: [Int]` and `logProb: Float`.

- [ ] **Step 3: Tests pass**

```bash
cd /Users/eric/vmlx/swift && swift test --filter DDTreeBuilderTests
```

- [ ] **Step 4: Commit**

---

## Task 9: JangDFlashSpecDec end-to-end step

**Files:**
- Create: `/Users/eric/vmlx/swift/Sources/vMLXLMCommon/DFlash/JangDFlashSpecDec.swift`

- [ ] **Step 1: Implement `step(state:) -> [Int]`**

Per-step flow:
1. Target forward on accepted prefix with `tapLayers=[10,22,34,46,58]`, grab `hTapDict`
2. Build `hTapsFlat: [1, T, 5*3072]` by concat(h_tap[10..58], axis=-1)
3. Slice last B positions → `hCtxBlock: [1, B, 5*3072]`
4. `blockInput: [1, B]` = [bonus, MASK, MASK, ..., MASK]
5. `logits = drafter(blockInput, hTaps: hCtxBlock)`
6. `probs = softmax(logits[:, 1:], axis=-1)`
7. `topKVals, topKIds = topK(probs, k=4, axis=-1)` (gather into 2D Swift arrays)
8. `paths = DDTreeBuilder.beamTopMLattice(vals, ids, m: 60)`
9. `(flat, mask, _) = DDTreeBuilder.flatten(paths)`
10. Build MLXArray tree mask from `[[Bool]]` (0/-inf additive bias)
11. Target forward on `acceptedPrefix + flat` with `treeMask`
12. Walk trie top-down, rejection sample per depth, return accepted ids

- [ ] **Step 2: Unit test with dummy target (2-layer Llama) + randomly-init drafter**

Verify step returns 1..B tokens, no NaNs, output ∈ vocab range.

- [ ] **Step 3: Commit**

---

## Task 10: Bench untrained drafter end-to-end (sanity)

**Files:**
- Modify: `Sources/vMLXCLI/main.swift` (add `--dflash-drafter PATH` flag to serve/chat)
- Create: `docs/superpowers/notes/2026-04-14-dflash-e2e-untrained-bench.md`

- [ ] **Step 1: Wire the flag through**

When `--dflash-drafter` is present, load the drafter safetensors via `JangLoader` + `JangDFlashDrafter`, wrap the target model's generate call in `JangDFlashSpecDec.step(...)`.

- [ ] **Step 2: Bench**

```bash
/Users/eric/vmlx/swift/.build/arm64-apple-macosx/release/vmlxctl serve \
  -m /Users/eric/models/MiniMax-M2.7-JANG_2L.jangspec \
  --dflash-drafter /tmp/dflash-drafter-smoke/drafter.safetensors \
  -p 8765
```

Run bench script (`/tmp/mm-isolate.sh`). Untrained drafter → acceptance ~0, effective tok/s probably LOWER than baseline 14.99 (draft overhead + rejected verification). Purpose of this task: confirm the pipeline completes, no crash, tokens are coherent when you feed the accepted prefix back (verification loop works mechanically even at low α).

- [ ] **Step 3: Record baseline in notes, commit**

---

## Task 11: Generate 5k-prompt distillation corpus (M3 Ultra)

- [ ] **Step 1: Build prompt list**

`prompts-5k.txt`: 1000 GSM8K train + 164 HumanEval + 1000 MMLU-STEM + 2836 OpenOrca reasoning.

- [ ] **Step 2: Run distill_data.py on M3 Ultra**

```bash
# On M3 Ultra
cd ~/jang && python -m jang_tools.dflash.distill_data \
  --model /Users/eric/models/MiniMax-M2.7-JANG_2L \
  --prompts prompts-5k.txt \
  --out /Volumes/External/dflash-distill-v1 \
  --max-tokens 256
```

Expected wall: ~14 hours at 42 tok/s. Dataset ~780 GB fp16.

- [ ] **Step 3: rsync to 5090**

---

## Task 12: Train on 5090, convert, ship

- [ ] **Step 1: Train**

```bash
# On 5090
cd /afagent/jang && python -m jang_tools.dflash.train \
  --data /data/dflash-distill-v1 \
  --out /data/dflash-drafter-v1 \
  --batch 16 --max-steps 2000 --lr 3e-4
```

Expected: ~4 hours, final loss < 2.0.

- [ ] **Step 2: Convert to MLX safetensors**

```bash
python -m jang_tools.dflash.convert_to_mlx \
  --ckpt /data/dflash-drafter-v1/drafter.pt \
  --out /data/dflash-drafter-v1/drafter.safetensors
```

- [ ] **Step 3: rsync back to M4 Max + M3 Ultra**

---

## Task 13: Trained bench, GSM8K drift check, publish

**Files:**
- Create: `docs/superpowers/notes/2026-04-14-dflash-v1-bench.md`

- [ ] **Step 1: Bench on MacBook M4 Max**

```bash
bash /tmp/mm-isolate.sh   # N=128 pure decode, prompt eval subtracted
```

Target: ≥ 60 tok/s (4× over 14.99 baseline).

- [ ] **Step 2: Bench on M3 Ultra** (user invokes; not on Mac Studio per feedback rule — user runs this themselves)

Target: ≥ 200 tok/s (4.8× over ~42 baseline).

- [ ] **Step 3: GSM8K accuracy drift**

Run 100 GSM8K prompts through target-only vs target+DFlash. Compute pass rate on both. Target: ≤ 1 pp drift.

- [ ] **Step 4: Document + commit**

```bash
git add docs/superpowers/notes/2026-04-14-dflash-v1-bench.md
git commit -m "dflash: v1 bench — M4 Max X tok/s, M3 Ultra Y tok/s, GSM8K Z"
```

---

## Completion

Plan done when Task 13 reports measured numbers and either:
- **Hit both targets** (M4 Max ≥ 60, M3 Ultra ≥ 200): ship v1, open Phase 2 for SSD-resident path on 400B+ models.
- **Miss**: document gap, iterate (bigger drafter, more data, larger B, different top-k).

## Constraints & reminders

- No AI attribution in any commit/PR
- Don't run MiniMax inference on Mac Studio (memory rule)
- All DFlash code under `Sources/vMLXLMCommon/DFlash/` (Swift) and `jang_tools/dflash/` (Python)
- Frequent small commits
- Verify every Swift change builds before moving on (fast on M4 Max)
