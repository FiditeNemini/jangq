# Swift / vMLX Integration — Nemotron-3-Nano-Omni-30B-A3B

Status as of 2026-04-28:

| Stage | Component | Status |
|---|---|---|
| Stage 1 | Text-only Swift LLM via `vmlx-swift-lm/Libraries/MLXLLM/Models/NemotronH.swift` | ✅ **WORKING** (affine path, used by Cascade-2) |
| Stage 2 | Swift JANGTQ wrapper for nemotron_h | ⏳ queued (~600 LOC) |
| Stage 3 | Swift native multimodal (RADIO + parakeet + projectors + bridge) | ⏳ queued (~3500 LOC) |

The text-only Swift path **already loads and runs** the omni-merged bundles
because `mlx_lm.nemotron_h.sanitize` (mirrored in our Swift loader) drops
multimodal-only keys at load time. Only the LLM portion runs; vision/audio
are silently ignored.

## Existing files

```
~/vmlx-swift-lm/Libraries/MLXLLM/Models/NemotronH.swift          1020 LOC  ✅
~/vmlx-swift-lm/Tests/MLXLMTests/NemotronHTests.swift             coverage
~/vmlx/swift/Sources/vMLXLLM/Models/NemotronH.swift              1251 LOC  ✅ (vmlx-tree mirror)
~/vmlx/swift/Tests/vMLXTests/NemotronHLatentMoETests.swift        coverage
```

These cover:
- `MambaCache` (size=2, conv + ssm state) and hybrid pattern parsing
- `NemotronHMamba2Mixer` with prefill (ssm_attn) + decode (ssm_update kernel) paths
- `NemotronHAttention` (GQA, no RoPE)
- `NemotronHMoE` with `switch_mlp.fc1/fc2` (already correct naming)
- `NemotronHModel.newCache()` returns `[MambaCache | KVCacheSimple | nil]` per layer

## Stage-2 Swift JANGTQ wrapper

Required to load JANGTQ4 / JANGTQ2 omni bundles fully (today they load via
the affine path and silently fall back to dequantized routed experts, which
is functional but loses the TQ-codec speed/quality benefit).

### Files to add

```swift
// vmlx-swift-lm/Libraries/MLXLLM/Models/NemotronHJANGTQ.swift  (~700 LOC)
// vmlx-swift-lm/Libraries/MLXLLM/JangLoader+NemotronH.swift    (~80 LOC)
```

### Skeleton

```swift
import MLX
import MLXNN
import MLXLLM
import vMLXJangTQ  // existing TurboQuantSwitchLinear, TurboQuantLinear

public class NemotronHJANGTQModel: NemotronHModel {

    /// Override sanitize to strip omni multimodal keys + remap per-expert
    /// .tq_packed / .tq_norms tensors into stacked switch_mlp.fc1/fc2 modules.
    override public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var w = super.sanitize(weights: weights)

        // Drop multimodal keys (matches Python mlx_lm.nemotron_h.sanitize patch)
        let omniPrefixes = ["mtp.", "vision_model.", "sound_encoder.",
                            "mlp1.", "sound_projection."]
        w = w.filter { k, _ in !omniPrefixes.contains(where: k.starts(with:)) }

        // Group per-expert TQ tensors → stacked switch_mlp.fc1/fc2
        // Pattern: backbone.layers.{N}.mixer.experts.{E}.{up_proj,down_proj}.tq_*
        // Stack along expert dim → switch_mlp.{fc1,fc2}.{tq_packed,tq_norms,tq_bits}
        // (See Python `nemo_pat` regex in load_jangtq.py:684)
        // ...

        return w
    }

    /// After load, walk modules and replace the SwitchLinear (fc1, fc2) with
    /// TurboQuantSwitchLinear when the corresponding tq_packed/.tq_norms are
    /// present in the loaded weights.
    public func wireTurboQuantExperts() {
        // For each MoE layer (block_type == .E):
        //   - Find switch_mlp.fc1.tq_packed → instantiate TurboQuantSwitchLinear
        //   - Same for fc2
        //   - Replace via `setValue(forKey:)`
        // ...
    }
}
```

### JangLoader detection hook

```swift
// JangLoader+NemotronH.swift
extension JangLoader {
    /// Detect omni-capable nemotron_h bundles by checking jang_config.json.
    static func detectNemotronOmni(at path: URL) -> Bool {
        // jang_config.json has "modality": "omni" and source_arch contains "Omni"
        // ...
    }

    /// Wire NemotronHJANGTQModel for nemotron_h + weight_format=mxtq paths.
    static func loadNemotronH(at path: URL) throws -> LLMModel {
        let cfg = try readJangConfig(at: path)
        if cfg.weight_format == "mxtq" {
            return try NemotronHJANGTQModel.from(directory: path)
        }
        return try NemotronHModel.from(directory: path)
    }
}
```

## Stage-3 Swift native multimodal

This is the biggest remaining piece. Plan:

| File | LOC est | Purpose |
|---|---:|---|
| `RADIOVision.swift` | 1500 | RADIO ViT-Huge: patch_generator + 32 transformer blocks (LN + Attn + MLP) |
| `RADIOPatchGenerator.swift` | 250 | Im2Patches + ViTPatchLinear + pos_embed interpolation + cls_token concat |
| `Parakeet.swift` | 1000 | 24-layer Conformer encoder: subsampling conv + attention + conv module + FF |
| `Projectors.swift` | 100 | mlp1 (LN+Linear+GELU+Linear) + sound_projection (RMSNorm+Linear+SqReLU+Linear) |
| `ImagePreprocess.swift` | 400 | NVLM 1-D tile preprocessing using `CIImage` + `Accelerate` |
| `VideoPreprocess.swift` | 350 | Frame extraction via `AVAsset` + EVS pruning |
| `AudioPreprocess.swift` | 250 | 16 kHz mono mel spectrogram via `Accelerate.vDSP` |
| `NemotronHOmni.swift` | 600 | Wrapper: holds vision + sound + projectors + LLM, manages inputs_embeds inject + multi-turn cache |

### RADIO ViT structure (for the Swift port)

The model is a **timm vit_huge_patch16_224** with NVIDIA's CPE patch +
ClsToken. Tensor naming on disk:

```
vision_model.radio_model.model.patch_generator.cls_token.token        (n_tokens, embed_dim) e.g. (1, 1280)
vision_model.radio_model.model.patch_generator.embedder.weight        (embed_dim, 3*16*16=768)
vision_model.radio_model.model.patch_generator.pos_embed              (1, num_patches=32*32=1024, embed_dim)
vision_model.radio_model.model.patch_generator.video_embedder.weight  (embed_dim, 3*16*16*T=1536) for T=2 temporal patches
vision_model.radio_model.model.blocks.{0..31}.norm1.{weight,bias}     LayerNorm
vision_model.radio_model.model.blocks.{0..31}.attn.qkv.{weight,bias}  Linear 1280→3840
vision_model.radio_model.model.blocks.{0..31}.attn.proj.{weight,bias} Linear 1280→1280
vision_model.radio_model.model.blocks.{0..31}.norm2.{weight,bias}     LayerNorm
vision_model.radio_model.model.blocks.{0..31}.mlp.fc1.{weight,bias}   Linear 1280→5120
vision_model.radio_model.model.blocks.{0..31}.mlp.fc2.{weight,bias}   Linear 5120→1280
vision_model.radio_model.model.norm.{weight,bias}                     LayerNorm (final)
vision_model.radio_model.summary_idxs                                 register buffer (k,)
```

Forward (eval mode, single tile at 512×512, patch_size=16):
1. **patch_generator(x)**:
   - `Im2Patches(16)`: reshape (B,3,512,512) → (B, 32×32=1024, 3×16×16=768)
   - `embedder` Linear: (B, 1024, 768) → (B, 1024, 1280)
   - `apply_pos_enc`: bilinear interpolate pos_embed (1,1024,1280) to current input dims (1024 patches → no resize since input matches max)
   - `cls_token` concat: prepend N_cls × 1280 → (B, 1024+N_cls, 1280)
2. **blocks** (32× LN+Attn+MLP residual stacks): standard ViT
3. **norm**: final LayerNorm
4. Output features = result[:, num_cls:, :] → (B, 1024, 1280)

### Parakeet structure (for the Swift port)

24-layer Conformer encoder. Per block:
- `pre_encode`: subsampling conv (3 layers, factor=8: stride=2 each)
- `layers.{0..23}`:
  - `feed_forward1`: half-FF (LN + Linear → SiLU → Linear, scale 0.5 residual)
  - `attention`: pre-LN + multi-head self-attn with relative pos
  - `conv`: pre-LN + Conformer conv module (pointwise conv → GLU → 1D conv kernel=9 → batch-norm → SiLU → pointwise conv)
  - `feed_forward2`: half-FF (same as ff1)
  - `final_norm`: LayerNorm

Audio frame rate: 16 kHz → 100 Hz mel → 100/8 = 12.5 Hz output.

### Multi-turn cache integration

Re-use existing `MambaCache` and `KVCacheSimple` from `NemotronH.swift`. The
omni wrapper just needs to:
1. Pass cache through to `NemotronHModel.callAsFunction`
2. Persist cache between calls (already supported by `MambaCache` reference semantics)
3. NEW: cache the encoded image/audio embeds per turn so re-asking about the
   same image doesn't re-run the encoder.

## Why isn't the full Swift native port done today?

Honest answer: ~3500 LOC of careful Swift + correctness verification against
PyTorch reference outputs is multi-day work. We have:
- **Stage 1 working today (Python hybrid, all 3 quants, multi-turn verified)**
- Architecture spec for stage 2/3 (this doc)
- Existing Swift LLM Stage-1 path that loads omni bundles (text-only)

The pragmatic order is:
1. ✅ Ship Python hybrid — done
2. ⏳ Swift JANGTQ for nemotron_h LLM (small port, 700 LOC)
3. ⏳ Native MLX RADIO Python (validate against PyTorch, 1500 LOC)
4. ⏳ Native MLX parakeet Python (1000 LOC)
5. ⏳ Swift native multimodal (after Python is correct)

Each step has clear validation: produce identical embeddings to the PyTorch
reference for a fixed image/audio input.

## How a Swift agent uses what's there today

```swift
import MLXLLM
import MLXLMCommon

// Load omni-capable bundle — text-only path works fully on Swift today.
let bundle = ModelConfiguration(
    id: "OsaurusAI/Nemotron-3-Nano-Omni-30B-A3B-MXFP4"
)
let container = try await LLMModelFactory.shared.loadContainer(
    configuration: bundle
)

let userMessage = UserInput(prompt: "What is the capital of France?")
let result = try await container.perform { context in
    let messages = try await context.processor.prepare(input: userMessage)
    let stream = try MLXLMCommon.generate(
        input: messages, parameters: .init(maxTokens: 50),
        context: context
    )
    var out = ""
    for await token in stream {
        out += token.chunk
    }
    return out
}
print(result)
```

For multi-turn: pass `MLXLMCommon.GenerateParameters(... withCache: ...)` to
keep KV+Mamba state across turns. The exact API depends on
`vmlx-swift-lm` version — see `Tests/MLXLMTests/NemotronHTests.swift` for
working examples.

## Multimodal in Swift TODAY (workaround)

Until stage 3 lands, Swift agents that need image / audio / video input
should:
1. Pre-encode the modalities **in Python** (one-time, write embeds to disk).
2. Load embeddings + use Swift LLM with custom inputs_embeds path.

Or just call the Python `nemotron_omni_session` runtime via a subprocess /
Process bridge. The Python runtime is the production multimodal path today.
