"""Build a text shim from a gemma4 bundle and generate, to confirm coherence."""
import json, sys, shutil
from pathlib import Path
import mlx.core as mx

BUN = Path(sys.argv[1]).expanduser()
SHIM = Path(sys.argv[2]).expanduser()
FRAG = ("vision_embedder", "embed_vision", "embed_audio", "vision_tower", "audio_tower")

SHIM.mkdir(parents=True, exist_ok=True)
cfg = json.loads((BUN / "config.json").read_text())
tcfg = dict(cfg.get("text_config", cfg))
tcfg["model_type"] = "gemma4_text"
tcfg["tie_word_embeddings"] = cfg.get("tie_word_embeddings", True)
# remap quantization block keys language_model.model.* -> model.*
q = cfg.get("quantization", {})
nq = {}
for k, v in q.items():
    if k in ("group_size", "bits", "mode"):
        nq[k] = v
    elif any(f in k for f in FRAG):
        continue
    else:
        nq[k.replace("language_model.model.", "model.", 1)] = v
if nq:
    tcfg["quantization"] = nq
(SHIM / "config.json").write_text(json.dumps(tcfg, indent=2))
for f in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
          "generation_config.json", "special_tokens_map.json"):
    if (BUN / f).exists():
        shutil.copy2(BUN / f, SHIM / f)

# load + remap + drop multimodal -> single model.safetensors
idx = json.loads((BUN / "model.safetensors.index.json").read_text())
weights = {}
for shard in sorted(set(idx["weight_map"].values())):
    for k, v in mx.load(str(BUN / shard)).items():
        if any(f in k for f in FRAG):
            continue
        weights[k.replace("language_model.model.", "model.", 1)] = v
mx.save_safetensors(str(SHIM / "model.safetensors"), weights)
print(f"shim built: {len(weights)} tensors -> {SHIM}")

from mlx_lm import load, generate
model, tok = load(str(SHIM))
print(f"loaded: {type(model).__name__}")
for qn in ["What is the capital of France? Answer in one word.",
           "What is 17 + 28?",
           "Write one sentence about the ocean."]:
    prompt = tok.apply_chat_template([{"role": "user", "content": qn}],
                                     add_generation_prompt=True, tokenize=False)
    out = generate(model, tok, prompt=prompt, max_tokens=64, verbose=False)
    print(f"\nQ: {qn}\nA: {out.strip()}")
