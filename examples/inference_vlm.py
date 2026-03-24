"""JANG VLM Inference — Process images with vision-language models.

Requirements:
    pip install "jang[mlx]" "mlx-vlm>=0.4.1"

Works with VLM models: Qwen3.5-VL, Mistral Small 4 (Pixtral), etc.
"""
import sys
import json
import gc
from pathlib import Path
import mlx.core as mx
import mlx.nn as nn
from mlx_vlm.utils import (
    get_model_and_args, update_module_configs,
    skip_multimodal_module, load_processor,
)
from mlx_vlm import generate
from mlx_vlm.prompt_utils import apply_chat_template
from jang_tools.loader import _get_v2_weight_files, _fix_quantized_bits

model_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("JANGQ-AI/Mistral-Small-4-119B-A6B-JANG_2L")
image_path = sys.argv[2] if len(sys.argv) > 2 else None
prompt_text = sys.argv[3] if len(sys.argv) > 3 else "Describe what you see in this image."

print(f"Loading VLM: {model_path}...")
vlm_config = json.loads((model_path / "config.json").read_text())

model_class, _ = get_model_and_args(config=vlm_config)
model_config = model_class.ModelConfig.from_dict(vlm_config)
modules = ["text", "vision", "perceiver", "projector", "audio"]
model_config = update_module_configs(model_config, model_class, vlm_config, modules)
model = model_class.Model(model_config)

def class_pred(p, m):
    if skip_multimodal_module(p): return False
    if "gate" in p and "gate_proj" not in p: return False
    return hasattr(m, "to_quantized")

qbits = vlm_config.get("quantization", {}).get("bits", 4)
nn.quantize(model, group_size=64, bits=qbits, class_predicate=class_pred)

for sf in _get_v2_weight_files(model_path):
    w = mx.load(str(sf))
    clean = {k: v for k, v in w.items()
             if not k.endswith(".importance") and not k.startswith("mtp.")
             and "activation_scale" not in k and "scale_inv" not in k}
    try: clean = model.sanitize(clean)
    except: pass
    model.load_weights(list(clean.items()), strict=False)
    del clean, w; gc.collect()

_fix_quantized_bits(model, {})
model.set_dtype(mx.bfloat16)
mx.eval(model.parameters())

processor = load_processor(model_path, processor_config=vlm_config)
model.config = model_config
print("Loaded!")

if image_path:
    prompt = apply_chat_template(processor, config=model_config,
        prompt=prompt_text, images=[image_path])
    output = generate(model, processor, prompt, max_tokens=200, verbose=True, image=[image_path])
else:
    prompt = apply_chat_template(processor, config=model_config, prompt=prompt_text)
    output = generate(model, processor, prompt, max_tokens=200, verbose=True)

print(f"\nOutput: {output.text if hasattr(output, 'text') else output}")
