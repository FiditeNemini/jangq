"""Minimal example: load a JANG or JANGTQ model and generate text.

Requirements:
    pip install 'jang[mlx]'          # for JANG dense + MoE models
    pip install 'jang[vlm]'          # additionally for VL/video models

Usage:
    python python.py /path/to/JANG-model "Your prompt here"

Function name reference:
    JANG text models:    jang_tools.loader.load_jang_model()
    JANGTQ text models:  jang_tools.load_jangtq.load_jangtq_model()
    VL/video models:     jang_tools.load_jangtq_vlm.load_jangtq_vlm_model()

Both load_jang_model and load_jangtq_model return (model, tokenizer) compatible
with mlx_lm.generate. VL loaders return (model, processor) compatible with
mlx_vlm.generate.

Author: Jinho Jang (eric@jangq.ai)
"""
import json
import sys
from pathlib import Path


def _detect_model_type(model_dir: Path):
    """Read jang_config.json and determine which loader to use."""
    config_path = model_dir / "jang_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"{model_dir} is not a JANG model (no jang_config.json)")

    with open(config_path) as f:
        cfg = json.load(f)

    fmt = cfg.get("format", "")
    if fmt not in ("jang", "jjqf", "mxq"):
        raise ValueError(f"Unexpected format value in jang_config.json: {fmt!r}")

    method = cfg.get("quantization", {}).get("method", "")
    is_jangtq = method == "jangtq"
    is_vl = (model_dir / "preprocessor_config.json").exists()

    return is_jangtq, is_vl


def main() -> int:
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <model_dir> <prompt>")
        return 1

    model_dir = Path(sys.argv[1])
    prompt = sys.argv[2]

    if not model_dir.is_dir():
        print(f"ERROR: {model_dir} is not a directory")
        return 1

    try:
        is_jangtq, is_vl = _detect_model_type(model_dir)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}")
        return 1

    model_path = str(model_dir)

    if is_vl:
        # Vision/language model — use mlx_vlm.generate
        from jang_tools.load_jangtq_vlm import load_jangtq_vlm_model
        from mlx_vlm import generate as generate_vl
        from PIL import Image

        model, processor = load_jangtq_vlm_model(model_path)

        # For a text-only prompt passed to a VL model, mlx_vlm.generate still
        # needs an image argument. Use a 1x1 white placeholder.
        image = Image.new("RGB", (1, 1), (255, 255, 255))
        response = generate_vl(model, processor, image=image, prompt=prompt, max_tokens=200)
        print(getattr(response, "text", response))

    elif is_jangtq:
        # JANGTQ (TurboQuant) text model — uses codebook-based expert weights
        from jang_tools.load_jangtq import load_jangtq_model
        from mlx_lm import generate

        model, tokenizer = load_jangtq_model(model_path)

        if getattr(tokenizer, "chat_template", None):
            messages = [{"role": "user", "content": prompt}]
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

        response = generate(model, tokenizer, prompt=prompt, max_tokens=200)
        print(response)

    else:
        # Standard JANG text model (affine quantization, all architectures)
        from jang_tools.loader import load_jang_model
        from mlx_lm import generate

        model, tokenizer = load_jang_model(model_path)

        if getattr(tokenizer, "chat_template", None):
            messages = [{"role": "user", "content": prompt}]
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

        response = generate(model, tokenizer, prompt=prompt, max_tokens=200)
        print(response)

    return 0


if __name__ == "__main__":
    sys.exit(main())
