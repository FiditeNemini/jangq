"""JANG Text Inference — Load and run any JANG model.

Usage:
    pip install "jang[mlx]"
    python inference_text.py /path/to/model "Your prompt here"

Works with all JANG models: Qwen, MiniMax, Nemotron, Mistral, etc.
"""
import sys
from jang_tools.loader import load_jang_model
from mlx_lm import generate

model_path = sys.argv[1] if len(sys.argv) > 1 else "JANGQ-AI/Qwen3.5-35B-A3B-JANG_4K"
prompt_text = sys.argv[2] if len(sys.argv) > 2 else "Write a Python function to check if a number is prime."

print(f"Loading {model_path}...")
model, tokenizer = load_jang_model(model_path)

messages = [{"role": "user", "content": prompt_text}]
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
output = generate(model, tokenizer, prompt=prompt, max_tokens=500, verbose=True)
print(output)
