"""JANG Reasoning Inference — Enable/disable step-by-step thinking.

Models with reasoning support:
  - Qwen3.5: <think>...</think> tags, enable_thinking=True/False
  - Mistral Small 4: [THINK]...[/THINK] tags, reasoning_effort="high"/"none"
  - Nemotron Cascade 2: <think>...</think> tags
"""
import sys
from jang_tools.loader import load_jang_model
from mlx_lm import generate

model_path = sys.argv[1] if len(sys.argv) > 1 else "JANGQ-AI/Nemotron-Cascade-2-30B-A3B-JANG_2L"

print(f"Loading {model_path}...")
model, tokenizer = load_jang_model(model_path)

question = "What is 17 * 24? Show your work."
messages = [{"role": "user", "content": question}]

# Try enabling thinking via chat template
try:
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=True,
    )
except TypeError:
    # Mistral 4 style: reasoning_effort in template
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    if "reasoning_effort" in prompt:
        prompt = prompt.replace('reasoning_effort": "none', 'reasoning_effort": "high')

print("=== Reasoning ON ===")
output = generate(model, tokenizer, prompt=prompt, max_tokens=500, verbose=True)
print(output)
