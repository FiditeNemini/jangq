"""
AWQ activation capture for JANGTQ.
Created by Jinho Jang (eric@jangq.ai)

Runs calibration text through a loaded JANGTQ model and captures
per-input-channel activation magnitudes at every TurboQuantLinear
and TurboQuantSwitchLinear module.

Output: awq_activations.safetensors — one tensor per module, shape
(in_features,), holding max(|x|) over all calibration tokens.

Usage:
    python3 -m jang_tools.awq_capture <model_path> [n_samples]
"""
import sys
import json
import time
from pathlib import Path

import numpy as np
import mlx.core as mx
from mlx.core import eval as _mx_force_compute
from safetensors.numpy import save_file

from jang_tools.load_jangtq import load_jangtq_model
from jang_tools.turboquant.tq_kernel import (
    TurboQuantLinear,
    TurboQuantSwitchLinear,
)

DEFAULT_N_SAMPLES = 32
DEFAULT_SEQ_LEN = 512

_CALIB_PROMPTS = [
    "The quick brown fox jumps over the lazy dog. This sentence contains every letter of the English alphabet, making it useful for typography and font testing.",
    "In the year 2025, artificial intelligence systems have become remarkably capable at natural language understanding, code generation, and mathematical reasoning.",
    "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)\n\n# This recursive implementation has exponential time complexity.",
    "The partial derivative of f(x, y) = x^2 + 3xy + y^2 with respect to x is 2x + 3y. Taking the second derivative gives us the Hessian matrix.",
    "Machine learning models, particularly transformers, rely on self-attention mechanisms to process sequences of tokens. The attention is computed via Q K^T / sqrt(d).",
    "SELECT user_id, COUNT(*) AS order_count FROM orders WHERE created_at >= '2025-01-01' GROUP BY user_id HAVING COUNT(*) > 10 ORDER BY order_count DESC;",
    "The Industrial Revolution began in Britain in the late 18th century, transforming agrarian societies into industrial ones through mechanization and factory production.",
    "Photosynthesis converts carbon dioxide and water into glucose and oxygen using light energy. The overall equation is 6CO2 + 6H2O + light -> C6H12O6 + 6O2.",
    "在人工智能领域,深度学习模型已经能够处理多种语言和任务。大语言模型通过在海量文本上训练,学会了理解和生成自然语言。",
    "Bonjour, je voudrais réserver une table pour deux personnes à 20 heures ce soir. Avez-vous des disponibilités? Le menu végétarien est-il disponible?",
    "The first law of thermodynamics states that energy cannot be created or destroyed, only transformed from one form to another. This is the law of energy conservation.",
    "Kubernetes pods are the smallest deployable units in the container orchestration platform. Each pod contains one or more containers that share network and storage.",
    "The United States Constitution, ratified in 1788, establishes the framework of federal government and enumerates fundamental rights of American citizens.",
    "Let f(x) = e^(x^2). By the chain rule, f'(x) = 2x * e^(x^2). The function has a global minimum at x=0 where f(0) = 1.",
    "In Shakespeare's Hamlet, the prince of Denmark contemplates existence in the famous soliloquy 'To be, or not to be, that is the question.'",
    "The mitochondrion is the powerhouse of the cell, responsible for producing ATP through oxidative phosphorylation in the inner membrane cristae.",
    "import numpy as np\n\ndef softmax(x):\n    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))\n    return exp_x / exp_x.sum(axis=-1, keepdims=True)",
    "The speed of light in vacuum is approximately 299,792,458 meters per second, denoted by the constant c in physics equations like E = mc^2.",
    "Climate change is driven primarily by greenhouse gas emissions from fossil fuel combustion, deforestation, and industrial processes, raising global temperatures.",
    "The Pythagorean theorem states that in a right-angled triangle, the square of the hypotenuse equals the sum of squares of the other two sides: a^2 + b^2 = c^2.",
    "한국어는 한글 문자를 사용하는 언어입니다. 세종대왕이 1443년에 창제했으며 과학적인 음소문자 체계를 갖추고 있습니다.",
    "Blockchain technology uses cryptographic hash functions and distributed consensus mechanisms to maintain an immutable, decentralized ledger of transactions.",
    "The human brain contains approximately 86 billion neurons, each forming thousands of synaptic connections with other neurons through chemical and electrical signaling.",
    "In organic chemistry, a carbonyl group consists of a carbon atom double-bonded to an oxygen atom. It appears in aldehydes, ketones, carboxylic acids, and esters.",
    "Assume arguendo that the defendant's intent was established beyond reasonable doubt. The jury must still consider whether the elements of the offense were met.",
    "The Fibonacci sequence appears throughout nature, from the arrangement of leaves on a stem to the spirals of seashells and the branching of trees.",
    "Quantum entanglement is a phenomenon where two or more particles become correlated in such a way that the quantum state of each cannot be described independently.",
    "Standard German word order follows subject-verb-object in main clauses but shifts the verb to final position in subordinate clauses introduced by dass or weil.",
    "The integral of sin(x) from 0 to pi equals 2. More generally, sin has period 2pi and amplitude 1, oscillating between -1 and +1.",
    "When designing distributed systems, consider the CAP theorem: you can have at most two of consistency, availability, and partition tolerance at any given time.",
    "Ancient Rome's legacy includes Roman law, concrete architecture, a network of roads, aqueducts, and the Latin language that evolved into the Romance languages.",
    "The logistic function sigmoid(x) = 1 / (1 + e^(-x)) maps real numbers to (0, 1) and is commonly used as an activation function in neural networks.",
]


def _register_hooks(model) -> tuple:
    """Monkey-patch TurboQuant modules to capture per-channel activation max.

    Returns (stats dict, originals dict) for later teardown.
    """
    stats = {}
    orig_calls = {}

    # Hook strategy: stay MLX-lazy during forward pass (no per-hook sync,
    # which serializes Metal kernels and caused hangs). After each forward
    # pass, force compute once on all accumulated stats, then convert to
    # numpy so the next forward starts with a clean graph.
    def _wrap_linear(name, module):
        orig = module.__call__

        def wrapped(x, *args, **kwargs):
            x_flat = x.reshape(-1, x.shape[-1])
            mag = mx.abs(x_flat).max(axis=0)
            prev = stats.get(name)
            if prev is None:
                stats[name] = mag
            else:
                # If prev is numpy (finalized from last sample), lift to mx.
                p = mx.array(prev) if isinstance(prev, np.ndarray) else prev
                stats[name] = mx.maximum(p, mag)
            return orig(x, *args, **kwargs)

        module.__call__ = wrapped
        orig_calls[id(module)] = orig

    def _wrap_switch(name, module):
        orig = module.__call__

        def wrapped(x, indices, *args, **kwargs):
            x_flat = x.reshape(-1, x.shape[-1])
            mag = mx.abs(x_flat).max(axis=0)
            prev = stats.get(name)
            if prev is None:
                stats[name] = mag
            else:
                p = mx.array(prev) if isinstance(prev, np.ndarray) else prev
                stats[name] = mx.maximum(p, mag)
            return orig(x, indices, *args, **kwargs)

        module.__call__ = wrapped
        orig_calls[id(module)] = orig

    n_linear = n_switch = 0
    for name, mod in model.named_modules():
        if isinstance(mod, TurboQuantSwitchLinear):
            _wrap_switch(name, mod); n_switch += 1
        elif isinstance(mod, TurboQuantLinear):
            _wrap_linear(name, mod); n_linear += 1

    print(f"  hooked {n_linear} TurboQuantLinear + {n_switch} TurboQuantSwitchLinear",
          flush=True)
    return stats, orig_calls


def _unregister_hooks(model, orig_calls):
    for _, mod in model.named_modules():
        if id(mod) in orig_calls:
            mod.__call__ = orig_calls[id(mod)]


def run_capture(model_path, n_samples=DEFAULT_N_SAMPLES,
                seq_len=DEFAULT_SEQ_LEN, output_path=None):
    model_path = Path(model_path)
    if output_path is None:
        output_path = model_path / "awq_activations.safetensors"
    output_path = Path(output_path)

    print("=" * 60)
    print("  AWQ Activation Capture")
    print("=" * 60)
    print(f"  Model:   {model_path}")
    print(f"  Output:  {output_path}")
    print(f"  Samples: {n_samples}  SeqLen: {seq_len}", flush=True)

    t0 = time.time()
    model, tokenizer = load_jangtq_model(str(model_path))
    print(f"  model loaded in {time.time()-t0:.1f}s", flush=True)

    prompts = list(_CALIB_PROMPTS)
    while len(prompts) < n_samples:
        prompts.append(_CALIB_PROMPTS[len(prompts) % len(_CALIB_PROMPTS)])
    prompts = prompts[:n_samples]

    stats, orig_calls = _register_hooks(model)

    print("  running forward passes...", flush=True)
    try:
        for i, prompt in enumerate(prompts):
            tokens = tokenizer.encode(prompt)
            if len(tokens) > seq_len:
                tokens = tokens[:seq_len]
            t1 = time.time()
            out = model(mx.array([tokens]), cache=None)
            # Force compute of forward output — hooks built a lazy graph
            # during the forward, this is where it actually runs.
            _mx_force_compute(out)
            # Force compute of all accumulated stats, then convert to numpy
            # so the next sample starts with a fresh MLX graph.
            if stats:
                mx_vals = [v for v in stats.values() if not isinstance(v, np.ndarray)]
                if mx_vals:
                    _mx_force_compute(mx_vals)
                for k in list(stats.keys()):
                    v = stats[k]
                    if not isinstance(v, np.ndarray):
                        stats[k] = np.array(v)
            elapsed = time.time() - t1
            if (i + 1) % 4 == 0 or i == 0:
                print(f"    sample {i+1}/{len(prompts)} "
                      f"({len(tokens)} tok, {elapsed:.1f}s)", flush=True)
    finally:
        _unregister_hooks(model, orig_calls)

    print(f"  captured {len(stats)} stats in {time.time()-t0:.1f}s", flush=True)

    out = {name: np.array(arr, dtype=np.float32) for name, arr in stats.items()}
    mags = sorted([float(a.max()) for a in out.values()], reverse=True)
    print(f"  top-5 per-channel max: {[f'{m:.3f}' for m in mags[:5]]}", flush=True)
    print(f"  bottom-5:              {[f'{m:.3f}' for m in mags[-5:]]}", flush=True)

    save_file(out, str(output_path))
    total_mb = sum(a.nbytes for a in out.values()) / 1e6
    print(f"  wrote {output_path} ({total_mb:.1f} MB)", flush=True)


if __name__ == "__main__":
    mp = sys.argv[1]
    ns = int(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_N_SAMPLES
    run_capture(mp, n_samples=ns)
