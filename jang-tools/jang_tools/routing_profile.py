"""
JANG Routing Profile — Expert routing statistics for TurboSmelt SSD inference.
Created by Jinho Jang (eric@jangq.ai)

Captures expert routing decisions during calibration and compresses them into
four tables that TurboSmelt uses for intelligent expert pre-loading:

  1. frequency[n_moe_layers, n_experts]  — Zipfian ordering, initial pre-load
  2. entropy[n_moe_layers]               — Adaptive n_load per layer
  3. coactivation[n_moe_layers, n_experts, K] — "expert 5 fired → also load 12,47,91"
  4. transition[n_moe_layers-1, n_experts, K] — Cross-layer speculation

Total overhead: ~15-40 MB on a 200+ GB model. Nothing.

Usage:
    from jang_tools.routing_profile import collect_routing_profile
    collect_routing_profile("./GLM-5-JANG_1L", n_samples=256)

Or standalone:
    python -m jang_tools profile ./GLM-5-JANG_1L
"""

import json
import time
from pathlib import Path
from typing import Optional

import numpy as np
from safetensors.numpy import save_file, load_file

# Defaults
DEFAULT_N_SAMPLES = 256
DEFAULT_SEQ_LEN = 512
DEFAULT_TOP_K_COACT = 16   # top co-activating experts to store per expert
DEFAULT_TOP_K_TRANS = 16   # top transition targets to store per expert
ROUTING_PROFILE_FILENAME = "routing_profile.safetensors"


def collect_routing_profile(
    model_path: str | Path,
    output_path: Optional[str | Path] = None,
    calibration_data: Optional[list[str]] = None,
    n_samples: int = DEFAULT_N_SAMPLES,
    seq_len: int = DEFAULT_SEQ_LEN,
    top_k_coact: int = DEFAULT_TOP_K_COACT,
    top_k_trans: int = DEFAULT_TOP_K_TRANS,
) -> dict:
    """
    Collect expert routing statistics by running calibration data through a model.

    Hooks into MoE gate modules to capture routing indices at every layer,
    then computes frequency, entropy, co-activation, and transition tables.

    Works with any MLX MoE model (DeepSeek-V2/V3, Mixtral, Qwen3.5, GLM-5, etc.).

    Args:
        model_path: path to JANG or MLX model directory
        output_path: where to save routing_profile.safetensors (default: model_path)
        calibration_data: list of text strings (default: built-in diverse corpus)
        n_samples: number of calibration samples
        seq_len: max sequence length per sample
        top_k_coact: number of co-activation entries per expert
        top_k_trans: number of transition entries per expert

    Returns:
        dict with profile metadata and file path
    """
    import mlx.core as mx
    from mlx_lm.utils import load as load_mlx_model

    model_path = Path(model_path)
    if output_path is None:
        output_path = model_path
    output_path = Path(output_path)

    print(f"\n{'='*60}")
    print(f"  JANG Routing Profile")
    print(f"  Created by Jinho Jang (eric@jangq.ai)")
    print(f"{'='*60}")
    print(f"  Model: {model_path}")
    print(f"  Samples: {n_samples}, seq_len: {seq_len}")
    print(f"  Co-activation K: {top_k_coact}, Transition K: {top_k_trans}")
    print(f"{'='*60}\n")

    # Load model config to get architecture info
    config = json.loads((model_path / "config.json").read_text())
    tc = config.get("text_config", {})
    n_experts = config.get("n_routed_experts", tc.get("n_routed_experts",
                 config.get("num_local_experts", tc.get("num_local_experts", 0))))
    top_k = config.get("num_experts_per_tok", tc.get("num_experts_per_tok", 8))
    n_layers = config.get("num_hidden_layers", tc.get("num_hidden_layers", 0))

    if n_experts == 0:
        raise ValueError(f"Model at {model_path} is not a MoE model (no experts found)")

    print(f"  Architecture: {n_layers} layers, {n_experts} experts, top-{top_k}")

    # Load model and tokenizer
    print(f"  Loading model...")
    t0 = time.perf_counter()
    model, tokenizer = load_mlx_model(str(model_path))
    print(f"  Loaded in {time.perf_counter() - t0:.1f}s")

    # Find MoE layers and patch their gates
    moe_info = _find_and_patch_moe_layers(model, n_layers)
    n_moe_layers = len(moe_info)
    print(f"  Found {n_moe_layers} MoE layers (indices: {moe_info[0]['layer_idx']}-{moe_info[-1]['layer_idx']})")

    # Prepare calibration data
    if calibration_data is None:
        calibration_data = _diverse_calibration_texts()
    calibration_data = calibration_data[:n_samples]

    # Run calibration forward passes and collect routing indices
    print(f"\n  Running calibration ({len(calibration_data)} samples)...")
    all_routing = _run_calibration(model, tokenizer, calibration_data, seq_len, moe_info)

    total_tokens = sum(len(r) for r in all_routing)
    print(f"  Collected routing for {total_tokens:,} tokens across {n_moe_layers} MoE layers")

    # Compute the four tables
    print(f"\n  Computing routing profile...")
    profile_tensors = _compute_profile(
        all_routing, n_moe_layers, n_experts, top_k,
        top_k_coact, top_k_trans, moe_info,
    )

    # Save
    output_file = output_path / ROUTING_PROFILE_FILENAME
    output_path.mkdir(parents=True, exist_ok=True)
    save_file(profile_tensors, str(output_file))

    # Update jang_config.json with routing profile metadata
    metadata = {
        "n_calibration_tokens": total_tokens,
        "n_moe_layers": n_moe_layers,
        "n_experts": n_experts,
        "top_k": top_k,
        "top_k_coactivation": top_k_coact,
        "top_k_transition": top_k_trans,
        "moe_layer_indices": [m["layer_idx"] for m in moe_info],
    }
    _update_jang_config(model_path, metadata)

    profile_size_mb = sum(t.nbytes for t in profile_tensors.values()) / (1024 ** 2)
    print(f"\n{'='*60}")
    print(f"  DONE — Routing Profile")
    print(f"  File: {output_file}")
    print(f"  Size: {profile_size_mb:.1f} MB")
    print(f"  Tokens profiled: {total_tokens:,}")
    print(f"{'='*60}\n")

    # Print per-layer entropy summary
    entropy = profile_tensors["routing.entropy"]
    freq = profile_tensors["routing.frequency"]
    print(f"  Per-layer routing entropy (higher = more chaotic):")
    print(f"  {'Layer':>6s}  {'Entropy':>8s}  {'Top-1 %':>8s}  {'Top-8 %':>8s}  {'Active':>7s}")
    for i, mi in enumerate(moe_info):
        e = entropy[i]
        f = freq[i]
        top1_pct = f.max() * 100
        top8_pct = np.sort(f)[-8:].sum() * 100
        active = int(np.sum(f > 0.001))
        print(f"  {mi['layer_idx']:>6d}  {e:>8.2f}  {top1_pct:>7.1f}%  {top8_pct:>7.1f}%  {active:>5d}/{n_experts}")

    return {
        "file": str(output_file),
        "size_mb": round(profile_size_mb, 1),
        **metadata,
    }


def _find_and_patch_moe_layers(model, n_layers: int) -> list[dict]:
    """
    Find MoE layers in the model and patch their gate calls to capture routing indices.

    Supports multiple MoE patterns:
    - DeepSeek-V2/V3/GLM-5: layer.mlp.gate (MoEGate)
    - Mixtral/Qwen3.5: layer.mlp.gate or layer.block_sparse_moe.gate
    - Any model with a .gate attribute that returns (indices, scores)

    Returns list of dicts with layer_idx and captured_indices list.
    """
    import mlx.core as mx

    layers = model.model.layers if hasattr(model, "model") else model.layers
    moe_info = []

    for idx in range(len(layers)):
        layer = layers[idx]
        if layer is None:
            continue

        # Find the MoE module — try common attribute paths
        moe = None
        if hasattr(layer, "mlp") and hasattr(layer.mlp, "gate"):
            moe = layer.mlp
        elif hasattr(layer, "block_sparse_moe") and hasattr(layer.block_sparse_moe, "gate"):
            moe = layer.block_sparse_moe
        elif hasattr(layer, "feed_forward") and hasattr(layer.feed_forward, "gate"):
            moe = layer.feed_forward

        if moe is None:
            continue

        # Create capture list and patch the MoE's __call__
        captured = []
        info = {"layer_idx": idx, "moe": moe, "captured": captured}

        original_call = moe.__call__

        def make_patched(orig, cap, moe_ref):
            def patched_call(x):
                # Call gate to get routing indices
                inds, scores = moe_ref.gate(x)
                # Materialize and capture indices (forces GPU sync — OK for calibration)
                mx.synchronize()
                cap.append(np.array(inds))  # [batch, seq_len, top_k]
                # Continue with original forward
                return orig(x)
            return patched_call

        moe.__call__ = make_patched(original_call, captured, moe)
        moe_info.append(info)

    if not moe_info:
        raise RuntimeError("No MoE layers found in model. Is this a MoE model?")

    return moe_info


def _run_calibration(model, tokenizer, texts, seq_len, moe_info):
    """
    Run calibration texts through the model and return per-layer routing indices.

    Returns: list of n_moe_layers arrays, each [total_tokens, top_k].
    """
    import mlx.core as mx
    from tqdm import tqdm

    for text in tqdm(texts, desc="  Profiling"):
        tokens = tokenizer.encode(text)
        if len(tokens) > seq_len:
            tokens = tokens[:seq_len]

        input_ids = mx.array([tokens])

        # Forward pass — routing indices captured by patched gates
        logits = model(input_ids)
        mx.synchronize()

    # Collect captured indices per layer
    per_layer = []
    for info in moe_info:
        if info["captured"]:
            # Concatenate all captured batches: each is [batch, seq, top_k]
            # Reshape to [total_tokens, top_k]
            layer_inds = []
            for arr in info["captured"]:
                reshaped = arr.reshape(-1, arr.shape[-1])
                layer_inds.append(reshaped)
            per_layer.append(np.concatenate(layer_inds, axis=0))
        else:
            per_layer.append(np.zeros((0, 8), dtype=np.int32))

    return per_layer


def _compute_profile(
    per_layer_indices: list[np.ndarray],
    n_moe_layers: int,
    n_experts: int,
    top_k: int,
    top_k_coact: int,
    top_k_trans: int,
    moe_info: list[dict],
) -> dict[str, np.ndarray]:
    """
    Compute the four routing profile tables from raw routing indices.

    Args:
        per_layer_indices: list of [n_tokens, top_k] arrays per MoE layer
        n_moe_layers: number of MoE layers
        n_experts: total number of routed experts
        top_k: experts per token (e.g. 8)
        top_k_coact: co-activation entries to store per expert
        top_k_trans: transition entries to store per expert

    Returns:
        dict of tensor name -> numpy array for safetensors
    """
    tensors = {}

    # 1. Frequency: how often each expert is selected (normalized)
    frequency = np.zeros((n_moe_layers, n_experts), dtype=np.float32)
    for i, inds in enumerate(per_layer_indices):
        if len(inds) == 0:
            continue
        for e in range(n_experts):
            frequency[i, e] = np.sum(inds == e)
        total = frequency[i].sum()
        if total > 0:
            frequency[i] /= total
    tensors["routing.frequency"] = frequency

    # 2. Entropy: Shannon entropy of the frequency distribution per layer
    entropy = np.zeros(n_moe_layers, dtype=np.float32)
    for i in range(n_moe_layers):
        f = frequency[i]
        f_nonzero = f[f > 0]
        if len(f_nonzero) > 0:
            entropy[i] = -np.sum(f_nonzero * np.log2(f_nonzero))
    tensors["routing.entropy"] = entropy

    # 3. Co-activation: for each expert, which other experts fire on the same token
    coact_indices = np.zeros((n_moe_layers, n_experts, top_k_coact), dtype=np.uint16)
    coact_scores = np.zeros((n_moe_layers, n_experts, top_k_coact), dtype=np.float32)

    for i, inds in enumerate(per_layer_indices):
        if len(inds) == 0:
            continue

        # inds: [n_tokens, top_k] — each row is the set of experts for one token
        # Vectorized co-occurrence: for each expert e, count co-occurrences
        # Build sparse co-occurrence from the top_k selections
        cooccur = np.zeros((n_experts, n_experts), dtype=np.float32)
        expert_counts = np.zeros(n_experts, dtype=np.float32)

        for token_experts in inds:
            valid = token_experts[(token_experts >= 0) & (token_experts < n_experts)]
            for e in valid:
                expert_counts[e] += 1
            for j, e in enumerate(valid):
                for e2 in valid:
                    if e != e2:
                        cooccur[e, e2] += 1

        # Normalize and extract top-K per expert
        for e in range(n_experts):
            if expert_counts[e] == 0:
                continue
            row = cooccur[e] / expert_counts[e]
            k = min(top_k_coact, int(np.sum(row > 0)))
            if k == 0:
                continue
            top_idx = np.argpartition(-row, k)[:k]
            top_idx = top_idx[np.argsort(-row[top_idx])]

            actual_k = min(len(top_idx), top_k_coact)
            coact_indices[i, e, :actual_k] = top_idx[:actual_k].astype(np.uint16)
            coact_scores[i, e, :actual_k] = row[top_idx[:actual_k]]

    tensors["routing.coactivation.indices"] = coact_indices
    tensors["routing.coactivation.scores"] = coact_scores

    # 4. Transition priors: given expert e at layer N, which experts fire at N+1
    n_transitions = n_moe_layers - 1
    trans_indices = np.zeros((max(n_transitions, 1), n_experts, top_k_trans), dtype=np.uint16)
    trans_scores = np.zeros((max(n_transitions, 1), n_experts, top_k_trans), dtype=np.float32)

    for i in range(n_transitions):
        inds_curr = per_layer_indices[i]
        inds_next = per_layer_indices[i + 1]

        if len(inds_curr) == 0 or len(inds_next) == 0:
            continue

        # Align lengths
        n_tokens = min(len(inds_curr), len(inds_next))
        inds_curr = inds_curr[:n_tokens]
        inds_next = inds_next[:n_tokens]

        # For each expert e in layer N, count which experts appear at N+1
        trans_count = np.zeros((n_experts, n_experts), dtype=np.float32)
        expert_count = np.zeros(n_experts, dtype=np.float32)

        for t in range(n_tokens):
            curr_valid = inds_curr[t][(inds_curr[t] >= 0) & (inds_curr[t] < n_experts)]
            next_valid = inds_next[t][(inds_next[t] >= 0) & (inds_next[t] < n_experts)]
            for e in curr_valid:
                expert_count[e] += 1
                for e2 in next_valid:
                    trans_count[e, e2] += 1

        # Normalize and extract top-K
        for e in range(n_experts):
            if expert_count[e] == 0:
                continue
            row = trans_count[e] / expert_count[e]
            k = min(top_k_trans, int(np.sum(row > 0)))
            if k == 0:
                continue
            top_idx = np.argpartition(-row, k)[:k]
            top_idx = top_idx[np.argsort(-row[top_idx])]

            actual_k = min(len(top_idx), top_k_trans)
            trans_indices[i, e, :actual_k] = top_idx[:actual_k].astype(np.uint16)
            trans_scores[i, e, :actual_k] = row[top_idx[:actual_k]]

    tensors["routing.transition.indices"] = trans_indices
    tensors["routing.transition.scores"] = trans_scores

    # Summary
    total_bytes = sum(t.nbytes for t in tensors.values())
    print(f"  Frequency table: {frequency.nbytes / 1024:.1f} KB")
    print(f"  Entropy table: {entropy.nbytes} bytes")
    print(f"  Co-activation table: {(coact_indices.nbytes + coact_scores.nbytes) / (1024**2):.1f} MB")
    print(f"  Transition table: {(trans_indices.nbytes + trans_scores.nbytes) / (1024**2):.1f} MB")
    print(f"  Total: {total_bytes / (1024**2):.1f} MB")

    return tensors


def _update_jang_config(model_path: Path, routing_metadata: dict):
    """Add routing profile metadata to jang_config.json."""
    config_path = model_path / "jang_config.json"
    if config_path.exists():
        config = json.loads(config_path.read_text())
        config["routing_profile"] = routing_metadata
        config_path.write_text(json.dumps(config, indent=2) + "\n")
        print(f"  Updated {config_path.name} with routing profile metadata")


def load_routing_profile(model_path: str | Path) -> Optional[dict]:
    """
    Load a routing profile from a JANG model directory.

    Returns dict with numpy arrays, or None if no profile exists.
    Used by TurboSmelt for expert pre-loading strategy.

    Keys:
        routing.frequency          — float32[n_moe_layers, n_experts]
        routing.entropy            — float32[n_moe_layers]
        routing.coactivation.indices — uint16[n_moe_layers, n_experts, K]
        routing.coactivation.scores  — float32[n_moe_layers, n_experts, K]
        routing.transition.indices   — uint16[n_moe_layers-1, n_experts, K]
        routing.transition.scores    — float32[n_moe_layers-1, n_experts, K]
    """
    profile_path = Path(model_path) / ROUTING_PROFILE_FILENAME
    if not profile_path.exists():
        return None

    tensors = load_file(str(profile_path))

    # Load metadata from jang_config.json
    config_path = Path(model_path) / "jang_config.json"
    metadata = {}
    if config_path.exists():
        config = json.loads(config_path.read_text())
        metadata = config.get("routing_profile", {})

    return {**tensors, "_metadata": metadata}


def get_adaptive_n_load(
    profile: dict,
    total_ram_experts: int,
) -> list[int]:
    """
    Compute adaptive n_load per layer based on entropy.

    Higher entropy layers get more pre-loaded experts, lower entropy layers
    get fewer. Total across all layers stays within budget.

    Args:
        profile: loaded routing profile dict
        total_ram_experts: total expert slots available across all layers

    Returns:
        list of n_load per MoE layer
    """
    entropy = profile["routing.entropy"]
    n_layers = len(entropy)

    # Allocate proportional to entropy (higher entropy = more experts needed)
    # With a minimum floor of 1 expert per layer
    min_per_layer = 1
    available = total_ram_experts - n_layers * min_per_layer
    if available <= 0:
        return [min_per_layer] * n_layers

    # Softmax-weighted allocation
    shifted = entropy - entropy.min() + 1.0
    weights = shifted / shifted.sum()

    n_load = np.full(n_layers, min_per_layer, dtype=np.int32)
    extra = np.round(weights * available).astype(np.int32)

    # Adjust to exactly hit budget
    while extra.sum() > available:
        idx = int(np.argmax(extra))
        extra[idx] -= 1
    while extra.sum() < available:
        idx = int(np.argmax(weights))
        extra[idx] += 1

    n_load += extra
    return n_load.tolist()


def get_preload_set(
    profile: dict,
    layer_idx: int,
    fired_experts: list[int],
    n_load: int,
) -> list[int]:
    """
    Get the optimal expert pre-load set for a given layer, using co-activation
    and transition data.

    Combines three signals:
    1. Frequency prior (baseline — always include top-frequency experts)
    2. Co-activation (if expert E fired this token, also load its co-activators)
    3. Transition prior (experts likely at NEXT layer given current layer's routing)

    Args:
        profile: loaded routing profile dict
        layer_idx: MoE layer index (0-based within MoE layers, not global)
        fired_experts: experts that fired at this layer for current token
        n_load: number of experts to pre-load

    Returns:
        sorted list of expert IDs to pre-load
    """
    freq = profile["routing.frequency"]
    coact_idx = profile["routing.coactivation.indices"]
    coact_scr = profile["routing.coactivation.scores"]
    trans_idx = profile["routing.transition.indices"]
    trans_scr = profile["routing.transition.scores"]

    n_experts = freq.shape[1]
    scores = np.zeros(n_experts, dtype=np.float32)

    # Signal 1: Frequency prior (weight: 0.3)
    scores += freq[layer_idx] * 0.3

    # Signal 2: Co-activation from fired experts (weight: 0.4)
    for e in fired_experts:
        if e < 0 or e >= n_experts:
            continue
        for k in range(coact_idx.shape[2]):
            co_e = int(coact_idx[layer_idx, e, k])
            co_s = float(coact_scr[layer_idx, e, k])
            if co_s > 0:
                scores[co_e] += co_s * 0.4

    # Signal 3: Transition from fired experts to NEXT layer (weight: 0.3)
    if layer_idx < len(trans_idx):
        for e in fired_experts:
            if e < 0 or e >= n_experts:
                continue
            for k in range(trans_idx.shape[2]):
                tr_e = int(trans_idx[layer_idx, e, k])
                tr_s = float(trans_scr[layer_idx, e, k])
                if tr_s > 0:
                    scores[tr_e] += tr_s * 0.3

    # Select top n_load experts
    if n_load >= n_experts:
        return list(range(n_experts))

    top_idx = np.argpartition(-scores, n_load)[:n_load]
    return sorted(top_idx.tolist())


def _diverse_calibration_texts() -> list[str]:
    """
    Diverse calibration corpus for routing profile.

    Covers: general knowledge, code, math, conversation, creative writing,
    technical docs, multilingual, reasoning. Each domain activates different
    expert specializations, giving representative routing statistics.
    """
    return [
        # General knowledge
        "The transformer architecture revolutionized natural language processing by introducing self-attention mechanisms that can capture long-range dependencies in text. Unlike recurrent neural networks, transformers process all positions in parallel, enabling much faster training on modern hardware.",
        "Photosynthesis is the process by which plants convert sunlight into chemical energy. Chlorophyll absorbs light, which drives a series of chemical reactions that convert carbon dioxide and water into glucose and oxygen. This process is essential for life on Earth.",
        "The history of computing began with mechanical calculators in the 17th century. Charles Babbage designed the Analytical Engine in the 1830s, often considered the first general-purpose computer. Ada Lovelace wrote what is recognized as the first computer program for this machine.",
        "Quantum mechanics describes the behavior of matter and energy at the smallest scales. The wave-particle duality, uncertainty principle, and quantum entanglement are fundamental concepts that challenge classical intuitions about the physical world.",
        "The human brain contains approximately 86 billion neurons, each connected to thousands of others through synapses. Neural plasticity allows the brain to reorganize itself by forming new neural connections throughout life.",

        # Code
        "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[len(arr) // 2]\n    left = [x for x in arr if x < pivot]\n    middle = [x for x in arr if x == pivot]\n    right = [x for x in arr if x > pivot]\n    return quicksort(left) + middle + quicksort(right)",
        "import torch\nimport torch.nn as nn\n\nclass TransformerBlock(nn.Module):\n    def __init__(self, d_model, n_heads):\n        super().__init__()\n        self.attn = nn.MultiheadAttention(d_model, n_heads)\n        self.ffn = nn.Sequential(nn.Linear(d_model, 4*d_model), nn.GELU(), nn.Linear(4*d_model, d_model))\n        self.norm1 = nn.LayerNorm(d_model)\n        self.norm2 = nn.LayerNorm(d_model)",
        "SELECT u.name, COUNT(o.id) as order_count, SUM(o.total) as total_spent\nFROM users u\nLEFT JOIN orders o ON u.id = o.user_id\nWHERE o.created_at >= DATE_SUB(NOW(), INTERVAL 30 DAY)\nGROUP BY u.id\nHAVING total_spent > 1000\nORDER BY total_spent DESC\nLIMIT 10;",
        "async function fetchWithRetry(url, maxRetries = 3) {\n  for (let i = 0; i < maxRetries; i++) {\n    try {\n      const response = await fetch(url);\n      if (!response.ok) throw new Error(`HTTP ${response.status}`);\n      return await response.json();\n    } catch (error) {\n      if (i === maxRetries - 1) throw error;\n      await new Promise(r => setTimeout(r, 1000 * Math.pow(2, i)));\n    }\n  }\n}",

        # Math and reasoning
        "Let's prove that the square root of 2 is irrational. Assume for contradiction that sqrt(2) = p/q where p and q are integers with no common factors. Then 2 = p squared / q squared, so p squared = 2 q squared. This means p squared is even, so p must be even. Write p = 2k. Then 4k squared = 2 q squared, so q squared = 2k squared, meaning q is also even. Contradiction.",
        "The traveling salesman problem asks: given a list of cities and distances between each pair, what is the shortest possible route that visits each city exactly once and returns to the origin? This is NP-hard, meaning no polynomial-time algorithm is known.",
        "Consider a Markov chain with states A, B, C and transition matrix P. To find the stationary distribution pi, we solve pi times P = pi subject to the constraint that the probabilities sum to 1. The chain is ergodic if it is irreducible and aperiodic.",
        "The fundamental theorem of calculus states that if f is continuous on the interval [a,b] and F is an antiderivative of f, then the integral from a to b of f(x) dx equals F(b) minus F(a). This connects differentiation and integration.",

        # Conversation
        "User: Can you explain how neural networks learn?\nAssistant: Neural networks learn through backpropagation. During training, the network makes predictions on input data, and the error between predictions and actual values is calculated. This error is then propagated backward through the network, adjusting each weight proportionally to its contribution.",
        "User: What's the difference between TCP and UDP?\nAssistant: TCP is connection-oriented and guarantees delivery through acknowledgments and retransmission. UDP is connectionless and does not guarantee delivery, but it is faster with less overhead. TCP is used for web browsing and email, while UDP is preferred for streaming and gaming.",

        # Creative writing
        "The old lighthouse keeper watched the storm approach from the north. Dark clouds rolled across the horizon like an army on the march. The waves grew taller, crashing against the rocky shore with increasing violence. He climbed the spiral staircase one more time, checking that the great lens was clean and the oil reservoir full.",
        "In the year 2157, humanity had spread across three star systems. The quantum entanglement network connected worlds separated by light-years in real-time. But the signal that arrived from the Kepler system that morning would change everything we thought we knew about the universe.",

        # Technical documentation
        "The Metal Performance Shaders framework provides highly optimized compute and rendering shaders for Apple GPUs. MPSMatrixMultiplication performs general matrix multiplication with support for mixed precision, including float16 inputs with float32 accumulation. This is critical for neural network inference on Apple Silicon.",
        "Docker containers share the host OS kernel but have isolated file systems, network interfaces, and process spaces. Unlike virtual machines, containers do not need a full OS installation, making them lightweight and fast to start. Kubernetes orchestrates containers across multiple hosts.",

        # Multilingual
        "La inteligencia artificial ha transformado multiples industrias. Los modelos de lenguaje grandes pueden traducir idiomas, escribir codigo y responder preguntas complejas. Sin embargo, tambien plantean desafios eticos importantes que la sociedad debe abordar.",
        "Die kuenstliche Intelligenz entwickelt sich rasant weiter. Grosse Sprachmodelle koennen komplexe Texte generieren, Code schreiben und mathematische Probleme loesen. Die ethischen Implikationen dieser Technologie werden weltweit diskutiert.",

        # Scientific
        "CRISPR-Cas9 is a revolutionary gene editing tool derived from bacterial immune systems. The Cas9 protein, guided by a short RNA sequence, can cut DNA at specific locations. This allows researchers to delete, modify, or insert genetic material with unprecedented precision.",
        "General relativity describes gravity as the curvature of spacetime caused by mass and energy. The Einstein field equations relate the geometry of spacetime to the distribution of matter within it. This theory predicts gravitational waves, black holes, and the expansion of the universe.",
        "The Standard Model of particle physics describes three of the four fundamental forces and classifies all known elementary particles. Quarks combine to form hadrons like protons and neutrons, while leptons such as electrons and neutrinos are fundamental particles.",

        # Legal/business
        "This Software License Agreement governs the use of the software provided by the Licensor. The Licensee is granted a non-exclusive, non-transferable license to use the software for internal business purposes only. Redistribution, reverse engineering, and modification are strictly prohibited without prior written consent.",

        # Instructions/how-to
        "To set up a Kubernetes cluster on AWS using EKS: First, install eksctl and kubectl CLI tools. Then create a cluster configuration YAML file specifying node groups, instance types, and networking. Run eksctl create cluster with the config file. Verify with kubectl get nodes. Finally deploy your application using Helm charts.",

        # Data/structured
        "Results summary: Model A achieved 86.4 percent accuracy on MMLU with 1.8 trillion parameters. Model B achieved 86.8 percent with 137 billion parameters. Model C achieved 79.5 percent with 70 billion parameters. The smaller models showed significantly better efficiency.",
    ]
