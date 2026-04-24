"""Expert importance scoring + per-layer prune selection.

Reads `routing_profile.safetensors` + sidecar JSON, produces:

  keep_mask[L, E]   bool array — which experts to keep per layer
  merge_plan[L][e_drop] = e_keep    which kept-expert absorbs a dropped one

The score combines:
  weighted_freq[e]            mass contribution signal (primary)
  freq[e]                     simple selection count
  output_energy[e]            activation-norm signal (if profiled, --with-energy)
  coact_bonus[e]              how much unique coverage e brings (penalty for
                              experts fully redundant with a kept one)

Selection per layer:
  * fixed ratio r: keep round((1-r) * E) highest-scoring experts
  * adaptive: adjust per-layer by router entropy — layers with more uniform
    routing get higher keep ratio (can't drop as many without large loss)

Absorb-merge (per dropped expert e):
  * find the kept expert k with highest coact[e, k]
  * plan[e] = k  — weight-merge handled by prune.py

Usage:
  python -m jang_tools.kimi_prune.score \\
      --profile <path/to/data-drive>/kimi_calib/routing_profile.safetensors \\
      --out <path/to/data-drive>/kimi_calib/prune_plan.json \\
      --ratio 0.30 [--adaptive] [--min-keep 150] [--max-keep 300]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _entropy(p):
    import numpy as np
    q = p / (p.sum() + 1e-20)
    q = np.clip(q, 1e-12, 1.0)
    return float(-(q * np.log(q)).sum())


def score_layer(freq, wfreq, coact, energy, alpha=1.0, beta=0.2, gamma=0.1):
    """importance[e] = alpha*wfreq + beta*freq + gamma*energy.

    All terms pre-normalized to their own max so weights are comparable.
    """
    import numpy as np
    def norm(a):
        m = a.max()
        return a / (m + 1e-20) if m > 0 else a
    return alpha * norm(wfreq) + beta * norm(freq) + gamma * norm(energy)


def plan_prune(profile_path: Path, ratio: float, adaptive: bool,
               min_keep: int, max_keep: int) -> dict:
    import mlx.core as mx
    import numpy as np

    t = mx.load(str(profile_path))
    freq = np.asarray(t["freq"])                  # (L, E)
    wfreq = np.asarray(t["weighted_freq"])        # (L, E)
    coact = np.asarray(t["coact"])                # (L, E, E)
    energy = np.asarray(t["energy"])              # (L, E)
    L, E = freq.shape

    sidecar_path = profile_path.with_suffix(".json")
    sidecar = json.loads(sidecar_path.read_text()) if sidecar_path.exists() else {}

    per_layer = []
    keep_count_total = 0
    for Li in range(L):
        s = score_layer(freq[Li], wfreq[Li], coact[Li], energy[Li])
        if adaptive:
            ent = _entropy(freq[Li])
            ent_max = np.log(E)
            uniform = ent / (ent_max + 1e-12)
            local_ratio = ratio * (1.0 - 0.5 * uniform)
        else:
            local_ratio = ratio
        n_drop = int(round(E * local_ratio))
        n_keep = E - n_drop
        n_keep = max(min_keep, min(max_keep, n_keep))
        order = np.argsort(-s)  # highest score first
        keep_ids = sorted(order[:n_keep].tolist())
        drop_ids = sorted(order[n_keep:].tolist())

        # Absorb plan per dropped expert: best-coact kept expert.
        plan = {}
        c = coact[Li]
        keep_set = set(keep_ids)
        for e in drop_ids:
            coact_e = c[e].copy()
            # Mask out non-kept experts.
            mask = np.full(E, -np.inf)
            for k in keep_ids:
                mask[k] = 0.0
            masked = coact_e + mask
            if np.all(np.isinf(masked) & (masked < 0)):
                target = keep_ids[0]  # fallback
            else:
                target = int(np.argmax(masked))
            plan[e] = target

        per_layer.append({
            "layer": Li,
            "n_keep": n_keep,
            "n_drop": len(drop_ids),
            "keep": keep_ids,
            "drop": drop_ids,
            "absorb_plan": plan,
            "router_entropy_norm": float(_entropy(freq[Li]) / (np.log(E) + 1e-12)),
            "score_kept_mean": float(s[keep_ids].mean()),
            "score_dropped_mean": float(s[drop_ids].mean()) if drop_ids else 0.0,
        })
        keep_count_total += n_keep

    return {
        "n_layers": L,
        "n_experts_per_layer": E,
        "base_ratio": ratio,
        "adaptive": adaptive,
        "total_experts_kept": keep_count_total,
        "total_experts_dropped": L * E - keep_count_total,
        "global_kept_fraction": keep_count_total / (L * E),
        "per_layer": per_layer,
        "source_profile": str(profile_path),
        "profile_sidecar": sidecar,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--ratio", type=float, default=0.30,
                    help="drop ratio per layer (0.30 => keep 70 pct)")
    ap.add_argument("--adaptive", action="store_true")
    ap.add_argument("--min-keep", type=int, default=128)
    ap.add_argument("--max-keep", type=int, default=320)
    args = ap.parse_args()

    plan = plan_prune(args.profile, args.ratio, args.adaptive,
                      args.min_keep, args.max_keep)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        json.dump(plan, f, indent=2)
    print(f"[score] ratio={args.ratio} adaptive={args.adaptive}  "
          f"total kept {plan['total_experts_kept']}/"
          f"{plan['n_layers'] * plan['n_experts_per_layer']} "
          f"({plan['global_kept_fraction']:.1%})  "
          f"-> {args.out}", flush=True)


if __name__ == "__main__":
    main()
