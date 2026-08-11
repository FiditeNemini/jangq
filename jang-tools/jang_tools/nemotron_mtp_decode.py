"""D2/D3 speculative decode driver for Nemotron 3.5 Lightning native MTP.

Depth naming (project convention):
    D1 = 1 token/step, MTP off (plain autoregressive)
    D2 = up to 2 tokens/step, 1 draft   — native, the head was trained for this
    D3 = up to 3 tokens/step, 2 drafts  — the SAME head re-applied to its own
         output. Out-of-distribution, so accept rate at depth 2 is expected to
         be much lower. Measure before trusting it.

Correctness invariant: speculative decoding is a pure speed optimization, so
greedy D2/D3 output MUST be token-identical to greedy D1. `compare()` asserts it.

    python -m jang_tools.nemotron_mtp_decode <bundle> [prompt] [--depth 2|3]
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import mlx.core as mx
from mlx_lm import load

from mlx_lm.models.cache import KVCache

from .nemotron_mtp import (
    cache_slots, load_mtp, restore_ssm, rewind_kv, snapshot_ssm,
)


def _greedy(logits: mx.array) -> int:
    return int(mx.argmax(logits[0, -1]).item())


def generate_d1(model, tokenizer, prompt_ids, max_tokens=128):
    """Plain autoregressive greedy — the reference."""
    cache = model.make_cache()
    out = []
    logits = model(mx.array([prompt_ids]), cache=cache)
    tok = _greedy(logits)
    eos = set(tokenizer.eos_token_ids or [tokenizer.eos_token_id])
    t0 = time.time()
    while len(out) < max_tokens:
        out.append(tok)
        if tok in eos:
            break
        logits = model(mx.array([[tok]]), cache=cache)
        tok = _greedy(logits)
    return out, time.time() - t0, {"accepted": 0, "proposed": 0, "steps": len(out)}


def generate_dn(model, tokenizer, mtp, prompt_ids, max_tokens=128, depth=2):
    """D2/D3 speculative decode with correct hybrid-cache rollback.

    depth=2 -> 1 draft/step, depth=3 -> 2 drafts/step.
    """
    n_draft = depth - 1
    assert n_draft >= 1
    cache = model.make_cache()
    ssm_slots, attn_slots = cache_slots(model)
    mtp_cache = [KVCache()]

    eos = set(tokenizer.eos_token_ids or [tokenizer.eos_token_id])
    embed = model.backbone.embeddings

    # Prefill.
    h = model.backbone(mx.array([prompt_ids]), cache=cache)
    logits = model.lm_head(h)
    tok = _greedy(logits)
    h_last = h[:, -1:, :]

    out, accepted, proposed, steps = [], 0, 0, 0
    t0 = time.time()
    while len(out) < max_tokens:
        out.append(tok)
        if tok in eos:
            break
        steps += 1

        # ── draft ──────────────────────────────────────────────────────────
        drafts = []
        mtp_off = mtp_cache[0].offset
        d_h, d_tok = h_last, tok
        for _ in range(n_draft):
            d_h = mtp(embed(mx.array([[d_tok]])), d_h, cache=mtp_cache[0])
            d_tok = _greedy(model.lm_head(d_h))
            drafts.append(d_tok)
        # Speculative MTP-cache entries are NOT real sequence positions.
        # Roll back unconditionally — including on full acceptance.
        mtp_cache[0].offset = mtp_off
        proposed += len(drafts)

        # ── verify: one batched forward over [tok, *drafts] ────────────────
        snap = snapshot_ssm(cache, ssm_slots)
        kv_off = cache[attn_slots[0]].offset
        batch = [tok] + drafts
        h = model.backbone(mx.array([batch]), cache=cache)
        vlogits = model.lm_head(h)

        # ── accept longest matching prefix ─────────────────────────────────
        # Single argmax over ALL verify positions, evaluated once. Doing this
        # per-position costs one device sync each, which at ~8 ms/token is a
        # material fraction of the step.
        picks = mx.argmax(vlogits[0], axis=-1)
        mx.eval(picks)
        picks = picks.tolist()
        n_acc = 0
        for i, d in enumerate(drafts):
            if picks[i] == d:
                n_acc += 1
            else:
                break

        if n_acc == len(drafts):
            out.extend(drafts)
            accepted += n_acc
            h_last = h[:, -1:, :]
            tok = picks[-1]
        else:
            out.extend(drafts[:n_acc])
            accepted += n_acc
            # Undo the rejected positions in every cache population.
            restore_ssm(cache, ssm_slots, snap)
            for i in attn_slots:
                cache[i].offset = kv_off
            keep = batch[: n_acc + 1]
            h = model.backbone(mx.array([keep]), cache=cache)
            h_last = h[:, -1:, :]
            tok = picks[n_acc]

        if len(out) >= max_tokens:
            break

    dt = time.time() - t0
    return out[:max_tokens], dt, {
        "accepted": accepted, "proposed": proposed, "steps": steps,
    }


def _sync_mtp(mtp, embed, mtp_cache, h, toks, upto):
    """Advance the MTP KV cache over accepted positions so it stays in lockstep
    with the backbone. Without this the head attends over a stale prefix and the
    accept rate silently decays over a long generation."""
    for i in range(upto):
        mtp(embed(mx.array([[toks[i + 1]]])), h[:, i:i + 1, :], cache=mtp_cache)


def compare(bundle: str, prompt: str = "Explain what a Mamba state space model is.",
            max_tokens: int = 96, depth: int = 2):
    model, tokenizer = load(bundle)
    mtp = load_mtp(model, bundle)
    if mtp is None:
        print("  NO MTP WEIGHTS IN BUNDLE — nothing to test")
        return 1

    ids = tokenizer.encode(tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True, tokenize=False, enable_thinking=False))

    print(f"  bundle: {Path(bundle).name}   depth=D{depth}   max_tokens={max_tokens}")
    ref, dt1, _ = generate_d1(model, tokenizer, ids, max_tokens)
    print(f"  D1 (reference): {len(ref)} tok in {dt1:.2f}s = {len(ref)/dt1:.1f} tok/s")

    spec, dt2, st = generate_dn(model, tokenizer, mtp, ids, max_tokens, depth=depth)
    rate = st["accepted"] / max(st["proposed"], 1)
    print(f"  D{depth}:            {len(spec)} tok in {dt2:.2f}s = {len(spec)/dt2:.1f} tok/s")
    print(f"  accept rate: {st['accepted']}/{st['proposed']} = {rate:.1%}  "
          f"over {st['steps']} steps")
    print(f"  speedup: {dt1/max(dt2,1e-9):.2f}x")

    n = min(len(ref), len(spec))
    identical = ref[:n] == spec[:n]
    print(f"  TOKEN-IDENTICAL to D1: {identical}")
    if not identical:
        for i in range(n):
            if ref[i] != spec[i]:
                print(f"    first divergence at {i}: ref={ref[i]} spec={spec[i]}")
                print(f"    ref : {tokenizer.decode(ref[:i+8])!r}")
                print(f"    spec: {tokenizer.decode(spec[:i+8])!r}")
                break
    return 0 if identical else 2


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    depth = 2
    for a in sys.argv[1:]:
        if a.startswith("--depth"):
            depth = int(a.split("=")[1]) if "=" in a else 2
    bundle = args[0]
    prompt = args[1] if len(args) > 1 else "Explain what a Mamba state space model is."
    raise SystemExit(compare(bundle, prompt, depth=depth))
