# Qwen3.8 MTP + which checkout to use (2026-08-14)

## Where the work happens

- **Use `~/mlx/vllm-mlx` on `erics-m5-max.local`**, branch `post-250-main`.
- Models come from `/Volumes/EricsLLMDrive/jangq-ai/`.
- **`~/vmlx` is a stale 1.6.9 clone — ignore it.** So is the `~/vmlx` checkout on
  the box. Wrong-checkout confusion has cost real time; check `git log -1` before
  believing anything you read there.
- **Do not serve models on max2.** max2 is for editing, tests and bundling.
- `Qwen3.6-27B-AGENT-KIT` is a 14 MB **docs kit, not a servable bundle**. Pointing
  `serve` at it hangs forever with no error worth reading.

## MTP facts, so they don't get rediscovered

- Qwen3.8 carries the **same v3 stamp as 3.6**.
- **MTP auto-engages with no flags.** Measured +21.7% on 4D (28.44 vs 23.36 t/s),
  **byte-identical output at temperature 0**.
- **Depth counts DRAFTS.** Read `recommended_num_drafts` or the tuning sidecar.
  Never use vLLM's `upstream_num_speculative_tokens` — their 2 equals kit D3,
  which is the forbidden depth (0.48x, must not ship).
- **MTP only engages at temperature 0 / deterministic sampling.** The app's
  default **Auto** mode sends `compatible-only`, so ordinary chat **skips MTP
  entirely** and you are measuring the OFF arm without realising it.
- `VMLX_NATIVE_MTP=0` was once half-wired (runtime off, gate and `/health` still
  on), so every A/B before that fix compared MTP to itself. Fixed; if you ever see
  an A/B come out at exactly 1.00x, suspect the switch before the kernels.

## Reporting a bundle bug

A `gs=64` vs `gs=128` group-size mismatch is a **genuine bug** — report it with
the bundle name attached.

## Running the MTP A/B yourself

On the box:

```
~/mlx/vllm-mlx/scripts/mtp-ab.sh /Volumes/EricsLLMDrive/jangq-ai/Qwen3.8-27B-JANG_4D
```

It serves the bundle twice on port 8014 — once with native MTP on, once with
`VMLX_NATIVE_MTP=0` — sends the same temperature-0 prompt to each, and prints the
decode rate for both arms plus whether the two outputs are byte-identical. Byte
divergence at temp 0 means something is wrong with the MTP path, not a speedup.

Takes roughly 6-8 minutes for a 27B, most of it model loading.
