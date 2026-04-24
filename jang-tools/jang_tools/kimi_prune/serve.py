"""OpenAI-compatible HTTP server for JANGTQ models (Kimi K2.6 and friends).

Thin wrapper around `mlx_lm.server` that:

  1. Applies the fp32 MLA-SDPA patch to `mlx_lm.models.deepseek_v3` on
     startup (idempotent; no-op if already patched).
  2. Monkey-patches `mlx_lm.server.ModelProvider.load` so any model path
     containing `jang_config.json` is loaded via `jang_tools.load_jangtq`
     instead of `mlx_lm.utils.load` — gets JANG's TurboQuant kernels,
     compile-friendly MoE, wired-limit auto-tuning, MLA bit-width fix, etc.
  3. Delegates the HTTP server, chat template, tool-call parsing,
     streaming, prompt caching, LRU eviction, speculative-draft, and
     distributed sharding to the stock `mlx_lm.server`.

Kimi K2's tool parser (`mlx_lm.tool_parsers.kimi_k2`) is auto-detected from
the chat template by mlx_lm's tokenizer loader, so tool-calling endpoints
return structured `tool_calls` automatically — no additional wiring needed
here.

Usage:
  python -m jang_tools.kimi_prune.serve \\
      --model /path/to/Kimi-K2.6-REAP-30-JANGTQ_1L \\
      --trust-remote-code \\
      --host 0.0.0.0 --port 8080

Then from any OpenAI-compatible client (curl / openai-python / etc.):
  POST http://localhost:8080/v1/chat/completions
  { "model": "default_model",
    "messages": [{"role": "user", "content": "Capital of France?"}],
    "max_tokens": 120 }

For tool calls, include `tools: [...]` per the OpenAI spec — the server
parses Kimi's `<|tool_call_begin|>functions.NAME:0<|tool_call_argument_begin|>{JSON}<|tool_call_end|>`
format out of the generated text and returns structured `tool_calls`.
"""

from __future__ import annotations

import sys
from pathlib import Path


def _apply_mla_patch_if_needed() -> None:
    """Apply the fp32 SDPA fix if mlx_lm.models.deepseek_v3 doesn't have it."""
    try:
        from jang_tools.kimi_prune.runtime_patch import apply
        apply(dry_run=False)
    except Exception as e:
        print(
            f"[serve] WARNING: MLA fp32 SDPA patch failed to apply: {e!r}\n"
            f"[serve] Decode on quantized Kimi / GLM / DSV3 may produce repetition loops.",
            file=sys.stderr,
        )


def _install_jangtq_loader_hook() -> None:
    """Route models with jang_config.json through load_jangtq_model.

    mlx_lm.server.ModelProvider.load normally calls mlx_lm.utils.load, which
    reads the safetensors via mlx's standard quantized layers. For JANGTQ
    bundles we need the TurboQuantLinear / TurboQuantSwitchLinear kernels
    from jang_tools, so we intercept and dispatch there.

    Non-JANGTQ model paths go through mlx_lm.utils.load unchanged.
    """
    from mlx_lm import server as _srv
    from mlx_lm.utils import load as _mlx_lm_load

    _original_load = _srv.ModelProvider.load

    def _is_jangtq_path(model_path: str) -> bool:
        if not model_path or model_path == "default_model":
            return False
        p = Path(model_path)
        if not p.exists():
            return False
        return (p / "jang_config.json").exists()

    def load(self, model_path, adapter_path=None, draft_model_path=None):
        target = self.default_model_map.get(model_path, model_path) if hasattr(self, "default_model_map") else model_path
        # For "default_model" sentinel we still need to check the resolved CLI
        # path — the default_model_map translation gives us that concrete path.
        probe = self.cli_args.model if target == "default_model" else target
        if _is_jangtq_path(probe):
            if self.model_key == (target, adapter_path, draft_model_path):
                return self.model, self.tokenizer
            print(f"[serve] JANGTQ path detected: {probe}", flush=True)
            from jang_tools.load_jangtq import load_jangtq_model
            self.model, self.tokenizer = load_jangtq_model(probe)
            self.model_key = (target, adapter_path, draft_model_path)
            self.draft_model = None
            return self.model, self.tokenizer
        # Fall back to stock mlx_lm.server loader (handles adapter, shard, draft).
        return _original_load(self, model_path, adapter_path=adapter_path, draft_model_path=draft_model_path)

    _srv.ModelProvider.load = load
    # Mark so we don't install twice if the module is reloaded.
    _srv.ModelProvider._jang_loader_installed = True


def main() -> int:
    _apply_mla_patch_if_needed()
    _install_jangtq_loader_hook()

    # Delegate to mlx_lm.server.main — it owns argparse, HTTP, streaming,
    # tool-call parsing, etc. Any future mlx_lm server improvements (e.g.
    # batching, new sampler flags) are picked up automatically.
    from mlx_lm.server import main as _srv_main
    return _srv_main() or 0


if __name__ == "__main__":
    sys.exit(main())
