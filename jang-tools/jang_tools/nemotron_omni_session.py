"""
Multi-turn chat session for Nemotron-3-Nano-Omni-30B-A3B.

Wraps OmniChat with persistent KV + Mamba caches across turns. Each turn:

  1. Build the user message (text + multimodal placeholders).
  2. Encode any new images / audio / video via PyTorch encoders.
  3. Tokenize the new user turn (only — assistant's previous reply stays in cache).
  4. Embed text tokens via MLX, inject multimodal embeddings at placeholders.
  5. Prefill via inputs_embeds, **using the existing cache** (so the cache picks
     up where the previous turn left off).
  6. Decode token-by-token until EOS or max_tokens.
  7. Cache now contains <prompt_turn1, reply_turn1, prompt_turn2, reply_turn2>
     and is ready for turn 3.

Key correctness points:
  - Mamba SSM state (ArraysCache) persists per layer across turns. The state
    encodes everything seen up to the previous turn.
  - KVCache for `*` attention layers grows monotonically — at turn N, K and V
    have accumulated all turns 1..N-1.
  - Position offsets are managed automatically by the cache classes' `offset`
    attribute used by `create_attention_mask`.

Usage:
    from jang_tools.nemotron_omni_session import OmniSession
    sess = OmniSession("OsaurusAI/Nemotron-3-Nano-Omni-30B-A3B-MXFP4")

    print(sess.turn("Capital of France?"))
    print(sess.turn("And of Germany?"))
    print(sess.turn("What did I just ask about?"))  # references prior turns
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Sequence, Union

import numpy as np

from jang_tools.nemotron_omni_chat import OmniChat


class OmniSession(OmniChat):
    """OmniChat extended with persistent multi-turn cache."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Persistent caches across turns. Built lazily on first turn.
        self._cache = None
        self._history_text: List[dict] = []  # [{"role":..., "content":...}]
        self._eos_ids = {11}  # nemotron-h <|im_end|>

    def reset(self):
        """Wipe the cache + history (start a fresh conversation)."""
        self._cache = None
        self._history_text = []

    def _ensure_cache(self):
        if self._cache is None:
            self._cache = self.mlx_model.make_cache()

    def _build_turn_prompt(
        self,
        user_text: str,
        n_image_tokens: int = 0,
        n_video_tokens: int = 0,
        n_audio_tokens: int = 0,
        is_first: bool = False,
    ) -> str:
        """Build the prompt fragment for THIS turn only.

        For turn-1: full chat template with system + user + generation prompt.
        For later turns: just the user turn + generation prompt (no system,
        no replay of earlier turns — the cache already has them).
        """
        media = ""
        if n_image_tokens > 0:
            media += "<img>" + ("<image>" * n_image_tokens) + "</img>\n"
        if n_video_tokens > 0:
            # Source processing.py reuses <img>/<image> placeholders for
            # video: <video> is plain text for this tokenizer, not an embed slot.
            media += "<img>" + ("<image>" * n_video_tokens) + "</img>\n"
        if n_audio_tokens > 0:
            media += "<sound>" + ("<so_embedding>" * n_audio_tokens) + "</sound>\n"
        msg_content = media + user_text

        # First turn: render full template (includes system if any).
        # Subsequent: render JUST this user turn + assistant generation marker.
        # We rely on the chat template producing a deterministic per-turn delta.
        if is_first:
            full = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": msg_content}],
                tokenize=False,
                add_generation_prompt=True,
            )
            return full
        # For follow-up turns, render two-turn template (assistant placeholder
        # + new user turn) and slice off everything before the latest user
        # turn. Different chat templates use different markers; we look for
        # the text-equivalent of the user-role open tag.
        prev_then_now = self.tokenizer.apply_chat_template(
            [
                {"role": "user", "content": "__PREV_USER__"},
                {"role": "assistant", "content": "__PREV_ASST__"},
                {"role": "user", "content": msg_content},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        # Find the second user turn opening — locate __PREV_ASST__ which marks
        # the end of the prior assistant. Everything after that token is the
        # new user turn delta.
        marker = "__PREV_ASST__"
        idx = prev_then_now.find(marker)
        if idx < 0:
            # Fallback: just render this user message in isolation.
            return self.tokenizer.apply_chat_template(
                [{"role": "user", "content": msg_content}],
                tokenize=False,
                add_generation_prompt=True,
            )
        return prev_then_now[idx + len(marker):]

    def turn(
        self,
        text: str,
        images: Optional[Sequence[Union[str, Path]]] = None,
        video: Optional[Union[str, Path]] = None,
        audio: Optional[Union[str, Path]] = None,
        max_tokens: int = 256,
        temperature: float = 0.6,
        top_p: float = 0.95,
    ) -> str:
        """Run a single conversational turn, advancing the persistent cache."""
        self._ensure_cache()
        from PIL import Image

        # Encode multimodal inputs
        image_embeds = None
        n_image_tokens = 0
        if images:
            pil_images = [Image.open(str(p)).convert("RGB") for p in images]
            image_embeds = self._extract_image_embeddings(pil_images)
            n_image_tokens = image_embeds.shape[0] * image_embeds.shape[1]

        video_embeds = None
        n_video_tokens = 0
        if video is not None:
            video_embeds = self._extract_video_embeddings(str(video))
            n_video_tokens = video_embeds.shape[0] * video_embeds.shape[1]

        audio_embeds = None
        n_audio_tokens = 0
        if audio is not None:
            audio_embeds = self._extract_audio_embeddings(str(audio))
            n_audio_tokens = audio_embeds.shape[0] * audio_embeds.shape[1]

        is_first = len(self._history_text) == 0
        prompt = self._build_turn_prompt(
            text,
            n_image_tokens=n_image_tokens,
            n_video_tokens=n_video_tokens,
            n_audio_tokens=n_audio_tokens,
            is_first=is_first,
        )
        input_ids = self.tokenizer(prompt, return_tensors="np")["input_ids"]

        # Embed text → mlx, inject multimodal embeddings
        mx = self.mx
        ids_mx = mx.array(input_ids)
        text_embeds = self.mlx_model.backbone.embeddings(ids_mx)
        text_embeds = self._inject_embeddings(
            input_ids, text_embeds, image_embeds, video_embeds, audio_embeds,
        )

        # Decode using the persistent cache (advanced in place by mlx_lm
        # backbone forward with cache=...).
        reply = self._decode_turn(
            text_embeds, max_tokens=max_tokens,
            temperature=temperature, top_p=top_p,
        )

        # Update history bookkeeping
        self._history_text.append({"role": "user", "content": text})
        self._history_text.append({"role": "assistant", "content": reply})
        return reply

    def _decode_turn(self, inputs_embeds, max_tokens, temperature, top_p):
        """Prefill (via inputs_embeds) + decode using self._cache."""
        mx = self.mx
        model = self.mlx_model
        backbone = model.backbone
        from mlx_lm.models.base import create_attention_mask, create_ssm_mask

        cache = self._cache  # persistent across turns!

        # Prefill via embed-injected sequence.
        h = inputs_embeds
        attn_mask = create_attention_mask(h, cache[backbone.fa_idx])
        ssm_mask = create_ssm_mask(h, cache[backbone.ssm_idx])
        ci = 0
        for layer in backbone.layers:
            if layer.block_type in ("M", "*"):
                c = cache[ci]; ci += 1
                mask_l = attn_mask if layer.block_type == "*" else ssm_mask
                h = layer(h, mask=mask_l, cache=c)
            else:
                h = layer(h)
        h = backbone.norm_f(h)
        logits = model.lm_head(h)
        next_logit = logits[:, -1, :]

        def sample(logit, temp, tp):
            if temp <= 0:
                return mx.argmax(logit, axis=-1, keepdims=True)
            logit = logit / temp
            if tp >= 1.0:
                return mx.random.categorical(logit)[..., None]
            sorted_idx = mx.argsort(-logit, axis=-1)
            sorted_logits = mx.take_along_axis(logit, sorted_idx, axis=-1)
            sorted_probs = mx.softmax(sorted_logits, axis=-1)
            cumprobs = mx.cumsum(sorted_probs, axis=-1)
            keep = mx.concatenate(
                [mx.ones_like(cumprobs[..., :1]) > 0, cumprobs[..., :-1] <= tp],
                axis=-1,
            )
            neg_inf = mx.full(sorted_logits.shape, -1e9, dtype=sorted_logits.dtype)
            filtered = mx.where(keep, sorted_logits, neg_inf)
            tok_in_sorted = mx.random.categorical(filtered)[..., None]
            return mx.take_along_axis(sorted_idx, tok_in_sorted, axis=-1)

        tokens: list[int] = []
        tok = sample(next_logit, temperature, top_p)
        tokens.append(int(tok.item()))
        for _ in range(max_tokens - 1):
            if tokens[-1] in self._eos_ids:
                break
            h = backbone.embeddings(tok)
            attn_mask = create_attention_mask(h, cache[backbone.fa_idx])
            ssm_mask = create_ssm_mask(h, cache[backbone.ssm_idx])
            ci = 0
            for layer in backbone.layers:
                if layer.block_type in ("M", "*"):
                    c = cache[ci]; ci += 1
                    mask_l = attn_mask if layer.block_type == "*" else ssm_mask
                    h = layer(h, mask=mask_l, cache=c)
                else:
                    h = layer(h)
            h = backbone.norm_f(h)
            logits = model.lm_head(h)
            tok = sample(logits[:, -1, :], temperature, top_p)
            tokens.append(int(tok.item()))
        return self.tokenizer.decode(tokens, skip_special_tokens=True)


def main():
    """Interactive multi-turn CLI."""
    import sys
    if len(sys.argv) < 2:
        print(
            "usage: python -m jang_tools.nemotron_omni_session <bundle_path>\n"
            "  Then type messages; commands: /image PATH, /audio PATH, /video PATH, /reset, /quit",
            file=sys.stderr,
        )
        sys.exit(2)
    sess = OmniSession(sys.argv[1])
    pending_images: List[str] = []
    pending_audio: Optional[str] = None
    pending_video: Optional[str] = None
    print("\nReady. Type messages. Commands: /image PATH, /audio PATH, "
          "/video PATH, /reset, /quit.\n", flush=True)
    while True:
        try:
            line = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[bye]")
            break
        if not line:
            continue
        if line == "/quit":
            break
        if line == "/reset":
            sess.reset()
            pending_images.clear()
            pending_audio = pending_video = None
            print("[reset]")
            continue
        if line.startswith("/image "):
            pending_images.append(line[len("/image "):].strip())
            print(f"[queued image: {pending_images[-1]}]")
            continue
        if line.startswith("/audio "):
            pending_audio = line[len("/audio "):].strip()
            print(f"[queued audio: {pending_audio}]")
            continue
        if line.startswith("/video "):
            pending_video = line[len("/video "):].strip()
            print(f"[queued video: {pending_video}]")
            continue
        reply = sess.turn(
            line,
            images=pending_images or None,
            audio=pending_audio,
            video=pending_video,
            max_tokens=200,
        )
        print(f"asst> {reply}\n", flush=True)
        pending_images.clear()
        pending_audio = pending_video = None


if __name__ == "__main__":
    main()
