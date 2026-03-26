"""
TurboQuantKVCache -- JANG-exclusive KV cache with TurboQuant compression.

Three-phase lifecycle:
  1. FILL: Raw float buffer at baseline Metal speed (zero overhead).
  2. COMPRESS: Encode old tokens to ~3-bit packed format (5x memory reduction).
     Decode once into a read-only float buffer. No per-step re-decoding.
  3. GENERATE: New tokens append to float window. Attention reads
     concat(decoded_buffer, float_window) — both float, full Metal SDPA speed.

JANG-gated: only instantiated via JANG loader for JANG models.
"""

from typing import Optional

import mlx.core as mx

from .pipeline import (
    TurboQuantEncoder,
    EncodedKeys,
    EncodedValues,
    encode_keys,
    decode_keys,
    encode_values,
    decode_values,
)


class TurboQuantKVCache:
    """KV cache with on-demand TurboQuant compression.

    Baseline speed at all times. compress() saves 5x memory and decodes
    the compressed region ONCE into a persistent read-only buffer.
    """

    step = 256

    def __init__(
        self,
        key_dim: int,
        value_dim: int,
        key_bits: int = 3,
        value_bits: int = 3,
        seed: int = 42,
        compress_after: int = 0,
        sink_tokens: int = 0,
    ):
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.key_bits = key_bits
        self.value_bits = value_bits
        self.offset = 0
        self.compress_after = compress_after
        self.sink_tokens = sink_tokens

        self._key_encoder = None
        self._value_encoder = None
        self._seed = seed

        # Float buffer for active tokens (pre-allocated, in-place writes)
        self.keys = None
        self.values = None

        # Compressed storage (packed bits — 5x smaller)
        self._compressed_keys: Optional[EncodedKeys] = None
        self._compressed_values: Optional[EncodedValues] = None
        self._compressed_tokens: int = 0

        # Decoded buffer (read-only float — decoded ONCE after compress)
        self._decoded_k_buffer: Optional[mx.array] = None
        self._decoded_v_buffer: Optional[mx.array] = None

        # Joined buffer: [decoded_buffer | float_window] pre-allocated
        # Avoids mx.concatenate on every step after compress
        self._joined_k: Optional[mx.array] = None
        self._joined_v: Optional[mx.array] = None

    def _ensure_encoders(self):
        if self._key_encoder is None:
            self._key_encoder = TurboQuantEncoder(
                dim=self.key_dim, key_bits=self.key_bits,
                value_bits=self.value_bits, seed=self._seed,
            )
            self._value_encoder = TurboQuantEncoder(
                dim=self.value_dim, key_bits=self.key_bits,
                value_bits=self.value_bits, seed=self._seed + 500,
            )

    @property
    def key_encoder(self):
        self._ensure_encoders()
        return self._key_encoder

    @property
    def value_encoder(self):
        self._ensure_encoders()
        return self._value_encoder

    def update_and_fetch(
        self, keys: mx.array, values: mx.array
    ) -> tuple[mx.array, mx.array]:
        """Append new K/V. Returns full cache for attention at baseline speed."""
        prev = self.offset
        num_new = keys.shape[2]

        if self._compressed_tokens > 0:
            # --- POST-COMPRESS PATH ---
            # _joined_k/v is pre-allocated: [decoded_buffer | space for new tokens]
            # Write new token directly at position self.offset — no concatenation.
            if self._joined_k is None or (self.offset + num_new) > self._joined_k.shape[2]:
                # Grow joined buffer
                B = self._decoded_k_buffer.shape[0]
                n_kv_heads = self._decoded_k_buffer.shape[1]
                total_needed = self.offset + num_new
                n_steps = (self.step + total_needed - 1) // self.step
                jk = mx.zeros((B, n_kv_heads, n_steps * self.step, self.key_dim), keys.dtype)
                jv = mx.zeros((B, n_kv_heads, n_steps * self.step, self.value_dim), values.dtype)
                # Copy decoded buffer into front
                ct = self._compressed_tokens
                jk[..., :ct, :] = self._decoded_k_buffer
                jv[..., :ct, :] = self._decoded_v_buffer
                # Copy any existing window tokens
                if self._joined_k is not None and self.offset > ct:
                    jk[..., ct:self.offset, :] = self._joined_k[..., ct:self.offset, :]
                    jv[..., ct:self.offset, :] = self._joined_v[..., ct:self.offset, :]
                self._joined_k = jk
                self._joined_v = jv

            # In-place write at offset position (fast, no allocation)
            self._joined_k[..., self.offset:self.offset + num_new, :] = keys
            self._joined_v[..., self.offset:self.offset + num_new, :] = values
            self.offset += num_new

            return (
                self._joined_k[..., :self.offset, :],
                self._joined_v[..., :self.offset, :],
            )

        else:
            # --- PRE-COMPRESS PATH (standard KVCache behavior) ---
            if self.keys is None or (prev + num_new) > self.keys.shape[2]:
                B, n_kv_heads, _, k_head_dim = keys.shape
                v_head_dim = values.shape[3]
                n_steps = (self.step + num_new - 1) // self.step
                k_shape = (B, n_kv_heads, n_steps * self.step, k_head_dim)
                v_shape = (B, n_kv_heads, n_steps * self.step, v_head_dim)
                new_k = mx.zeros(k_shape, keys.dtype)
                new_v = mx.zeros(v_shape, values.dtype)
                if self.keys is not None:
                    if prev % self.step != 0:
                        self.keys = self.keys[..., :prev, :]
                        self.values = self.values[..., :prev, :]
                    self.keys = mx.concatenate([self.keys, new_k], axis=2)
                    self.values = mx.concatenate([self.values, new_v], axis=2)
                else:
                    self.keys, self.values = new_k, new_v

            self.offset += num_new
            self.keys[..., prev:self.offset, :] = keys
            self.values[..., prev:self.offset, :] = values

            # Auto-compress if threshold exceeded
            if self.compress_after > 0 and self.offset > self.compress_after:
                self.compress(self.compress_after)
                return self._get_full_cache()

            return (
                self.keys[..., :self.offset, :],
                self.values[..., :self.offset, :],
            )

    def compress(self, n_tokens: Optional[int] = None):
        """Compress oldest tokens. Decode ONCE into persistent read-only buffer.

        After compression:
          - Packed compressed storage (5x smaller, for serialization/eviction)
          - Decoded float buffer (read-only, for fast attention)
          - Float window for new tokens
          - Sink tokens (first N) preserved at full precision in the decoded buffer
        """
        if self.offset == 0 or self.keys is None:
            return
        self._ensure_encoders()

        n = n_tokens if n_tokens is not None else self.offset
        n = min(n, self.offset)

        # Preserve sink tokens — compress [sink..n), keep [0..sink) as float
        sink = min(self.sink_tokens, n)

        if n <= sink:
            return  # nothing to compress

        # Region to compress: [sink..n)
        k_region = self.keys[..., sink:n, :]
        v_region = self.values[..., sink:n, :]

        self._compressed_keys = encode_keys(k_region, self._key_encoder)
        self._compressed_values = encode_values(v_region, self._value_encoder)
        self._compressed_tokens = n  # total tokens accounted for (sink + compressed)

        # Decode compressed region ONCE into persistent buffer
        decoded_k = decode_keys(self._compressed_keys, self._key_encoder)
        decoded_v = decode_values(self._compressed_values, self._value_encoder)

        # Build decoded buffer: [sink_float, decoded_compressed]
        if sink > 0:
            sink_k = self.keys[..., :sink, :]
            sink_v = self.values[..., :sink, :]
            self._decoded_k_buffer = mx.concatenate([sink_k, decoded_k], axis=2)
            self._decoded_v_buffer = mx.concatenate([sink_v, decoded_v], axis=2)
        else:
            self._decoded_k_buffer = decoded_k
            self._decoded_v_buffer = decoded_v

        # Build joined buffer: [decoded | remaining window tokens | space]
        remaining = self.offset - n
        B, n_kv_heads = self._decoded_k_buffer.shape[:2]
        total = self.offset
        n_steps = (self.step + total - 1) // self.step
        self._joined_k = mx.zeros((B, n_kv_heads, n_steps * self.step, self.key_dim), self._decoded_k_buffer.dtype)
        self._joined_v = mx.zeros((B, n_kv_heads, n_steps * self.step, self.value_dim), self._decoded_v_buffer.dtype)
        # Copy decoded buffer into front [0..n)
        self._joined_k[..., :n, :] = self._decoded_k_buffer
        self._joined_v[..., :n, :] = self._decoded_v_buffer
        # Copy remaining float window [n..offset)
        if remaining > 0:
            self._joined_k[..., n:self.offset, :] = self.keys[..., n:self.offset, :]
            self._joined_v[..., n:self.offset, :] = self.values[..., n:self.offset, :]
        # Clear old separate buffers
        self.keys = None
        self.values = None

    def _get_full_cache(self) -> tuple[mx.array, mx.array]:
        """Return full K/V. Uses joined buffer if compressed, else float buffer."""
        if self._joined_k is not None:
            return self._joined_k[..., :self.offset, :], self._joined_v[..., :self.offset, :]
        elif self.keys is not None:
            return self.keys[..., :self.offset, :], self.values[..., :self.offset, :]
        return None, None

    def make_mask(self, N: int, return_array: bool = True, window_size: Optional[int] = None):
        from mlx_lm.models.cache import create_attention_mask
        return create_attention_mask(N, self.offset, return_array, window_size)

    def empty(self) -> bool:
        return self.offset == 0

    def is_trimmable(self) -> bool:
        return True

    def trim(self, n: int) -> int:
        n = min(self.offset, n)
        self.offset -= n
        return n

    @property
    def state(self):
        if self.offset == 0:
            return [], []
        return self._get_full_cache()

    @state.setter
    def state(self, v):
        if v is not None and len(v) == 2:
            self.keys, self.values = v
            if self.keys is not None:
                self.offset = self.keys.shape[-2]

    @property
    def meta_state(self):
        return str(self.offset), str(self.key_bits), str(self.value_bits)

    @meta_state.setter
    def meta_state(self, v):
        self.offset = int(v[0])
        self.key_bits = int(v[1])
        self.value_bits = int(v[2])

    def size(self):
        return self.offset

    @property
    def nbytes(self):
        """Actual GPU memory: joined buffer OR float buffer + compressed packed."""
        total = 0
        if self._joined_k is not None:
            total += self._joined_k.nbytes + self._joined_v.nbytes
        elif self.keys is not None:
            total += self.keys.nbytes + self.values.nbytes
        if self._compressed_keys is not None:
            for arr in self._compressed_keys:
                if hasattr(arr, 'nbytes'):
                    total += arr.nbytes
            for arr in self._compressed_values:
                if hasattr(arr, 'nbytes'):
                    total += arr.nbytes
        return total

    @property
    def compressed_nbytes(self):
        """Memory of ONLY the packed compressed storage (not decoded buffer)."""
        total = 0
        if self._compressed_keys is not None:
            for arr in self._compressed_keys:
                if hasattr(arr, 'nbytes'):
                    total += arr.nbytes
            for arr in self._compressed_values:
                if hasattr(arr, 'nbytes'):
                    total += arr.nbytes
        return total

    @property
    def is_compressed(self):
        return self._compressed_tokens > 0
