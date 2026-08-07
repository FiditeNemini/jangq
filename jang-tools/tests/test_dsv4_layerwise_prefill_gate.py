"""Context-aware default for DSV4 layerwise prefill materialization.

The per-layer ``mx.eval`` barrier costs 25-30% prefill throughput (live A/B:
267 vs 346 pp/s at 15k) while attention sub-chunking already bounds the
dominant score temporaries.  Unset env must therefore skip barriers up to the
standalone-proven 24,576-token curve and engage them beyond it; explicit env
values keep absolute on/off semantics.
"""

import os
from types import SimpleNamespace
from unittest import mock

from jang_tools.dsv4.mlx_model import _layerwise_prefill_materialization_enabled


def _ids(n):
    return SimpleNamespace(shape=(1, n))


def _call(chunk, final_context, env):
    clean = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith("DSV4_LAYERWISE_PREFILL")
        and k != "DSV4_ATTN_SUBCHUNK"
    }
    clean.update(env)
    with mock.patch.dict(os.environ, clean, clear=True):
        return _layerwise_prefill_materialization_enabled(
            _ids(chunk), final_context
        )


def test_unset_env_skips_barrier_below_auto_threshold():
    assert _call(2048, 15_000, {}) is False


def test_unset_env_engages_barrier_beyond_auto_threshold():
    assert _call(2048, 39_000, {}) is True


def test_unset_env_engages_barrier_at_threshold_boundary():
    assert _call(2048, 24_576, {}) is False
    assert _call(2048, 24_577, {}) is True


def test_unset_env_with_subchunk_disabled_keeps_barrier_everywhere():
    assert _call(2048, 4_096, {"DSV4_ATTN_SUBCHUNK": "0"}) is True


def test_explicit_on_is_absolute_regardless_of_context():
    assert _call(2048, 4_096, {"DSV4_LAYERWISE_PREFILL": "1"}) is True


def test_explicit_off_is_absolute():
    assert _call(2048, 100_000, {"DSV4_LAYERWISE_PREFILL": "0"}) is False


def test_short_chunks_never_barrier():
    assert _call(128, 100_000, {}) is False
    assert _call(128, 100_000, {"DSV4_LAYERWISE_PREFILL": "1"}) is False


def test_auto_threshold_env_override():
    env = {"DSV4_LAYERWISE_PREFILL_AUTO_TOKENS": "8192"}
    assert _call(2048, 9_000, env) is True
    assert _call(2048, 8_000, env) is False
