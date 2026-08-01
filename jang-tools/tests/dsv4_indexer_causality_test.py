from types import SimpleNamespace

import numpy as np
import pytest


mx = pytest.importorskip("mlx.core")


@pytest.fixture(autouse=True)
def _run_indexer_contract_on_cpu():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        yield
    finally:
        mx.set_default_device(previous)


class _IdentityRope:
    dims = 32

    def __call__(self, value, offset=0, inverse=False, positions=None):
        del offset, inverse, positions
        return value


def _pool(batch=2, rows=130, width=32):
    # Rows become progressively more attractive. At absolute query position
    # 12 only rows 0, 1, and 2 are causal for ratio=4, so rows >=3 are a
    # deliberate high-score trap for top-k-before-mask implementations.
    values = np.arange(1, rows + 1, dtype=np.float32)
    return mx.array(
        np.broadcast_to(values[None, :, None], (batch, rows, width))
    )


@pytest.mark.parametrize(
    ("offset", "seq_len"),
    [
        (0, 13),
        (8, 5),
    ],
    ids=["one_shot_prefill", "nonzero_offset_chunk"],
)
def test_dense_indexer_excludes_future_rows_before_topk(offset, seq_len):
    from jang_tools.dsv4.mlx_model import Indexer

    cfg = SimpleNamespace(
        hidden_size=32,
        q_lora_rank=32,
        qk_rope_head_dim=32,
        rms_norm_eps=1e-6,
        index_n_heads=1,
        index_head_dim=32,
        index_topk=2,
    )
    indexer = Indexer(cfg, compress_ratio=4)
    indexer.wq_b.weight = mx.eye(32, dtype=mx.float32)
    indexer.weights_proj.weight = mx.ones((1, 32), dtype=mx.float32)

    selected = indexer.select(
        mx.ones((2, seq_len, 32), dtype=mx.float32),
        mx.ones((2, seq_len, 32), dtype=mx.float32),
        _IdentityRope(),
        _pool(),
        start_pos=offset,
    )
    mx.eval(selected)

    # Both calls end at absolute query position 12. Rows 1 and 2 are the two
    # highest-scoring causal rows; every higher-scoring row is still future.
    assert [set(row.tolist()) for row in np.asarray(selected)[:, -1]] == [
        {1, 2},
        {1, 2},
    ]


@pytest.mark.parametrize(
    ("offset", "seq_len"),
    [
        (0, 13),
        (8, 5),
    ],
    ids=["one_shot_prefill", "nonzero_offset_chunk"],
)
def test_native_q8_tiled_indexer_excludes_future_rows_before_topk(
    monkeypatch,
    offset,
    seq_len,
):
    import jang_tools.dsv4.mlx_model as model_module
    from jang_tools.dsv4.mlx_model import _dsv4_tiled_index_topk
    from jang_tools.dsv4.pool_quant_cache import _StateProxy

    state = _StateProxy()
    state._replace_quantized(_pool())
    pooled = state.pool_view()
    assert state._pooled_bf16 is None
    assert len(state._pooled_q_segments) == 3

    # Force 64-row scan tiles so the test also covers global candidate merging
    # across three native-q8 segments, not just one materialized tile.
    monkeypatch.setattr(model_module, "_DSV4_POOL_TILE_TARGET_BYTES", 1)
    selected = _dsv4_tiled_index_topk(
        mx.ones((2, 1, seq_len, 32), dtype=mx.float32),
        mx.ones((2, seq_len, 1), dtype=mx.float32),
        pooled,
        scale=1.0,
        top_k=2,
        offset=offset,
        ratio=4,
    )
    mx.eval(selected)

    assert [set(row.tolist()) for row in np.asarray(selected)[:, -1]] == [
        {1, 2},
        {1, 2},
    ]
