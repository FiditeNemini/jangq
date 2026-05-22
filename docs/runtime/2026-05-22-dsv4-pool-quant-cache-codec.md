# DSV4 Pool Quant Cache Codec - 2026-05-22

Status: experimental, not a release-default path.

## Finding

`PoolQuantizedV4Cache` was quantizing the whole accumulated DSV4 compressor
and indexer pool on each appended pool row. The bad path was:

1. `DeepseekV4Cache.update_pool()` reads `state["pooled"]`.
2. `_StateProxy.__getitem__("pooled")` dequantizes the existing full pool.
3. `update_pool()` concatenates the new row.
4. `_StateProxy.__setitem__("pooled")` quantizes the whole enlarged pool.

That makes decode cost grow with accumulated pool length and repeatedly
round-trips old pool rows through the lossy codec.

## Fix

`PoolQuantizedV4Cache.update_pool()` now appends only newly produced rows
through `_StateProxy.append_pooled()`. Old rows stay in their existing quantized
segments, so they are not requantized on every decode append.

Regression test:

```sh
PYTHONPATH=/Users/eric/jang/jang-tools \
  /Users/eric/mlx/vllm-mlx-finite-launch-guard/.venv/bin/python \
  -m pytest -q \
  jang-tools/tests/test_dsv4_pool_quant_cache.py::test_pool_quant_cache_appends_new_rows_without_requantizing_old_pool
```

## Live vMLX Evidence

After installing this local JANG source into the vMLX worktree venv:

```sh
uv pip install --python /Users/eric/mlx/vllm-mlx-finite-launch-guard/.venv/bin/python \
  --force-reinstall --no-deps /Users/eric/jang/jang-tools
```

DSV4 native prefix/paged/L2 with pool quant still remained too slow:

- `DSV4_POOL_QUANT=1`: `build/current-dsv4-pool-quant-append-stream-spacing-20260522.json`
  - 192 streamed content deltas in 80.21s, about 2.39 deltas/s
  - raw streamed HTML spacing looked normal in this probe
- `DSV4_POOL_QUANT=0`: `build/current-dsv4-pooloff-stream-spacing-20260522.json`
  - 192 streamed content deltas in 29.87s, about 6.43 deltas/s
  - raw streamed HTML spacing looked normal in this probe

Earlier non-stream vMLX A/B remained consistent with this:

- native prefix/paged/L2, pool quant off: about 10.33 tok/s
- native prefix/paged/L2, pool quant on before this fix: about 2.97 tok/s

## Release Rule

Keep `DSV4_POOL_QUANT=0` as the app/release default. The append-only codec fix
removes repeated requantization and should reduce quality drift risk, but the
pool-quant path still dequantizes and concatenates the full pool for attention
reads and remains too slow for production.

Do not re-enable DSV4 pool quant in vMLX UI or startup defaults until a second
runtime design avoids full-pool dequantize/concat on every decode read and a
live cache-on tool/code/spacing gate proves both speed and output fidelity.
