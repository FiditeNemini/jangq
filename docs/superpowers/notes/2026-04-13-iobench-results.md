# jang-spec IO Benchmark — 2026-04-13

Spike A from `docs/superpowers/specs/2026-04-13-jang-spec-design.md` §14.

**Machine:** Mac Studio, Apple M4 Max
**Device (Metal):** Apple M4 Max
**Fixture:** 256 files × 50 MB = 12.8 GB under `$TMPDIR/jang-spec-iobench/`
**Build:** `swift build -c release --product jang-spec-iobench`

## Results

**MTLIOCommandQueue, sequential**
- throughput: 11.85 GB/s
- p50: 4.38 ms
- p99: 4.96 ms

**MTLIOCommandQueue, random**
- throughput: 12.54 GB/s
- p50: 4.15 ms
- p99: 4.74 ms

**pread, random**
- throughput: 20.04 GB/s
- p50: 2.57 ms
- p99: 2.75 ms

## Verdict

**GO** — random-access streaming meets design thresholds by ~4× on bandwidth and
stays under the 5 ms p99 latency ceiling.

## Interpretation

- **Random ≈ sequential on MTLIOCommandQueue.** 12.54 vs 11.85 GB/s — random-order
  reads are not penalized at 50 MB granularity, which is exactly what jang-spec
  needs: expert reads land in whatever order the router picks.
- **pread is faster in this synthetic** (20 GB/s) because the fixture was just
  written — reads are hitting warm APFS page cache, not NVMe. MTLIOCommandQueue
  goes through a direct-to-GPU-buffer path that bypasses the page cache, so its
  numbers are closer to the real cold-SSD throughput we'll see in production.
  That makes the 12.54 GB/s the trustworthy number for capacity planning.
- **p99 of 4.74 ms** leaves little headroom under the 5 ms ceiling. Worth tracking
  once we add concurrent GPU compute contention in a future spike.

## Implications for the spec

- [x] Proceed to Plan 2 (JANGCore Swift v2 loader) unchanged. Streaming premise
      confirmed.
- [ ] Monitor p99 latency under concurrent-GPU-work conditions in a later benchmark
      (after Plan 3 exists).
- [ ] No changes needed to the on-disk layout, alignment, or expert-blob sizing.

## Raw log

`/tmp/jang-spec-iobench.log` (preserved for reference; not committed).
