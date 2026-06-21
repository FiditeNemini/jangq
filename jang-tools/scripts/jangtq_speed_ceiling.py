#!/usr/bin/env python3
"""Compute whole-model JANGTQ speed ceilings from measured component fractions."""

from __future__ import annotations

import argparse


def total_speedup(fraction: float, optimized_speedup: float) -> float:
    return 1.0 / ((1.0 - fraction) + fraction / optimized_speedup)


def required_fraction(target_speedup: float, optimized_speedup: float) -> float | None:
    if target_speedup > optimized_speedup:
        return None
    if target_speedup == optimized_speedup:
        return 1.0
    return (1.0 - 1.0 / target_speedup) / (1.0 - 1.0 / optimized_speedup)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Amdahl calculator for JANGTQ M5 TensorOps work. Use this with a "
            "measured prompt-processing profile: fraction is the share of PP "
            "wall time covered by the optimized expert cluster."
        )
    )
    parser.add_argument(
        "--fraction",
        type=float,
        action="append",
        default=[],
        help="Measured accelerated fraction, e.g. 0.8. Can be repeated.",
    )
    parser.add_argument(
        "--optimized-speedup",
        type=float,
        action="append",
        default=[],
        help="Speedup of the optimized region, e.g. 6. Can be repeated.",
    )
    parser.add_argument(
        "--target",
        type=float,
        action="append",
        default=[2.0, 3.0, 4.0],
        help="Target total speedup. Defaults to 2, 3, and 4.",
    )
    args = parser.parse_args()

    fractions = args.fraction or [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    speeds = args.optimized_speedup or [2.0, 3.0, 4.0, 6.0, 8.0, 12.0]

    print("Required accelerated fraction")
    print("optimized_speedup\ttarget_total\trequired_fraction")
    for speed in speeds:
        for target in args.target:
            req = required_fraction(target, speed)
            req_text = "impossible" if req is None else f"{req:.3f}"
            print(f"{speed:.3f}\t{target:.3f}\t{req_text}")

    print()
    print("Total speedup for measured fractions")
    header = "fraction\t" + "\t".join(f"s={speed:g}" for speed in speeds)
    print(header)
    for fraction in fractions:
        values = [f"{total_speedup(fraction, speed):.3f}" for speed in speeds]
        print(f"{fraction:.3f}\t" + "\t".join(values))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

