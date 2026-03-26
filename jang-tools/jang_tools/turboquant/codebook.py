"""
Optimal scalar codebook for TurboQuant.

After Hadamard rotation, each coordinate of a unit-sphere vector follows:
    f_X(x) = Gamma(d/2) / (sqrt(pi) * Gamma((d-1)/2)) * (1-x^2)^((d-3)/2)

which converges to N(0, 1/d) for large d. We compute Lloyd-Max optimal
quantizers for this distribution, giving near-optimal distortion within
2.7x of the Shannon bound.

Codebooks are precomputed per (dimension, bits) pair and cached.

Reference: TurboQuant Theorem 1 -- D_mse <= sqrt(3)*pi/(2*4^b)
"""

import math
from functools import lru_cache

import mlx.core as mx

# NumPy 2.0 renamed np.trapz → _trapezoid
import numpy as np
_trapezoid = getattr(np, 'trapezoid', getattr(np, 'trapz', None))
import numpy as np


def _beta_pdf(x: np.ndarray, d: int) -> np.ndarray:
    """PDF of a coordinate on the unit sphere S^(d-1).

    f_X(x) = Gamma(d/2) / (sqrt(pi) * Gamma((d-1)/2)) * (1-x^2)^((d-3)/2)
    Defined on [-1, 1].
    """
    if d <= 2:
        return np.ones_like(x) * 0.5
    # Use log-space to avoid overflow for large d
    log_const = (
        math.lgamma(d / 2.0)
        - 0.5 * math.log(math.pi)
        - math.lgamma((d - 1) / 2.0)
    )
    # Clamp 1-x^2 to avoid log(0)
    safe = np.maximum(1.0 - x**2, 1e-30)
    log_pdf = log_const + (d - 3) / 2.0 * np.log(safe)
    return np.exp(log_pdf)


@lru_cache(maxsize=64)
def compute_codebook(dim: int, bits: int, n_iter: int = 200) -> list[float]:
    """Compute Lloyd-Max optimal codebook for post-rotation distribution.

    Args:
        dim: Vector dimension (head_dim). Determines the distribution shape.
        bits: Number of quantization bits. Codebook has 2^bits entries.
        n_iter: Lloyd-Max iterations.

    Returns:
        List of 2^bits centroid values, sorted ascending.
    """
    if bits < 1:
        return [0.0]

    n_codes = 1 << bits
    # Dense grid for numerical integration
    n_grid = 10000
    grid = np.linspace(-1.0, 1.0, n_grid)
    pdf = _beta_pdf(grid, dim)
    # Normalize
    total = _trapezoid(pdf, grid)
    if total > 0:
        pdf = pdf / total

    # Initialize centroids uniformly within the effective support
    # For high d, support ~ [-3/sqrt(d), 3/sqrt(d)]
    support = 3.0 / np.sqrt(max(dim, 1))
    centroids = np.linspace(-support, support, n_codes)

    for _ in range(n_iter):
        # Compute boundaries (midpoints between consecutive centroids)
        boundaries = np.concatenate(
            [[-1.0], (centroids[:-1] + centroids[1:]) / 2.0, [1.0]]
        )

        # Update centroids: E[X | X in region_i]
        new_centroids = np.zeros(n_codes)
        for i in range(n_codes):
            lo, hi = boundaries[i], boundaries[i + 1]
            mask = (grid >= lo) & (grid < hi)
            if i == n_codes - 1:
                mask = (grid >= lo) & (grid <= hi)
            if mask.sum() <= 1:
                new_centroids[i] = centroids[i]  # keep old if region is empty
                continue
            mass = _trapezoid(pdf[mask], grid[mask])
            moment = _trapezoid(grid[mask] * pdf[mask], grid[mask])
            new_centroids[i] = moment / max(mass, 1e-10)

        if np.allclose(centroids, new_centroids, atol=1e-10):
            break
        centroids = new_centroids

    return sorted(centroids.tolist())


def _compute_boundaries(codebook: mx.array) -> mx.array:
    """Precompute decision boundaries (midpoints between adjacent centroids)."""
    # boundaries[i] = (codebook[i] + codebook[i+1]) / 2
    return (codebook[:-1] + codebook[1:]) / 2.0


def quantize_scalar(x: mx.array, codebook: mx.array) -> mx.array:
    """Quantize each element to nearest codebook entry.

    Uses precomputed boundaries for O(b) comparisons per element instead
    of O(2^b) distance computations. For small codebooks (4-8 entries),
    vectorized boundary comparison is faster than argmin over distances.

    Args:
        x: Input tensor (any shape).
        codebook: 1D sorted array of centroid values, shape (2^b,).

    Returns:
        Integer indices, same shape as x, dtype uint8.
    """
    boundaries = _compute_boundaries(codebook)
    # Count how many boundaries each element exceeds
    # x > boundary[0] → at least index 1, x > boundary[1] → at least index 2, etc.
    # This is equivalent to searchsorted but fully vectorized in MLX
    indices = mx.zeros(x.shape, dtype=mx.uint8)
    for b in boundaries:
        indices = indices + (x > b).astype(mx.uint8)
    return indices


def dequantize_scalar(indices: mx.array, codebook: mx.array) -> mx.array:
    """Dequantize by looking up codebook entries.

    Args:
        indices: Integer indices from quantize_scalar().
        codebook: Same codebook used for quantization.

    Returns:
        Reconstructed values, same shape as indices.
    """
    return mx.take(codebook, indices.astype(mx.uint32))
