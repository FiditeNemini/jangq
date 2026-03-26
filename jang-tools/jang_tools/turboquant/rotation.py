"""
Randomized Hadamard Transform for TurboQuant.

Rotates vectors so coordinates follow Beta((d-1)/2, (d-1)/2) ~ N(0, 1/d),
enabling near-optimal scalar quantization per coordinate.

For non-power-of-2 dimensions (e.g., 192 for Mistral 4 keys), we pad to
the next power of 2, apply Hadamard, then slice back. The padded zeros
don't affect the rotation of the original coordinates.

Reference: TurboQuant (arXiv:2504.19874), QuIP# (arXiv:2402.04396)
"""

import mlx.core as mx
import numpy as np


def _next_power_of_2(n: int) -> int:
    """Return smallest power of 2 >= n."""
    if n <= 0:
        return 1
    return 1 << (n - 1).bit_length()


def _decompose_into_pow2_blocks(dim: int) -> list[int]:
    """Decompose dim into sum of distinct powers of 2 (binary representation).

    E.g., 192 = 128 + 64 → [128, 64]
    E.g., 128 = 128 → [128]
    E.g., 320 = 256 + 64 → [256, 64]
    """
    blocks = []
    remaining = dim
    while remaining > 0:
        p = 1 << (remaining.bit_length() - 1)  # largest power of 2 <= remaining
        blocks.append(p)
        remaining -= p
    return blocks


def generate_random_signs(dim: int, seed: int = 0) -> mx.array:
    """Generate random +/-1 signs for Randomized Hadamard Transform.

    For non-power-of-2 dims, generates signs for each power-of-2 block
    (e.g., 192 → 128-dim signs + 64-dim signs, concatenated).

    Args:
        dim: The dimension of vectors to rotate.
        seed: Random seed for reproducibility. Same seed = same rotation.

    Returns:
        mx.array of shape (dim,) with values in {-1, +1}.
    """
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=dim).astype(np.float32)
    return mx.array(signs)


def _hadamard_transform(x: mx.array) -> mx.array:
    """Apply unnormalized Hadamard transform along last dimension.

    Uses the recursive butterfly structure: O(d*log d) operations.
    Input last dimension must be a power of 2.
    """
    d = x.shape[-1]
    assert d > 0 and (d & (d - 1)) == 0, f"Dimension {d} must be power of 2"

    h = 1
    while h < d:
        # Reshape to pair adjacent blocks of size h
        shape_prefix = x.shape[:-1]
        x = x.reshape(*shape_prefix, d // (2 * h), 2, h)
        a = x[..., 0, :]  # even indices
        b = x[..., 1, :]  # odd indices
        x = mx.concatenate([a + b, a - b], axis=-1).reshape(*shape_prefix, d)
        h *= 2

    # Normalize to make it orthogonal: divide by sqrt(d)
    return x * (1.0 / (d ** 0.5))


def hadamard_rotate(x: mx.array, signs: mx.array) -> mx.array:
    """Apply Randomized Hadamard Transform: H * diag(signs) * x.

    This is an orthogonal rotation that spreads coordinate energy uniformly,
    making all coordinates follow the same distribution ~ N(0, 1/d).

    For non-power-of-2 dims (e.g., 192 = 128+64), applies block-wise
    Hadamard to each power-of-2 block independently. This is perfectly
    invertible and still spreads outliers within each block.

    Args:
        x: Input tensor, any shape. Last dimension is the one being rotated.
        signs: Random +/-1 vector from generate_random_signs(), shape (dim,).

    Returns:
        Rotated tensor, same shape as x.
    """
    dim = x.shape[-1]
    blocks = _decompose_into_pow2_blocks(dim)

    if len(blocks) == 1:
        # Power-of-2: single Hadamard on full dimension
        return _hadamard_transform(x * signs)

    # Non-power-of-2: apply Hadamard per block
    parts = []
    offset = 0
    for block_size in blocks:
        block_x = x[..., offset:offset + block_size]
        block_signs = signs[offset:offset + block_size]
        parts.append(_hadamard_transform(block_x * block_signs))
        offset += block_size

    return mx.concatenate(parts, axis=-1)


def hadamard_inverse(y: mx.array, signs: mx.array) -> mx.array:
    """Inverse Randomized Hadamard Transform: diag(signs) * H^T * y.

    Since H is symmetric and orthogonal (H = H^T = H^{-1} up to normalization),
    the inverse is: signs * H(y) (same transform, then multiply by signs).

    Args:
        y: Rotated tensor.
        signs: Same signs used in hadamard_rotate().

    Returns:
        Original tensor (within floating point precision).
    """
    dim = y.shape[-1]
    blocks = _decompose_into_pow2_blocks(dim)

    if len(blocks) == 1:
        # Power-of-2: single inverse Hadamard
        return _hadamard_transform(y) * signs

    # Non-power-of-2: inverse Hadamard per block, then undo signs
    parts = []
    offset = 0
    for block_size in blocks:
        block_y = y[..., offset:offset + block_size]
        block_signs = signs[offset:offset + block_size]
        parts.append(_hadamard_transform(block_y) * block_signs)
        offset += block_size

    return mx.concatenate(parts, axis=-1)
