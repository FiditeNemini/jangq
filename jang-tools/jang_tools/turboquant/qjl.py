"""
QJL -- Quantized Johnson-Lindenstrauss Transform for TurboQuant.

Encodes residual vectors into 1-bit sign representations that provide
UNBIASED inner product estimation. Used for key cache compression.

The key insight: sign(S*x) loses magnitude information, but combined
with ||x||_2 and the projection matrix S, we can estimate inner products
without bias. The variance decreases as 1/d.

Q_qjl(x) = sign(S*x)
Dequant: sqrt(pi/2)/d * ||x||_2 * S^T * sign(S*x)
E[<y, dequant(qjl(x))>] = <y, x>  (UNBIASED)

Reference: TurboQuant Lemma 4, QJL paper (arXiv:2406.03482)
"""

import math

import mlx.core as mx
import numpy as np


def generate_qjl_projection(dim: int, seed: int = 0) -> mx.array:
    """Generate random Gaussian projection matrix for QJL.

    Args:
        dim: Dimension of vectors.
        seed: Random seed for reproducibility.

    Returns:
        Projection matrix S of shape (dim, dim), entries ~ N(0, 1).
    """
    rng = np.random.default_rng(seed)
    S = rng.standard_normal((dim, dim)).astype(np.float32)
    return mx.array(S)


def qjl_encode(x: mx.array, S: mx.array) -> tuple[mx.array, mx.array]:
    """Encode a vector using QJL: sign(S*x) and ||x||_2.

    Args:
        x: Input vector(s). Shape (..., dim).
        S: Projection matrix from generate_qjl_projection(). Shape (dim, dim).

    Returns:
        signs: mx.array of {-1, +1}, shape (..., dim).
        norm: mx.array, the L2 norm(s) of x. Shape (..., 1).
    """
    # Project: S*x
    projected = x @ S.T  # (..., dim) @ (dim, dim)^T = (..., dim)

    # Sign quantization: +1 or -1
    signs = mx.where(projected >= 0, mx.array(1.0), mx.array(-1.0))

    # Store norm
    norm = mx.sqrt(mx.sum(x * x, axis=-1, keepdims=True))

    return signs, norm


def qjl_decode(signs: mx.array, norm: mx.array, S: mx.array) -> mx.array:
    """Decode QJL representation back to an approximate vector.

    x_hat = sqrt(pi/2)/d * ||x||_2 * S^T * signs

    Args:
        signs: {-1, +1} array from qjl_encode(). Shape (..., dim).
        norm: L2 norm(s) from qjl_encode(). Shape (..., 1).
        S: Same projection matrix used in encoding.

    Returns:
        Approximate reconstruction of x. Shape (..., dim).
    """
    dim = signs.shape[-1]
    scale = math.sqrt(math.pi / 2.0) / dim

    # S^T * signs
    reconstructed = signs @ S  # (..., dim) @ (dim, dim) = (..., dim)

    return scale * norm * reconstructed


def qjl_inner_product(
    query: mx.array,
    key_signs: mx.array,
    key_norm: mx.array,
    S: mx.array,
) -> mx.array:
    """Estimate <query, key> from QJL-encoded key WITHOUT full reconstruction.

    <y, x_hat_qjl> = sqrt(pi/2)/d * ||x||_2 * <S*y, signs>

    This avoids the O(d^2) cost of S^T*signs by computing S*query once
    and dotting with the sign vector.

    Args:
        query: Query vector(s). Shape (..., dim).
        key_signs: QJL signs of key. Shape (..., dim).
        key_norm: L2 norm of key. Shape (..., 1).
        S: Projection matrix.

    Returns:
        Inner product estimate(s). Shape (..., 1).
    """
    dim = query.shape[-1]
    scale = math.sqrt(math.pi / 2.0) / dim

    # S * query (project query into QJL space)
    q_projected = query @ S.T  # (..., dim)

    # Dot with signs: <S*q, sign(S*k)>
    dot = mx.sum(q_projected * key_signs, axis=-1, keepdims=True)

    return scale * key_norm * dot
