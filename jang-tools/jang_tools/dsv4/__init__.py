"""DeepSeek-V4 quantization + runtime (MLX)."""

# Register `deepseek_v4` model type with mlx_lm on first import so that
# `load_jangtq_model` finds the model class via the standard mlx_lm factory.
from . import mlx_register  # noqa: F401
