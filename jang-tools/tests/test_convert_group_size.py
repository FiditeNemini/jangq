from jang_tools.convert import _get_mlx_compatible_group_size


def test_mlx_group_size_shrinks_for_dsv4_tiny_projection():
    assert _get_mlx_compatible_group_size(32, 128) == 32


def test_mlx_group_size_prefers_largest_supported_divisor_not_exceeding_request():
    assert _get_mlx_compatible_group_size(192, 128) == 64
    assert _get_mlx_compatible_group_size(128, 64) == 64


def test_mlx_group_size_returns_none_when_unquantizable_by_mlx():
    assert _get_mlx_compatible_group_size(24, 128) is None
