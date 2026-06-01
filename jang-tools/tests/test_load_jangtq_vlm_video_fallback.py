import numpy as np

from jang_tools.load_jangtq_vlm import _install_video_fallback


class FakeImageProcessor:
    temporal_patch_size = 2
    merge_size = 2

    def __call__(self, images):
        frame_count = len(images)
        return {
            "pixel_values": np.arange(frame_count * 2, dtype=np.float32).reshape(frame_count, 2),
            "image_grid_thw": np.array([[1, 2, 2]] * frame_count, dtype=np.int32),
        }


def test_video_fallback_handles_processors_without_video_processor_attribute():
    class ProcessorWithoutVideoProcessor:
        image_processor = FakeImageProcessor()

        def __call__(self, images=None, text=None, videos=None, **kwargs):
            if videos is not None:
                raise TypeError("image_processor got an unexpected keyword argument 'videos'")
            return {"input_ids": np.array([[1, 2, 3]], dtype=np.int32)}

    processor = ProcessorWithoutVideoProcessor()
    _install_video_fallback(processor)

    result = processor(videos=[["frame-a", "frame-b"]])

    assert result["input_ids"].tolist() == [[1, 2, 3]]
    assert result["pixel_values_videos"].shape == (2, 2)
    assert result["video_grid_thw"].tolist() == [[1, 2, 2]]


def test_video_fallback_uses_real_video_processor_until_optional_dependency_fails():
    class BrokenVideoProcessor:
        pass

    class ProcessorWithBrokenVideoProcessor:
        image_processor = FakeImageProcessor()
        video_processor = BrokenVideoProcessor()

        def __call__(self, images=None, text=None, videos=None, **kwargs):
            if videos is not None:
                raise ImportError("PyAV is not installed")
            return {"input_ids": np.array([[4, 5, 6]], dtype=np.int32)}

    processor = ProcessorWithBrokenVideoProcessor()
    _install_video_fallback(processor)

    result = processor(videos=[["frame-a", "frame-b", "frame-c", "frame-d"]])

    assert result["input_ids"].tolist() == [[4, 5, 6]]
    assert result["pixel_values_videos"].shape == (4, 2)
    assert result["video_grid_thw"].tolist() == [[2, 2, 2]]


def test_video_fallback_handles_mlx_vlm_frame_list_shape_error():
    class ProcessorWithShapeStrictVideoProcessor:
        image_processor = FakeImageProcessor()
        video_processor = object()

        def __call__(self, images=None, text=None, videos=None, **kwargs):
            if videos is not None:
                raise ValueError("Expected video as (T, C, H, W), got shape (4,).")
            return {"input_ids": np.array([[7, 8, 9]], dtype=np.int32)}

    processor = ProcessorWithShapeStrictVideoProcessor()
    _install_video_fallback(processor)

    result = processor(videos=[["frame-a", "frame-b", "frame-c", "frame-d"]])

    assert result["input_ids"].tolist() == [[7, 8, 9]]
    assert result["pixel_values_videos"].shape == (4, 2)
    assert result["video_grid_thw"].tolist() == [[2, 2, 2]]
