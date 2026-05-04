"""jang_tools.nemotron_omni — native MLX components for Nemotron-3-Nano-Omni-30B.

Stage 1 (PyTorch hybrid bridge — works today):
    from jang_tools.nemotron_omni_chat import OmniChat        # single-turn
    from jang_tools.nemotron_omni_session import OmniSession   # multi-turn

Stage 2 (native MLX, queued — multi-day port):
    from jang_tools.nemotron_omni.radio import RADIOVisionModel
    from jang_tools.nemotron_omni.parakeet import ParakeetEncoder
    from jang_tools.nemotron_omni.projectors import VisionMLP, SoundProjection
    from jang_tools.nemotron_omni.image_processor import NemotronImageProcessor
    from jang_tools.nemotron_omni.video_processor import NemotronVideoProcessor
    from jang_tools.nemotron_omni.audio_features import ParakeetFeatureExtractor
    from jang_tools.nemotron_omni.model import NemotronHOmni

This package's classes are stub/placeholder until stage-2 lands. The
production runtime today is via stage-1 hybrid (OmniChat / OmniSession at
the top-level jang_tools).
"""
