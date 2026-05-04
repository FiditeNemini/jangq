"""Native numpy mel-spectrogram features for parakeet (Nemotron audio path).

Mirrors `transformers.ParakeetFeatureExtractor`:
  - 16 kHz mono audio
  - preemphasis filter (coef=0.97)
  - Short-time Fourier transform (n_fft=512, win_length=400, hop_length=160,
    hann window with `periodic=False`, constant pad mode)
  - Magnitude squared (power spectrum)
  - Mel filterbank (slaney norm, 128 mels, sr=16000, fmin=0, fmax=8000)
  - log(mel + 2^-24)

Output shape: (batch, num_frames, 128) — same as the source extractor's
`input_features`.

We use librosa once at startup to compute the mel filterbank (cached as a
numpy array). Then the runtime path is pure numpy. This avoids the torch
dependency in the audio preprocessing.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

LOG_ZERO_GUARD = 2 ** -24


_MEL_CACHE: dict[tuple, np.ndarray] = {}


def _hann_window(n: int, periodic: bool = False) -> np.ndarray:
    """torch.hann_window equivalent. periodic=False matches scipy/numpy."""
    if periodic:
        return np.hanning(n + 1)[:-1].astype(np.float32)
    return np.hanning(n).astype(np.float32)


def _mel_filters(
    sr: int = 16000,
    n_fft: int = 512,
    n_mels: int = 128,
    fmin: float = 0.0,
    fmax: Optional[float] = None,
) -> np.ndarray:
    """Slaney-normalized mel filterbank, shape (n_mels, n_fft//2+1)."""
    if fmax is None:
        fmax = sr / 2
    key = (sr, n_fft, n_mels, fmin, fmax)
    if key in _MEL_CACHE:
        return _MEL_CACHE[key]
    try:
        import librosa
        fb = librosa.filters.mel(
            sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax,
            norm="slaney",
        ).astype(np.float32)
    except ImportError:
        # Inlined slaney mel filterbank if librosa unavailable.
        fb = _slaney_mel_filterbank(sr, n_fft, n_mels, fmin, fmax)
    _MEL_CACHE[key] = fb
    return fb


def _hz_to_mel_slaney(freqs: np.ndarray) -> np.ndarray:
    """Slaney-style hz → mel: piecewise linear up to 1000 Hz, then log."""
    f_min = 0.0
    f_sp = 200.0 / 3
    min_log_hz = 1000.0
    min_log_mel = (min_log_hz - f_min) / f_sp
    logstep = np.log(6.4) / 27.0
    mels = (freqs - f_min) / f_sp
    log_t = freqs >= min_log_hz
    mels[log_t] = min_log_mel + np.log(freqs[log_t] / min_log_hz) / logstep
    return mels


def _mel_to_hz_slaney(mels: np.ndarray) -> np.ndarray:
    f_min = 0.0
    f_sp = 200.0 / 3
    min_log_hz = 1000.0
    min_log_mel = (min_log_hz - f_min) / f_sp
    logstep = np.log(6.4) / 27.0
    freqs = f_min + f_sp * mels
    log_t = mels >= min_log_mel
    freqs[log_t] = min_log_hz * np.exp(logstep * (mels[log_t] - min_log_mel))
    return freqs


def _slaney_mel_filterbank(
    sr: int, n_fft: int, n_mels: int, fmin: float, fmax: float,
) -> np.ndarray:
    """Pure-numpy fallback for librosa.filters.mel with norm='slaney'."""
    fft_freqs = np.linspace(0, sr / 2, n_fft // 2 + 1)
    mel_min = _hz_to_mel_slaney(np.array([fmin]))[0]
    mel_max = _hz_to_mel_slaney(np.array([fmax]))[0]
    mel_pts = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_pts = _mel_to_hz_slaney(mel_pts)
    fb = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for i in range(n_mels):
        l, c, r = hz_pts[i], hz_pts[i + 1], hz_pts[i + 2]
        # Triangle filter
        rising = (fft_freqs - l) / (c - l + 1e-12)
        falling = (r - fft_freqs) / (r - c + 1e-12)
        fb[i] = np.maximum(0, np.minimum(rising, falling))
        # Slaney normalization: divide by triangle width (in Hz) × 0.5
        fb[i] *= 2.0 / (r - l + 1e-12)
    return fb


def _stft(
    waveform: np.ndarray,
    n_fft: int,
    hop_length: int,
    win_length: int,
    window: np.ndarray,
) -> np.ndarray:
    """torch.stft equivalent with center=True (default), pad_mode='constant'.

    Returns complex-valued spectrogram of shape (n_fft//2+1, n_frames).
    """
    pad = n_fft // 2
    waveform = np.pad(waveform, (pad, pad), mode="constant")
    n_frames = 1 + (len(waveform) - n_fft) // hop_length
    # Pad window to n_fft (centered)
    if win_length < n_fft:
        offset = (n_fft - win_length) // 2
        full_window = np.zeros(n_fft, dtype=np.float32)
        full_window[offset:offset + win_length] = window
    else:
        full_window = window
    # Build (n_frames, n_fft) by sliding
    starts = np.arange(n_frames) * hop_length
    frames = np.stack([waveform[s:s + n_fft] for s in starts], axis=0)
    frames = frames * full_window  # broadcast over n_frames
    # rFFT
    spec = np.fft.rfft(frames, n=n_fft, axis=1)  # (n_frames, n_fft//2+1)
    return spec.T  # (n_fft//2+1, n_frames)


def _preemphasis(waveform: np.ndarray, coef: float) -> np.ndarray:
    """y[t] = x[t] - coef * x[t-1], y[0] = x[0]."""
    if coef == 0.0:
        return waveform
    out = np.empty_like(waveform)
    out[0] = waveform[0]
    out[1:] = waveform[1:] - coef * waveform[:-1]
    return out


def extract_mel_features(
    waveform: np.ndarray,
    *,
    sampling_rate: int = 16000,
    n_fft: int = 512,
    hop_length: int = 160,
    win_length: int = 400,
    n_mels: int = 128,
    preemphasis_coef: float = 0.97,
    normalize: bool = True,
) -> np.ndarray:
    """Compute parakeet mel features for a single waveform.

    Mirrors `transformers.ParakeetFeatureExtractor.__call__`:
      1. Optional mono downmix
      2. Preemphasis filter
      3. STFT (n_fft, win, hop, hann periodic=False, center pad constant)
      4. |X|² power spectrum
      5. Mel filterbank @ power
      6. log(mel + 2^-24)
      7. Per-sample zero-mean unit-variance normalize (with EPSILON=1e-5)

    Args:
        waveform: 1-D float32 numpy array, mono audio at `sampling_rate`.
        normalize: apply per-sample mean/std normalization (default True,
            matches the source feature extractor).

    Returns:
        (1, num_frames, n_mels) float32 array — same shape as
        ParakeetFeatureExtractor's `input_features`.
    """
    EPSILON = 1e-5
    if waveform.dtype != np.float32:
        waveform = waveform.astype(np.float32)
    # Mono guard
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=-1)

    # Preemphasis: y[0]=x[0], y[t]=x[t]-coef*x[t-1]
    waveform = _preemphasis(waveform, preemphasis_coef)

    # Hann window (periodic=False)
    window = _hann_window(win_length, periodic=False)

    # STFT
    spec = _stft(waveform, n_fft, hop_length, win_length, window)
    # |X|^2 power spectrum
    power = np.abs(spec) ** 2  # (n_fft//2+1, n_frames)

    # Mel filterbank
    fb = _mel_filters(sampling_rate, n_fft, n_mels)  # (n_mels, n_fft//2+1)
    mel = fb @ power  # (n_mels, n_frames)

    # log mel
    log_mel = np.log(mel + LOG_ZERO_GUARD)

    # (n_mels, n_frames) → (1, n_frames, n_mels)
    out = log_mel.T[None, :, :].astype(np.float32)

    # Per-sample zero-mean unit-variance normalize (matches source)
    if normalize:
        n_frames = out.shape[1]
        mean = out.mean(axis=1, keepdims=True)                                # (1, 1, n_mels)
        # Use Bessel-corrected variance (n-1) to match source exactly:
        #   variance = sum((x - mean)^2) / (n_frames - 1)
        variance = ((out - mean) ** 2).sum(axis=1, keepdims=True) / (n_frames - 1)
        std = np.sqrt(variance)
        out = (out - mean) / (std + EPSILON)
    return out


def extract_mel_features_batch(
    waveforms: list[np.ndarray], **kwargs,
) -> tuple[np.ndarray, np.ndarray]:
    """Process a batch with right-pad to the longest sequence.

    Returns:
        input_features: (batch, max_frames, n_mels)
        attention_mask: (batch, max_frames) — 1 = real, 0 = pad
    """
    feats = [extract_mel_features(w, **kwargs)[0] for w in waveforms]
    max_frames = max(f.shape[0] for f in feats)
    n_mels = feats[0].shape[1]
    batch = len(feats)
    input_features = np.zeros((batch, max_frames, n_mels), dtype=np.float32)
    attention_mask = np.zeros((batch, max_frames), dtype=np.int32)
    for i, f in enumerate(feats):
        n = f.shape[0]
        input_features[i, :n] = f
        attention_mask[i, :n] = 1
    return input_features, attention_mask
