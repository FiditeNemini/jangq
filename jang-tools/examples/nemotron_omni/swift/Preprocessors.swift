// Preprocessors.swift
// Image / video / audio preprocessors for Nemotron-3-Nano-Omni.
//
// Apple-native: CIImage + AVFoundation + Accelerate (vDSP, vImage). No
// PyTorch / torchvision dependency.

import Foundation
import CoreImage
import Accelerate
import AVFoundation
import MLX

// MARK: - Image preprocessing (NVLM 1-D dynamic tile, NVLM-style)

/// CLIP normalization mean (matches source `norm_mean`)
public let CLIP_MEAN: [Float] = [0.48145466, 0.4578275, 0.40821073]
/// CLIP normalization std (matches source `norm_std`)
public let CLIP_STD: [Float] = [0.26862954, 0.26130258, 0.27577711]

/// NVLM dynamic tile preprocessing.
///
/// Produces (num_tiles, 3, H, W) pixel values normalized via CLIP mean/std.
@available(macOS 14.0, *)
public func preprocessImages(
    _ images: [CIImage],
    imageSize: Int = 512,
    minNum: Int = 1,
    maxNum: Int = 12,
    useThumbnail: Bool = true
) -> (pixelValues: MLXArray, tileCounts: [Int]) {
    var allTiles: [CIImage] = []
    var tileCounts: [Int] = []
    for img in images {
        let tiles = dynamicPreprocess(
            img, imageSize: imageSize,
            minNum: minNum, maxNum: maxNum,
            useThumbnail: useThumbnail,
        )
        allTiles.append(contentsOf: tiles)
        tileCounts.append(tiles.count)
    }
    // Stack to (N, 3, H, W) MLXArray, normalize via CLIP mean/std.
    // TODO: Use vImage to convert each CIImage tile to (3, H, W) Float32 and
    // build the final MLXArray.
    fatalError("TODO: vImage-based tile rasterization + CLIP normalization")
}

/// Pick the best (cols, rows) tile grid for the input aspect ratio.
private func dynamicPreprocess(
    _ image: CIImage,
    imageSize: Int,
    minNum: Int,
    maxNum: Int,
    useThumbnail: Bool
) -> [CIImage] {
    // Mirror Python `dynamic_preprocess` logic:
    // 1. Build candidate (cols, rows) pairs with cols*rows in [minNum, maxNum]
    // 2. Pick the closest aspect ratio match
    // 3. Resize image to (cols*image_size, rows*image_size) bicubic
    // 4. Crop into cols × rows tiles
    // 5. If useThumbnail and tiles > 1, append (image_size × image_size) thumbnail
    fatalError("TODO: implement tile selection")
}

// MARK: - Audio preprocessing (parakeet mel STFT)

/// Mel STFT for parakeet input — validated bit-exact against the Python port
/// (jang_tools/nemotron_omni/audio_features.py extract_mel_features).
///
/// Pipeline (matches transformers.ParakeetFeatureExtractor):
///   1. Preemphasis: y[t] = x[t] - 0.97 * x[t-1]
///   2. Hann window (periodic=False, length=400, padded to n_fft=512 centered)
///   3. STFT via vDSP_fft_zrip: complex frames (n_fft/2+1, n_frames)
///   4. |X|² power spectrum
///   5. Mel filterbank (slaney norm, 128 mels, fmin=0, fmax=8000) @ power
///   6. log(mel + 2⁻²⁴)
///   7. **Per-sample normalize** to zero-mean unit-variance (CRITICAL — without
///      this the model gets nonsense audio. Use Bessel-corrected variance:
///      var = sum((x - mean)²) / (n_frames - 1), then `(x - mean) / (std + 1e-5)`)
///   8. Output (1, n_frames, n_mels=128)
@available(macOS 14.0, *)
public func extractMelFeatures(
    _ waveform: [Float],
    sampleRate: Int = 16000,
    nFFT: Int = 512,
    hopLength: Int = 160,
    winLength: Int = 400,
    nMels: Int = 128,
    preemphasisCoef: Float = 0.97
) -> MLXArray {
    // TODO: implement steps 1-8 above. The Slaney mel filterbank can be
    // precomputed once at startup and cached as an MLXArray of shape
    // (n_mels=128, n_fft//2+1=257). Use `vDSP_fft_zrip` for the STFT and
    // `cblas_sgemv` for the mel projection.
    //
    // CRITICAL: do not skip step 7 (per-sample normalize). The Python port
    // wasted ~30 minutes debugging when this was missing — output became
    // "the sound of a door opening and closing" instead of speech text.
    fatalError("TODO: vDSP STFT + slaney mel filterbank + per-sample normalize")
}

/// Load a 16 kHz mono Float32 array from an audio file.
@available(macOS 14.0, *)
public func loadAudioFile(_ url: URL, targetSampleRate: Double = 16000) throws -> [Float] {
    let file = try AVAudioFile(forReading: url)
    let format = AVAudioFormat(
        commonFormat: .pcmFormatFloat32,
        sampleRate: targetSampleRate,
        channels: 1,
        interleaved: false,
    )!
    // Use AVAudioConverter to resample + downmix to mono if needed.
    fatalError("TODO: AVAudioConverter resample to 16 kHz mono Float32")
}

// MARK: - Video preprocessing (frame extraction + EVS)

/// Native Swift video preprocessor — mirror of the Python
/// `jang_tools/nemotron_omni/video_processor.py` (validated working).
///
/// Pipeline:
///   1. Extract frames via AVAssetImageGenerator (uniform sampling)
///   2. Pad N to multiple of T=2 (video_temporal_patch_dim) by repeating last frame
///   3. Bicubic resize each frame to 512×512 via vImage
///   4. CLIP normalize per channel
///   5. Stack T frames into channel dim → (N/T, T*3, H, W) MLXArray
///   6. (After RADIO with video=true): apply EVS retention mask via
///      cosine-similarity prune
@available(macOS 14.0, *)
public func preprocessVideo(
    url: URL,
    imageSize: Int = 512,
    targetFrames: Int = 32,
    videoTemporalPatchDim: Int = 2
) async throws -> (pixelValues: MLXArray, metadata: [String: Any]) {
    let asset = AVURLAsset(url: url)
    let duration = try await asset.load(.duration).seconds

    // Build uniform sample times across the video duration
    let sampleCount = max(1, targetFrames)
    let times: [CMTime] = (0..<sampleCount).map { i in
        let t = duration * Double(i) / Double(max(1, sampleCount - 1))
        return CMTime(seconds: t, preferredTimescale: 600)
    }

    let imageGen = AVAssetImageGenerator(asset: asset)
    imageGen.appliesPreferredTrackTransform = true
    imageGen.requestedTimeToleranceBefore = .zero
    imageGen.requestedTimeToleranceAfter = .zero

    // Extract frames as CIImage (or CGImage), then convert to (3, H, W) Float32
    var frames: [[Float]] = []
    for cmTime in times {
        // generateCGImagesAsynchronously(forTimes:) is the Swift-Concurrency
        // path. For simplicity here use the synchronous variant and post-rotate.
        var actualTime = CMTime.zero
        let cgImage = try imageGen.copyCGImage(at: cmTime, actualTime: &actualTime)
        // TODO: bicubic resize to (imageSize, imageSize) via vImage
        // TODO: extract RGB Float32 pixel values, normalize via CLIP mean/std
        // TODO: append to frames in (3*H*W) flat layout
        _ = cgImage
        fatalError("TODO: vImage-based resize + RGB Float32 extraction + CLIP normalize")
    }

    // Pad to multiple of T by repeating last frame
    var n = frames.count
    while n % videoTemporalPatchDim != 0 {
        frames.append(frames.last!)
        n += 1
    }

    // Stack T frames into channel dim: (N, 3, H, W) → (N/T, T*3, H, W)
    // TODO: Build MLXArray of shape (N/videoTemporalPatchDim, videoTemporalPatchDim*3, imageSize, imageSize)
    fatalError("TODO: build MLXArray with stacked T-frame channel layout")
}

/// EVS retention mask — drop fraction `q` of patch tokens with highest
/// cosine similarity to previous frame's same spatial position.
/// Mirrors Python `compute_evs_retention_mask`.
@available(macOS 14.0, *)
public func computeEVSRetentionMask(
    videoEmbeds: MLXArray,
    nTemporalGroups: Int,
    gridH: Int,
    gridW: Int,
    spatialMergeSize: Int = 2,
    q: Float = 0.7
) -> MLXArray {
    let Tg = nTemporalGroups
    let Hm = gridH / spatialMergeSize
    let Wm = gridW / spatialMergeSize
    let hidden = videoEmbeds.shape.last!

    // Reshape (T*Hm*Wm, hidden) → (T, Hm, Wm, hidden)
    let reshaped = videoEmbeds.reshaped([Tg, Hm, Wm, hidden])

    // Cosine similarity between frame t and frame t-1 per spatial patch
    let a = reshaped[1..., 0..., 0..., 0...]  // (T-1, Hm, Wm, hidden)
    let b = reshaped[0..<(Tg - 1), 0..., 0..., 0...]
    let aNorm = a / (MLX.sqrt(MLX.sum(a * a, axes: [-1], keepDims: true)) + 1e-8)
    let bNorm = b / (MLX.sqrt(MLX.sum(b * b, axes: [-1], keepDims: true)) + 1e-8)
    let sim = MLX.sum(aNorm * bNorm, axes: [-1])  // (T-1, Hm, Wm)
    let dis = 1.0 - sim

    // Always keep first frame's tokens (mark dissimilarity=255)
    let first = MLXArray.full([1, Hm, Wm], values: Float(255), dtype: dis.dtype)
    let dissimilarity = MLX.concatenated([first, dis], axis: 0).reshaped([-1])

    let minNum = Hm * Wm
    let evsNum = Int(Float(Tg * minNum) * (1.0 - q))
    let nKeep = max(minNum, evsNum)

    // Top-k highest dissimilarity → retention mask
    let order = MLX.argSort(-dissimilarity, axis: 0)
    let keepIdx = order[0..<nKeep]
    var mask = MLXArray.zeros(dissimilarity.shape, dtype: .bool)
    // TODO: scatter true into mask at keepIdx — MLX-Swift may not have
    // direct scatter; use a where-style construction instead:
    //   mask = MLXArray(arange) < nKeep_at_position
    fatalError("TODO: finalize scatter via where + argSort indices")
}
