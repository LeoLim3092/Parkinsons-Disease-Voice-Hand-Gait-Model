import numpy as np
import subprocess
import soundfile as sf
import os
import pysptk
from speechScoring import score_pronunciation


def convert_to_wav(input_file):
    output_file = f'{input_file[:-4]}.wav'

    # Run the ffmpeg command as a subprocess
    subprocess.run(["ffmpeg", "-i", input_file, output_file])

    return output_file


def buffer(X, n, p=0, opt=None):
    '''Mimic MATLAB routine to generate buffer array

    MATLAB docs here: https://se.mathworks.com/help/signal/ref/buffer.html

    Parameters
    ----------
    x: ndarray
        Signal array
    n: int
        Number of data segments
    p: int
        Number of values to overlap
    opt: str
        Initial condition options. default sets the first `p` values to zero,
        while 'nodelay' begins filling the buffer immediately.

    Returns
    -------
    result : (n,n) ndarray
        Buffer array created from X
    '''
    import numpy as np

    if opt not in [None, 'nodelay']:
        raise ValueError('{} not implemented'.format(opt))

    i = 0
    first_iter = True
    while i < len(X):
        if first_iter:
            if opt == 'nodelay':
                # No zeros at array start
                result = X[:n]
                i = n
            else:
                # Start with `p` zeros
                result = np.hstack([np.zeros(p), X[:n - p]])
                i = n - p
            # Make 2D array and pivot
            result = np.expand_dims(result, axis=0).T
            first_iter = False
            continue

        # Create next column, add `p` results from last col if given
        col = X[i:i + (n - p)]
        if p != 0:
            col = np.hstack([result[:, -1][-p:], col])
        i += n - p

        # Append zeros if last row and not length `n`
        if len(col) < n:
            col = np.hstack([col, np.zeros(n - len(col))])

        # Combine result with next row
        result = np.hstack([result, np.expand_dims(col, axis=0).T])

    return result


def audio_feature(waveFile, debug=False):
    if debug:
        print("=" * 60)
        print("audio_feature() called")
        print(f"waveFile: {waveFile}")

    # --- Read audio file ---
    y, fs = sf.read(waveFile)
    if debug:
        print("\n[1] Audio loaded")
        print(f"    sample rate (fs): {fs}")
        print(f"    raw shape: {y.shape}")
        print(f"    dtype: {y.dtype}")

    # --- Frame settings ---
    frameSize = int(0.025 * fs)   # 25 ms
    overlap = frameSize // 2      # 50% overlap
    if debug:
        print("\n[2] Frame configuration")
        print(f"    frameSize: {frameSize}")
        print(f"    overlap: {overlap}")

    # --- Normalize audio ---
    max_abs = np.max(np.abs(y)) if len(y) > 0 else 0
    if max_abs > 0:
        y = y / max_abs
    if debug:
        print("\n[3] Audio normalization")
        print(f"    max abs before normalization: {max_abs}")
        print(f"    normalized: {max_abs > 0}")

    # --- Convert stereo to mono if needed ---
    if y.ndim > 1:
        if debug:
            print("\n[4] Convert to mono")
            print(f"    original channels: {y.shape[1]}")
        y = y.mean(axis=1)
        if debug:
            print(f"    new shape after mono conversion: {y.shape}")
    else:
        if debug:
            print("\n[4] Convert to mono")
            print("    already mono")

    # --- Create frame matrix ---
    frameMat = buffer(y, frameSize, overlap)
    frameNum = frameMat.shape[1]
    volume1 = np.zeros(frameNum)

    if debug:
        print("\n[5] Frame matrix")
        print(f"    frameMat shape: {frameMat.shape}")
        print(f"    frameNum: {frameNum}")

    # --- Compute frame-wise volume ---
    if debug:
        print("\n[6] Compute frame-wise volume")

    for i in range(frameNum):
        frame = frameMat[:, i] - np.mean(frameMat[:, i])  # zero-mean
        volume1[i] = np.sum(np.abs(frame))

        if debug and (i < 5 or i == frameNum - 1):
            print(f"    frame {i}: mean={np.mean(frameMat[:, i]):.6f}, volume={volume1[i]:.6f}")

    if debug and frameNum > 5:
        print("    ...")
        print(f"    last frame {frameNum - 1}: volume={volume1[-1]:.6f}")

    # --- Threshold ---
    bond = 10
    if debug:
        print("\n[7] Threshold")
        print(f"    bond: {bond}")

    # --- Pause / Speech frame classification ---
    ave = 0
    aveNum = 0
    pauseNum = 0

    if debug:
        print("\n[8] Pause / speech classification")

    for idx, v in enumerate(volume1):
        if v > bond:
            ave += v
            aveNum += 1
            if debug and idx < 5:
                print(f"    frame {idx}: speech (volume={v:.6f})")
        else:
            pauseNum += 1
            if debug and idx < 5:
                print(f"    frame {idx}: pause  (volume={v:.6f})")

    volume = ave / aveNum if aveNum > 0 else 0

    if debug:
        print(f"    speech frames: {aveNum}")
        print(f"    pause frames: {pauseNum}")
        print(f"    accumulated speech volume: {ave:.6f}")
        print(f"    average speech volume: {volume:.6f}")

    # --- Pause duration ---
    frame_hop = frameSize - overlap
    frame_hop_duration = frame_hop / fs
    pause = pauseNum * frame_hop_duration

    total_duration = len(y) / fs
    pause_percentage = (pause / total_duration) * 100 if total_duration > 0 else 0

    if debug:
        print("\n[9] Pause statistics")
        print(f"    frame_hop: {frame_hop}")
        print(f"    frame_hop_duration: {frame_hop_duration:.6f} sec")
        print(f"    pause duration: {pause:.6f} sec")
        print(f"    total duration: {total_duration:.6f} sec")
        print(f"    pause percentage: {pause_percentage:.2f}%")

    # --- Volume change between first and second half ---
    mid = frameNum // 2

    first_half = volume1[3:mid]
    second_half = volume1[mid+1:]

    first_half_valid = first_half[first_half > bond]
    second_half_valid = second_half[second_half > bond]

    aveF = np.mean(first_half_valid) if len(first_half_valid) > 0 else 0
    aveB = np.mean(second_half_valid) if len(second_half_valid) > 0 else 0
    volumn_change = ((aveB - aveF) / volume * 100) if volume > 0 else 0

    if debug:
        print("\n[10] Volume change analysis")
        print(f"    mid frame index: {mid}")
        print(f"    first half valid frames: {len(first_half_valid)}")
        print(f"    second half valid frames: {len(second_half_valid)}")
        print(f"    aveF: {aveF:.6f}")
        print(f"    aveB: {aveB:.6f}")
        print(f"    volumn_change: {volumn_change:.2f}%")

    if debug:
        print("\n[11] Final output")
        print(f"    volume = {volume:.6f}")
        print(f"    pause = {pause:.6f}")
        print(f"    pause_percentage = {pause_percentage:.2f}")
        print(f"    volumn_change = {volumn_change:.2f}")
        print("=" * 60)

    return volume, pause, pause_percentage, volumn_change



def pitch(x, fs, method='NCF', winLength=400, overlapLength=200):
    hop_length = winLength - overlapLength

    # Pre-emphasis filter
    preemph_coeff = 0.97
    x = np.append(x[0], x[1:] - preemph_coeff * x[:-1])

    if method == 'NCF':
        f0 = pysptk.swipe(x, fs=fs, hopsize=hop_length, min=60, max=400, threshold=0.25, otype="f0")
    elif method == 'ACF':
        f0 = pysptk.rapt(x, fs=fs, hopsize=hop_length, min=60, max=400)
    else:
        raise ValueError('Invalid method')

    return f0
    

import numpy as np
import soundfile as sf

def pitch_feature(waveFile, debug=False):
    if debug:
        print("=" * 60)
        print("pitch_feature() called")
        print(f"waveFile: {waveFile}")

    # --- Read audio file ---
    x, fs = sf.read(waveFile)
    if debug:
        print("\n[1] Audio loaded")
        print(f"    sample rate (fs): {fs}")
        print(f"    raw shape: {x.shape}")
        print(f"    dtype: {x.dtype}")

    # --- Convert stereo to mono if needed ---
    if x.ndim > 1:
        if debug:
            print("\n[2] Convert to mono")
            print(f"    original shape: {x.shape}")
        x = x.mean(axis=1)
        if debug:
            print(f"    new shape after mono conversion: {x.shape}")
    else:
        if debug:
            print("\n[2] Convert to mono")
            print("    already mono")

    # --- Optional noise clipping ---
    if debug:
        print("\n[3] Noise clipping")
        print("    rule: abs(x) > 0.2 -> 0")
        before_nonzero = np.count_nonzero(x)

    x = np.where(np.abs(x) > 0.2, 0, x)

    if debug:
        after_nonzero = np.count_nonzero(x)
        print(f"    nonzero samples before: {before_nonzero}")
        print(f"    nonzero samples after : {after_nonzero}")

    # --- Frame parameters ---
    winLength = int(0.025 * fs)      # 25 ms
    overlapLength = int(0.015 * fs)  # as in your original code
    if debug:
        print("\n[4] Frame configuration")
        print(f"    winLength: {winLength}")
        print(f"    overlapLength: {overlapLength}")

    # --- Pitch extraction ---
    f0 = pitch(x, fs, method='NCF', winLength=winLength, overlapLength=overlapLength)

    if debug:
        print("\n[5] Pitch extraction")
        print(f"    f0 shape: {f0.shape}")
        preview_n = min(10, len(f0))
        print(f"    first {preview_n} f0 values: {f0[:preview_n]}")

    # --- Compute volume envelope over same framing ---
    frameMat = buffer(x, winLength, overlapLength)
    frameNum = frameMat.shape[1]
    volume1 = np.zeros(frameNum)

    if debug:
        print("\n[6] Volume envelope framing")
        print(f"    frameMat shape: {frameMat.shape}")
        print(f"    frameNum: {frameNum}")

    for i in range(frameNum):
        frame = frameMat[:, i] - np.mean(frameMat[:, i])
        volume1[i] = np.sum(np.abs(frame))

        if debug and (i < 5 or i == frameNum - 1):
            print(f"    frame {i}: mean={np.mean(frameMat[:, i]):.6f}, volume={volume1[i]:.6f}")

    if debug and frameNum > 5:
        print("    ...")
        print(f"    last frame {frameNum - 1}: volume={volume1[-1]:.6f}")

    # --- Threshold for pause/silence ---
    bond = 10
    valid_idx = volume1 > bond

    if debug:
        print("\n[7] Silence / speech mask")
        print(f"    bond: {bond}")
        print(f"    valid speech-like frames: {np.sum(valid_idx)} / {len(valid_idx)}")
        print(f"    silent/low-volume frames: {len(valid_idx) - np.sum(valid_idx)}")

    # --- Align lengths if needed ---
    min_len = min(len(f0), len(valid_idx))
    if len(f0) != len(valid_idx) and debug:
        print("\n[8] Length alignment")
        print(f"    len(f0): {len(f0)}")
        print(f"    len(valid_idx): {len(valid_idx)}")
        print(f"    using min_len: {min_len}")

    f0 = f0[:min_len]
    valid_idx = valid_idx[:min_len]

    # --- Mask F0 values where volume is low ---
    f0[~valid_idx] = np.nan

    if debug:
        print("\n[9] Apply volume mask to f0")
        nan_count = np.sum(np.isnan(f0))
        print(f"    NaN count after masking: {nan_count}")
        preview_n = min(10, len(f0))
        print(f"    first {preview_n} masked f0 values: {f0[:preview_n]}")

    # --- Clean F0 values ---
    f0_clean = f0[~np.isnan(f0)]
    f0_in_range = f0_clean[(f0_clean >= 70) & (f0_clean <= 270)]

    if debug:
        print("\n[10] Clean and filter pitch range")
        print(f"    non-NaN f0 count: {len(f0_clean)}")
        print(f"    in-range f0 count (70~270 Hz): {len(f0_in_range)}")
        if len(f0_in_range) > 0:
            preview_n = min(10, len(f0_in_range))
            print(f"    first {preview_n} in-range f0 values: {f0_in_range[:preview_n]}")

    # --- Pitch statistics ---
    if len(f0_in_range) == 0:
        average_pitch = 0
        pitch_change = 0
        if debug:
            print("\n[11] Pitch statistics")
            print("    no valid pitch values in range")
    else:
        diffs = np.abs(np.diff(f0_in_range))
        average_pitch = np.mean(f0_in_range)
        pitch_change = np.mean(diffs) if len(diffs) > 0 else 0

        if debug:
            print("\n[11] Pitch statistics")
            print(f"    average_pitch: {average_pitch:.6f}")
            print(f"    diff count: {len(diffs)}")
            if len(diffs) > 0:
                preview_n = min(10, len(diffs))
                print(f"    first {preview_n} pitch diffs: {diffs[:preview_n]}")
            print(f"    pitch_change: {pitch_change:.6f}")

    # --- Average volume over valid frames ---
    average_vol = np.mean(volume1[valid_idx]) if np.any(valid_idx) else 0

    if debug:
        print("\n[12] Average volume")
        print(f"    average_vol: {average_vol:.6f}")

        print("\n[13] Final output")
        print(f"    average_vol   = {average_vol:.6f}")
        print(f"    pitch_change  = {pitch_change:.6f}")
        print(f"    average_pitch = {average_pitch:.6f}")
        print("=" * 60)

    return average_vol, pitch_change, average_pitch


def voice_features_extraction(voice_file, debug=False):
    wave_file = f"{voice_file[:-4]}.wav"

    if os.path.isfile(wave_file):
        pass
    else:
        wave_file = convert_to_wav(voice_file)

    volume, pause, pause_percentage, volumn_change = audio_feature(wave_file, debug=debug)
    average_vol, pitch_change, average_pitch = pitch_feature(wave_file, debug=debug)

    score = score_pronunciation(wave_file)
    read_duration = calculate_duration(wave_file)

    return [read_duration, score, pause_percentage, volumn_change, pitch_change, average_pitch]


def calculate_average_volume(file_path):
    audio_data, _ = sf.read(file_path)
    squared_samples = np.square(audio_data)  # Square the audio samples
    mean_squared = np.mean(squared_samples)  # Calculate the mean of squared samples
    root_mean_square = np.sqrt(mean_squared)  # Take the square root to get RMS
    return root_mean_square


def calculate_duration(file_path):
    audio_data, fs = sf.read(file_path)
    total_samples = len(audio_data)  # Number of samples in all channels
    duration = total_samples / fs

    return duration


def sound_checking(voice_file):
    wave_file = f"{voice_file[:-4]}.wav"
    failed_process = False

    if os.path.isfile(wave_file):
        pass
    else:
        wave_file = convert_to_wav(voice_file)

    duration = calculate_duration(wave_file)

    try:
        audio_feature(wave_file)
        average_vol, _, _ = pitch_feature(wave_file)

    except:
        failed_process = True
        average_vol = calculate_average_volume(wave_file)

    return duration, average_vol, failed_process
