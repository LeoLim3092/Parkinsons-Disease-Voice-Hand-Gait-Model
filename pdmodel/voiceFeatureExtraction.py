import numpy as np
import subprocess
import soundfile as sf
import os
import pysptk


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


def audio_feature(waveFile):

    y, fs = sf.read(waveFile)
    frameSize = int(0.025 * fs) 
    overlap = frameSize // 2 

    # Normalize audio and convert to mono if needed
    y = y / np.max(np.abs(y)) if np.max(np.abs(y)) > 0 else y
    if y.ndim > 1:
        y = y.mean(axis=1)

    frameMat = buffer(y, frameSize, overlap)
    frameNum = frameMat.shape[1]
    volume1 = np.zeros(frameNum)

    # Compute frame-wise volume
    for i in range(frameNum):
        frame = frameMat[:, i] - np.mean(frameMat[:, i])  # zero-justified
        volume1[i] = np.sum(np.abs(frame))

    # --- threshold ---
    bond = 10

    # --- Pause / Speech frame classification ---
    ave, aveNum, pauseNum = 0, 0, 0
    for v in volume1:
        if v > bond:
            ave += v
            aveNum += 1
        else:
            pauseNum += 1

    volume = ave / aveNum if aveNum > 0 else 0
    
    frame_hop = frameSize - overlap
    frame_hop_duration = frame_hop / fs
    pause = pauseNum * frame_hop_duration
    
    total_duration = len(y) / fs
    pause_percentage = (pause / total_duration) * 100


    # --- Volume change between first and second half ---
    mid = frameNum // 2
    aveF = np.mean(volume1[3:mid][volume1[3:mid] > bond]) if np.any(volume1[3:mid] > bond) else 0
    aveB = np.mean(volume1[mid+1:][volume1[mid+1:] > bond]) if np.any(volume1[mid+1:] > bond) else 0
    volumn_change = ((aveB - aveF) / volume * 100) if volume > 0 else 0

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
    

def pitch_feature(waveFile):
    x, fs = sf.read(waveFile)
    if x.ndim > 1:
        x = x.mean(axis=1)  # Convert stereo to mono

    # Optional: clip noise (you had this)
    x = np.where(np.abs(x) > 0.2, 0, x)

    # Frame parameters: 25ms frame, 15ms hop
    winLength = int(0.025 * fs)
    overlapLength = int(0.015 * fs)

    f0 = pitch(x, fs, method='NCF', winLength=winLength, overlapLength=overlapLength)

    # Compute volume envelope over same framing
    frameMat = buffer(x, winLength, overlapLength)
    frameNum = frameMat.shape[1]
    volume1 = np.zeros(frameNum)

    for i in range(frameNum):
        frame = frameMat[:, i] - np.mean(frameMat[:, i])
        volume1[i] = np.sum(np.abs(frame))

    # threshold for pause/silence
    bond = 10

    # Mask F0 values where volume is low
    valid_idx = volume1 > bond
    f0[~valid_idx] = np.nan

    # Now compute pitch stats
    f0_clean = f0[~np.isnan(f0)]
    f0_in_range = f0_clean[(f0_clean >= 70) & (f0_clean <= 270)]

    if len(f0_in_range) == 0:
        average_pitch = 0
        pitch_change = 0
    else:
        diffs = np.abs(np.diff(f0_in_range))
        average_pitch = np.mean(f0_in_range)
        pitch_change = np.mean(diffs) if len(diffs) > 0 else 0

    average_vol = np.mean(volume1[valid_idx]) if np.any(valid_idx) else 0

    return average_vol, pitch_change, average_pitch


def voice_features_extraction(voice_file):
    wave_file = f"{voice_file[:-4]}.wav"

    if os.path.isfile(wave_file):
        pass
    else:
        wave_file = convert_to_wav(voice_file)

    volume, pause, pause_percentage, volumn_change = audio_feature(wave_file)
    average_vol, pitch_change, average_pitch = pitch_feature(wave_file)

    return [pause_percentage, volumn_change, pitch_change, average_pitch]


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
