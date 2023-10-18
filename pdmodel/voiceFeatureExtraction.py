import numpy as np
import subprocess
import soundfile as sf
import os
import pysptk


def convert_to_wav(input_file):
    output_file = f'{input_file[:-4]}.wav'

    # Run the ffmpeg command as a subprocess
    print(input_file, output_file)
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
    frameSize = 11025
    overlap = 0
    y, fs = sf.read(waveFile)

    frameMat = buffer(y, frameSize, overlap)

    frameNum = frameMat.shape[1]
    volume1 = np.zeros(frameNum)

    ave = 0
    aveF = 0
    aveB = 0
    aveNum = 0
    pauseNum = 0
    aveFNum = 0
    aveBNum = 0
    bond = 35

    for i in range(frameNum):
        frame = frameMat[:, i]
        frame = frame - np.mean(frame)  # zero-justified
        volume1[i] = np.sum(np.abs(frame))  # method 1

    for i in range(1, frameNum - 1):
        if volume1[i] > bond:
            if volume1[i] < 300:
                ave += volume1[i]
                aveNum += 1
        else:
            pauseNum += 1

    if ave / (aveNum) < 60 and ave / (aveNum) > 45:
        bond = 25
        ave = 0
        aveNum = 0
        pauseNum = 0
        for i in range(1, frameNum - 1):
            if volume1[i] > bond:
                ave += volume1[i]
                aveNum += 1
            else:
                pauseNum += 1

    if ave / (aveNum) < 45:
        bond = 20
        ave = 0
        aveNum = 0
        pauseNum = 0
        for i in range(1, frameNum - 1):
            if volume1[i] > bond:
                ave += volume1[i]
                aveNum += 1
            else:
                pauseNum += 1

    pause = pauseNum * 0.25
    pause_percentage = pauseNum * 25 * fs / len(y)
    volume = ave / (aveNum)

    for i in range(3, round(frameNum / 2)):
        if volume1[i] > bond:
            aveF += volume1[i]
            aveFNum += 1

    for i in range(round(frameNum / 2 + 1), frameNum - 1):
        if volume1[i] > bond:
            aveB += volume1[i]
            aveBNum += 1

    volumn_change = (((aveB / aveBNum) - (aveF / aveFNum)) / (ave / aveNum)) * 100

    return volume, pause, pause_percentage, volumn_change


def pitch(x, fs, method='NCF', winLength=400, overlapLength=200):
    frame_length = winLength
    hop_length = frame_length - overlapLength

    # Pre-emphasis
    preemph_coeff = 0.97
    x = np.append(x[0], x[1:] - preemph_coeff * x[:-1])

    # Compute F0 using SPTK library
    if method == 'NCF':
        f0 = pysptk.swipe(x, fs=fs, hopsize=hop_length, min=60, max=400, threshold=0.25, otype="f0")
    elif method == 'ACF':
        f0 = pysptk.rapt(x, fs=fs, hopsize=hop_length, min=60, max=400)
    else:
        raise ValueError('Invalid method')

    return np.array(f0)


def pitch_feature(waveFile):
    x, fs = sf.read(waveFile)
    x = np.where(np.abs(x) > 0.2, 0, x)

    # readtime = len(x)
    winLength = round(fs / 10)
    overlapLength = round(0 * fs)

    f0 = pitch(x, fs, method='NCF', winLength=winLength, overlapLength=overlapLength)

    frameMat = buffer(x, winLength, overlapLength)
    frameNum = frameMat.shape[1]
    volume1 = np.zeros(frameNum)
    ave = 0
    aveNum = 0
    bond = 25

    for i in range(frameNum):
        frame = frameMat[:, i]
        frame = frame - np.mean(frame)  # zero-justified
        volume1[i] = np.sum(np.abs(frame))  # method 1

    for i in range(1, frameNum - 1):
        if volume1[i] > bond:
            ave += volume1[i]
            aveNum += 1

    if (ave / aveNum < 50) and (ave / aveNum > 35):
        bond = 17
        ave = 0
        aveNum = 0

        for i in range(1, frameNum - 1):
            if (volume1[i] > bond) and (volume1[i] < 300):
                ave += volume1[i]
                aveNum += 1

    if ave/aveNum < 35:
        bond = 11
        ave = 0
        aveNum = 0

        for i in range(1, frameNum - 1):
            if volume1[i] > bond:
                ave += volume1[i]
                aveNum += 1

    f0[volume1 < bond] = np.nan
    p = 0
    pitchnum = 0
    change = 0
    changenum = 0

    for i in range(1, len(f0)):
        if (f0[i] >= 60) and (f0[i] <= 270):
            p += f0[i]
            pitchnum += 1
            if (f0[i - 1] >= 60) and (f0[i - 1] <= 270):
                change += np.abs(f0[i] - f0[i - 1])
                changenum += 1
        else:
            f0[i] = np.nan

    if p / pitchnum > 180:
        p = 0
        pitchnum = 0
        change = 0
        changenum = 0
        for i in range(1, len(f0)):
            if 150 <= f0[i] <= 250:
                p += f0[i]
                pitchnum += 1
                if 150 <= f0[i - 1] <= 250:
                    change += abs(f0[i] - f0[i - 1])
                    changenum += 1
            else:
                f0[i] = float('nan')
    elif 140 < p / pitchnum < 180:
        p = 0
        pitchnum = 0
        change = 0
        changenum = 0
        for i in range(1, len(f0)):
            if 100 <= f0[i] <= 230:
                p += f0[i]
                pitchnum += 1
                if 100 <= f0[i - 1] <= 230:
                    change += abs(f0[i] - f0[i - 1])
                    changenum += 1
            else:
                f0[i] = float('nan')
    else:
        p = 0
        pitchnum = 0
        change = 0
        changenum = 0
        for i in range(1, len(f0)):
            if 70 <= f0[i] <= 180:
                p += f0[i]
                pitchnum += 1
                if 70 <= f0[i - 1] <= 180:
                    change += abs(f0[i] - f0[i - 1])
                    changenum += 1
            else:
                f0[i] = float('nan')

    average_vol = ave / aveNum
    pitch_change = change / changenum
    average_pitch = p / pitchnum

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
