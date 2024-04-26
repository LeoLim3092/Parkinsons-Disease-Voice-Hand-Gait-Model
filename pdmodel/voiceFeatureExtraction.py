import numpy as np
import subprocess
import soundfile as sf
import os
import json
import numpy as np
import pysptk


def convert_to_wav(input_file):
    output_file = f'{input_file[:-4]}.wav'

    # Run the ffmpeg command as a subprocess
    print(input_file, output_file)
    subprocess.run(["ffmpeg", "-i", input_file, output_file])

    return output_file

"""
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
"""

def buffer(signal, frame_length, overlap=0, initial_condition=None):
    """
    Generate a buffer array similar to MATLAB's buffer function, creating overlapping frames from the signal.
    
    Parameters:
    - signal: ndarray, the input signal array.
    - frame_length: int, number of data points in each frame.
    - overlap: int, number of data points to overlap between frames.
    - initial_condition: str, handling of the initial condition ('nodelay' or None).
      'nodelay' starts filling the buffer immediately without initial zeros.
    
    Returns:
    - buffer_array: ndarray, the buffer array created from the input signal.
    """
    
    if initial_condition not in [None, 'nodelay']:
        raise ValueError(f"Initial condition '{initial_condition}' is not implemented.")
    
    signal_length = len(signal)
    step = frame_length - overlap
    first_iter = True
    buffer_list = []

    for start in range(0, signal_length, step):
        end = start + frame_length
        if first_iter:
            if initial_condition == 'nodelay':
                frame = signal[start:end]
            else:
                frame = np.hstack([np.zeros(overlap), signal[start:end-overlap]])
            first_iter = False
        else:
            frame = signal[start-overlap:end] if end <= signal_length else signal[start-overlap:]
            if len(frame) < frame_length:
                frame = np.hstack([frame, np.zeros(frame_length - len(frame))])
        
        buffer_list.append(frame)
    
    buffer_array = np.column_stack(buffer_list)
    return buffer_array

"""
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
"""
    
"""
def new_audio_feature(audio_file_path):
"""
"""
    Calculate volume, pause duration, pause percentage, and volume change of an audio file.
    
    Parameters:
    - audio_file_path: Path to the audio file.
    
    Returns:
    - volume: Average volume of the audio.
    - total_pause_duration: Total duration of pauses in seconds.
    - pause_percentage: Percentage of the audio duration that is silent.
    - volume_change_percentage: Percentage change in volume between the first and second half of the audio.
"""
"""
    frame_size = 11025
    overlap = 0  # Set overlap to 0 or adjust as needed
    audio_data, sample_rate = sf.read(audio_file_path)
    
    # Use the refactored buffer function for creating frames
    frames = buffer(audio_data, frame_size, overlap)
    num_frames = frames.shape[1]
    volume_per_frame = np.sum(np.abs(frames - np.mean(frames, axis=0)), axis=0)

    for frame in frames:
        frame = frame - np.mean(frame)  # Zero-justification

    def volume_band_filter(volume_per_frame, low, high=None):
        volume_per_frame = volume_per_frame[1:-1]
        filtered = volume_per_frame[(volume_per_frame > low)]
        if low and high:
            average_volume = np.mean(filtered[(filtered < high)])
        else:
            average_volume = np.mean(filtered)

        return average_volume, filtered

    [lower_threshold, upper_threshold] = [35, 300]
    threshold_adjustments = [(60, 45, 25, None), (45, -np.inf, 20, None)]
    for max_vol, min_vol, new_lower_threshold, new_upper_threshold in threshold_adjustments:
        average_volume, _ = volume_band_filter(volume_per_frame, lower_threshold, upper_threshold)
        if min_vol < average_volume < max_vol:
            lower_threshold = new_lower_threshold
            upper_threshold = new_upper_threshold

    # print(lower_threshold, upper_threshold)
    average_volume, valid_volumes = volume_band_filter(volume_per_frame, lower_threshold, upper_threshold)
    pause_count = num_frames - len(valid_volumes) - 2 
    # print(pause_count)

    total_pause_duration = pause_count * 0.25 #(frame_size / sample_rate)
    pause_percentage = (pause_count / len(audio_data) * 25 * sample_rate)  # / num_frames) * 100

    # Calculate volume change between the first and second half
    # half = round(num_frames / 2)
    first_half = volume_per_frame[3:round(num_frames / 2)]
    second_half = volume_per_frame[round(num_frames / 2+1):-1]
    average_volume_first_half = np.mean(first_half[(first_half > lower_threshold)])
    average_volume_second_half = np.mean(second_half[(second_half > lower_threshold)])
    volume_change_percentage = ((average_volume_second_half - average_volume_first_half) / average_volume) * 100

    return average_volume, total_pause_duration, pause_percentage, volume_change_percentage
"""
    
def new_audio_feature_without_filter(audio_file_path):
    audio_data, sample_rate = sf.read(audio_file_path)
    frame_size, overlap = round(sample_rate / 10), 0

    frames = buffer(audio_data, frame_size, overlap)
    num_frames = frames.shape[1]
    volume_per_frame = np.sum(np.abs(frames - np.mean(frames, axis=0)), axis=0)
    # print(volume_per_frame)

    average_volume = np.mean(volume_per_frame)
    half = round(num_frames / 2)
    first_half = np.mean(volume_per_frame[:half])
    second_half = np.mean(volume_per_frame[half:])
    volume_change_percentage = (first_half - second_half) / average_volume * 100

    pause_threshold = np.min(volume_per_frame) * 2 #volume threshold for determine pause
    pause_frames = volume_per_frame[(volume_per_frame < pause_threshold)]
    pause_percentage = (len(pause_frames)) / num_frames

    return average_volume, volume_change_percentage, pause_percentage

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


"""
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

    # print('orginal: ', f0)
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
        # print('old: ', p / pitchnum)
        # print(f0)
        # print('150, 250')
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
        # print('old: ', p / pitchnum)
        # print(f0)
        # print('100, 230')
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
        # print('old: ', p / pitchnum)
        # print(f0)
        # print('70, 180')
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
    # print('old:', change, ' ',changenum)
    average_vol = ave / aveNum
    pitch_change = change / changenum
    average_pitch = p / pitchnum

    return average_vol, pitch_change, average_pitch
"""

"""
def new_pitch_features(audio_file):
    audio_data, sample_rate = sf.read(audio_file)
    audio_data = np.where(np.abs(audio_data) > 0.2, 0, audio_data)
    window_length, overlap_length = round(sample_rate / 10), 0

    fundamental_frequency = pitch(audio_data, sample_rate, method='NCF', 
                                  winLength=window_length, overlapLength=overlap_length)
    frame_matrix = buffer(audio_data, window_length, overlap_length)
    num_frames = frame_matrix.shape[1]

    # Simplifying volume calculation
    frame_volumes = np.sum(np.abs(frame_matrix - np.mean(frame_matrix, axis=0)), axis=0)

    def volume_band_filter(frame_volumes, low, high=None):
        frame_volumes = frame_volumes[1:-1]
        filtered = frame_volumes[(frame_volumes > low)]
        if low and high:
            average_volume = np.mean(frame_volumes[(frame_volumes > low) & (frame_volumes < high)])
        else:
            average_volume = np.mean(filtered)

        return average_volume

    [lower_threshold, upper_threshold] = [25, None]
    threshold_adjustments = [(50, 35, 17, 300), (35, -np.inf, 11, None)]
    for max_vol, min_vol, new_lower_threshold, new_upper_threshold in threshold_adjustments:
        average_volume = volume_band_filter(frame_volumes, lower_threshold, upper_threshold)
        if min_vol < average_volume < max_vol:
            lower_threshold = new_lower_threshold
            upper_threshold = new_upper_threshold

    average_volume = volume_band_filter(frame_volumes, lower_threshold, upper_threshold)

    fundamental_frequency[frame_volumes < lower_threshold] = np.nan
    def pitch_band_filter(fundamental_frequency, low, high):
        fundamental_frequency[(fundamental_frequency <= low) | (fundamental_frequency >= high)] = np.nan
        average_pitch = np.mean(fundamental_frequency[~np.isnan(fundamental_frequency)])
        fundamental_difference = np.abs(np.diff(fundamental_frequency))
        # print('new:', np.sum(np.abs(fundamental_difference[(~np.isnan(fundamental_difference))])), ' ',len(fundamental_difference[(~np.isnan(fundamental_difference))]))
        pitch_differece = np.mean(np.abs(fundamental_difference[(~np.isnan(fundamental_difference))]))
        return average_pitch, pitch_differece

    [lower_threshold, upper_threshold] = [60, 270]
    average_pitch, pitch_differece = pitch_band_filter(fundamental_frequency, lower_threshold, upper_threshold)
    threshold_adjustments = [(180, np.inf, 150, 250), (140, 180, 100, 230), (-np.inf, 140, 70, 180)]
    for min_pitch, max_pitch, new_lower_threshold, new_upper_threshold in threshold_adjustments:
        if min_pitch < average_pitch < max_pitch:
            lower_threshold = new_lower_threshold
            upper_threshold = new_upper_threshold

    average_pitch, pitch_differece = pitch_band_filter(fundamental_frequency, lower_threshold, upper_threshold)

    return average_volume, pitch_differece, average_pitch
"""

def new_pitch_feature_without_filter(audio_file):
    audio_data, sample_rate = sf.read(audio_file)
    window_length, overlap_length = round(sample_rate / 10), 0

    fundamental_frequency = pitch(audio_data, sample_rate, method='NCF', 
                                  winLength=window_length, overlapLength=overlap_length)
 
    average_pitch = np.mean(fundamental_frequency)
    pitch_changes = np.mean(np.abs(np.diff(fundamental_frequency)))

    return average_pitch, pitch_changes


def voice_features_extraction(voice_file):
    wave_file = f"{voice_file[:-4]}.wav"

    if os.path.isfile(wave_file):
        pass
    else:
        wave_file = convert_to_wav(voice_file)

    # volume, pause, pause_percentage, volumn_change = audio_feature(wave_file)
    # average_vol, pitch_change, average_pitch = pitch_feature(wave_file)
    volume, volumn_change, pause_percentage, = new_audio_feature_without_filter(wave_file)
    average_pitch, pitch_change, = new_pitch_feature_without_filter(wave_file)

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
        # audio_feature(wave_file)
        # average_vol, _, _ = pitch_feature(wave_file)
        average_vol, volumn_change, pause_percentage, = new_audio_feature_without_filter(wave_file)
        average_pitch, pitch_change, = new_pitch_feature_without_filter(wave_file)

    except:
        failed_process = True
        average_vol = calculate_average_volume(wave_file)

    return duration, average_vol, failed_process

if __name__ == '__main__':

    data_dir = '../PD/AudioKoreaDrKim/'

    """
    with open('../metadata.json', 'r') as f:
        patient_dict = json.load(f)

    def data_to_path(data):
        paths = []
        prefix = '{}/{}/{}-{}-{}-{}-{}-'.format(data_dir, pat, pat.split(' ')[0], data['ord'], data['age'], data['gender'], data['date'])
        for rec in data['recordings']:
            paths.append(prefix+rec.split(':')[0])

        return paths

    for pat, val in patient_dict.items():
        paths = data_to_path(val)
        print(pat)
        for path in paths:
            # volume, pause, pause_percentage, volumn_change = audio_feature(path)
            # average_vol, pitch_change, average_pitch = pitch_feature(path)

            # new_volume, new_pause, new_pause_percentage, new_volumn_change = new_audio_feature(path)
            # new_average_vol, new_pc, new_ap = new_pitch_features(path)
            # if ~np.isclose(volume, new_volume):
            #     print('Exception: volume on ', path, volume, ' ', new_volume)
            # if ~np.isclose(pause, new_pause):
            #     print('Exception: pause on ', path, pause, ' ', new_pause)     
            # if ~np.isclose(pause_percentage, new_pause_percentage):
            #     print('Exception: pause_percentage on ', path, pause_percentage, ' ', new_pause_percentage)     
            # if ~np.isclose(volumn_change, new_volumn_change):
            #     print('Exception: volumn_change on ', path, volumn_change, ' ', new_volumn_change)     
            # if ~np.isclose(average_vol, new_average_vol):
            #     print('Exception: average_vol on ', path, average_vol, ' ', new_average_vol)     
            # if ~np.isclose(pitch_change, new_pc):
            #     print('Exception: pitch_change on ', path, pitch_change, ' ', new_pc)     
            # if ~np.isclose(average_pitch, new_ap):
            #     print('Exception: average_pitch on ', path, average_pitch, ' ', new_ap) 

            volume, volumn_change, pause_percentage, = new_audio_feature_without_filter(path)
            pitch_change, average_pitch = new_pitch_feature_without_filter(path)
            print('-'*20)                      
    """
            

    voice_file = '../PD/AudioKoreaDrKim/1-001/1-001-LBU-70-M-20220704-PD_014.wav'
    # [pause_percentage, volumn_change, pitch_change, average_pitch] = voice_features_extraction(voice_path)

    wave_file = f"{voice_file[:-4]}.wav"

    if os.path.isfile(wave_file):
        pass
    else:
        wave_file = convert_to_wav(voice_file)


    # volume, pause, pause_percentage, volumn_change = audio_feature(wave_file)
    # average_vol, pitch_change, average_pitch = pitch_feature(wave_file)

    # new_volume, new_pause, new_pause_percentage, new_volumn_change = new_audio_feature(wave_file)
    # new_average_vol, new_pc, new_ap = new_pitch_features(wave_file)

    volume, volumn_change, pause_percentage, = new_audio_feature_without_filter(wave_file)
    average_pitch, pitch_change, = new_pitch_feature_without_filter(wave_file)
    print('volume:', volume)
    # print('pause:', pause)
    print('pause percentage: ', pause_percentage)
    print('volume change: ', volumn_change)
    print('-'*20)
    # print('average volume: ',  average_vol)
    print('pitch change: ', pitch_change)
    print('average pitch: ', average_pitch)

    