from scipy.signal import find_peaks, correlate, stft
import numpy as np
import matplotlib.pyplot as plt
import math


def autocorr(x):
    result = correlate(x, x)
    return result[int(result.size / 2):]


def find_period(arr, debug=False):

    if debug:
        print("=" * 60)
        print("[find_period] Start")
        print(f"    input type : {type(arr)}")
        try:
            print(f"    input len  : {len(arr)}")
        except:
            print("    input len  : not available")

    # =========================================================
    # 1. Remove mean
    # =========================================================
    mean = np.mean(arr)

    if debug:
        print("\n[STEP 1] Remove mean")
        print(f"    mean value: {mean}")

    centered = arr - mean

    # =========================================================
    # 2. Compute autocorrelation
    # =========================================================
    if debug:
        print("\n[STEP 2] Compute autocorrelation")

    acf_result = autocorr(centered)

    if debug:
        try:
            print(f"    acf_result len : {len(acf_result)}")
            preview_n = min(10, len(acf_result))
            print(f"    first {preview_n} acf values: {acf_result[:preview_n]}")
        except:
            pass

    # =========================================================
    # 3. Detect peaks in ACF
    # =========================================================
    if debug:
        print("\n[STEP 3] Find peaks in autocorrelation")

    peaks, properties = find_peaks(acf_result)

    if debug:
        print(f"    peaks found: {len(peaks)}")
        print(f"    peak indices: {peaks}")

    # =========================================================
    # 4. Determine period
    # =========================================================
    if debug:
        print("\n[STEP 4] Determine period")

    if len(peaks) == 0:
        period = 1
        if debug:
            print("    WARNING: no peaks found -> default period = 1")

    elif len(peaks) == 1:
        period = peaks[0]
        if debug:
            print("    only one peak found")
            print(f"    period = {period}")

    else:
        if peaks[0] > 50:
            period = peaks[0]
            if debug:
                print("    using peaks[0] (>50)")
        else:
            period = peaks[1]
            if debug:
                print("    peaks[0] too small -> using peaks[1]")

        if debug:
            print(f"    selected period: {period}")

    if debug:
        print("\n[find_period] Finished")
        print("=" * 60)

    return period, acf_result


def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)


def turningpoints(lst):
    dx = np.diff(lst)
    return np.sum(dx[1:] * dx[:-1] < 0)


def my_find_peaks_2(arr, height=1, range_=0.7):
    T, _ = find_period(arr)
    p, _ = find_peaks(arr, distance=int(T * range_), height=height)
    return p


def my_find_peaks(arr):
    T, _ = find_period(arr)
    amax = np.argmax(arr)
    locs = [amax]

    while amax + (T * (1.3)) <= len(arr):
        new_range = (int(amax + (T * (0.7))), int(amax + (T * (1.3))))
        amax = new_range[0] + np.argmax(arr[new_range[0]:new_range[1]])
        locs.append(amax)

    while amax - (T * (1.3)) >= 0:
        new_range = (int(amax - T * (0.7)), int(amax - T * (1.3)))
        amax = new_range[1] + np.argmax(arr[new_range[1]:new_range[0]])
        locs.append(amax)

    locs = np.array(locs)
    locs = np.unique(locs)

    asort = np.argsort(locs)
    locs = locs[asort]

    return locs


def calculate_angle(x1, y1, z1,
                   x2, y2, z2,
                   x3, y3, z3):
    # Find direction ratio of line AB
    ABx = x1 - x2;
    ABy = y1 - y2;
    ABz = z1 - z2;

    # Find direction ratio of line BC
    BCx = x3 - x2;
    BCy = y3 - y2;
    BCz = z3 - z2;

    # Find the dotProduct
    # of lines AB & BC
    dotProduct = (ABx * BCx +
                  ABy * BCy +
                  ABz * BCz);

    # Find magnitude of
    # line AB and BC
    magnitudeAB = (ABx * ABx +
                   ABy * ABy +
                   ABz * ABz);
    magnitudeBC = (BCx * BCx +
                   BCy * BCy +
                   BCz * BCz);

    # Find the cosine of
    # the angle formed
    # by line AB and BC
    angle = dotProduct;
    angle /= math.sqrt(magnitudeAB *
                       magnitudeBC);

    # Find angle in radian
    angle = (angle * 180) / 3.14;

    # Print angle
    return round(abs(angle), 4)


def get_anglefpose3d(pose, two_line_id=None):
    line = two_line_id
    out = []
    for i in range(pose.shape[0]):
        p = pose[i, line[0], 0], pose[i, line[0], 1], pose[i, line[0], 2], \
            pose[i, line[1], 0], pose[i, line[1], 1], pose[i, line[1], 2], \
            pose[i, line[2], 0], pose[i, line[2], 1], pose[i, line[2], 2],
        out.append(calculate_angle(*p))

    return out


def cal_angles(pose_3d):
    dt = {
        "l_leg": [4, 5, 6],
        "r_leg": [1, 2, 3],
        "l_arm": [11, 12, 13],
        "r_arm": [14, 15, 16],
        "core": [8, 7, 0],
    }
    out_dt = {}
    for k, v in dt.items():
        max_name = k + "_max_angles"
        min_name = k + "_min_angles"
        angle = get_anglefpose3d(pose_3d, v)

        out_dt[max_name] = np.max(angle)
        out_dt[min_name] = np.min(angle)

    return out_dt


def clarity(wave):
    try:
        mean = np.mean(wave)
        gacr = autocorr(wave - mean)
        N = 10
        gacr = np.convolve(gacr, np.ones(N) / N, mode='valid')
        peak_id, _ = find_peaks(gacr, height=np.mean(gacr))
        peaks = gacr[peak_id]
        
        if peaks[0] > peaks[1]:
            new_peaks_id, _ = find_peaks(gacr, height=np.mean(gacr), distance=peak_id[0] * 0.3)
        else:
            new_peaks_id, _ = find_peaks(gacr, height=np.mean(gacr), distance=peak_id[np.argmax(peaks[:3])] * 0.3)

        new_peaks = gacr[new_peaks_id]
        
        return gacr[new_peaks_id[0]] / np.max(gacr[:new_peaks_id[0]])
        
    except:
        return np.nan


def clarity2(wave, vis=False):
    try:
        # Calculate the mean of the waveform
        mean = np.mean(wave)
        
        # Calculate the autocorrelation of the waveform after subtracting the mean
        gacr = np.correlate(wave - mean, wave - mean, mode='full')
        gacr = gacr[len(gacr) // 2:]  # Keep the second half of the autocorrelation
        
        # Smoothing the autocorrelation using a convolution
        N = 10
        gacr = np.convolve(gacr, np.ones(N) / N, mode='valid')  
        
        # Find peaks in the smoothed autocorrelation
        peak_id, _ = find_peaks(gacr, height=np.mean(gacr))
        peaks = gacr[peak_id]
        
        # Determine new peak locations based on the conditions
        if len(peaks) > 1:
            if peaks[0] > peaks[1]:
                new_peaks_id, _ = find_peaks(gacr, height=np.mean(gacr), distance=peak_id[0] * 0.3)
            else:
                new_peaks_id, _ = find_peaks(gacr, height=np.mean(gacr), distance=peak_id[np.argmax(peaks[:3])] * 0.3)
        else:
            new_peaks_id = peak_id
            
        new_peaks = gacr[new_peaks_id]

        if vis:
            print(new_peaks_id)
            plt.plot(gacr)
            for p, v in zip(new_peaks_id, new_peaks):
                plt.plot(p, v, 'r+')
            plt.show()
            
        # Return the clarity metric
        return (gacr[new_peaks_id[0]] / gacr[0])
            
    except Exception as e:
        # Return NaN in case of any exception
        print(f"An error occurred: {e}")
        return 0


def get_freq_inten(arr, fs=30, nperseg=300):

    # Perform STFT
    f, t, Zxx = stft(arr, fs=fs, nperseg=nperseg)
    
    # Determine midpoint in time for the first half of the signal
    mid_point = np.where(t >= t.max() / 2)[0][0]
    
    # Extract STFT data for the first half
    Zxx_first_half = Zxx[:, :mid_point]
    Zxx_last_half = Zxx[:, mid_point:]
    
    # Calculate magnitude
    magnitude = np.abs(Zxx_first_half)
    last_msg = np.abs(Zxx_last_half)
    
    # Calculate the weighted average frequency
    _1st_half_average_frequency = np.mean(np.sum(magnitude * f[:, None], axis=0) / np.sum(magnitude, axis=0))
    _last_half_average_frequency = np.mean(np.sum(last_msg * f[:, None], axis=0) / np.sum(last_msg, axis=0))
    avr_frq = (_1st_half_average_frequency + _last_half_average_frequency)/2
    frq_dff = _1st_half_average_frequency - _last_half_average_frequency
    
    # df, dt = f[1] - f[0], t[1] - t[0]
    # e = sum(np.sum(Zxx.real**2 + Zxx.imag**2, axis=0) * df) * dt

    T, _ = find_period(arr)
    p, values = find_peaks(arr, height=np.mean(arr),
                           distance=int(T * 0.4))
    height = values["peak_heights"]

    return avr_frq, np.mean(height), avr_frq*np.mean(height), frq_dff