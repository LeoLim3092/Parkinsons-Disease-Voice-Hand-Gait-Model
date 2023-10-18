from scipy.signal import find_peaks, correlate
import numpy as np
import math


def autocorr(x):
    result = correlate(x, x)
    return result[int(result.size / 2):]


def find_period(arr):
    mean = np.mean(arr)

    acf_result = autocorr(arr - mean)
    peaks, _ = find_peaks(acf_result)

    if peaks[0] > 50:
        period = peaks[0]
    else:
        period = peaks[1]

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
