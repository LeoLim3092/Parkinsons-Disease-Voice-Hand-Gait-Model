import os
import joblib
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks
import pandas as pd
from itertools import combinations
from utils import find_period
from scipy.signal import stft


def get_thumb_index_dis(hand_landmarks):
    hand_thumb = hand_landmarks[:, 4, :-1]
    hand_index = hand_landmarks[:, 8, :-1]

    hand_4_8_dis = []
    for t, i in zip(hand_thumb, hand_index):
        hand_4_8_dis.append(np.linalg.norm(i - t))
    hand_4_8_dis = np.array(hand_4_8_dis)

    return hand_4_8_dis


def get_thumb_pinky_dis(hand_landmarks):
    hand_thumb = hand_landmarks[:, 4, :]
    hand_index = hand_landmarks[:, 20, :]

    hand_4_20_dis = []
    for t, i in zip(hand_thumb, hand_index):
        hand_4_20_dis.append(np.linalg.norm(i - t))
    hand_4_20_dis = np.array(hand_4_20_dis)

    return hand_4_20_dis


def extract_thumb_index_periods(hand_pose, ax=None, title="", fps=59, debug=False):
    if debug:
        print("=" * 70)
        print("[extract_thumb_index_periods] Start")
        print(f"    title: {title}")
        print(f"    fps  : {fps}")
        print(f"    hand_pose type: {type(hand_pose)}")
        try:
            print(f"    hand_pose shape: {hand_pose.shape}")
        except Exception:
            print("    hand_pose shape: not available")

    # =========================================================
    # 1. Compute thumb-index distance
    # =========================================================
    if debug:
        print("\n[STEP 1] Compute thumb-index distance")

    hand_dis = get_thumb_index_dis(hand_pose)
    mean_hand_dis = np.mean(hand_dis)

    if debug:
        print(f"    hand_dis type : {type(hand_dis)}")
        try:
            print(f"    hand_dis len  : {len(hand_dis)}")
        except Exception:
            print("    hand_dis len  : not available")
        print(f"    mean_hand_dis : {mean_hand_dis}")
        try:
            preview_n = min(10, len(hand_dis))
            print(f"    first {preview_n} hand_dis values: {hand_dis[:preview_n]}")
        except Exception:
            pass

    # =========================================================
    # 2. Compute normalization factor
    # =========================================================
    if debug:
        print("\n[STEP 2] Compute normalization factor")
        print("    using distance between landmark 0 and 5")

    norm_factor = np.mean(
        np.linalg.norm(hand_pose[:, 0, :-1] - hand_pose[:, 5, :-1], axis=1)
    )

    if debug:
        print(f"    norm_factor: {norm_factor}")

    hand_dis = hand_dis / norm_factor if norm_factor > 0 else hand_dis

    if debug:
        print("    normalization applied" if norm_factor > 0 else "    normalization skipped (norm_factor <= 0)")
        try:
            preview_n = min(10, len(hand_dis))
            print(f"    first {preview_n} normalized hand_dis values: {hand_dis[:preview_n]}")
        except Exception:
            pass

    # =========================================================
    # 3. Estimate period
    # =========================================================
    if debug:
        print("\n[STEP 3] Estimate signal period with find_period()")

    T, _ = find_period(hand_dis, debug=debug)

    if debug:
        print(f"    estimated period T: {T}")
        try:
            peak_distance = int(T * 0.4)
            print(f"    peak minimum distance: {peak_distance}")
        except Exception:
            pass

    # =========================================================
    # 4. Find peaks
    # =========================================================
    if debug:
        print("\n[STEP 4] Find peaks")

    peak_height = np.mean(hand_dis)
    peak_distance = max(1, int(T * 0.4))  # safer than allowing 0

    p, values = find_peaks(
        hand_dis,
        height=peak_height,
        distance=peak_distance
    )

    if debug:
        print(f"    peak height threshold: {peak_height}")
        print(f"    number of peaks found : {len(p)}")
        print(f"    peak indices          : {p}")
        if isinstance(values, dict):
            print(f"    peak info keys        : {list(values.keys())}")
            if "peak_heights" in values:
                print(f"    peak heights          : {values['peak_heights']}")

    # =========================================================
    # 5. Plot if axis is provided
    # =========================================================
    if ax is not None:
        if debug:
            print("\n[STEP 5] Plot results on axis")

        ax.plot(hand_dis)
        ax.plot(p, np.array(hand_dis)[p], "x", ms=10)
        ax.set_xlabel("Frames")
        ax.set_ylabel("Relative thumb-index distance")
        ax.set_title(title)

    else:
        if debug:
            print("\n[STEP 5] Plot skipped (ax is None)")

    # =========================================================
    # 6. Compute temporal features
    # =========================================================
    if debug:
        print("\n[STEP 6] Compute period-based features")

    if len(p) == 0:
        avg_period_sec = 0
        decay_feature = 0

        if debug:
            print("    no peaks found -> avg_period_sec = 0, decay_feature = 0")

    else:
        peak_intervals = np.diff(p, prepend=0)

        avg_period_sec = peak_intervals.mean() / fps if fps > 0 else 0

        first_part = peak_intervals[:4]
        last_part = peak_intervals[-5:]

        first_mean = first_part.mean() / fps if len(first_part) > 0 and fps > 0 else 0
        last_mean = last_part.mean() / fps if len(last_part) > 0 and fps > 0 else 0

        decay_feature = (first_mean - last_mean) / 2

        if debug:
            print(f"    peak_intervals: {peak_intervals}")
            print(f"    avg_period_sec: {avg_period_sec}")
            print(f"    first_part    : {first_part}")
            print(f"    last_part     : {last_part}")
            print(f"    first_mean    : {first_mean}")
            print(f"    last_mean     : {last_mean}")
            print(f"    decay_feature : {decay_feature}")

    if debug:
        print("\n[STEP 7] Final output")
        print(f"    avg_period_sec = {avg_period_sec}")
        print(f"    decay_feature  = {decay_feature}")
        print(f"    mean_hand_dis  = {mean_hand_dis}")
        print("=" * 70)

    return avg_period_sec, decay_feature, mean_hand_dis


def preprocess_landmarks(dt):
    hand_pose_arr = list(dt["landmarks"].values())
    hand_pose_ls = []

    for a, arr in enumerate(hand_pose_arr):
        if isinstance(arr, list):
            if arr:
                hand_pose_ls.append(arr)
            else:
                hand_pose_ls.append([np.zeros((21, 3))])
        else:
            if arr[0].shape[0] != 0:
                hand_pose_ls.append(arr)
            else:
                hand_pose_ls.append([np.zeros((21, 3))])

    hand_pose_arr = np.array(hand_pose_ls)
    return hand_pose_arr


def get_freq_inten(arr):

    dis = get_thumb_index_dis(arr)
    norm_factor = np.mean(np.linalg.norm(arr[:, 0, :-1] - arr[:, 5, :-1], axis=1))
    dis = dis / norm_factor if norm_factor > 0 else dis
    
    # Perform STFT
    fs = 30
    f, t, Zxx = stft(dis, fs=fs, nperseg=300)
    
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

    T, _ = find_period(dis)
    p, values = find_peaks(dis, height=np.mean(dis),
                           distance=int(T * 0.4))
    height = values["peak_heights"]

    return avr_frq, np.mean(height), avr_frq*np.mean(height), frq_dff


def single_thumb_index_hand(r_path, l_path, out_dir, fps=59, debug=False):
    if debug:
        print("=" * 70)
        print("[single_thumb_index_hand] Start")
        print(f"    r_path : {r_path}")
        print(f"    l_path : {l_path}")
        print(f"    out_dir: {out_dir}")
        print(f"    fps    : {fps}")

    # =========================================================
    # 1. Prepare output directory and figure
    # =========================================================
    os.makedirs(out_dir, exist_ok=True)

    if debug:
        print("\n[STEP 1] Create figure and output directory")
        print(f"    ensured output directory exists: {out_dir}")

    fig, ax = plt.subplots(2, 1, figsize=(20, 20))

    if debug:
        print("    figure created successfully")

    # =========================================================
    # 2. Load right-hand landmark data
    # =========================================================
    if debug:
        print("\n[STEP 2] Load right-hand joblib data")

    r_dt = joblib.load(r_path)

    if debug:
        print(f"    right raw type: {type(r_dt)}")

    # =========================================================
    # 3. Preprocess right-hand landmarks
    # =========================================================
    if debug:
        print("\n[STEP 3] Preprocess right-hand landmarks")

    right_preprocessed = preprocess_landmarks(r_dt)
    right_hand_arr = right_preprocessed[:, 0, :, :]

    if debug:
        print(f"    right_preprocessed type : {type(right_preprocessed)}")
        try:
            print(f"    right_preprocessed shape: {right_preprocessed.shape}")
        except Exception:
            print("    right_preprocessed shape: not available")

        print(f"    right_hand_arr type     : {type(right_hand_arr)}")
        try:
            print(f"    right_hand_arr shape    : {right_hand_arr.shape}")
        except Exception:
            print("    right_hand_arr shape    : not available")

    # =========================================================
    # 4. Extract right-hand thumb-index periods
    # =========================================================
    if debug:
        print("\n[STEP 4] Extract right-hand thumb-index periods")

    r_hf = extract_thumb_index_periods(
        right_hand_arr,
        ax=ax[0],
        title="Thumb-index right hand",
        fps=fps,
        debug=debug
    )

    if debug:
        print(f"    r_hf type: {type(r_hf)}")
        try:
            print(f"    r_hf len : {len(r_hf)}")
            print(f"    r_hf vals: {list(r_hf)}")
        except Exception:
            print(f"    r_hf val : {r_hf}")

    # =========================================================
    # 5. Load left-hand landmark data
    # =========================================================
    if debug:
        print("\n[STEP 5] Load left-hand joblib data")

    l_dt = joblib.load(l_path)

    if debug:
        print(f"    left raw type: {type(l_dt)}")

    # =========================================================
    # 6. Preprocess left-hand landmarks
    # =========================================================
    if debug:
        print("\n[STEP 6] Preprocess left-hand landmarks")

    left_preprocessed = preprocess_landmarks(l_dt)
    left_hand_arr = left_preprocessed[:, 0, :, :]

    if debug:
        print(f"    left_preprocessed type : {type(left_preprocessed)}")
        try:
            print(f"    left_preprocessed shape: {left_preprocessed.shape}")
        except Exception:
            print("    left_preprocessed shape: not available")

        print(f"    left_hand_arr type     : {type(left_hand_arr)}")
        try:
            print(f"    left_hand_arr shape    : {left_hand_arr.shape}")
        except Exception:
            print("    left_hand_arr shape    : not available")

    # =========================================================
    # 7. Extract left-hand thumb-index periods
    # =========================================================
    if debug:
        print("\n[STEP 7] Extract left-hand thumb-index periods")

    l_hf = extract_thumb_index_periods(
        left_hand_arr,
        ax=ax[1],
        title="Thumb-index left hand",
        fps=fps,
        debug=debug
    )

    if debug:
        print(f"    l_hf type: {type(l_hf)}")
        try:
            print(f"    l_hf len : {len(l_hf)}")
            print(f"    l_hf vals: {list(l_hf)}")
        except Exception:
            print(f"    l_hf val : {l_hf}")

    # =========================================================
    # 8. Save visualization
    # =========================================================
    save_fig_path = os.path.join(out_dir, "vis_hand_features_extraction_.png")

    if debug:
        print("\n[STEP 8] Save visualization figure")
        print(f"    save_fig_path: {save_fig_path}")

    plt.savefig(save_fig_path)
    plt.close(fig)

    if debug:
        print("    figure saved and closed")

    # =========================================================
    # 9. Extract frequency / intensity features
    # =========================================================
    if debug:
        print("\n[STEP 9] Extract frequency / intensity features")

    l_hf2 = get_freq_inten(left_hand_arr)
    r_hf2 = get_freq_inten(right_hand_arr)

    if debug:
        print(f"    l_hf2 type: {type(l_hf2)}")
        try:
            print(f"    l_hf2 len : {len(l_hf2)}")
            print(f"    l_hf2 vals: {list(l_hf2)}")
        except Exception:
            print(f"    l_hf2 val : {l_hf2}")

        print(f"    r_hf2 type: {type(r_hf2)}")
        try:
            print(f"    r_hf2 len : {len(r_hf2)}")
            print(f"    r_hf2 vals: {list(r_hf2)}")
        except Exception:
            print(f"    r_hf2 val : {r_hf2}")

    # =========================================================
    # 10. Merge output
    # =========================================================
    output = list(r_hf) + list(r_hf2) + list(l_hf) + list(l_hf2)

    if debug:
        print("\n[STEP 10] Merge output features")
        print(f"    output length: {len(output)}")
        print(f"    output: {output}")

        print("\n[single_thumb_index_hand] Finished")
        print("=" * 70)

    return output


if __name__ == '__main__':
    right_path = "/mnt/pd_app/results/test/right_hand_202008069_AR.txt"
    left_path = "/mnt/pd_app/results/test/left_hand_202008069_AR.txt"
    out_d = "/mnt/pd_app/results/test/"
    single_thumb_index_hand(right_path, left_path, out_d)
