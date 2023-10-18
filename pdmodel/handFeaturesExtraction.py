import os
import joblib
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks
import pandas as pd
from itertools import combinations
from .utils import find_period


fps = 59
shift = fps
coverage = 0.5


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


def extract_thumb_index_periods(hand_pose, ax=None, title=""):
    hand_dis = get_thumb_index_dis(hand_pose)
    mean_hand_dis = np.mean(hand_dis)
    hand_dis = hand_dis / hand_dis.max()

    T, _ = find_period(hand_dis)
    p, values = find_peaks(hand_dis, height=np.mean(hand_dis),
                           distance=int(T * 0.4))

    if ax:
        ax.plot(hand_dis)
        ax.plot(p, np.array(hand_dis)[p], "x", ms=10)
        ax.set_xlabel("Frames")
        ax.set_ylabel("Relative thumb-index distance")
        ax.set_title(title)

    return np.diff(p, prepend=0).mean() / fps, (
                np.diff(p, prepend=0)[:4].mean() / fps - np.diff(p, prepend=0)[-5:].mean() / fps) / 2, mean_hand_dis


def preprocess_landmarks(dt):
    hand_pose_arr = np.array(list(dt["landmarks"].values()))
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


# def extract_hand_turning(hand_pose, ax=None, title=""):
#     output = []
#     h_p = hand_pose[:, [8, 7], 0].sum(axis=1)
#     T, _ = find_period(h_p)
#     p, values = find_peaks(h_p, height=np.mean(h_p),
#                            distance=int(T * 0.4))
#
#     if ax:
#         ax.plot(h_p)
#         ax.set_xlabel("Frame")
#         ax.plot(p, np.array(h_p)[p], "x", ms=10)
#         ax.set_ylabel("finger x-axis displacement")
#         ax.set_title(title)
#
#     h, h_d = np.diff(p, prepend=0).mean() / fps, (
#                 np.diff(p, prepend=0)[:4].mean() / fps - np.diff(p, prepend=0)[-5:].mean() / fps) / 2
#     output.append(h)
#     output.append(h_d)
#
#     return output


# def extract_hand_turning_all(hand_pose, ax=None, title=""):
#     output = []
#
#     for i in range(21):
#         h_p = hand_pose[:, i, 0]
#         T, _ = find_period(h_p)
#         p, values = find_peaks(h_p, height=np.mean(h_p),
#                                distance=int(T * 0.4))
#
#         if ax:
#             ax.plot(h_p)
#             ax.set_xlabel("Frame")
#             ax.plot(p, np.array(h_p)[p], "x", ms=10)
#             ax.set_ylabel("finger x-axis displacement")
#             ax.set_title(title)
#
#         h, h_d = np.diff(p, prepend=0).mean() / fps, (
#                     np.diff(p, prepend=0)[:4].mean() / fps - np.diff(p, prepend=0)[-5:].mean() / fps) / 2
#         output.append(h)
#         output.append(h_d)
#
#     return output


# def extract_hand_turning_v_lm(hand_pose, ax=None, title="", lms=[4, 8, 16, 20], pick=4):
#     output = []
#     comb = combinations(lms, pick)
#
#     for i, c in enumerate(comb):
#
#         h_p = hand_pose[:, list(c), 0].sum(axis=1)
#         T, _ = find_period(h_p)
#         p, values = find_peaks(h_p, height=np.mean(h_p),
#                                distance=int(T * 0.4))
#
#         if ax:
#             ax.plot(h_p)
#             ax.set_xlabel("Frame")
#             ax.plot(p, np.array(h_p)[p], "x", ms=10)
#             ax.set_ylabel("finger x-axis displacement")
#             ax.set_title(title)
#
#         h, h_d = np.diff(p, prepend=0).mean() / fps, (
#                     np.diff(p, prepend=0)[:4].mean() / fps - np.diff(p, prepend=0)[-5:].mean() / fps) / 2
#         output.append(h)
#         output.append(h_d)
#
#     return output, i


# def hand_feature_extraction(path_ls):
#     date_ls = []
#     pid_ls = []
#     out_result = []
#     all_hand_id = ["AR", "AL", "BR", "BL"]
#     hand_id = ["Right", "Left"]
#
#     for e, path in enumerate(path_ls):
#         date = path.split("_")[0]
#         pid = path.split("_")[1]
#         all_path = [f"../handOutput3/{date}_{pid}_{hid}_hand.txt" for hid in all_hand_id]
#         fig, ax = plt.subplots(2, 2, figsize=(20, 20))
#         out_ls = []
#
#         if pid[-1] in ["A", "a", "B", "b"]:
#             dt = joblib.load(f"../handOutput3/{date}_{pid}_hand.txt")
#             hand_pose_arr = preprocess_landmarks(dt)[:, 0, :, :]
#             q = hand_pose_arr.shape[0] // 4
#             t1 = q
#             t2 = 2 * q
#             t3 = 3 * q
#
#             try:
#                 hf1 = list(extract_thumb_index_periods(hand_pose_arr[5 * 59:10 * 59, :, :], ax=ax[0, 0],
#                                                        title=f"Thumb-index {hand_id[0]}"))
#             except:
#                 hf1 = (np.ones((3)) * np.nan).tolist()
#             try:
#                 hf2 = list(extract_thumb_index_periods(hand_pose_arr[15 * 59:20 * 59, :, :], ax=ax[0, 1],
#                                                        title=f"Thumb-index {hand_id[1]}"))
#             except:
#                 hf2 = (np.ones((3)) * np.nan).tolist()
#             try:
#                 hf3 = list(extract_hand_turning(hand_pose_arr[t2 * 59:t2 + 5 * 59, :, :], ax=ax[1, 0],
#                                                 title=f"Thumb-Hand_turning {hand_id[0]}"))
#             except:
#                 hf3 = (np.ones((2)) * np.nan).tolist()
#             try:
#                 hf4 = list(extract_hand_turning(hand_pose_arr[t3 * 59:t3 + 5 * 59, :, :], ax=ax[1, 1],
#                                                 title=f"Thumb-Hand_turning {hand_id[1]}"))
#             except:
#                 hf4 = (np.ones((2)) * np.nan).tolist()
#             ax[0, 0].text(0, 0.5, str(4 * q))
#             out_ls = hf1 + hf2 + hf3 + hf4
#
#         else:
#             for i, all_hand_path in enumerate(all_path):
#                 if os.path.isfile(all_hand_path):
#
#                     dt = joblib.load(all_hand_path)
#                     hand_pose_arr = preprocess_landmarks(dt)[:, 0, :, :]
#
#                     if i <= 1:
#                         hf = extract_thumb_index_periods(hand_pose_arr, ax=ax[0, i], title=f"Thumb-index {hand_id[i]}")
#                         out_ls += list(hf)
#                     else:
#                         hf = extract_hand_turning(hand_pose_arr, ax=ax[1, i % 2], title=f"Hand_turning {hand_id[i % 2]}")
#                         out_ls += list(hf)
#
#                 else:
#                     if i <= 1:
#                         out_ls += (np.ones((3)) * np.nan).tolist()
#                     else:
#                         out_ls += (np.ones((2)) * np.nan).tolist()
#
#         out_result.append(out_ls)
#         date_ls.append(date)
#         pid_ls.append(pid)
#
#         plt.savefig(f"../handOutput3/outfig/{date}_{pid}.png")
#         plt.close()
#
#         df = pd.DataFrame(
#             np.concatenate([np.array(date_ls).reshape(-1, 1), np.array(pid_ls).reshape(-1, 1), np.array(out_result)],
#                            axis=1))
#
#         col = df.columns.tolist()
#         col[0] = "Date"
#         col[1] = "PID"
#
#         df.columns = col
#         df.Date = df.Date.apply(lambda x: f"{x[:4]}-{x[4:6]}-{x[-2:]}")
#         df.PID = df.PID.astype(int)
#
#         return df


def single_thumb_index_hand(r_path, l_path, out_dir):

    # fig, ax = plt.subplots(2, 1, figsize=(20, 20))

    r_dt = joblib.load(r_path)
    right_hand_arr = preprocess_landmarks(r_dt)[:, 0, :, :]
    r_hf = extract_thumb_index_periods(right_hand_arr, title=f"Thumb-index right hand")

    # r_hf = extract_thumb_index_periods(right_hand_arr, ax=ax[0], title=f"Thumb-index right hand")

    l_dt = joblib.load(l_path)
    left_hand_arr = preprocess_landmarks(l_dt)[:, 0, :, :]
    l_hf = extract_thumb_index_periods(left_hand_arr, title=f"Thumb-index left hand")

    # l_hf = extract_thumb_index_periods(left_hand_arr, ax=ax[1], title=f"Thumb-index left hand")

    # plt.savefig(f"{out_dir}vis_hand_features_extraction_.png")
    # plt.close()

    return list(r_hf) + list(l_hf)


if __name__ == '__main__':
    right_path = "/mnt/pd_app/results/test/right_hand_202008069_AR.txt"
    left_path = "/mnt/pd_app/results/test/left_hand_202008069_AR.txt"
    out_d = "/mnt/pd_app/results/test/"
    single_thumb_index_hand(right_path, left_path, out_d)
