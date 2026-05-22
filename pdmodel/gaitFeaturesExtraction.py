import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from sklearn.preprocessing import minmax_scale
from utils import cal_angles, moving_average, my_find_peaks_2, find_period
import joblib


def load_pose_data(pose_2d_pth, pose_3d_pth):
    pose = np.load(pose_2d_pth, allow_pickle=True)
    pose_3d = np.load(pose_3d_pth)["reconstruction"][0]
    pose_2d = np.zeros((len(pose), 17, 3))

    for i, p in enumerate(pose):
        pose_2d[i] = p["keypoint"]
    pose_2d = pose_2d.astype(np.float32)

    return pose_2d, pose_3d


def get_2d_y_axis(pose_2d, thres=0.3, ):
    r_leg_x = []
    r_leg_y = []
    l_leg_x = []
    l_leg_y = []
    p_rlx = 0
    p_rly = 0
    p_llx = 0
    p_lly = 0

    delta = lambda x, y, t: y if np.absolute(x - y) > t else x

    for i in range(pose_2d.shape[0]):

        if pose_2d[i, 16, -1] > thres:
            r_leg_x.append(pose_2d[i, 16, 0])
            r_leg_y.append(pose_2d[i, 16, 1])


        else:
            r_leg_x.append(p_rlx)
            r_leg_y.append(p_rly)

        if pose_2d[i, 15, -1] > thres:
            l_leg_x.append(pose_2d[i, 15, 0])
            l_leg_y.append(pose_2d[i, 15, 1])

        else:
            l_leg_x.append(p_llx)
            l_leg_y.append(p_lly)

        p_rlx = pose_2d[i, 16, 0]
        p_rly = pose_2d[i, 16, 1]
        p_llx = pose_2d[i, 15, 0]
        p_lly = pose_2d[i, 15, 1]

    return r_leg_y, l_leg_y


def cal_foot_ground(pose_3d, ax=None):
    # Get Z-coordinates of left and right foot joints
    left_z = pose_3d[:, 3, -1]  # joint 3 = left foot
    right_z = pose_3d[:, 6, -1]  # joint 6 = right foot

    # Estimate dynamic ground threshold based on 5th percentile
    baseline = np.percentile(np.concatenate([left_z, right_z]), 5)
    thres = baseline + 0.05  # Slight buffer above minimum

    # Determine contact frames
    left = left_z <= thres
    right = right_z <= thres

    # Optional visualization
    if ax:
        ax.plot(left, alpha=0.6, label='left on ground')
        ax.plot(right, alpha=0.6, label='right on ground')
        ax.plot(left_z, '--', alpha=0.4, label='left foot Z')
        ax.plot(right_z, '--', alpha=0.4, label='right foot Z')
        ax.axhline(thres, color='r', linestyle='--', label=f'thres={thres:.2f}')
        ax.set_title("Feet on ground detection")
        ax.set_xlabel("frames")
        ax.set_ylabel("Z-height / Boolean")
        ax.legend()

    return np.sum(left) / len(left), np.sum(right) / len(right)


def cal_leftrightfoot_len(pose_3d):
    l = 3
    r = 6
    d_ls = []
    for i in range(pose_3d.shape[0]):
        dist = np.linalg.norm(pose_3d[i, l, :] - pose_3d[i, r, :]) * 100
        d_ls.append(dist)

    return np.array(d_ls)


def cal_turning_speed(pose_3d, ax=None, mask_theres=0.2, fps=30):
    l = 4
    r = 1
    d_4_1 = []

    for i in range(pose_3d.shape[0]):
        d_4_1.append(pose_3d[i, l, :] - pose_3d[i, r, :])

    d_4_1 = np.sum(d_4_1, axis=1)

    if mask_theres:
        d_4_1[d_4_1 >= mask_theres] = mask_theres
        d_4_1[d_4_1 <= -mask_theres] = -mask_theres

    ma = moving_average(minmax_scale(d_4_1), fps)
    g = np.gradient(ma)
    p = my_find_peaks_2(-g, 0.01)

    peaks1, _ = find_peaks(d_4_1, height=0.1, distance=200)
    peaks2, _ = find_peaks(-d_4_1, height=0.1, distance=200)

    peaks = np.sort(np.concatenate([peaks1, peaks2]))

    distances = []
    plot_peaks = []

    for i in range(len(peaks)):
        if i + 1 < len(peaks):
            if d_4_1[peaks[i]] > 0 and d_4_1[peaks[i + 1]] < 0:
                distances.append(peaks[i + 1] - peaks[i])
                if ax:
                    plot_peaks.append(peaks[i])
                    plot_peaks.append(peaks[i + 1])

    if ax:
        ax.plot(np.arange(0, len(d_4_1)), d_4_1, c="g")
        ax.plot(plot_peaks, np.array(d_4_1)[plot_peaks], "x", ms=10)

        ax.legend(["pelvis distances"])
        ax.set_ylabel("distances")
        ax.set_xlabel("frame")
        ax.set_title("point distances from pelvis vs frame")

    return np.mean(np.array(distances) / fps), np.mean(g[p])


def cal_step_per_rounds(x, peaks, fps=30, verbose=False):
    total_steps = 0
    num_cycles = 0
    local_x = []

    # Smoothing window: 0.3 seconds
    w = max(3, int(fps * 0.3))  # at least 3 frames to avoid too-short window

    for i in range(len(peaks) - 1):
        local_x = x[peaks[i]:peaks[i + 1]]
        if len(local_x) < w:
            continue  # skip if not enough data for smoothing

        # Smooth the signal
        smoothed = moving_average(minmax_scale(local_x), w)
        gradient = np.abs(np.gradient(smoothed))
        gradient_smooth = moving_average(gradient, w)

        # Adaptive threshold for peaks
        dynamic_height = np.percentile(gradient_smooth, 70)
        _p, _ = find_peaks(gradient_smooth, height=dynamic_height)

        total_steps += len(_p)
        num_cycles += 1

    if num_cycles == 0:
        return 0  # avoid division by zero

    if verbose:
        fig, axs = plt.subplots(1, 3, figsize=(30, 10))
        axs[0].plot(local_x)
        axs[0].set_title("Original y-axis location")

        axs[1].plot(smoothed)
        axs[1].set_title(f"Moving average (w = {w})")

        axs[2].plot(gradient_smooth)
        axs[2].axhline(dynamic_height, color='r', linestyle='--', label=f'Threshold={dynamic_height:.4f}')
        axs[2].plot(_p, gradient_smooth[_p], "x", c="b", ms=10)
        axs[2].set_title("Adaptive peak detection")
        axs[2].legend()

        plt.tight_layout()

    return total_steps / num_cycles


def cal_speed_round(pose_2d, thres=0.3, ax=None, verbose=False, fps=30):
    r_leg_x = []
    r_leg_y = []
    l_leg_x = []
    l_leg_y = []
    p_rlx = 0
    p_rly = 0
    p_llx = 0
    p_lly = 0

    delta = lambda x, y, t: y if np.absolute(x - y) > t else x

    for i in range(pose_2d.shape[0]):

        if pose_2d[i, 16, -1] > thres:
            r_leg_x.append(pose_2d[i, 16, 0])
            r_leg_y.append(pose_2d[i, 16, 1])
            p_rlx = pose_2d[i, 16, 0]
            p_rly = pose_2d[i, 16, 1]

        else:
            r_leg_x.append(p_rlx)
            r_leg_y.append(p_rly)

        if pose_2d[i, 15, -1] > thres:
            l_leg_x.append(pose_2d[i, 15, 0])
            l_leg_y.append(pose_2d[i, 15, 1])
            p_llx = pose_2d[i, 15, 0]
            p_lly = pose_2d[i, 15, 1]

        else:
            l_leg_x.append(p_llx)
            l_leg_y.append(p_lly)

    l_T, l_acf = find_period(moving_average(l_leg_y, 20))
    r_T, r_acf = find_period(moving_average(r_leg_y, 20))
    l_peaks, _ = find_peaks(l_leg_y, distance=l_T * 0.8)
    r_peaks, _ = find_peaks(r_leg_y, distance=r_T * 0.8)

    if ax is not None:
        ax.plot(np.arange(0, len(l_leg_y)), l_leg_y, c="g")
        ax.plot(np.arange(0, len(r_leg_y)), r_leg_y, c="b")
        ax.plot(l_peaks, np.array(l_leg_y)[l_peaks], "x", ms=10)
        ax.plot(r_peaks, np.array(r_leg_y)[r_peaks], "x", c="b", ms=10)
        ax.legend(["left", "right"])
        ax.set_ylabel("y-axis")
        ax.set_xlabel("frame")
        ax.set_title("Foot location vs frame")

    l_mean_steps = cal_step_per_rounds(l_leg_y, l_peaks, verbose=verbose)
    r_mean_steps = cal_step_per_rounds(r_leg_y, r_peaks, verbose=verbose)

    average_speed = (np.mean(np.diff(l_peaks) / fps) + np.mean(np.diff(r_peaks) / fps)) / 2
    speed_change_l = ((np.mean(np.diff(l_peaks)[:3] / fps)) - (np.mean(np.diff(l_peaks)[-3:] / fps))) / (
        np.mean(np.diff(l_peaks)[:3] / fps))
    speed_change_r = ((np.mean(np.diff(r_peaks)[:3] / fps)) - (np.mean(np.diff(r_peaks)[-3:] / fps))) / (
        np.mean(np.diff(r_peaks)[:3] / fps))
    speed_change = (speed_change_l + speed_change_r) / 2

    return average_speed, speed_change, l_mean_steps, r_mean_steps


def pose_features_extract(pth_2d, pth_3d, fps=30, plot_results=False, save_fig_pth=None, debug=False):
    if debug:
        print("=" * 70)
        print("[pose_features_extract] Start")
        print(f"    pth_2d        : {pth_2d}")
        print(f"    pth_3d        : {pth_3d}")
        print(f"    fps           : {fps}")
        print(f"    plot_results  : {plot_results}")
        print(f"    save_fig_pth  : {save_fig_pth}")

    # =========================================================
    # 1. Prepare plotting axes
    # =========================================================
    if plot_results:
        if debug:
            print("\n[STEP 1] Create figure and axes for plotting")
        fig, ax = plt.subplots(2, 2, figsize=(20, 20))
        plot_fog = ax[0][0]
        plot_as = ax[0][1]
        plot_lts = ax[1][0]
        plot_rts = ax[1][1]

        if debug:
            print("    figure created successfully")
    else:
        if debug:
            print("\n[STEP 1] Plotting disabled")
        fig = None
        plot_fog = None
        plot_as = None
        plot_lts = None
        plot_rts = None

    # =========================================================
    # 2. Load pose data
    # =========================================================
    if debug:
        print("\n[STEP 2] Load pose data")

    pose_2d, pose_3d = load_pose_data(pth_2d, pth_3d)

    if debug:
        print(f"    pose_2d type  : {type(pose_2d)}")
        print(f"    pose_3d type  : {type(pose_3d)}")
        try:
            print(f"    pose_2d shape : {pose_2d.shape}")
        except Exception:
            print("    pose_2d shape : not available")
        try:
            print(f"    pose_3d shape : {pose_3d.shape}")
        except Exception:
            print("    pose_3d shape : not available")

    # =========================================================
    # 3. Foot on ground
    # =========================================================
    if debug:
        print("\n[STEP 3] Calculate foot-on-ground features")

    fog_l, fog_r = cal_foot_ground(pose_3d, ax=plot_fog)

    if debug:
        print(f"    fog_l: {fog_l}")
        print(f"    fog_r: {fog_r}")

    # =========================================================
    # 4. Left-right foot length
    # =========================================================
    if debug:
        print("\n[STEP 4] Calculate left-right foot length")

    left_right_foot_len = cal_leftrightfoot_len(pose_3d)

    if debug:
        print(f"    left_right_foot_len type: {type(left_right_foot_len)}")
        try:
            print(f"    left_right_foot_len len : {len(left_right_foot_len)}")
        except Exception:
            print("    left_right_foot_len len : not available")
        try:
            print(f"    mean left_right_foot_len: {np.mean(left_right_foot_len)}")
            print(f"    max  left_right_foot_len: {np.max(left_right_foot_len)}")
        except Exception as e:
            print(f"    failed to summarize left_right_foot_len: {e}")

    # =========================================================
    # 5. Left turning speed
    # =========================================================
    if debug:
        print("\n[STEP 5] Calculate left turning speed")

    left_turning_speed, left_turning_slope = cal_turning_speed(pose_3d, ax=plot_lts)

    if debug:
        print(f"    left_turning_speed: {left_turning_speed}")
        print(f"    left_turning_slope: {left_turning_slope}")

    # =========================================================
    # 6. Right turning speed
    # =========================================================
    if debug:
        print("\n[STEP 6] Calculate right turning speed")

    right_turning_speed, right_turning_slope = cal_turning_speed(-pose_3d, ax=plot_rts)

    if debug:
        print(f"    right_turning_speed: {right_turning_speed}")
        print(f"    right_turning_slope: {right_turning_slope}")

    # =========================================================
    # 7. Angle features
    # =========================================================
    if debug:
        print("\n[STEP 7] Calculate angle features")

    angles_dict = cal_angles(pose_3d)
    angles_ls = list(angles_dict.values())

    if debug:
        print(f"    angles_dict keys : {list(angles_dict.keys())}")
        print(f"    number of angles : {len(angles_ls)}")
        print(f"    angle values     : {angles_ls}")

    # =========================================================
    # 8. Speed-round features
    # =========================================================
    if debug:
        print("\n[STEP 8] Calculate average speed and step features")

    average_speed, speed_change, l_mean_steps, r_mean_steps = cal_speed_round(
        pose_2d, thres=0.3, ax=plot_as
    )

    if debug:
        print(f"    average_speed: {average_speed}")
        print(f"    speed_change : {speed_change}")
        print(f"    l_mean_steps : {l_mean_steps}")
        print(f"    r_mean_steps : {r_mean_steps}")

    # =========================================================
    # 9. Build output
    # =========================================================
    if debug:
        print("\n[STEP 9] Assemble output feature vector")

    output = [
        fog_l,
        fog_r,
        np.mean(left_right_foot_len),
        np.max(left_right_foot_len),
        left_turning_speed,
        left_turning_slope,
        right_turning_speed,
        right_turning_slope,
    ] + angles_ls + [
        average_speed,
        speed_change,
        l_mean_steps,
        r_mean_steps,
    ]

    if debug:
        print(f"    output length: {len(output)}")
        print(f"    output: {output}")

    # =========================================================
    # 10. Save figure
    # =========================================================
    if plot_results:
        if debug:
            print("\n[STEP 10] Save figure")
        fig.tight_layout()

        if save_fig_pth is not None:
            fig.savefig(save_fig_pth)
            if debug:
                print(f"    figure saved to: {save_fig_pth}")
        else:
            if debug:
                print("    save_fig_pth is None, figure not saved")

        plt.close(fig)
        if debug:
            print("    figure closed")

    if debug:
        print("\n[pose_features_extract] Finished")
        print("=" * 70)

    return output


if __name__ == '__main__':
    pth_2d = "/mnt/pd_app/results/test/2d_20200806_3C.npy"
    pth_3d = "/mnt/pd_app/results/test/3D_20200806_3C.npz"
    pose_features_extract(pth_2d, pth_3d, plot_results=True, save_fig_pth="/mnt/pd_app/results/test/vis_extration.png")

