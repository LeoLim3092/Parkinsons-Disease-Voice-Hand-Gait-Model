import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from sklearn.preprocessing import minmax_scale
from .utils import cal_angles, moving_average, my_find_peaks_2, find_period
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


def cal_foot_ground(pose_3d, thres=0.1, ax=None):
    left = pose_3d[:, 3, -1] <= thres
    right = pose_3d[:, 6, -1] <= thres

    if ax:
        ax.plot(left, alpha=0.5, c="b")
        #         ax.plot(right, alpha=0.5, c="r")
        ax.set_title("feet on ground")
        ax.set_xlabel("frames")
        ax.set_ylabel("boolean")
        ax.legend(["left", "right"])

    return sum(left) / pose_3d.shape[0], sum(right) / pose_3d.shape[0]


def cal_leftrightfoot_len(pose_3d):
    l = 3
    r = 6
    d_ls = []
    for i in range(pose_3d.shape[0]):
        dist = np.linalg.norm(pose_3d[i, l, :] - pose_3d[i, r, :]) * 100
        d_ls.append(dist)

    return np.array(d_ls)


def cal_turning_speed(pose_3d, ax=None, mask_theres=0.2):
    l = 4
    r = 1
    d_4_1 = []

    for i in range(pose_3d.shape[0]):
        d_4_1.append(pose_3d[i, l, :] - pose_3d[i, r, :])

    d_4_1 = np.sum(d_4_1, axis=1)

    if mask_theres:
        d_4_1[d_4_1 >= mask_theres] = mask_theres
        d_4_1[d_4_1 <= -mask_theres] = -mask_theres

    ma = moving_average(minmax_scale(d_4_1), 30)
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

    return np.mean(np.array(distances) / 30), np.mean(g[p])


def cal_step_per_rounds(x, peaks, verbose=False):
    total_steps = 0
    local_x = []
    i = 0
    for i in range(len(peaks)):
        if i + 1 < len(peaks):
            local_x = x[peaks[i]:peaks[i + 1]]
            _p, _ = find_peaks(moving_average(np.absolute(np.gradient(moving_average(minmax_scale(local_x), 10))), 10),
                               height=0.005)
            total_steps += len(_p)

    if verbose:
        fig, axs = plt.subplots(1, 3, figsize=(30, 10))
        axs[0].plot(local_x)
        axs[0].set_title("Original y-axis location")

        axs[1].plot(moving_average(minmax_scale(local_x), 10))
        axs[1].set_title("Moving average with 10 frames window")

        axs[2].plot(moving_average(np.absolute(np.gradient(moving_average(minmax_scale(local_x), 10))), 10))
        _p, _ = find_peaks(moving_average(np.absolute(np.gradient(moving_average(minmax_scale(local_x), 10))), 10),
                           height=0.005)
        axs[2].plot(_p, moving_average(np.absolute(np.gradient(moving_average(minmax_scale(local_x), 10))), 10)[_p],
                    "x", c="b", ms=10)
        axs[2].set_title("counting step per round")

    return total_steps / i


def cal_speed_round(pose_2d, thres=0.3, ax=None, verbose=False):
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

    average_speed = (np.mean(np.diff(l_peaks) / 30) + np.mean(np.diff(r_peaks) / 30)) / 2
    speed_change_l = ((np.mean(np.diff(l_peaks)[:3] / 30)) - (np.mean(np.diff(l_peaks)[-3:] / 30))) / (
        np.mean(np.diff(l_peaks)[:3] / 30))
    speed_change_r = ((np.mean(np.diff(r_peaks)[:3] / 30)) - (np.mean(np.diff(r_peaks)[-3:] / 30))) / (
        np.mean(np.diff(r_peaks)[:3] / 30))
    speed_change = (speed_change_l + speed_change_r) / 2

    return average_speed, speed_change, l_mean_steps, r_mean_steps


def pose_features_extract(pth_2d, pth_3d, plot_results=False, save_fig_pth=None):

    if plot_results:
        fig, ax = plt.subplots(2, 2, figsize=(20, 20))
        plot_fog = ax[0][0]
        plot_as = ax[0][1]
        plot_lts = ax[1][0]
        plot_rts = ax[1][1]

    else:
        plot_fog = None
        plot_as = None
        plot_lts = None
        plot_rts = None

    # load pose data
    pose_2d, pose_3d = load_pose_data(pth_2d, pth_3d)

    # foot on ground
    fog_l, fog_r = cal_foot_ground(pose_3d, thres=0.1, ax=plot_fog)

    # left to right foot length (mean, max)
    left_right_foot_len = cal_leftrightfoot_len(pose_3d)

    # left and right turning speed
    left_turning_speed, left_turning_slope = cal_turning_speed(pose_3d, ax=plot_lts)
    right_turning_speed, right_turning_slope = cal_turning_speed(-pose_3d, ax=plot_rts)

    # angles
    angles_ls = list(cal_angles(pose_3d).values())

    # average speed
    average_speed, speed_change, l_mean_steps, r_mean_steps = cal_speed_round(pose_2d, thres=0.3, ax=plot_as)

    output = [fog_l, fog_r, np.mean(left_right_foot_len), np.max(left_right_foot_len),
              left_turning_speed, left_turning_slope, right_turning_speed, right_turning_slope] + angles_ls + [
                 average_speed, speed_change, l_mean_steps, r_mean_steps]

    if plot_results:
        fig.tight_layout()
        fig.savefig(save_fig_pth)
        plt.close(fig)

    return output


if __name__ == '__main__':
    pth_2d = "/mnt/pd_app/results/test/2d_20200806_3C.npy"
    pth_3d = "/mnt/pd_app/results/test/3D_20200806_3C.npz"
    pose_features_extract(pth_2d, pth_3d, plot_results=True, save_fig_pth="/mnt/pd_app/results/test/vis_extration.png")

