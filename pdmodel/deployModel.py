from .gaitExtraction import gait_extraction, gait_checking
from .handExtraction import hand_extraction, hand_checking
from .voiceFeatureExtraction import voice_features_extraction, sound_checking
from . import handFeaturesExtraction
from . import gaitFeaturesExtraction
import os
from datetime import datetime
import joblib
import numpy as np
import matplotlib.pyplot as plt
import settings


model_paths = settings.MODEL_PATHS

TRAINED_MODELS_LS = ["C4.5 DT", "RF", "LogReg", "AdaBoost", "NB", "SVM"]  # remove KNN, "GBM", "LightGBM",

gait_feature_name = ['left_foot_ground', 'right_foot_ground', 'left_right_foot_len_average', 'left_right_foot_len_max',
                     'left_turning_duration', 'left_turning_slope', 'right_turning_duration', 'right_turning_slope',
                     'l_leg_max_angles', 'l_leg_min_angles', 'r_leg_max_angles', 'r_leg_min_angles', 'l_arm_max_angles',
                     'l_arm_min_angles', 'r_arm_max_angles', 'r_arm_min_angles', 'core_max_angles', "core_min_angles",
                     'average_duration_per_rounds', "duration_change", "l_mean_steps", "r_mean_steps"]

hand_features_name = ["right_finger_open_duration", 'right_finger_open_change', "right_finger_open_distance",
                      "left_finger_open_duration", 'left_finger_open_change', "left_finger_open_distance"]

voice_feature_name = ['pause(%)', 'volume change', 'pitch change', 'Average pitch']


def extract_gait(gait_video_pth, out_dir):
    gait_extraction(gait_video_pth, out_dir)
    pth_2d = f"{out_dir}2d_{os.path.basename(gait_video_pth)[:-4]}.npy"
    pth_3d = f"{out_dir}3d_{os.path.basename(gait_video_pth)[:-4]}.npz"
    gait_feature = gaitFeaturesExtraction.pose_features_extract(pth_2d, pth_3d)
    return gait_feature


def extract_hand(hand_video_pth, out_dir, hand):
    hand_extraction(hand_video_pth, out_video_root=out_dir, hand=hand)


def model_extraction(gait_video_pth, left_video_path, right_video_path, voice_path, out_dir):

    voice_feature = voice_features_extraction(voice_path)

    print(f'Start gait features extraction')
    gait_feature = extract_gait(gait_video_pth, out_dir)

    print(f'Start left hand features extraction')
    extract_hand(left_video_path, out_dir, 'left')

    print(f'Start right hand features extraction')
    extract_hand(right_video_path, out_dir, 'right')

    r_path = f"{out_dir}right_hand_{os.path.basename(right_video_path)[:-4]}.txt"
    l_path = f"{out_dir}left_hand_{os.path.basename(left_video_path)[:-4]}.txt"

    hand_feature = handFeaturesExtraction.single_thumb_index_hand(r_path, l_path, out_dir)
    print(f"Finish hand features extraction")

    print(f"{gait_feature}, \n {hand_feature}, \n {voice_feature} \n")

    all_features = gait_feature + hand_feature + voice_feature
    np.save(f'{out_dir}all_feature.npy', all_features)
    print(f"save! \n {all_features}")


def predict_models(all_features_pth, age, gender, out_dir):
    sfs_idx = joblib.load(settings.sfs_pth)
    _temp_voice_sfs_idx = joblib.load(settings.temp_voice_sfs_pth)
    gait_len = len(gait_feature_name)
    hand_len = len(hand_features_name)
    voice_len = len(voice_feature_name)
    gait_result = {}
    hand_result = {}
    voice_result = {}
    all_result = {}

    for r in [gait_result, hand_result, voice_result, all_result]:
        for k in TRAINED_MODELS_LS:
            r[k] = [0, 0]

    data = np.load(all_features_pth)
    gait_feature = np.concatenate([np.array([age, gender]), data[:gait_len]])
    hand_feature = np.concatenate([np.array([age, gender]), data[gait_len:gait_len + hand_len]])
    voice_feature = np.concatenate([np.array([age, gender]), data[-voice_len:]])

    gait_result = deploy(np.array([gait_feature]), sfs_idx['gait_sfs_idx'], modal="gait", fold=10)
    hand_result = deploy(np.array([hand_feature]), sfs_idx['hand_sfs_idx'], modal="hand", fold=10)
    voice_result = deploy(np.array([voice_feature]), _temp_voice_sfs_idx, modal="voice2", fold=10)
    weight = [4, 6, 5]

    all_result = {}

    for k in TRAINED_MODELS_LS:
        all_result[k] = np.average(np.array([[gait_result[k]], [hand_result[k]],
                                             [voice_result[k]]]), weights=weight, axis=0)[0]

    results = np.array([gait_result["SVM"][0], hand_result["SVM"][0], voice_result["SVM"][0],
                        all_result["SVM"][0]]) * 100


    return results


def deploy(data, feature_idx_dt, modal="gait", fold=10):
    save_file_dir = f"{model_paths}/Save_model_{modal}/"
    proba_out_dt = {}

    for model_name in TRAINED_MODELS_LS:
        proba_ls = []

        for i in range(fold):
            clf_pth = f'{save_file_dir}{i}_{model_name}.joblib'
            clf = joblib.load(clf_pth)
            print(clf.n_features_in_)
            f_idx = feature_idx_dt[model_name]
            predict_proba = clf.predict_proba(data[:, f_idx])
            proba_ls.append(predict_proba)

        mean_test = np.array(proba_ls).mean(0)
        proba_out_dt[model_name] = mean_test[:, 1]

    return proba_out_dt


def data_checking(gait_file_pth, l_hand_file_pth, r_hand_file_pth, sound_file_pth):
    success, error = "success", ""
    gait_score = gait_checking(gait_file_pth)
    left_hand_score = hand_checking(l_hand_file_pth)
    right_hand_score = hand_checking(r_hand_file_pth)
    sound_duration, sound_score, failed_sound = sound_checking(sound_file_pth)

    if right_hand_score != "too short!":
        if right_hand_score < 0.3:
            success = "failed"
            error += "右手影像，"
    else:
        success = "failed"
        error += "右手影片過短，"

    if left_hand_score != "too short!":
        if left_hand_score < 0.3:
            success = "failed"
            error += "左手影像，"
    else:
        success = "failed"
        error += "左手影片過短，"

    if gait_score != "too short!":
        if gait_score < 0.3:
            success = "failed"
            error += "步態影像，"
    else:
        success = "failed"
        error += "步態影片過短，"

    if failed_sound:
        success = "failed"
        error += "閱讀聲音，"
    else:
        if sound_score < 50:
            error += "閱讀聲音太小，"

    if sound_duration < 15:
        success = "failed"
        error += "閱讀聲音過短，"

    if error != "":
        error = error[:-1] + "。"

    return success, error


if __name__ == "__main__":
    gait_v_pth = "/mnt/pd_app/walk/20200818_5C.mp4"
    left_v_path = "/mnt/pd_app/gesture/202008069_AL.mp4"
    right_v_path = "/mnt/pd_app/gesture/202008069_AR.mp4"
    out_d = "/mnt/pd_app/results/test/"

    ini_time_for_now = datetime.now()
    asyncio.run(model_extraction(gait_v_pth, left_v_path, right_v_path, out_d))
    finish_time = datetime.now()
    print((finish_time - ini_time_for_now).seconds)

    # data = np.load(f'{out_d}all_feature.npy')
    # deploy(data, feature_idx_dt, modal="gait", fold=10)
