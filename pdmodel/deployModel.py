from gaitExtraction import gait_extraction, gait_checking
from handExtraction import hand_extraction, hand_checking
from voiceFeatureExtraction import voice_features_extraction, sound_checking
import handFeaturesExtraction
import gaitFeaturesExtraction
from pd_calibration_only import calibrate_new_predictions
import os
from datetime import datetime
import joblib
import numpy as np
import matplotlib.pyplot as plt
import cv2


import settings

MODEL_PATHS = str(settings.MODEL_PATHS)
if not MODEL_PATHS.endswith(("/", "\\")):
    MODEL_PATHS += "/"
CALIBRATOR_PATH = os.path.join(os.path.dirname(__file__), "pd_calibrators.json")
HAND_LANDMARK_PATH = getattr(settings, 'HAND_LANDMARK_PATH', os.path.join(os.path.dirname(__file__), 'handLandmarks'))
GAIT_LANDMARK_PATH = getattr(settings, 'GAIT_LANDMARK_PATH', os.path.join(os.path.dirname(__file__), 'gaitLandmarks'))
HAND_LANDMARK_PATH = str(HAND_LANDMARK_PATH)
GAIT_LANDMARK_PATH = str(GAIT_LANDMARK_PATH)
if not HAND_LANDMARK_PATH.endswith(('/', '\\')):
    HAND_LANDMARK_PATH += '/'
if not GAIT_LANDMARK_PATH.endswith(('/', '\\')):
    GAIT_LANDMARK_PATH += '/'
TRAINED_MODELS_LS = ["RF"]  # remove KNN, "GBM", "LightGBM", "C4.5 DT", "LogReg", "NB", "RF", "AdaBoost",

gait_feature_name = ['left_foot_ground', 'right_foot_ground', 'left_right_foot_len_average', 'left_right_foot_len_max',
                     'left_turning_duration', 'left_turning_slope', 'right_turning_duration', 'right_turning_slope',
                     'l_leg_max_angles', 'l_leg_min_angles', 'r_leg_max_angles', 'r_leg_min_angles', 'l_arm_max_angles',
                     'l_arm_min_angles', 'r_arm_max_angles', 'r_arm_min_angles', 'core_max_angles', "core_min_angles",
                     'average_duration_per_rounds', "duration_change", "l_mean_steps", "r_mean_steps"]

hand_features_name = ["Right Tapping Time", "Right Tapping Time Change", "Right Tapping Distance", "Right Tapping Frequency",
                      "Right Tapping Intensity", "Right Tapping Power", "Right Tapping Frequency Change", "Left Tapping Time",
                      "Left Tapping Time Change", "Left Tapping Distance", "Left Tapping Frequency", 
                      "Left Tapping Intensity", "Left Tapping Power", "Left Tapping Frequency Change"]

voice_feature_name = ['reading_time' , 'score', 'pause(%)', 'volume change', 'pitch change', 'Average pitch']


def _landmark_dir(path):
    """Normalize a landmark directory path with a trailing separator."""
    path = str(path)
    if not path.endswith(("/", "\\")):
        path += "/"
    return path


def extract_gait(gait_video_pth, out_dir):
    gait_extraction(gait_video_pth, out_dir)
    

def gait_features_extraction(gait_video_pth, fps, out_dir, debug=False, gait_landmark_dir=None):
    gait_dir = _landmark_dir(gait_landmark_dir or GAIT_LANDMARK_PATH)
    pth_2d = f"{gait_dir}2d_{os.path.basename(gait_video_pth)[:-4]}.npy"
    pth_3d = f"{gait_dir}3d_{os.path.basename(gait_video_pth)[:-4]}.npz"
    
    gait_feature = gaitFeaturesExtraction.pose_features_extract(pth_2d, pth_3d, fps=fps, plot_results=True,
                                                                save_fig_pth=f"{out_dir}vis_gait_extraction.png", debug=debug)
    np.save(f'{out_dir}gait_feature.npy', gait_feature)
    return gait_feature


def extract_hand(hand_video_pth, out_dir, hand):
    hand_extraction(hand_video_pth, out_video_root=out_dir, hand=hand)


def hand_features_extraction(left_video_path, right_video_path, fps, out_dir, debug=False, hand_landmark_dir=None):
    hand_dir = _landmark_dir(hand_landmark_dir or HAND_LANDMARK_PATH)

    if debug:
        print("=" * 70)
        print("[hand_features_extraction] Start")
        print(f"    left_video_path : {left_video_path}")
        print(f"    right_video_path: {right_video_path}")
        print(f"    fps             : {fps}")
        print(f"    out_dir         : {out_dir}")

    # =========================================================
    # 1. Construct landmark file paths
    # =========================================================
    if debug:
        print("\n[STEP 1] Build landmark file paths")

    r_name = f"right_hand_{os.path.basename(right_video_path)[:-4]}.txt"
    l_name = f"left_hand_{os.path.basename(left_video_path)[:-4]}.txt"

    r_path = os.path.join(hand_dir, r_name)
    l_path = os.path.join(hand_dir, l_name)

    if debug:
        print(f"    right landmark path: {r_path}")
        print(f"    left landmark path : {l_path}")

    # Check if files exist
    if not os.path.exists(r_path):
        raise FileNotFoundError(f"Right hand landmark file not found: {r_path}")

    if not os.path.exists(l_path):
        raise FileNotFoundError(f"Left hand landmark file not found: {l_path}")

    if debug:
        print("    landmark files exist")

    # =========================================================
    # 2. Extract hand features
    # =========================================================
    if debug:
        print("\n[STEP 2] Extract hand features")

    hand_feature = handFeaturesExtraction.single_thumb_index_hand(
        r_path,
        l_path,
        out_dir,
        fps=fps,
        debug=debug
    )

    if debug:
        print(f"    hand_feature type: {type(hand_feature)}")
        try:
            print(f"    hand_feature length: {len(hand_feature)}")
        except Exception:
            print("    hand_feature length: not available")

    # =========================================================
    # 3. Save hand features
    # =========================================================
    save_path = os.path.join(out_dir, "hand_feature.npy")

    if debug:
        print("\n[STEP 3] Save hand features")
        print(f"    save path: {save_path}")

    np.save(save_path, hand_feature)

    if debug:
        print("    hand features saved successfully")

    if debug:
        print("\n[hand_features_extraction] Finished")
        print("=" * 70)

    return hand_feature


def features_extraction(
    gait_video_pth,
    left_video_path,
    right_video_path,
    voice_path,
    out_dir,
    debug=False,
    gait_landmark_dir=None,
    hand_landmark_dir=None,
):
    """Build gait, hand, and voice features and save ``all_feature.npy`` under ``out_dir``.

    Expects gait and hand landmarks to already exist (default dirs: ``GAIT_LANDMARK_PATH``,
    ``HAND_LANDMARK_PATH``). Does not run ``extract_gait`` or ``extract_hand``.

    For a full local run (extract landmarks first), use ``model_extraction`` instead.
    """
    if debug:
        print("=" * 70)
        print("[features_extraction] Start features extraction")
        print("=" * 70)
        print("\n[DEBUG] Input paths")
        print(f"    gait_video_pth : {gait_video_pth}")
        print(f"    left_video_path: {left_video_path}")
        print(f"    right_video_path: {right_video_path}")
        print(f"    voice_path     : {voice_path}")
        print(f"    out_dir        : {out_dir}")

    # --- Ensure output directory exists ---
    os.makedirs(out_dir, exist_ok=True)
    
    if debug:
        print("\n[STEP 1] Output directory check")
        print(f"    ensured output directory exists: {out_dir}")

    # =========================================================
    # 1. Voice feature extraction
    # =========================================================


    voice_feature = voice_features_extraction(voice_path, debug=debug)

    voice_save_path = os.path.join(out_dir, "voice_feature.npy")
    
    if debug:
        print("\n[STEP 3] Voice feature extraction finished")
        print(f"    voice_feature type   : {type(voice_feature)}")
        try:
            print(f"    voice_feature length : {len(voice_feature)}")
        except Exception:
            print("    voice_feature length : not available")
        print(f"    saving voice features to: {voice_save_path}")

    np.save(voice_save_path, voice_feature)

    # =========================================================
    # 2. Read gait FPS
    # =========================================================

    cap = cv2.VideoCapture(gait_video_pth)
    
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {gait_video_pth}")

    gait_fps = int(cap.get(cv2.CAP_PROP_FPS))
    gait_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    gait_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    gait_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    if gait_fps <= 0:
        raise ValueError(f"Invalid FPS: {gait_video_pth}")

    if debug:
        print(f"    gait_fps        : {gait_fps}")
        print(f"    gait_frame_count: {gait_frame_count}")
        print(f"    gait_resolution : {gait_width} x {gait_height}")
        if gait_fps > 0:
            print(f"    gait_duration   : {gait_frame_count / gait_fps:.3f} sec")

    # =========================================================
    # 3. Gait feature extraction
    # =========================================================

    gait_feature = gait_features_extraction(
        gait_video_pth, gait_fps, out_dir, debug=debug, gait_landmark_dir=gait_landmark_dir
    )

    if debug:
        print("\n[STEP 6] Gait feature extraction finished")
        print(f"    gait_feature type   : {type(gait_feature)}")
        try:
            print(f"    gait_feature length : {len(gait_feature)}")
        except Exception:
            print("    gait_feature length : not available")

    # =========================================================
    # 4. Read hand FPS from left video
    # =========================================================

    cap = cv2.VideoCapture(left_video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {left_video_path}")

    hand_fps = int(cap.get(cv2.CAP_PROP_FPS))
    hand_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    hand_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    hand_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    if hand_fps <= 0:
        raise ValueError(f"Invalid FPS: {left_video_path}")

    if debug:
        print(f"    hand_fps        : {hand_fps}")
        print(f"    hand_frame_count: {hand_frame_count}")
        print(f"    hand_resolution : {hand_width} x {hand_height}")
        if hand_fps > 0:
            print(f"    hand_duration   : {hand_frame_count / hand_fps:.3f} sec")

    # =========================================================
    # 5. Hand feature extraction
    # =========================================================

    hand_feature = hand_features_extraction(
        left_video_path,
        right_video_path,
        hand_fps,
        out_dir,
        debug=debug,
        hand_landmark_dir=hand_landmark_dir,
    )

    if debug:
        print("\n[STEP 9] Hand feature extraction finished")
        print(f"    hand_feature type   : {type(hand_feature)}")
        try:
            print(f"    hand_feature length : {len(hand_feature)}")
        except Exception:
            print("    hand_feature length : not available")

    # =========================================================
    # 6. Merge all features
    # =========================================================

    if debug:
        try:
            print(f"    gait_feature length : {len(gait_feature)}")
        except Exception:
            print("    gait_feature length : not available")

        try:
            print(f"    hand_feature length : {len(hand_feature)}")
        except Exception:
            print("    hand_feature length : not available")

        try:
            print(f"    voice_feature length: {len(voice_feature)}")
        except Exception:
            print("    voice_feature length: not available")

    all_features = gait_feature + hand_feature + voice_feature

    if debug:
        print(f"    all_features type   : {type(all_features)}")
        try:
            print(f"    all_features length : {len(all_features)}")
        except Exception:
            print("    all_features length : not available")

    # =========================================================
    # 7. Save all features
    # =========================================================
    all_save_path = os.path.join(out_dir, "all_feature.npy")
    np.save(all_save_path, all_features)

    return all_features

    
def predict_sound(all_features_pth, age, gender):
    sfs_idx = joblib.load(f'{MODEL_PATHS}nvp_sfs_idx.txt')
    voice_len = len(voice_feature_name)

    data = np.load(all_features_pth)
    voice_feature = np.concatenate([np.array([age, gender]), data[-voice_len:]])
    voice_result = deploy(np.array([voice_feature]), sfs_idx['voice_sfs_idx'], modal="voice", fold=10)

    return voice_result


def predict_gait(all_features_pth, age, gender):
    sfs_idx = joblib.load(f'{MODEL_PATHS}nvp_sfs_idx.txt')
    gait_len = len(gait_feature_name) 

    data = np.load(all_features_pth)
    gait_feature = np.concatenate([np.array([age, gender]), data[:gait_len]])
    gait_result = deploy(np.array([gait_feature]), sfs_idx['gait_sfs_idx'], modal="gait", fold=10)

    return gait_result

def predict_hand(all_features_pth, age, gender):
    sfs_idx = joblib.load(f'{MODEL_PATHS}nvp_sfs_idx.txt')
    gait_len = len(gait_feature_name)
    hand_len = len(hand_features_name)

    data = np.load(all_features_pth)
    hand_feature = np.concatenate([np.array([age, gender]), data[gait_len:gait_len + hand_len]])
    hand_result = deploy(np.array([hand_feature]), sfs_idx['hand_sfs_idx'], modal="hand", fold=10)

    return hand_result


def predict_models(all_features_pth, age, gender, out_dir=""):
    sfs_idx = joblib.load(f'{MODEL_PATHS}nvp_sfs_idx.txt')
    # _temp_voice_sfs_idx = joblib.load(f'/home/pdapp/voice_noscore_sfs.txt')

    gait_len = len(gait_feature_name)
    hand_len = len(hand_features_name)
    voice_len = len(voice_feature_name)

    data = np.load(all_features_pth)
    gait_feature = np.concatenate([np.array([age, gender]), data[:gait_len]])
    hand_feature = np.concatenate([np.array([age, gender]), data[gait_len:gait_len + hand_len]])
    voice_feature = np.concatenate([np.array([age, gender]), data[-voice_len:]])

    gait_result = deploy(np.array([gait_feature]), sfs_idx['gait_sfs_idx'], modal="gait", fold=10)
    hand_result = deploy(np.array([hand_feature]), sfs_idx['hand_sfs_idx'], modal="hand", fold=10)
    voice_result = deploy(np.array([voice_feature]), sfs_idx['voice_sfs_idx'], modal="voice", fold=10)
    
    weight = [4, 6, 5]

    all_result = {}

    for k in TRAINED_MODELS_LS:
        all_result[k] = np.average(np.array([[gait_result[k]], [hand_result[k]],
                                             [voice_result[k]]]), weights=weight, axis=0)[0]

    raw_results = {
        "gait": float(gait_result["RF"][0]) * 100,
        "voice": float(voice_result["RF"][0]) * 100,
        "hand": float(hand_result["RF"][0]) * 100,
        "ensemble": float(all_result["RF"][0]) * 100,
    }
    calibrated_results = calibrate_new_predictions(raw_results, CALIBRATOR_PATH)
    results = np.array([
        calibrated_results["gait"],
        calibrated_results["hand"],
        calibrated_results["voice"],
        calibrated_results["ensemble"],
    ]) * 100
        
    if out_dir:

        # plot results
        plt.figure()
        plt.bar(["Gait", "Hand", "Voice", "All"], results)

        for i, r in enumerate(results):
            plt.text(i - 0.2, 105, f"{r:.2f}%")

        for i, s in enumerate(["Gait", "Hand", "Voice", "All"]):
            plt.text(i - 0.2, 115, f"{s}")

        plt.ylim([0, 120])
        plt.axis("off")
        plt.savefig(f"{out_dir}result.png")
        plt.close()

    return results


def deploy(data, feature_idx_dt, modal="gait", fold=10):
    save_file_dir = f"{MODEL_PATHS}Save_model_{modal}/"
    proba_out_dt = {}
    
    for model_name in TRAINED_MODELS_LS:
        proba_ls = []

        for i in range(fold):

            clf_pth = f'{save_file_dir}{i}_{model_name}.joblib'
            clf = joblib.load(clf_pth)
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


def model_extraction(
    gait_video_pth,
    left_video_path,
    right_video_path,
    voice_path,
    out_dir,
    debug=False,
    skip_landmark_extraction=False,
    gait_landmark_dir=None,
    hand_landmark_dir=None,
):
    """Full pipeline: optional landmark extraction, then multimodal feature extraction.

    By default (``skip_landmark_extraction=False``):
      1. ``extract_gait`` → ``gait_landmark_dir`` (default ``GAIT_LANDMARK_PATH``)
      2. ``extract_hand`` (left + right) → ``hand_landmark_dir`` (default ``HAND_LANDMARK_PATH``)
      3. ``features_extraction`` → saves features under ``out_dir``

    Set ``skip_landmark_extraction=True`` when landmarks are already on disk (same as the
    Django API calling ``features_extraction`` after background upload jobs).

    Returns:
        Combined feature list (also written to ``{out_dir}/all_feature.npy``).
    """
    gait_dir = _landmark_dir(gait_landmark_dir or GAIT_LANDMARK_PATH)
    hand_dir = _landmark_dir(hand_landmark_dir or HAND_LANDMARK_PATH)

    if not skip_landmark_extraction:
        os.makedirs(gait_dir, exist_ok=True)
        os.makedirs(hand_dir, exist_ok=True)
        if debug:
            print("[model_extraction] Extracting gait landmarks...")
        extract_gait(gait_video_pth, gait_dir)
        if debug:
            print("[model_extraction] Extracting hand landmarks...")
        extract_hand(left_video_path, hand_dir, "left")
        extract_hand(right_video_path, hand_dir, "right")

    return features_extraction(
        gait_video_pth,
        left_video_path,
        right_video_path,
        voice_path,
        out_dir,
        debug=debug,
        gait_landmark_dir=gait_dir,
        hand_landmark_dir=hand_dir,
    )

