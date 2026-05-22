import os
import json
import joblib
import numpy as np
import cv2
import ast
from typing import Tuple, List

from handExtraction import hand_extraction
import handFeaturesExtraction
from deployModel import MODEL_PATHS, hand_features_name, TRAINED_MODELS_LS, deploy


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Directory that contains the input hand videos
# Expected file names (default):
#   <ID>_left_hand.mp4
#   <ID>_right_hand.mp4
BASE_HAND_VIDEO_DIR = "/mnt/other/hand"

# Directory where all intermediate and output .npy / joblib files will be saved
BASE_RESULT_DIR = "/mnt/other/results"

# Output summary file (per-ID hand probability) under /mnt/other/
SUMMARY_JSON_PATH = "/mnt/other/hand_prediction_proba.json"
SUMMARY_NPY_PATH = "/mnt/other/hand_prediction_proba.npy"

# Video name patterns – adjust if your filenames differ
LEFT_VIDEO_PATTERN = "{id}_left_hand.mp4"
RIGHT_VIDEO_PATTERN = "{id}_right_hand.mp4"


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def build_video_paths(pid: str) -> Tuple[str, str]:
    """
    Build left/right hand video paths for a given ID.

    Adjust LEFT_VIDEO_PATTERN / RIGHT_VIDEO_PATTERN above if needed.
    """
    left_video = os.path.join(BASE_HAND_VIDEO_DIR, LEFT_VIDEO_PATTERN.format(id=pid))
    right_video = os.path.join(BASE_HAND_VIDEO_DIR, RIGHT_VIDEO_PATTERN.format(id=pid))
    return left_video, right_video


def extract_hand_landmarks_for_id(pid: str) -> Tuple[str, str]:
    """
    Run MediaPipe hand landmark extraction for a single ID.

    Returns:
        (right_landmark_path, left_landmark_path)
    """
    left_video, right_video = build_video_paths(pid)

    out_dir = os.path.join(BASE_RESULT_DIR, pid)
    ensure_dir(out_dir)

    # Extract landmarks (joblib .txt) for left and right hands.
    # These calls will save:
    #   <out_dir>/left_hand_<video-basename>.txt
    #   <out_dir>/right_hand_<video-basename>.txt
    hand_extraction(left_video, out_video_root=out_dir, hand="left")
    hand_extraction(right_video, out_video_root=out_dir, hand="right")

    left_name = f"left_hand_{os.path.basename(left_video)[:-4]}.txt"
    right_name = f"right_hand_{os.path.basename(right_video)[:-4]}.txt"

    left_path = os.path.join(out_dir, left_name)
    right_path = os.path.join(out_dir, right_name)

    return right_path, left_path


def extract_hand_features_for_id(pid: str, debug: bool = False) -> np.ndarray:
    """
    Extract hand-tapping features for a single ID and save them as hand_feature.npy.

    Returns:
        feature_array (1D numpy array)
    """
    out_dir = os.path.join(BASE_RESULT_DIR, pid)
    ensure_dir(out_dir)

    # Build original video paths to read FPS from the left-hand video
    left_video, _ = build_video_paths(pid)
    cap = cv2.VideoCapture(left_video)
    if not cap.isOpened():
        raise ValueError(f"Cannot open hand video for FPS reading: {left_video}")
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    cap.release()
    if fps <= 0:
        raise ValueError(f"Invalid FPS ({fps}) read from video: {left_video}")

    right_path, left_path = extract_hand_landmarks_for_id(pid)

    features = handFeaturesExtraction.single_thumb_index_hand(
        right_path,
        left_path,
        out_dir,
        fps=fps,
        debug=debug,
    )

    features = np.asarray(features, dtype=float)

    # Save per-ID feature file under its own subdirectory
    np.save(os.path.join(out_dir, "hand_feature.npy"), features)

    # Also save a flat copy directly under BASE_RESULT_DIR for convenience:
    #   /mnt/other/results/hand_feature_<ID>.npy
    base_feature_name = f"hand_feature_{pid}.npy"
    np.save(os.path.join(BASE_RESULT_DIR, base_feature_name), features)

    return features


def predict_hand_from_features(
    hand_features: np.ndarray,
    age: float = 0.0,
    gender: float = 0.0,
) -> dict:
    """
    Run hand-only PD prediction using the pretrained models.

    Args:
        hand_features: 1D numpy array from single_thumb_index_hand
        age: numerical age (default 0.0 if unknown)
        gender: encoded gender (0/1, default 0.0 if unknown)

    Returns:
        Dictionary: model_name -> probability_of_PD (class 1)
    """
    sfs_idx = joblib.load(os.path.join(MODEL_PATHS, "nvp_sfs_idx.txt"))

    # Feature vector structure used in the original model:
    #   [age, gender] + hand_features
    vec = np.concatenate([np.array([age, gender], dtype=float), hand_features])
    data = np.expand_dims(vec, axis=0)

    hand_result = deploy(
        data,
        sfs_idx["hand_sfs_idx"],
        modal="hand",
        fold=10,
    )

    # Convert numpy values to plain Python floats
    out = {}
    for model_name in TRAINED_MODELS_LS:
        proba_arr = hand_result[model_name]
        out[model_name] = float(proba_arr[0])

    return out


def run_batch_hand_prediction(
    id_list: List[Tuple[str, float, float]],
    debug: bool = False,
) -> dict:
    """
    Run hand-tapping prediction for a list of IDs.

    All intermediate .npy / .txt files and per-ID results are saved under:
        /mnt/other/results/<ID>/

    The final probability summary is saved under:
        /mnt/other/hand_prediction_proba.json
        /mnt/other/hand_prediction_proba.npy
    """
    ensure_dir(BASE_RESULT_DIR)

    summary = {}
    for pid, age, gender in id_list:
        try:
            print(f"[run_batch_hand_prediction] Processing ID: {pid}, age={age}, gender={gender}")

            hand_feat = extract_hand_features_for_id(pid, debug=debug)
            proba_dt = predict_hand_from_features(
                hand_feat,
                age=age,
                gender=gender,
            )

            summary[pid] = proba_dt

            # Save per-ID probability as .npy in its own folder
            id_dir = os.path.join(BASE_RESULT_DIR, pid)
            np.save(
                os.path.join(id_dir, "hand_prediction_proba.npy"),
                np.array([proba_dt], dtype=object),
            )

        except Exception as e:
            print(f"[run_batch_hand_prediction] ERROR for ID {pid}: {e}")

    # Save global summary under /mnt/other/
    try:
        with open(SUMMARY_JSON_PATH, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"Failed to write JSON summary: {e}")

    try:
        # Convert to a structured numpy representation for convenience
        ids = list(summary.keys())
        rf_vals = [summary[i]["RF"] for i in ids] if ids else []
        np.save(
            SUMMARY_NPY_PATH,
            {
                "ids": np.array(ids),
                "RF": np.array(rf_vals, dtype=float),
                "all": summary,
            },
        )
    except Exception as e:
        print(f"Failed to write NPY summary: {e}")

    return summary


if __name__ == "__main__":
    # -----------------------------------------------------------------------
    # Example usage:
    #   1. Edit ID_LIST to include the IDs you want to process.
    #   2. Ensure the hand videos exist at:
    #        /mnt/other/hand/<ID>_left_hand.mp4
    #        /mnt/other/hand/<ID>_right_hand.mp4
    #   3. Run this script with the Python environment that has all
    #      project dependencies installed.
    # -----------------------------------------------------------------------
    id_list_txt = "/mnt/other/IDlist.txt"

    if os.path.exists(id_list_txt):
        with open(id_list_txt, "r", encoding="utf-8") as f:
            content = f.read()
        try:
            ID_LIST = ast.literal_eval(content)
        except Exception as e:
            raise ValueError(f"Failed to parse ID list from {id_list_txt}: {e}")

        if not ID_LIST:
            print(f"ID list file {id_list_txt} is empty. Nothing to run.")
        else:
            print(f"Loaded {len(ID_LIST)} IDs from {id_list_txt}")
            run_batch_hand_prediction(ID_LIST, debug=False)
    else:
        print(f"ID list file not found: {id_list_txt}")


