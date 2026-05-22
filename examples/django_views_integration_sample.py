"""
Django API integration sample for pdmodel
=========================================

This file documents how the production PD screening API uses ``pdmodel``
functions. It mirrors the pattern in:

    pd-api-Mirlab-server/api/views.py

It is **not** meant to run as-is inside this repo (no Django project here).
Copy the patterns into your Django app when embedding ``pdmodel`` as a package
(e.g. ``from api.pdModel.deployModel import ...`` with relative imports).

Production layout (Linux server)
--------------------------------
    /mnt/pd_app/
        walk/              # raw gait videos
        gesture/           # raw left/right hand videos
        sound/             # raw voice recordings
        gaitLandmarks/     # extract_gait output (2d_*.npy, 3d_*.npz)
        handLandmarks/     # extract_hand output (*_hand_*.txt)
        results/{patient}/ # features + prediction outputs per run

Path constants in views.py must match deployModel.py:

    views:     gaitlandmark_pth = "/mnt/pd_app/gaitLandmarks/"
               handlandmark_pth = "/mnt/pd_app/handLandmarks/"

    deployModel:
               GAIT_LANDMARK_PATH = "/mnt/pd_app/gaitLandmarks/"
               HAND_LANDMARK_PATH = "/mnt/pd_app/handLandmarks/"
"""

from __future__ import annotations

import datetime
import os
import threading
from typing import Optional

# ---------------------------------------------------------------------------
# In production Django (pd-api-Mirlab-server/api/views.py):
#
#   from .pdModel.deployModel import (
#       features_extraction,
#       extract_gait,
#       extract_hand,
#       predict_models,
#       data_checking,
#   )
#
# For local scripts in this repo, use:
#
#   from deployModel import ...
# ---------------------------------------------------------------------------

try:
    from deployModel import (
        data_checking,
        extract_gait,
        extract_hand,
        features_extraction,
        predict_models,
    )
except ImportError:
    # When copied into Django, switch to package imports above.
    raise ImportError(
        "Run from pdmodel/ or install the package. "
        "See examples/django_views_integration_sample.py docstring."
    )


# --- Same paths as pd-api-Mirlab-server/api/views.py (lines 43-45) -----------

BASE_PATH = "/mnt/pd_app"
GAIT_LANDMARK_PTH = f"{BASE_PATH}/gaitLandmarks/"
HAND_LANDMARK_PTH = f"{BASE_PATH}/handLandmarks/"


# =============================================================================
# 1. Upload endpoints — background landmark extraction
# =============================================================================
# Production classes: UploadWalk, UploadGesture (views.py ~193-265)
#
# Videos are saved under walk/ or gesture/, then extraction runs in a
# daemon thread so the HTTP response returns immediately.


def on_gait_video_uploaded(full_video_path: str) -> None:
    """
    Called after saving a walk video (UploadWalk.post).

    Equivalent to:
        threading.Thread(
            target=extract_gait,
            args=(full_file_path, gaitlandmark_pth),
            daemon=True,
        ).start()
    """
    threading.Thread(
        target=extract_gait,
        args=(full_video_path, GAIT_LANDMARK_PTH),
        daemon=True,
    ).start()


def on_hand_video_uploaded(full_video_path: str, side: str) -> None:
    """
    Called after saving a gesture video (UploadGesture.post).

    Args:
        full_video_path: absolute path under /mnt/pd_app/gesture/
        side: ``'left'`` or ``'right'`` (from POST ``type``: 左手 / 右手)
    """
    threading.Thread(
        target=extract_hand,
        args=(full_video_path, HAND_LANDMARK_PTH, side),
        daemon=True,
    ).start()


# =============================================================================
# 2. Quality check — before prediction
# =============================================================================
# Production class: CheckRecording (views.py ~409-456)


def check_patient_recordings(
    gait_file_pth: str,
    l_hand_file_pth: str,
    r_hand_file_pth: str,
    sound_file_pth: str,
) -> tuple[str, str]:
    """
    Returns:
        (success, error) where success is ``'success'`` or ``'failed'``.
    """
    return data_checking(gait_file_pth, l_hand_file_pth, r_hand_file_pth, sound_file_pth)


# =============================================================================
# 3. Predict — features then models (landmarks must already exist)
# =============================================================================
# Production function: run_predict_model (views.py ~824-899)
# Production class: PredictModel.post → run_predict_model(pid)


def run_predict_model(
    *,
    gait_file_pth: str,
    l_hand_file_pth: str,
    r_hand_file_pth: str,
    sound_file_pth: str,
    out_dir: str,
    age: int,
    gender: int,
    debug: bool = False,
) -> dict:
    """
    Full prediction for one patient session.

    Steps (same as production):
      1. features_extraction — reads landmarks from GAIT_/HAND_LANDMARK_PATH
      2. predict_models — RF + calibration, saves result.png if out_dir set

    Args:
        gait_file_pth, l_hand_file_pth, r_hand_file_pth, sound_file_pth:
            Absolute paths to the latest uploaded media files.
        out_dir: e.g. ``/mnt/pd_app/results/{patient_name}/{timestamp}/``
        age, gender: patient demographics for model input.

    Returns:
        dict with keys gait_result, hand_result, voice_result, multimodal_results
        (production stores these on the Results model).

    Note:
        ``predict_models`` returns a 1D numpy array
        ``[gait%, hand%, voice%, ensemble%]``. Production unpacks it as:
        ``gait_result, hand_result, voice_results, all_results = predict_models(...)``
        which works because numpy scalars unpack from a length-4 array.
    """
    os.makedirs(out_dir, exist_ok=True)

    # Does NOT call extract_gait / extract_hand — expects upload jobs finished.
    features_extraction(
        gait_file_pth,
        l_hand_file_pth,
        r_hand_file_pth,
        sound_file_pth,
        out_dir,
        debug=debug,
    )

    all_features_pth = os.path.join(out_dir, "all_feature.npy")
    gait_pct, hand_pct, voice_pct, ensemble_pct = predict_models(
        all_features_pth,
        age,
        gender,
        out_dir,
    )

    return {
        "gait_result": float(gait_pct),
        "hand_result": float(hand_pct),
        "voice_result": float(voice_pct),
        "multimodal_results": float(ensemble_pct),
        "all_features_path": all_features_pth,
        "result_plot": os.path.join(out_dir, "result.png"),
    }


# =============================================================================
# 4. Predict from cached features only
# =============================================================================
# Production: PredictWithoutModelExtraction, run_predict_from_latest_extracted_features
# (views.py ~459-476, ~902-937)


def run_predict_from_existing_features(
    all_features_pth: str,
    age: int,
    gender: int,
    out_dir: str = "",
) -> dict:
    """Re-run models when ``all_feature.npy`` already exists."""
    gait_pct, hand_pct, voice_pct, ensemble_pct = predict_models(
        all_features_pth,
        age,
        gender,
        out_dir,
    )
    return {
        "gait_result": float(gait_pct),
        "hand_result": float(hand_pct),
        "voice_result": float(voice_pct),
        "multimodal_results": float(ensemble_pct),
    }


# =============================================================================
# 5. Mapping: production APIView → pdmodel calls
# =============================================================================
"""
+-----------------------------+-----------------------------------------------+
| Django (views.py)           | pdmodel function                              |
+-----------------------------+-----------------------------------------------+
| UploadWalk                  | extract_gait(video, gaitlandmark_pth)         |
| UploadGesture               | extract_hand(video, handlandmark_pth, side) |
| CheckRecording              | data_checking(gait, L, R, sound)              |
| PredictModel                | run_predict_model → features + predict        |
| PredictWithoutModelExtract. | predict_models(existing all_feature.npy)    |
| RedoPatientFeaturesAndPred. | run_predict_model (full pipeline)             |
+-----------------------------+-----------------------------------------------+

NOT used in production views:
    model_extraction()  — use in notebooks/local runs instead; production
                          splits extract (upload) vs features (predict).

Equivalent local call for full offline pipeline:
    from deployModel import model_extraction, predict_models
    model_extraction(gait, left, right, voice, out_dir)
    predict_models(f"{out_dir}all_feature.npy", age, gender, out_dir)
"""


# =============================================================================
# Example (pseudo patient paths)
# =============================================================================

if __name__ == "__main__":
    # Illustrative paths only — adjust before running on a machine with data.
    PATIENT = "demo_patient"
    TS = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    OUT = f"{BASE_PATH}/results/{PATIENT}/{TS}/"

    paths = {
        "gait": f"{BASE_PATH}/walk/example_gait.mp4",
        "left": f"{BASE_PATH}/gesture/example_left.mp4",
        "right": f"{BASE_PATH}/gesture/example_right.mp4",
        "sound": f"{BASE_PATH}/sound/example_voice.wav",
    }

    print("=== Django-style flow (production) ===")
    print("1) On upload:")
    print("   on_gait_video_uploaded(paths['gait'])")
    print("   on_hand_video_uploaded(paths['left'], 'left')")
    print("   on_hand_video_uploaded(paths['right'], 'right')")
    print("   # wait for background threads to finish")
    print("2) On predict:")
    print("   run_predict_model(..., out_dir=OUT, age=65, gender=1)")

    print("\n=== Standalone flow (this repo / notebooks) ===")
    print("   model_extraction(...)  # extract + features in one call")
    print("   predict_models(...)")
