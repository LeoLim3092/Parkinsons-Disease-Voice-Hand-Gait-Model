from pathlib import Path, PureWindowsPath

# Base directory for this package
_PDMODEL_DIR = Path(__file__).resolve().parent

# Path
# models path
MODEL_PATHS = _PDMODEL_DIR / "PD_pretrained_models"

sfs_pth = str(MODEL_PATHS / "nvp_sfs_idx.txt")
temp_voice_sfs_pth = str(MODEL_PATHS / "voice_noscore_sfs.txt")

# Landmark storage (override for production paths, e.g. /mnt/pd_app/handLandmarks/)
HAND_LANDMARK_PATH = _PDMODEL_DIR / "handLandmarks"
GAIT_LANDMARK_PATH = _PDMODEL_DIR / "gaitLandmarks"

det_config_pth = str(_PDMODEL_DIR / "checkpoint" / "faster_rcnn_r50_fpn_coco.py")
det_checkpoint_pth = str(_PDMODEL_DIR / "checkpoint" / "faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth")
pose_config_pth = str(_PDMODEL_DIR / "checkpoint" / "hrnet_w48_coco_wholebody_384x288_dark_plus.py")
pose_checkpoint_pth = str(_PDMODEL_DIR / "checkpoint" / "hrnet_w48_coco_wholebody_384x288_dark-f5726563_20200918.pth")

chk_pth = str(_PDMODEL_DIR / "checkpoint" / "detected81f.bin")
