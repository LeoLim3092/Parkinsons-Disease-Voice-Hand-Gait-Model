from pathlib import PureWindowsPath

# Path
# models path

MODEL_PATHS = PureWindowsPath("../pd_model/pdmodel/PD_pretrained_models")

sfs_pth = '../pd_model/pdmodel/PD_pretrained_models/nvp_sfs_idx.txt'
temp_voice_sfs_pth = '../pd_model/pdmodel/PD_pretrained_models/voice_noscore_sfs.txt'

det_config_pth = "../pd_model/pdmodel/checkpoint/faster_rcnn_r50_fpn_coco.py"
det_checkpoint_pth = "../pd_model/pdmodel/checkpoint/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth"
pose_config_pth = "../pd_model/pdmodel/checkpoint/hrnet_w48_coco_wholebody_384x288_dark_plus.py"
pose_checkpoint_pth = "../pd_model/pdmodel/checkpoint/hrnet_w48_coco_wholebody_384x288_dark-f5726563_20200918.pth"

chk_pth = r'../pd_model/pdmodel/checkpoint/detected81f.bin'
