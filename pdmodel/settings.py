from pathlib import PureWindowsPath

# Path
# models path

MODEL_PATHS = PureWindowsPath("../Parkinsons-Disease-Voice-Hand-Gait-Model-master/pdmodel/PD_pretrained_models")

sfs_pth = '../Parkinsons-Disease-Voice-Hand-Gait-Model-master/pd_model/PD_pretrained_models/nvp_sfs_idx.txt'
temp_voice_sfs_pth = '../Parkinsons-Disease-Voice-Hand-Gait-Model-master/pdmodel/PD_pretrained_models/voice_noscore_sfs.txt'

det_config_pth = "/HDD3/leo/NTUH_PD/Parkinsons-Disease-Voice-Hand-Gait-Model-master/pdmodel/checkpoint/faster_rcnn_r50_fpn_coco.py"
det_checkpoint_pth = "/HDD3/leo/NTUH_PD/Parkinsons-Disease-Voice-Hand-Gait-Model-master/pdmodel/checkpoint/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth"
pose_config_pth = "/HDD3/leo/NTUH_PD/Parkinsons-Disease-Voice-Hand-Gait-Model-master/pdmodel/checkpoint/hrnet_w48_coco_wholebody_384x288_dark_plus.py"
pose_checkpoint_pth = "/HDD3/leo/NTUH_PD/Parkinsons-Disease-Voice-Hand-Gait-Model-master/pdmodel/checkpoint/hrnet_w48_coco_wholebody_384x288_dark-f5726563_20200918.pth"

chk_pth = r'/HDD3/leo/NTUH_PD/Parkinsons-Disease-Voice-Hand-Gait-Model-master/pdmodel/checkpoint/detected81f.bin'
