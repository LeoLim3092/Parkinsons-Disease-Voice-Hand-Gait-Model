from gaitExtraction import gait_extraction
import gaitFeaturesExtraction
import os
import re
import json

ls_2d = []
ls_3d = []

for r, f, files in os.walk("/HDD3/Leo/Pose2D/ViTPose/TW_gait_2D_pose_data/"):
    for file in files:
        if file[-4:] == ".npy":
            ls_2d.append(file)
            
for r, f, files in os.walk("/HDD3/Leo/Pose2D/TW_gait_3D_pose_data/"):
    for file in files:
        if file[-4:] == ".npz":
            ls_3d.append(file)
            
pth_2d = "/HDD3/Leo/Pose2D/ViTPose/TW_gait_2D_pose_data/"
pth_3d = "/HDD3/Leo/Pose2D/TW_gait_3D_pose_data/"
json_file = 'gaitdata_20250605.json'
result_dt = {}
error_ls = []

for file2d in ls_2d:
    for file3d in ls_3d:
        if file2d[:-4] in file3d:
            file_3d = file3d
    file_pth_2d = f"{pth_2d}{file2d}"
    file_pth_3d = f"{pth_3d}{file_3d}"
    full_id = "_".join(file2d[:-4].split("_")[1:])
    # Extract only date_digits, e.g., 20200611_20
    
    match = re.match(r'^(\d+_\d+)', full_id)
    pid_name = match.group(1) if match else full_id
    
    try:
        gait_feature = gaitFeaturesExtraction.pose_features_extract(file_pth_2d, file_pth_3d,  plot_results=True, save_fig_pth=f"./Test/Gait/{pid_name}.png")
        result_dt[pid_name] = gait_feature
        
        # Save after each successful extraction
        with open(json_file, 'w') as f:
            json.dump(result_dt, f, indent=2)
            
    except Exception as e:
                print(f"Error processing {pid_name}: {e}")
                error_ls.append(pid_name)
