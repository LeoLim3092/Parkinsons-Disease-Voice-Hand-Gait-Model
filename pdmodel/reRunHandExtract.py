import handFeaturesExtraction
import os
import re
import json
import numpy as np
import glob
import joblib
import cv2


path3 = r'../../handOutput3/*_A*_hand.txt'
files3_ls = glob.glob(path3)

pid_ls = []

json_file = 'handdata_20250605.json'
result_dt = {}
error_ls = []

for filepth in files3_ls:
    date = filepth.split("_")[0].split("/")[-1]
    pid = filepth.split("_")[1]

    l_hand = f"../../handOutput3/{date}_{pid}_AL_hand.txt"
    r_hand = f"../../handOutput3/{date}_{pid}_AR_hand.txt"

    if os.path.isfile(l_hand) & os.path.isfile(r_hand):
        pid_ls.append((date,pid))

pid_ls = set(pid_ls)

for i , all_id in enumerate(pid_ls):
    date = all_id[0]
    pid = all_id[1]

    pid_name = f"{date}_{pid}"

    l_hand = f"../../handOutput3/{date}_{pid}_AL_hand.txt"
    r_hand = f"../../handOutput3/{date}_{pid}_AR_hand.txt"

    video_path = f"/HDD3/leo/Data/PD/PD_Data_Hand/{date}/{date}_{pid}AR.mp4"
    
    if not os.path.isfile(video_path):
        fps = 59
    else:
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))

    try:
        arr = handFeaturesExtraction.single_thumb_index_hand(r_hand, l_hand, f"./Test/Hand/{date}_{pid}_", fps=fps)
        result_dt[pid_name] = arr
        
        # Save after each successful extraction
        with open(json_file, 'w') as f:
            json.dump(result_dt, f, indent=2)
            
    except Exception as e:
                print(f"Error processing {pid_name}: {e}")
                error_ls.append(pid_name)
    
