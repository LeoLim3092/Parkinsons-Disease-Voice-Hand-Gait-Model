import os
from voiceFeatureExtraction import *
import json
import numpy as np
import subprocess
import soundfile as sf
import os
import pysptk
import matplotlib.pyplot as plt

result_dt = {}
error_ls = []

json_file = 'audiodata_20250603.json'

# Load existing data if the file exists
if os.path.exists(json_file):
    with open(json_file, 'r') as f:
        result_dt = json.load(f)

# Walk through directory
for r, f, files in os.walk("/HDD3/leo/Data/PD/AudioOnly/"):
    for file in files:
        if file.endswith(".mp3") and file not in result_dt:
            file_pth = os.path.join(r, file)
            try:
                print(f"Processing file: {file}", end='\r')
                result_dt[file] = voice_features_extraction(file_pth)

                # Save after each successful extraction
                with open(json_file, 'w') as f:
                    json.dump(result_dt, f, indent=2)

            except Exception as e:
                print(f"Error processing {file}: {e}")
                error_ls.append(file)