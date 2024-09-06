import os
import pandas as pd
import cv2
import numpy as np
import time
from scipy.signal import find_peaks, correlate

def autocorr(x):
    result = correlate(x, x)
    return result[int(result.size / 2):]

def find_period(arr):
    mean = np.mean(arr)
    acf_result = autocorr(arr - mean)
    peaks, _ = find_peaks(acf_result)

    if peaks[0] > 50:
        period = peaks[0]
    else:
        period = peaks[1]

    return period, acf_result

def sobel_variance(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_skin = np.array([0, 48, 80])
    upper_skin = np.array([20, 255, 255])
    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    result = cv2.bitwise_and(image, image, mask=mask)
    
    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel_magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    sobel_magnitude = cv2.convertScaleAbs(sobel_magnitude)
    
    sobel_magnitude[mask == 0] = 0
    
    return np.var(sobel_magnitude[sobel_magnitude != 0])

def get_edge_frequency_and_time(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()
    non_zero_var = []

    frame_number = 0
    total_sobel_time = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_number += 1
        
        start_time = time.time()
        sobel_var = sobel_variance(frame)
        sobel_time = time.time() - start_time
        
        non_zero_var.append(sobel_var)
        total_sobel_time += sobel_time
        print(f'Processed frame {frame_number}, Sobel Variance: {sobel_var:.2f}, Time: {sobel_time:.4f} seconds')
    
    cap.release()
    
    period, acf_result = find_period(non_zero_var)
    
    return 1.0 / period, total_sobel_time

def clarity(wave):
    try:
        mean = np.mean(wave)
        gacr = autocorr(wave - mean)
        N = 10
        gacr = np.convolve(gacr, np.ones(N) / N, mode='valid')
        peak_id, _ = find_peaks(gacr, height=np.mean(gacr))
        peaks = gacr[peak_id]
        
        if peaks[0] > peaks[1]:
            new_peaks_id, _ = find_peaks(gacr, height=np.mean(gacr), distance=peak_id[0] * 0.3)
        else:
            new_peaks_id, _ = find_peaks(gacr, height=np.mean(gacr), distance=peak_id[np.argmax(peaks[:3])] * 0.3)

        new_peaks = gacr[new_peaks_id]
        
        return gacr[new_peaks_id[0]] / np.max(gacr[:new_peaks_id[0]])
        
    except:
        return np.nan

def update_csv(video_name, new_psnr_time, new_ssim_time, csv_file):
    df = pd.read_csv(csv_file)
    
    row_index = df.index[df['video_name'] == video_name].tolist()
    
    if row_index:
        df.loc[row_index, 'total_psnr_time'] = new_psnr_time
        df.loc[row_index, 'total_ssim_time'] = new_ssim_time
        
        df.to_csv(csv_file, index=False)
        print(f'Updated total_psnr_time and total_ssim_time for video_name: {video_name}')
    else:
        print(f'Video name {video_name} not found in CSV file.')

# dates = "20200521"
dates = """
20200429  20200521  20200611  20200707  20200716  20200730  20200818  20200915
"""
# 20200429  20200521  20200611  20200707  20200716  20200730  20200818  20200915  20201008  20201023  20201103  20201117  20201201  20201218
# 20200430  20200528  20200618  20200709  20200721  20200806  20200825  20200929  20201013  20201027  20201106  20201124  20201208  20201222
# 20200514  20200604  20200702  20200714  20200723  20200811  20200908  20201006  20201015  20201029  20201110  20201127  20201211  20201229

date_list = [f"{date}" for date in dates.split() if date]

csv_path = 'edge_clarity.csv'
df = pd.read_csv(csv_path)

for date in date_list:
    directory = f'/HDD3/PD_Data/TW/PD_Data_Hand/{date}'
    video_names = os.listdir(directory)
    video_names = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f)) and not f.startswith('.') and "A" not in f]
    print(video_names)
    for video_name in video_names:
        video_path = f"/HDD3/PD_Data/TW/PD_Data_Hand/{date}/{video_name}"
        # psnr_freq, ssim_freq, total_psnr_time, total_ssim_time = get_winer_frequency_and_time(video_path=video_path)
        edge_detect_freq, edge_detect_time = get_edge_frequency_and_time(video_path)
        # new_row = {'video_name': video_name, 'winer_psnr_freq': psnr_freq, 'winer_ssim_freq': ssim_freq, 
        #            'total_psnr_time' : total_psnr_time, 'total_ssim_time' : total_ssim_time,
        #            'edge_detect_freq' : edge_detect_freq, 'edge_detect_time' : edge_detect_time}
        
        new_row = {'video_name': video_name, 'edge_detect_freq' : edge_detect_freq, 'edge_detect_time' : edge_detect_time}
        df = df.append(new_row, ignore_index=True)
        
        df.to_csv(csv_path, index=False)
        
