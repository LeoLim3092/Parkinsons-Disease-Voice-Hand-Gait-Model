import os
from matplotlib import pyplot as plt
import pandas as pd
import cv2
import numpy as np
import time
from scipy.signal import find_peaks, correlate, detrend
import mediapipe as mp
import argparse

# process video
def process_video_by_edgede(video_path):
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
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()
    non_zero_var = []

    frame_number = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_number += 1
        
        sobel_var = sobel_variance(frame)
        
        non_zero_var.append(sobel_var)
        
    print(f'process video by edge detection compelete, wave length : {len(non_zero_var)}')
    return non_zero_var
        
def process_video_by_mediapipe(video_path, landmark_point = 4):
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    
    cap = cv2.VideoCapture(video_path)
    out_dt = {}
    
    with mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5) as hands:
        i=0
        while cap.isOpened():
            ret, frame = cap.read()
            
            if not ret:
                break
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            results = hands.process(frame_rgb)
            
            all_hand = []
            
            if results.multi_hand_landmarks:
                for h_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    hand_ls = []
                    for l_idx, landmark in enumerate(hand_landmarks.landmark):
                        hand_ls.append([landmark.x, landmark.y, landmark.z])

                    hand_arr = np.array(hand_ls)
                    all_hand.append(hand_arr)
                    break
                i += 1
                out_dt[f"{i}"] = all_hand
                
    landmark_x = []

    for frame in out_dt:
        landmark_x.append(out_dt[frame][0][landmark_point][0])
    
    print(f'process video by mediapipe compelete, wave length : {len(landmark_x)}')
    return landmark_x

# mid process
def mid_by_poly_detrend(wave, degree = 10):
    t = np.arange(len(wave))
    t_normalized = (t - t.min()) / (t.max() - t.min())
    
    p = np.polyfit(t_normalized, wave, degree)
    trend = np.polyval(p, t_normalized)
    
    detrend = wave - trend
    
    print(f'mid process by poly detrend compelete')
    return detrend

# get feature
def get_feature_by_autocorr(wave):
    def autocorr(x):
        result = correlate(x, x)
        return result[int(result.size / 2):]
    
    autocorrlation = autocorr(wave)
    
    sampling_rate = 60 # 影片取樣率

    peaks, _ = find_peaks(autocorrlation,height=0)

    if len(peaks) > 0:
        first_peak = peaks[0]
        frequency = sampling_rate / first_peak  # 計算頻率    
        clarity = autocorrlation[peaks[0]] / autocorrlation[0]
        
        print(f"autocorr freq : {frequency} clarity : {clarity}")
        return frequency, clarity
    else:
        return 0, 0

def get_feature_by_fft(wave):
    velocity = np.diff(wave)
    
    fft_result = np.fft.fft(velocity)
    frequencies = np.fft.fftfreq(len(velocity))
    
    power_spectrum = np.abs(fft_result)**2
    
    max_power_index = np.argmax(power_spectrum)  # 找到最大功率對應的索引
    max_power_frequency = frequencies[max_power_index]  # 獲取對應的頻率

    print(f"The frequency with maximum power is: {max_power_frequency}")
    
    return max_power_frequency

# def find_period(arr):
#     mean = np.mean(arr)
#     acf_result = autocorr(arr - mean)
#     peaks, _ = find_peaks(acf_result)

#     if peaks[0] > 50:
#         period = peaks[0]
#     else:
#         period = peaks[1]

#     return period, acf_result


# def mediapipe_process_landmark(video_path, landmark_point):
#     cap = cv2.VideoCapture(video_path)
    
#     out_dt = {}
    
#     with mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5) as hands:
#         i=0
#         while cap.isOpened():
#             ret, frame = cap.read()
            
#             if not ret:
#                 break

#             # 將 BGR 顏色空間轉換為 RGB
#             frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

#             # 使用 Mediapipe Hands 進行手勢檢測
#             results = hands.process(frame_rgb)
            
#             all_hand = []
            
#             # 如果檢測到手部
#             if results.multi_hand_landmarks:
#                 for h_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
#                     hand_ls = []
#                     for l_idx, landmark in enumerate(hand_landmarks.landmark):
#                         hand_ls.append([landmark.x, landmark.y, landmark.z])

#                     hand_arr = np.array(hand_ls)
#                     all_hand.append(hand_arr)
#                     break
#                 i += 1
#                 out_dt[f"{i}"] = all_hand
                
#     landmark_x = []

#     for frame in out_dt:
#         landmark_x.append(out_dt[frame][0][landmark_point][0])
    
#     return landmark_x

# def poly_detrend(wave, degree):
#     t = np.arange(len(wave))
#     t_normalized = (t - t.min()) / (t.max() - t.min())
    
#     p = np.polyfit(t_normalized, wave, degree)
#     trend = np.polyval(p, t_normalized)
    
#     detrend = wave - trend
    
#     return detrend

    


# def mediapipe_4_keypoint_frequency_and_fft_speed(video_path):
#     landmark = mediapipe_process_landmark(video_path, landmark_point=4)
#     detrend = poly_detrend(wave=landmark, degree=10)
    
#     autocorrlation = autocorr(detrend)
    
#     freq, clarity = find_freq_clarity(autocorr=autocorrlation)
    
#     video_fft_speed = fft_speed(detrend)
    
#     return freq, clarity, video_fft_speed
    
    
# def get_edge_frequency_and_time_and_clarity_by_blur(video_path):
#     cap = cv2.VideoCapture(video_path)
#     if not cap.isOpened():
#         print("Error: Could not open video.")
#         exit()
#     non_zero_var = []

#     frame_number = 0
#     total_sobel_time = 0
    
#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break
#         frame_number += 1
        
#         start_time = time.time()
#         sobel_var = sobel_variance(frame)
#         sobel_time = time.time() - start_time
        
#         non_zero_var.append(sobel_var)
#         total_sobel_time += sobel_time
#         print(f'Processed frame {frame_number}, Sobel Variance: {sobel_var:.2f}, Time: {sobel_time:.4f} seconds')
    
#     cap.release()
    
#     t = np.arange(len(non_zero_var))
    
#     degree = 10  
#     p = np.polyfit(t, non_zero_var, degree)
#     trend = np.polyval(p, t)
    
#     detrended_data_poly = non_zero_var - trend
    
#     autocorrlation = autocorr(detrended_data_poly)
    
#     return find_freq_clarity(autocorr=autocorrlation)

# def find_freq_clarity(autocorr):
#     sampling_rate = 60 # 影片取樣率

#     peaks, _ = find_peaks(autocorr,height=0)

#     if len(peaks) > 0:
#         first_peak = peaks[0]
#         frequency = sampling_rate / first_peak  # 計算頻率    
#         clarity = autocorr[peaks[0]] / autocorr[0]
        
#         return frequency, clarity
#     else:
#         return 0, 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Script for processing parameters")
    parser.add_argument("--video_process", type=str, required=True, help="")
    parser.add_argument("--mid_process", type=str, required=True, help="")
    parser.add_argument("--get_feature", nargs="*" ,type=str, required=True, help="")
    
    args = parser.parse_args()
    
    video_process_method_group = {
        "edgede": process_video_by_edgede,
        "mediapipe": process_video_by_mediapipe
    }
    
    mid_process_method_group = {
        "polydetrend": mid_by_poly_detrend,
    }
    
    get_feature_method_group = {
        "autocorr" : get_feature_by_autocorr,
        "fft" : get_feature_by_fft
    }
    
    feature_dict = {
        "autocorr_freq" : None,
        "autocorr_clarity" : None,
        "fft_max_power_freq" : None
    }
    
    video_path = "/HDD3/PD_Data/TW/PD_Data_Hand/20200716/20200716_19BL.mp4"

    if args.video_process in video_process_method_group:
        wave = video_process_method_group[args.video_process](video_path)
        
    if args.mid_process in mid_process_method_group:
        processed_wave = mid_process_method_group[args.mid_process](wave=wave)
    
    for get_feature_method in args.get_feature:
        if get_feature_method == "autocorr":
            feature_dict["autocorr_freq"], feature_dict["autocorr_clarity"] = get_feature_method_group[get_feature_method](processed_wave)
        if get_feature_method == "fft":
            feature_dict["fft_max_power_freq"] = get_feature_method_group[get_feature_method](processed_wave)
    
    print(feature_dict)
        
# dates = "20200521"
# dates = """
# 20200429  20200521  20200611  20200707  20200716  20200730  20200818  20200915
# """
# 20200429  20200521  20200611  20200707  20200716  20200730  20200818  20200915  20201008  20201023  20201103  20201117  20201201  20201218
# 20200430  20200528  20200618  20200709  20200721  20200806  20200825  20200929  20201013  20201027  20201106  20201124  20201208  20201222
# 20200514  20200604  20200702  20200714  20200723  20200811  20200908  20201006  20201015  20201029  20201110  20201127  20201211  20201229

# date_list = [f"{date}" for date in dates.split() if date]

# csv_path = 'new_freq_clarity.csv'
# df = pd.read_csv(csv_path)

# for date in date_list:
#     directory = f'/HDD3/PD_Data/TW/PD_Data_Hand/{date}'
#     video_names = os.listdir(directory)
#     video_names = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f)) and not f.startswith('.') and "A" not in f]
#     print(video_names)
#     for video_name in video_names:
#         video_path = f"/HDD3/PD_Data/TW/PD_Data_Hand/{date}/{video_name}"
#         # psnr_freq, ssim_freq, total_psnr_time, total_ssim_time = get_winer_frequency_and_time(video_path=video_path)
#         edge_detect_freq, edge_detect_clarity = get_edge_frequency_and_time_and_clarity(video_path)
#         # new_row = {'video_name': video_name, 'winer_psnr_freq': psnr_freq, 'winer_ssim_freq': ssim_freq, 
#         #            'total_psnr_time' : total_psnr_time, 'total_ssim_time' : total_ssim_time,
#         #            'edge_detect_freq' : edge_detect_freq, 'edge_detect_time' : edge_detect_time}
        
#         new_row = {'video_name': video_name, 'edge_detect_freq' : edge_detect_freq, 'edge_detect_clarity' : edge_detect_clarity}
#         df = df.append(new_row, ignore_index=True)
        
#         df.to_csv(csv_path, index=False)


