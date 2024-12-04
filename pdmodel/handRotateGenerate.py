import cv2
import mediapipe as mp
import time
import numpy as np
from matplotlib import pyplot as plt
# 初始化 Mediapipe Hands 模組和繪圖工具
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# 打開影片文件（替換為您的影片路徑）
video_path = '/HDD3/PD_Data/TW/PD_Data_Hand/20200429/20200429_1BL.mp4'  # 請替換為你自己的影片路徑
cap = cv2.VideoCapture(video_path)

# 創建 VideoWriter 用於保存辨識後的影片
output_path = 'output_video.mp4'
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# 計算程式執行時間
start_time = time.time()
out_dt = {}

# 初始化 Mediapipe Hands 模組
with mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5) as hands:
    i=0
    while cap.isOpened():
        ret, frame = cap.read()
        
        if not ret:
            break

        # 將 BGR 顏色空間轉換為 RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 使用 Mediapipe Hands 進行手勢檢測
        results = hands.process(frame_rgb)
        
        all_hand = []
        
        # 如果檢測到手部
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
                
        out.write(frame)

        # 顯示影像
        # cv2.imshow('Hand Gesture Recognition', frame)

        # # 將處理後的影像寫入輸出影片
        

        # # 按 'q' 鍵退出
        # if cv2.waitKey(10) & 0xFF == ord('q'):
        #     break


four_landmark_x = []

for frame in out_dt:
    four_landmark_x.append(out_dt[frame][0][4][0])

print(four_landmark_x)
# 計算並顯示總執行時間
end_time = time.time()
execution_time = end_time - start_time
print(f"Total execution time: {execution_time:.2f} seconds")

# 釋放資源
cap.release()
out.release()
cv2.destroyAllWindows()

plt.figure(figsize=(10, 6))
plt.plot(range(len(four_landmark_x)), four_landmark_x, label='four_landmark_x')
        
plt.xlabel('Frame Index')
plt.ylabel('four_landmark_x')
plt.title('four_landmark_x')
plt.legend()
plt.grid(True)
plt.savefig("four landmark output.png")


