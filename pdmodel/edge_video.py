import cv2
import numpy as np
import matplotlib.pyplot as plt

def sobel_variance(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 定義皮膚色的 HSV 範圍
    lower_skin = np.array([0, 48, 80])
    upper_skin = np.array([20, 255, 255])

    # 創建遮罩
    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    
    result = cv2.bitwise_and(image, image, mask=mask)

    # 使用遮罩提取皮膚色區域
    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel_magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    sobel_magnitude_mask = cv2.convertScaleAbs(sobel_magnitude)
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel_magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    sobel_magnitude = cv2.convertScaleAbs(sobel_magnitude)
    
    sobel_magnitude[sobel_magnitude_mask == 0] = 0
    
    return np.var(sobel_magnitude[sobel_magnitude != 0])

video_path = '20200429_1BL.mp4'  # 请替换为你的视频路径
cap = cv2.VideoCapture(video_path)

# 获取视频的基本信息
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# 定义输出视频编解码器及其输出文件名
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 或者使用 'XVID', 'MJPG', 'X264', 'XVID'
out = cv2.VideoWriter('output_video.mp4', fourcc, fps, (width, height))

# 检查视频是否打开成功
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()
non_zero_var = []

# 逐帧读取视频并进行处理
frame_number = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 处理当前帧 (例如，绘制帧号)
    frame_number += 1
    sobel_var = sobel_variance(frame)
    non_zero_var.append(sobel_var)
    text = str(frame_number) + " " + str(sobel_var)
    
    # 在帧上绘制文本
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, text, (50, 50), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # 将处理后的帧写入输出视频文件
    out.write(frame)

# 释放所有资源
cap.release()
out.release()
cv2.destroyAllWindows()

plt.figure(figsize=(10, 6))
plt.plot(range(len(non_zero_var)), non_zero_var, label='Sobel Variance')

# # 设置 x 轴刻度间隔
# plt.xticks(ticks=np.arange(0, len(non_zero_var), step=max(1, len(non_zero_var) // 100)))

plt.xlabel('Frame Index')
plt.ylabel('Sobel Variance')
plt.title('Sobel Variance Over Time')
plt.legend()
plt.grid(True)
plt.show()

print("视频处理完成并保存为 output_video.mp4")