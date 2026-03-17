import os
import pandas as pd
import cv2
import os
import numpy as np
import mediapipe as mp
import joblib


def hand_extraction(path, visualize=True, out_video_root="", hand='right'):
    """
    path: the path of the video need to be processed
    out_video_root: The path for saving visualized landmarks video
    """
    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands
    cap = cv2.VideoCapture(path)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    print(fps)

    if visualize:
        video_writer = cv2.VideoWriter(os.path.join(out_video_root, f'vis_{hand}_hand_{os.path.basename(path)}'),
                                    cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    out_dt = {"landmarks": {}}
    hand_id = []
    with mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5) as hands:
        i = 0
        while True:
            success, image = cap.read()
            if not success:
                # If loading a video, use 'break' instead of 'continue'.
                break;

            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image)

            # Draw the face mesh annotations on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            all_hand = []

            if not results.multi_hand_landmarks:
                out_dt["landmarks"][f"{i}"] = []
                continue

            for h_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                if results.multi_handedness[h_idx].classification[0].score > 0.95:
                    mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    hand_ls = []
                    hand_id.append(results.multi_handedness[h_idx].classification[0].label)
                    for l_idx, landmark in enumerate(hand_landmarks.landmark):
                        hand_ls.append([landmark.x, landmark.y, landmark.z])

                    hand_arr = np.array(hand_ls)
                    all_hand.append(hand_arr)
                    break

            if visualize:
                video_writer.write(image)
                print("processing frame {}".format(i), end="\r")

            i += 1
            out_dt["landmarks"][f"{i}"] = all_hand

            if i >= 80 * 59:
                break

    out_dt["hand_labels"] = hand_id

    if visualize:
        video_writer.release()

    cap.release()
    joblib.dump(out_dt, f"{out_video_root}{hand}_hand_{os.path.basename(path)[:-4]}.txt")


def hand_checking(path):
    mp_hands = mp.solutions.hands
    cap = cv2.VideoCapture(path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    seconds = frames/fps
    
    if seconds < 5:
        return "too short!"

    with mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5) as hands:
        i = 0
        score = 0
        while True:
            success, image = cap.read()
            if not success:
                # If loading a video, use 'break' instead of 'continue'.
                break

            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image)

            if results.multi_handedness:
                score += results.multi_handedness[0].classification[0].score
            else:
                score += 0

            i += 1
            if i == 5 * fps:
                return score/i


if __name__ == '__main__':
    left_video_path = "/mnt/pd_app/gesture/202008069_AL.mp4"
    right_video_path = "/mnt/pd_app/gesture/202008069_AR.mp4"
    out_path = "/mnt/pd_app/results/test/"
    hand_extraction(left_video_path, out_video_root=out_path, hand='left')
    hand_extraction(right_video_path, out_video_root=out_path, hand='right')
