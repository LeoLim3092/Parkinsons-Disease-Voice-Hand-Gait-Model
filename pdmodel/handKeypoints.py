import cv2
import numpy as np
import mediapipe as mp
import json


def deploy_hand_key_points(path, save=True):
    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands
    # Todo read correct file format

    pid = path.split("/")[-1]
    cap = cv2.VideoCapture(path)
    save_filename = f"../handOutput3/{pid}.mp4"

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    video_writer = cv2.VideoWriter(save_filename, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    out_dt = {"landmarks": {}}
    hand_id = []

    with mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5) as hands:
        i = 0
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

            video_writer.write(image)
            print("processing frame {}".format(i), end="\r")

            i += 1
            out_dt["landmarks"][f"{i}"] = all_hand

            if i >= 80 * 59:
                break

        out_dt["hand_labels"] = hand_id
        video_writer.release()
        cap.release()

    if save:
        json.dump(out_dt, "../output/hand/{}.json".format(pid))

    return out_dt

