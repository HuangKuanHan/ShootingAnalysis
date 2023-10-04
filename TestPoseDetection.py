#可以跑
import cv2
import mediapipe as mp
import math


# 計算座標直線距離
def findDistance(x1, y1, x2, y2):
    dist = math.sqrt((x2-x1)**2+(y2-y1)**2)
    return dist


# 計算角度
def angle_calc(x1, y1, x2, y2, x3, y3):
    try:
        a = (x2 - x1) ** 2 + (y2 - y1) ** 2
        b = (x2 - x3) ** 2 + (y1 - y2) ** 2
        c = (x3 - x1) ** 2 + (y3 - y1) ** 2
        angle = math.acos((a + b - c) / math.sqrt(4 * a * b)) * 180 / math.pi
    except:
        return 0
    return int(angle)


def draw_str(dst, xxx_todo_changeme, s, color, scale):
    (x, y) = xxx_todo_changeme
    if (color[0] + color[1] + color[2] == 255 * 3):
        cv2.putText(dst, s, (x + 1, y + 1), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness=4, lineType=5)
    else:
        cv2.putText(dst, s, (x + 1, y + 1), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness=4, lineType=5)
    # cv2.line
    cv2.putText(dst, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (216, 230, 0), lineType=5)


mp_drawing = mp.solutions.drawing_utils          # mediapipe 繪圖方法
mp_drawing_styles = mp.solutions.drawing_styles  # mediapipe 繪圖樣式
mp_pose = mp.solutions.pose                      # mediapipe 姿勢偵測

# 紀錄影片位置
# video_name = "media/input_video/shia_right.mp4"
video_name = "media/input_video/shia_frontleft.mp4"
# video_name = "media/input_video/shia_rightfront.mp4"

# 選擇要使用影片或是直接使用camera (0)
# cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture(1)
cap = cv2.VideoCapture(video_name)

# 根據影片大小抓禎數、長、寬
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_size = (width, height)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

# Video writer. cv2輸出有骨架的影片
# video_output = cv2.VideoWriter('Test3_側面2.mp4', fourcc, fps, frame_size)

# 字體顏色
red_color = (255, 0, 0)
# c = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

# 啟用姿勢偵測
with mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as pose:

    # if not cap.isOpened():
    #     print("Cannot open camera")
    #     exit()

    # for i in range(33):
    #     c[i] = mp_pose[i]
    #     print(f"c[{i}] = {c[i]}")

    while True:
        ret, img = cap.read()
        # if not ret:
        #     print("Cannot receive frame")
        #     break

        # Get fps.
        fps = cap.get(cv2.CAP_PROP_FPS)
        # Get height and width of the frame.
        h, w = img.shape[:2]

        img = cv2.resize(img, frame_size)               # 縮小尺寸，加快演算速度
        # img = cv2.resize(img, (720, 800))               # 縮小尺寸，加快演算速度
        img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)   # 將 BGR 轉換成 RGB
        results = pose.process(img2)                  # 取得姿勢偵測結果

        lm = results.pose_landmarks
        lmPose = mp_pose.PoseLandmark

        # 右肩膀 座標
        r_shoulder_x = int(lm.landmark[lmPose.RIGHT_SHOULDER].x * w)
        r_shoulder_y = int(lm.landmark[lmPose.RIGHT_SHOULDER].y * h)

        # 左肩膀 座標
        l_shoulder_x = int(lm.landmark[lmPose.LEFT_SHOULDER].x * w)
        l_shoulder_y = int(lm.landmark[lmPose.LEFT_SHOULDER].y * h)

        # 右骨盆 座標
        r_hip_x = int(lm.landmark[lmPose.RIGHT_HIP].x * w)
        r_hip_y = int(lm.landmark[lmPose.RIGHT_HIP].y * h)

        # 左骨盆 座標
        l_hip_x = int(lm.landmark[lmPose.LEFT_HIP].x * w)
        l_hip_y = int(lm.landmark[lmPose.LEFT_HIP].y * h)

        # 右手肘 座標
        r_elbow_x = int(lm.landmark[lmPose.RIGHT_ELBOW].x * w)
        r_elbow_y = int(lm.landmark[lmPose.RIGHT_ELBOW].y * h)

        # 左手肘 座標
        l_elbow_x = int(lm.landmark[lmPose.LEFT_ELBOW].x * w)
        l_elbow_y = int(lm.landmark[lmPose.LEFT_ELBOW].y * h)

        # 右膝蓋 座標
        r_knee_x = int(lm.landmark[lmPose.RIGHT_KNEE].x * w)
        r_knee_y = int(lm.landmark[lmPose.RIGHT_KNEE].y * h)

        # 左膝蓋 座標
        l_knee_x = int(lm.landmark[lmPose.LEFT_KNEE].x * w)
        l_knee_y = int(lm.landmark[lmPose.LEFT_KNEE].y * h)

        # 右手腕 座標
        r_wrist_x = int(lm.landmark[lmPose.RIGHT_WRIST].x * w)
        r_wrist_y = int(lm.landmark[lmPose.RIGHT_WRIST].y * h)

        # 左手腕 座標
        l_wrist_x = int(lm.landmark[lmPose.LEFT_WRIST].x * w)
        l_wrist_y = int(lm.landmark[lmPose.LEFT_WRIST].y * h)

        # 右手指 座標
        r_knuckle_x = int(lm.landmark[lmPose.RIGHT_INDEX].x * w)
        r_knuckle_y = int(lm.landmark[lmPose.RIGHT_INDEX].y * h)

        # 左手指 座標
        l_knuckle_x = int(lm.landmark[lmPose.LEFT_INDEX].x * w)
        l_knuckle_y = int(lm.landmark[lmPose.LEFT_INDEX].y * h)

        # 右腳踝 座標
        r_ankle_x = int(lm.landmark[lmPose.RIGHT_ANKLE].x * w)
        r_ankle_y = int(lm.landmark[lmPose.RIGHT_ANKLE].y * h)

        # 左腳踝 座標
        l_ankle_x = int(lm.landmark[lmPose.LEFT_ANKLE].x * w)
        l_ankle_y = int(lm.landmark[lmPose.LEFT_ANKLE].y * h)

        '''
        # 左邊骨盆肩膀 距離
        l_dist_hip_shoulder = findDistance(l_shoulder_x, l_shoulder_y, l_hip_x, l_hip_y)
        # 左肩膀盆手肘 距離
        l_dist_shoulder_elbow = findDistance(l_shoulder_x, l_shoulder_y, l_elbow_x, l_elbow_y)

        # 右邊骨盆肩膀 距離
        r_dist_hip_shoulder = findDistance(r_shoulder_x, r_shoulder_y, r_hip_x, r_hip_y)
        # 右肩膀盆手肘 距離
        r_dist_shoulder_elbow = findDistance(r_shoulder_x, r_shoulder_y, r_elbow_x, r_elbow_y)
        '''

        # 左邊肩膀夾角
        l_shoulder_angle = angle_calc(l_hip_x, l_hip_y, l_shoulder_x, l_shoulder_y, l_elbow_x, l_elbow_y)
        # 右邊肩膀夾角
        r_shoulder_angle = angle_calc(r_hip_x, r_hip_y, r_shoulder_x, r_shoulder_y, r_elbow_x, r_elbow_y)

        # 左邊髖關節夾角
        l_hip_angle = angle_calc(l_elbow_x, l_elbow_y, l_hip_x, l_hip_y, l_knee_x, l_knee_y)
        # 右邊髖關節夾角
        r_hip_angle = angle_calc(r_elbow_x, r_elbow_y, r_hip_x, r_hip_y, r_knee_x, r_knee_y)

        # 左邊肘關節夾角
        l_elbow_angle = angle_calc(l_shoulder_x, l_shoulder_y, l_elbow_x, l_elbow_y, l_wrist_x, l_wrist_y)
        # 右邊肘關節夾角
        r_elbow_angle = angle_calc(r_shoulder_x, r_shoulder_y, r_elbow_x, r_elbow_y, r_wrist_x, r_wrist_y)

        # 左邊手腕夾角
        l_wrist_angle = angle_calc(l_elbow_x, l_elbow_y, l_wrist_x, l_wrist_y, l_knuckle_x, l_knuckle_y)
        # 右邊手腕夾角
        r_wrist_angle = angle_calc(r_elbow_x, r_elbow_y, r_wrist_x, r_wrist_y, r_knuckle_x, r_knuckle_y)

        # 左邊膝蓋夾角
        l_knee_angle = angle_calc(l_hip_x, l_hip_y, l_knee_x, l_knee_y, l_ankle_x, l_ankle_y)
        # 右邊膝蓋夾角
        r_knee_angle = angle_calc(r_hip_x, r_hip_y, r_knee_x, r_knee_y, r_ankle_x, r_ankle_y)

        shoulder_angles = "Left Shoulder:" + str(l_shoulder_angle) + "  Right Shoulder:" + str(r_shoulder_angle)
        hip_angles = "Left Hip:" + str(l_hip_angle) + "  Right Hip:" + str(r_hip_angle)
        elbow_angles = "Left Elbow:" + str(l_elbow_angle) + "  Right Elbow:" + str(r_elbow_angle)
        wrist_angles = "Left Wrist:" + str(l_wrist_angle) + "  Right Wrist:" + str(r_wrist_angle)
        knee_angles = "Left Knee:" + str(l_knee_angle) + "  Right Knee:" + str(r_knee_angle)

        # draw_str(img, (20, 50), shoulder_angles, red_color, 1)
        # draw_str(img, (20, 50), hip_angles, red_color, 1)
        # draw_str(img, (20, 50), elbow_angles, red_color, 1)
        # draw_str(img, (20, 50), wrist_angles, red_color, 1)
        # draw_str(img, (20, 50), knee_angles, red_color, 1)

        print("左邊肩膀夾角:", l_shoulder_angle, "右邊肩膀夾角:", r_shoulder_angle)
        print("左邊髖關節夾角:", l_hip_angle, "右邊髖關節夾角:", r_hip_angle)
        print("左邊肘關節夾角:", l_elbow_angle, "右邊肘關節夾角:", r_elbow_angle)
        print("左邊手腕夾角:", l_wrist_angle, "右邊手腕夾角:", r_wrist_angle)
        print("左邊膝蓋夾角:", l_knee_angle, "右邊膝蓋夾角:", r_knee_angle)
        print()


        # 根據姿勢偵測結果，標記身體節點和骨架
        mp_drawing.draw_landmarks(
            img,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

        # video_output.write(img)
        cv2.imshow('camera_base', img)

        if cv2.waitKey(5) == ord('q'):
            break     # 按下 q 鍵停止
cap.release()
cv2.destroyAllWindows()