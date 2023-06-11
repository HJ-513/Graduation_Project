import cv2
import mediapipe as mp
import numpy as np


def calculate_angle(a, b, c):
    a = np.array(a) # 랜드마크 좌표를 numpy 배열로 변환
    b = np.array(b)
    c = np.array(c)
    
    # 두 벡터 생성
    ab = a - b
    cb = c - b
    
    # 각도 계산
    angle_rad = np.arccos(np.dot(ab, cb) / (np.linalg.norm(ab) * np.linalg.norm(cb)))
    angle_deg = angle_rad * 180 / np.pi
    
    return angle_deg


mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

cap = cv2.VideoCapture(0)

with mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

    # Flip the image horizontally for a later selfie-view display, and convert
    # the BGR image to RGB.
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    results = hands.process(image)

    # Draw the hand annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Extract finger landmarks
        thumb = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
        index = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
        middle = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
        ring = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
        pinky = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]

        # Calculate finger angles
        thumb_angle = calculate_angle(thumb, index, middle)
        index_angle = calculate_angle(index, thumb, middle)
        middle_angle = calculate_angle(middle, index, ring)
        ring_angle = calculate_angle(ring, middle, pinky)
        pinky_angle = calculate_angle(pinky, ring, middle)

        # Draw finger angle annotations on the image.
        cv2.putText(image, str(int(thumb_angle)), tuple(np.multiply(thumb, [640, 480]).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(image, str(int(index_angle)), tuple(np.multiply(index, [640, 480]).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(image, str(int(middle_angle)), tuple(np.multiply(middle, [640, 480]).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(image, str(int(ring_angle)), tuple(np.multiply(ring, [640, 480]).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(image, str(int(pinky_angle)), tuple(np.multiply(pinky, [640, 480]).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)