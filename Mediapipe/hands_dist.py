import cv2
import mediapipe as mp

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
            continue
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

        results = hands.process(image)

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                wrist = int(hand_landmarks.landmark[0].y * 100 )
                thumb_tip = int(hand_landmarks.landmark[4].y * 100 )
                thumb_dist = abs(wrist - thumb_tip)
                index_tip = int(hand_landmarks.landmark[8].y * 100 )
                index_dist = abs(wrist - index_tip)
                middle_tip = int(hand_landmarks.landmark[12].y * 100 )
                middle_dist = abs(wrist - middle_tip)
                ring_tip = int(hand_landmarks.landmark[16].y * 100 )
                ring_dist = abs(wrist - ring_tip)
                pinky_tip = int(hand_landmarks.landmark[20].y * 100 )
                pinky_dist = abs(wrist - pinky_tip)
                cv2.putText(
                    image, text='thumb=%d index=%d' %(thumb_dist, index_dist), org=(10, 30),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
                    color=255, thickness=2)
                # cv2.putText(
                #     image, text='middle=%d ring=%d pinky=%d' %( middle_dist, ring_dist, pinky_dist), org=(10, 100),
                #     fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
                #     color=255, thickness=2)
                cv2.putText(
                    image, text='landmark = %d' %(hand_landmarks.landmark[0].x * 100), org=(10, 100),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
                    color=255, thickness=2)
                mp_drawing.draw_landmarks(
                    image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        cv2.imshow('image', image)
        if cv2.waitKey(1) == ord('q'):
            break

cap.release()