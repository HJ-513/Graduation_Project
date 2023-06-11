import cv2
import mediapipe as mp
import numpy as np

max_num_hands = 1

#MediaPipe hands model
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=max_num_hands,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

#Gesture recognition model
# file = np.genfromtxt('data/gesture_train.csv', delimiter=',')
# angle = file[:,:-1].astype(np.float32)
# label = file[:, -1].astype(np.float32)
# knn = cv2.ml.KNearest_create()
# knn.train(angle, cv2.ml.ROW_SAMPLE, label)

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, img = cap.read()
    if not ret:
        continue

    img = cv2.flip(img, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    result = hands.process(img)

    img =cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    if result.multi_hand_landmarks is not None:
        for res in result.multi_hand_landmarks:
            joint = np.zeros((21,3))
            for j, lm in enumerate(res.landmark):
                joint[j] = [lm.x, lm.y, lm.z]

            #Compute angles between joints
            v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19, 5,5],:] # Parent joint
            v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20, 6,9],:]
            v = v2 - v1 # [20,3]

            # Normalize v(벡터 정규화)
            # 벡터 정규화 : 한 벡터를 벡터의 길이로 나누어 방향값만 남도록 길이를 1로 만드는 것.
            # 정규화를 통해 만들어진 벡터를 단위벡터라고 함.
            v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

            # Get angle using arcos of dot product
            # 단위벡터에 arcos를 곱해 각도가 나오도록 함.
            # einsum: 아인슈타인 합계. 벡터의 연산을 편하게 하도록하는 식
            angle = np.arccos(np.einsum('nt, nt->n',
                v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18, 20],:],
                v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19, 21],:])) # [0 ~ 14 +1,]

            # Convert radian to degree
            angle = np.degrees(angle) 
            angle = angle.astype(int)

            thumb_cmc = angle[15]
            thumb = (angle[1]+angle[2])//2
            index = (angle[3]+angle[4]+angle[5])//3
            middle = (angle[6]+angle[7]+angle[8])//3
            ring = (angle[9]+angle[10]+angle[11])//3
            pinky = (angle[12]+angle[13]+angle[14])//3
            angle5 = angle[15] - 80
            
            # 70<angle5<90 일때 prev에 저장하고 문자열로 변환
            if angle5<-10:
                angle5 = -10
                prev = angle5
                # angle5 값을 모터로 전송
            elif angle5>10:
                angle5 = 10
                prev = angle5
            elif index>20:
                angle5 = 0
                prev = angle5
            else:
                prev = angle5
                # prev값을 모터로 전송
            

            cv2.putText(img, text='angle5 : '+str(angle5)+'prev : '+str(prev), org=(int(res.landmark[0].x * img.shape[1]), int(res.landmark[0].y * img.shape[0])+20), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0,255,255), thickness=2)

            cv2.putText(img, text=str(thumb_cmc), org=(int(res.landmark[1].x * img.shape[1]), int(res.landmark[1].y * img.shape[0])), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255,0,0), thickness=2)
            cv2.putText(img, text=str(thumb), org=(int(res.landmark[4].x * img.shape[1]), int(res.landmark[4].y * img.shape[0])), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255,0,0), thickness=2)
            cv2.putText(img, text=str(index), org=(int(res.landmark[8].x * img.shape[1]), int(res.landmark[8].y * img.shape[0])), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255,0,255), thickness=2)
            cv2.putText(img, text=str(middle), org=(int(res.landmark[12].x * img.shape[1]), int(res.landmark[12].y * img.shape[0])), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255,0,0), thickness=2)
            cv2.putText(img, text=str(ring), org=(int(res.landmark[16].x * img.shape[1]), int(res.landmark[16].y * img.shape[0])), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255,0,0), thickness=2)
            cv2.putText(img, text=str(pinky), org=(int(res.landmark[20].x * img.shape[1]), int(res.landmark[20].y * img.shape[0])), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255,0,0), thickness=2)

            mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)
    
    cv2.imshow('Game', img)
    if cv2.waitKey(1) == ord('q'):
        break