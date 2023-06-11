import cv2
import mediapipe as mp
import numpy as np
import serial

seri = serial.Serial(port='COM10', baudrate=9600,
                    parity=serial.PARITY_NONE,
                    stopbits=serial.STOPBITS_ONE,
                    bytesize=serial.EIGHTBITS)

max_num_hands = 1

#MediaPipe hands model
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=max_num_hands,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0)

def sendToArduino(key):
    print(f'Send : {key}')
    seri.write(bytes(key, encoding='ascii'))

def hand_index():
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
                v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19,0,5,9],:] # Parent joint
                v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,2,13,17],:]
                v = v2 - v1 # [20,3]
                # thumb : 20
                # index-ring : 21
                # middle-pinky : 22

                # Normalize v(벡터 정규화)
                # 벡터 정규화 : 한 벡터를 벡터의 길이로 나누어 방향값만 남도록 길이를 1로 만드는 것.
                # 정규화를 통해 만들어진 벡터를 단위벡터라고 함.
                v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

                # Get angle using arcos of dot product
                # 단위벡터에 arcos를 곱해 각도가 나오도록 함.
                # einsum: 아인슈타인 합계. 벡터의 연산을 편하게 하도록하는 식
                angle = np.arccos(np.einsum('nt, nt->n',
                    v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18,20,5,9,13,17],:],
                    v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19,4,21,21,22,22],:]))
                # (angle[15]=20-4) (angle[16]=5-21) (angle[17]=9-21) (angle[18]=13-22) (angle[19]=17-22)
                
                # Convert radian to degree
                angle = np.degrees(angle) 
                angle = angle.astype(int)

                # angle_list = [thumb_cmc, thumb, index, middle, ring, pinky,
                #               angle[15], angle[16], angle[17], angle[18], angle[19]]
                thumb_cmc = str(angle[15])
                thumb = str((angle[1]+angle[2])//2)
                index = str((angle[3]+angle[4]+angle[5])//3)
                middle = str((angle[6]+angle[7]+angle[8])//3)
                ring = str((angle[9]+angle[10]+angle[11])//3)
                pinky = str((angle[12]+angle[13]+angle[14])//3)

                angle1 = str(angle[0])
                angle5 = str(angle[16])
                angle9 = str(angle[17])
                angle13 = str(angle[18])
                angle17 = str(angle[19])
                
                ang_ls = [[int(thumb),int(index),int(middle),int(ring),int(pinky)],[int(angle1),int(angle5),int(angle9),int(angle13),int(angle17)]]

                
                cv2.putText(img, text=thumb_cmc, org=(int(res.landmark[1].x * img.shape[1]), int(res.landmark[1].y * img.shape[0])), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255,0,0), thickness=2)

                cv2.putText(img, text=thumb, org=(int(res.landmark[4].x * img.shape[1]), int(res.landmark[4].y * img.shape[0])), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255,0,0), thickness=2)

                cv2.putText(img, text=index, org=(int(res.landmark[8].x * img.shape[1]), int(res.landmark[8].y * img.shape[0])), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255,0,0), thickness=2)

                cv2.putText(img, text=middle, org=(int(res.landmark[12].x * img.shape[1]), int(res.landmark[12].y * img.shape[0])), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255,0,0), thickness=2)

                cv2.putText(img, text=ring, org=(int(res.landmark[16].x * img.shape[1]), int(res.landmark[16].y * img.shape[0])), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255,0,0), thickness=2)

                cv2.putText(img, text=pinky, org=(int(res.landmark[20].x * img.shape[1]), int(res.landmark[20].y * img.shape[0])), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255,0,0), thickness=2)
                
                cv2.putText(img, text=angle1, org=(int(res.landmark[1].x * img.shape[1]), int(res.landmark[1].y * img.shape[0])), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0,255,0), thickness=1)

                cv2.putText(img, text=angle5, org=(int(res.landmark[5].x * img.shape[1]), int(res.landmark[5].y * img.shape[0])), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0,255,0), thickness=1)

                cv2.putText(img, text=angle9, org=(int(res.landmark[9].x * img.shape[1]), int(res.landmark[9].y * img.shape[0])), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0,255,0), thickness=1)

                cv2.putText(img, text=angle13, org=(int(res.landmark[13].x * img.shape[1]), int(res.landmark[13].y * img.shape[0])), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0,255,0), thickness=1)

                cv2.putText(img, text=angle17, org=(int(res.landmark[17].x * img.shape[1]), int(res.landmark[17].y * img.shape[0])), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0,255,0), thickness=1)

                mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)

                sendToArduino(ang_ls)
        cv2.imshow('Game', img)
        if cv2.waitKey(1) == ord('q'):
            break
    