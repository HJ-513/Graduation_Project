import numpy as np
import cv2
import time
import mediapipe as mp
import os
import csv

DIR = 'images_saved'

try:
    os.mkdir(DIR)
except FileExistsError:
    pass

max_num_hands = 1

#MediaPipe hands model
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=max_num_hands,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

# 비디오 캡처 객체 생성
cap = cv2.VideoCapture(0)
# 이미지 저장 카운터 변수 초기화
cnt = 0
start = False

time.sleep(10)
while True:
    # 영상 프레임 읽기
    ret, img = cap.read()
    if not ret:
        continue

    # 영상 출력
    img = cv2.flip(img, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    result = hands.process(img)

    img =cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    cv2.rectangle(img, (0, 0), (400, 400), (255, 255, 255), 2)

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

            ang_ls = []
            thumb = (angle[1]+angle[2])//2
            index = (angle[3]+angle[4]+angle[5])//3
            middle = (angle[6]+angle[7]+angle[8])//3
            ring = (angle[9]+angle[10]+angle[11])//3
            pinky = (angle[12]+angle[13]+angle[14])//3

            angle1 = angle[0] # angle1 제외 (thumb_cmc와 같은 모터)
            angle5 = angle[16]
            angle9 = angle[17]
            angle13 =angle[18]
            angle17 =angle[19]
            
            ang_ls = [thumb, index, middle, ring, pinky, angle1, angle5, angle9, angle13, angle17]
            
    if start:
        roi = img[0:400, 0:400]
        save_path = (f'{DIR}\{cnt+1}.jpg')
        cv2.imwrite(save_path, roi)
        cnt += 1
        print(ang_ls)
        with open(f'{DIR}\hand_train.csv', 'a',newline='') as f: 
        # using csv.writer method from CSV package 
            write = csv.writer(f) 
            write.writerow(ang_ls)

    cv2.putText(img, text=f"Collecting {cnt}",
            org=(5, 50), fontScale=0.7, color=(0, 255, 255), thickness=2, fontFace=cv2.FONT_HERSHEY_SIMPLEX)

    cv2.imshow("Collecting images", img)

    # 'q' 키 누르면 종료
    # 's' 키 누르면 저장/중지
    key = cv2.waitKey(10)
    if key == ord('s'):
        start = not start
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()