import cv2
import numpy as np
import os

DIR = 'images_saved'

try:
    os.mkdir(DIR)
except FileExistsError:
    pass

cap = cv2.VideoCapture(0)

start = False
cnt = 0

while cap.isOpened():
    ret, img = cap.read()
    if not ret:
        continue
    img = cv2.flip(img,1)
    cv2.rectangle(img, (0, 0), (400, 400), (255, 255, 255), 2)

    if start:
        roi = img[0:400, 0:400]
        save_path = (f'{DIR}\{cnt+1}.jpg')
        cv2.imwrite(save_path, roi)
        cnt += 1

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, text=f"Collecting {cnt}",
            org=(5, 50), fontScale=0.7, color=(0, 255, 255), thickness=2, fontFace=cv2.FONT_HERSHEY_SIMPLEX)

    cv2.imshow("Collecting images", img)

    key = cv2.waitKey(10)
    if key == ord('s'):
        start = not start
    elif key == ord('q'):
        break