import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime


def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 人脸识别库要求的格式
        encode = face_recognition.face_encodings(img)[0]  # 人脸编码
        encodeList.append(encode)
    return encodeList


def markAttendance(name):
    with open("出勤表.csv", 'r+') as f:
        my_data_list = f.readlines()
        name_list = []
        for line in my_data_list:
            entry = line.split(',')
            name_list.append(entry[0])
        if name not in name_list:
            now = datetime.now()
            date_str = now.strftime('%Y-%m-%d %H:%M:%S')
            f.writelines(f"\n{name},{date_str}")


path = 'face'
images = []
class_names = []
my_list = os.listdir(path)
print(my_list)
for cls in my_list:
    cur_img = cv2.imread(f'{path}/{cls}')
    images.append(cur_img)
    class_names.append(os.path.splitext(cls)[0])
print(class_names)

encoding_list_known = findEncodings(images)
print("解码完成")

cap = cv2.VideoCapture(0)
while True:
    success, img = cap.read()
    imgs = cv2.resize(img, (0, 0), None, 0.25, 0.25)  # 缩小图片，加快识别速度
    imgs = cv2.cvtColor(imgs, cv2.COLOR_BGR2RGB)  # 人脸识别库要求的格式

    facesCurFrame = face_recognition.face_locations(imgs)  # 人脸位置
    encodeCurFrame = face_recognition.face_encodings(imgs, facesCurFrame)  # 人脸编码

    for encodeFace, faceLoc in zip(encodeCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encoding_list_known, encodeFace)
        faceDis = face_recognition.face_distance(encoding_list_known, encodeFace)
        print(faceDis)
        matcheIndex = np.argmin(faceDis)

        if matches[matcheIndex]:
            name = class_names[matcheIndex].upper()
            print(name)
            y1, x2, y2, x1 = faceLoc
            # 人脸识别库识别出来的人脸位置是缩小后的，所以要还原\
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, \
                        1, (255, 255, 255), 2)
            markAttendance(name)

    cv2.imshow("Webcam", img)
    cv2.waitKey(1)
