import numpy as np
from ultralytics import YOLO
import cv2
import cvzone
import math
import torch
from sort import *

# 查看版本
print(torch.__version__)
# 查看gpu是否可用
gpuAvailable = torch.cuda.is_available()
print(gpuAvailable)
# 返回设备gpu个数
gpu = torch.cuda.get_device_name()
print(gpu)

# cap = cv2.VideoCapture(0)  # For Webcam
# cap.set(3, 1280)
# cap.set(4, 720)
cap = cv2.VideoCapture("Videos/people.mp4")  # For Video
mask = cv2.imread("Images/mask_people.png")

# Tracking
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
totalCounterUp = []
totalCounterDown = []


limitUp = [130, 199, 300, 199]
limitDown = [450, 297, 573, 297]

module = YOLO("YOLO_Weights/yolov8l.pt")

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

while True:
    success, img = cap.read()
    imgRegion = cv2.bitwise_and(img, mask)

    results = module(imgRegion, stream=True)

    detections = np.empty((0, 5))

    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            w, h = x2 - x1, y2 - y1

            #  Confidence
            conf = math.ceil(box.conf[0] * 100) / 100
            print(conf)

            # Class
            cls = int(box.cls[0])
            currentClass = classNames[cls]

            if currentClass == "person" and conf > 0.3:
                # cvzone.putTextRect(img, f"{classNames[cls]}{conf}",
                #                    (max(0, x1), max(35, y1 - 10)), 1.2, 1, offset=8)
                # cvzone.cornerRect(img, (x1, y1, w, h), l=8, rt=5)
                currentArray = np.array([[x1, y1, x2, y2, conf]])
                detections = np.vstack((detections, currentArray))

    resultsTracker = tracker.update(detections)
    cv2.line(img, (limitUp[0], limitUp[1]), (limitUp[2], limitUp[3]), (0, 255, 255), 5)
    cv2.line(img, (limitDown[0], limitDown[1]), (limitDown[2], limitDown[3]), (0, 255, 255), 5)

    for r in resultsTracker:
        x1, y1, x2, y2, id = r
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        w, h = x2 - x1, y2 - y1
        cvzone.cornerRect(img, (x1, y1, w, h), l=8, rt=2, colorR=(255, 0, 0))
        cvzone.putTextRect(img, f"{id}",
                           (max(0, x1), max(35, y1 - 10)), 2, 3, offset=10)
        print(r)

        cx, cy = x1 + w // 2, y1 + h // 2
        cv2.circle(img, (cx, cy), 5, (0, 255, 0), cv2.FILLED)

        if limitUp[0] < cx < limitUp[2] and limitUp[1] - 30 < cy < limitUp[3] + 30:
            if totalCounterUp.count(id) == 0:
                totalCounterUp.append(id)
                cv2.line(img, (limitUp[0], limitUp[1]), (limitUp[2], limitUp[3]),
                         (0, 255, 0), 3)

        if limitDown[0]<cx<limitDown[2] and limitDown[1]-30<cy<limitDown[3]+30:
            if totalCounterDown.count(id) == 0:
                totalCounterDown.append(id)
                cv2.line(img, (limitDown[0], limitDown[1]), (limitDown[2], limitDown[3]),
                         (0, 255, 0), 3)

    cvzone.putTextRect(img, f"TotalUp Count: {len(totalCounterUp)}, TotalDown Count: {len(totalCounterDown)}",
                       (50, 50), 2, 3, offset=10)

    cv2.imshow("Image", img)
    # cv2.imshow("Image Region", imgRegion)
    cv2.waitKey(1)
