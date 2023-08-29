from ultralytics import YOLO
import cv2
import cvzone
import math
import torch

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
cap = cv2.VideoCapture("Videos/ppe-3.mp4")  # For Video

module = YOLO("YOLO_Weights/HardHel.pt")

# classNames = ['Excavator', 'Gloves', 'Hardhat', 'Ladder', 'Mask',
#               'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest', 'Person',
#               'SUV', 'Safety Cone', 'Safety Vest', 'bus', 'dump truck',
#               'fire hydrant', 'machinery', 'mini-van', 'sedan', 'semi',
#               'trailer', 'truck and trailer', 'truck', 'van', 'vehicle',
#               'wheel loader'
#               ]
classNames = ['head', 'helmet', 'person']
myColor = (0, 0, 255)

while True:
    success, img = cap.read()
    results = module(img, stream=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            w, h = x2 - x1, y2 - y1
            # cvzone.cornerRect(img, (x1, y1, w, h))

            #  Confidence
            conf = math.ceil(box.conf[0] * 100) / 100
            print(conf)

            # Class
            cls = int(box.cls[0])
            currentClass = classNames[cls]
            if conf > 0.5:
                if currentClass == "Hardhat" or currentClass == 'Safety Vest':
                    myColor = (0, 255, 0)
                else:
                    myColor = (0, 0, 255)

                cvzone.putTextRect(img, f"{classNames[cls]}{conf}",
                                   (max(0, x1), max(35, y1 - 10)), 1, 1,
                                   offset=5, colorB=myColor, colorT=(255, 255, 255), colorR=myColor)
                cv2.rectangle(img, (x1, y1), (x2, y2), myColor, 3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
