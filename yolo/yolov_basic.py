from ultralytics import YOLO
import cv2

modle = YOLO('YOLO_Weights/yolov8l.pt')
results = modle("Images/3.png", show=True)
cv2.waitKey(0)
