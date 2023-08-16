import cvzone
from cvzone.ColorModule import ColorFinder
import cv2
import socket

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

success, img = cap.read()
h, w, _ = img.shape

myColor = ColorFinder(False)
hsvVals = {'hmin': 77, 'smin': 36, 'vmin': 72, 'hmax': 92, 'smax': 213, 'vmax': 230}

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
server_address_port = ("127.0.0.1", 8888)

while True:
    success, img = cap.read()
    imgColor, mask = myColor.update(img, hsvVals)
    imgContours, contours = cvzone.findContours(img, mask)

    if contours:
        data = contours[0]['center'][0], \
            h - contours[0]['center'][1], \
            int(contours[0]['area'])
        print(data)
        sock.sendto(str(data).encode(), server_address_port)

    imgStack = cvzone.stackImages([img, imgColor, mask, imgContours], 2, 0.5)
    cv2.imshow("Image", imgStack)
    cv2.waitKey(1)
