import cv2
import cvzone
import numpy as np
from cvzone.HandTrackingModule import HandDetector
import math
import random


class SnakeGameClass():
    def __init__(self, path="Images/Donut.png"):
        self.points = []  # All points of the snake
        self.length = []  # distance between two points
        self.currentLength = 0  # total length of the snake
        self.allowedLength = 150  # total allowed length of the snake
        self.previousHead = [0, 0]  # previous head of the snake

        self.imgFood = cv2.imread(path, cv2.IMREAD_UNCHANGED)  # food image
        self.hFood, self.wFood, _ = self.imgFood.shape  # height and width of the food image
        self.foodPos = [0, 0]  # position of the food image
        self.randomFoodLocaion()

        self.score = 0
        self.GameOver = False

    def randomFoodLocaion(self):
        self.foodPos = random.randint(100, 1000), random.randint(100, 600)

    def update(self, imgMain, currentHead):
        if self.GameOver:
            print(self.score)
            cvzone.putTextRect(imgMain, "Game Over", [300, 400], 7, 5, offset=50, border=10)
            cvzone.putTextRect(imgMain, "Your Score: " + str(self.score), [300, 550], 5,
                               3, offset=30, border=5)
        else:
            px, py = self.previousHead
            cx, cy = currentHead

            self.points.append([cx, cy])
            distance = math.hypot(cx - px, cy - py)
            self.length.append(distance)
            self.currentLength += distance
            self.previousHead = currentHead

            # length reduction
            if self.currentLength > self.allowedLength:
                for i, length in enumerate(self.length):
                    self.currentLength -= length
                    self.length.pop(i)
                    self.points.pop(i)

                    if self.currentLength < self.allowedLength:
                        break

            # Check if snake is eating the food
            rx, ry = self.foodPos
            if (rx - self.wFood // 2 < cx < rx + self.wFood // 2 and \
                    ry - self.hFood // 2 < cy < ry + self.hFood // 2):
                self.randomFoodLocaion()
                self.allowedLength += 50
                self.score += 1

            # Draw snake
            if self.points:
                for i, point in enumerate(self.points):
                    if i != 0:
                        cv2.line(imgMain, self.points[i], self.points[i - 1], (0, 0, 255), 15)
                cv2.circle(img, self.points[-1], 15, (200, 0, 200), cv2.FILLED)

            # Check if snake is touching itself
            pts = np.array(self.points[:-2], np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(imgMain, [pts], False, (0, 200, 0), 3)
            minDistance = cv2.pointPolygonTest(pts, (cx, cy), True)
            # print(minDistance)

            if -1 <= minDistance <= 1:
                print("HIT")
                self.GameOver = True
                self.points = []  # All points of the snake
                self.length = []  # distance between two points
                self.currentLength = 0  # total length of the snake
                self.allowedLength = 150  # total allowed length of the snake
                self.previousHead = [0, 0]  # previous head of the snake
                self.randomFoodLocaion()

            # Draw food
            imgMain = cvzone.overlayPNG(imgMain, self.imgFood,
                                        (rx - self.wFood // 2, ry - self.hFood // 2))

            cvzone.putTextRect(imgMain, "Score: " + str(self.score), [50, 80], 3, 3, offset=10, border=1)

        return imgMain


cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

detector = HandDetector(detectionCon=0.8, maxHands=1)

game = SnakeGameClass()

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    hands, img = detector.findHands(img, flipType=False)

    if hands:
        lmList = hands[0]["lmList"]  # List of 21 Landmark points
        ponintIndexX = lmList[8][:2]
        img = game.update(img, ponintIndexX)

    cv2.imshow("Image", img)
    key = cv2.waitKey(1)
    if key == 27:
        break
    if key == ord("r"):
        game.GameOver = False
        game.score = 0
