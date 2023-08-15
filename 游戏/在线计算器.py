import cv2
from cvzone.HandTrackingModule import HandDetector
import time


class Button:
    def __init__(self, pos, width, height, value):
        self.pos = pos
        self.width = width
        self.height = height
        self.value = value

    def draw(self, img):
        cv2.rectangle(img, self.pos, (self.pos[0] + self.width, self.pos[1] + self.height), \
                      (225, 225, 225), cv2.FILLED)
        cv2.rectangle(img, self.pos, (self.pos[0] + self.width, self.pos[1] + self.height), \
                      (50, 50, 50), 3)
        cv2.putText(img, self.value, (self.pos[0] + 40, self.pos[1] + 62), cv2.FONT_HERSHEY_PLAIN, \
                    2, (50, 50, 50), 2)

    def clicked(self, x,y):
        if self.pos[0] < x < self.pos[0] + self.width and \
                self.pos[1] < y < self.pos[1] + self.height:
            cv2.rectangle(img, self.pos, (self.pos[0] + self.width, self.pos[1] + self.height), \
                          (255, 255, 255), cv2.FILLED)
            cv2.rectangle(img, self.pos, (self.pos[0] + self.width, self.pos[1] + self.height), \
                          (50, 50, 50), 3)
            cv2.putText(img, self.value, (self.pos[0] + 40, self.pos[1] + 62), cv2.FONT_HERSHEY_PLAIN, \
                        5, (0, 0, 0), 5)
            return True
        else:
            return False

# Webcom
cap = cv2.VideoCapture(0)
cap.set(3, 1280)  # width
cap.set(4, 720)  # height
detector = HandDetector(detectionCon=0.8, maxHands=1)

# Create Buttons
bottonListVal = [["7", "8", "9", "*"],
                 ["4", "5", "6", "-"],
                 ["1", "2", "3", "+"],
                 ["0", "/", ".", "="]]

buttons = []
for x in range(4):
    for y in range(4):
        xpos = x * 100 + 800
        ypos = y * 100 + 150
        button = Button([xpos, ypos], 100, 100, bottonListVal[y][x])
        buttons.append(button)

# Variables
myEquation = ""
delayCounter = 0

# Loop
while True:
    # Get image frame
    success, img = cap.read()
    img = cv2.flip(img, 1)

    # Detect of the Hand
    hands, img = detector.findHands(img, flipType=False)

    # Draw the button
    cv2.rectangle(img, (800, 50), (800 + 400, 70 + 100), \
                  (225, 225, 225), cv2.FILLED)
    cv2.rectangle(img, (800, 50), (800 + 400, 70 + 100), \
                  (50, 50, 50), 3)
    for button in buttons:
        button.draw(img)

    # Check for hand
    if hands:
        lmList = hands[0]['lmList']
        length, _, img = detector.findDistance(lmList[8][:2], lmList[12][:2], img)
        # print(length)
        x,y = lmList[8][:2]
        if length < 50:
            for i,button in enumerate(buttons):
                if button.clicked(x, y) and delayCounter == 0:
                    myValue = bottonListVal[int(i%4)][int(i/4)]
                    if myValue == '=':
                        myEquation = str(eval(myEquation))
                    else:
                        myEquation += myValue
                    delayCounter = 1

    # Avoid Duplicate Display
    if delayCounter > 0:
        delayCounter += 1
        if delayCounter > 15:
            delayCounter = 0

    # Display the Equations/Answers
    cv2.putText(img, myEquation, (810, 120), cv2.FONT_HERSHEY_PLAIN, \
                3, (50, 50, 50), 3)

    # Display the image
    cv2.imshow("Image", img)
    key = cv2.waitKey(1)
    if key == ord('c'):
        myEquation = ""
    if key == ord('q'):
        break