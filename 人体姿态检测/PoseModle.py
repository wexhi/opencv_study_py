import cv2
import mediapipe as mp
import time


class PoseDetector():
    def __init__(self, mode=False, upBody=False, smooth=True,
                 detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.upBody = upBody
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpDraw = mp.solutions.drawing_utils
        self.selmpPose = mp.solutions.pose
        self.pose = self.selmpPose.Pose(static_image_mode=self.mode, enable_segmentation=self.upBody,
                                        smooth_segmentation=self.smooth,
                                        min_detection_confidence=self.detectionCon,
                                        min_tracking_confidence=self.trackCon)

    def findPose(self, img, draw_id=False, draw_conect=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.pose.process(imgRGB)
        PoseList = []
        # print(results.pose_landmarks)
        if results.pose_landmarks:
            if draw_conect:
                self.mpDraw.draw_landmarks(img, results.pose_landmarks, self.selmpPose.POSE_CONNECTIONS)
                for id, lm in enumerate(results.pose_landmarks.landmark):
                    h, w, c = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    # print(id, cx, cy)
                    PoseList.append([id, cx, cy])
                    if draw_id:
                        cv2.circle(img, (cx, cy), 15, \
                                   (255, 0, 255), cv2.FILLED)
        return PoseList, img


def main():
    pTime = 0
    cTime = 0

    cap = cv2.VideoCapture(0)
    detector = PoseDetector()
    while True:
        success, img = cap.read()
        listP, img = detector.findPose(img=img, draw_id=False, draw_conect=True)
        if len(listP) != 0:
            print(listP[0])
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, str(int(fps)),
                    (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,
                    (255, 0, 255), 3)

        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
