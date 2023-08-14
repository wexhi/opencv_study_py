import cv2
import numpy as np

detection = False
frame_counter = 0

cap = cv2.VideoCapture(0)
img_target = cv2.imread("test.jpg")
my_vid = cv2.VideoCapture("vtest.avi")

success, img_video = my_vid.read()
hT, wT, cT = img_video.shape
img_target = cv2.resize(img_target, (wT, hT))

orb = cv2.ORB_create(nfeatures=2000)
kp1, des1 = orb.detectAndCompute(img_target, None)
img_target = cv2.drawKeypoints(img_target, kp1, None)

while True:
    success1, img_webcam = cap.read()
    img_Aug = img_webcam.copy()
    kp2, des2 = orb.detectAndCompute(img_webcam, None)
    # img_webcam = cv2.drawKeypoints(img_webcam, kp2, None)

    if detection == False:
        my_vid.set(cv2.CAP_PROP_POS_FRAMES, 0)
        frame_counter = 0
    else:
        if frame_counter == my_vid.set(cv2.CAP_PROP_FRAME_COUNT, 0):
            my_vid.set(cv2.CAP_PROP_POS_FRAMES, 0)
            frame_counter = 0
        success, img_video = my_vid.read()
        img_target = cv2.resize(img_target, (wT, hT))

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)
    print(len(good))
    imgFeatures = cv2.drawMatches(img_target, kp1, img_webcam, kp2, good, None, flags=2)

    if len(good) > 25:
        detection = True
        srcPoints = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dstPoints = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        matrix, mask = cv2.findHomography(srcPoints, dstPoints, cv2.RANSAC, 5)
        print(matrix)

        pts = np.float32([[0, 0], [0, hT], [wT, hT], [wT, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, matrix)
        img2 = cv2.polylines(img_webcam, [np.int32(dst)], True, (255, 0, 255), 3)

        img_wrap = cv2.warpPerspective(img_video, matrix, (img_webcam.shape[1], img_webcam.shape[0]))

        mask_new = np.zeros((img_webcam.shape[0], img_webcam.shape[1]), np.uint8)
        cv2.fillPoly(mask_new, [np.int32(dst)], [255, 255, 255])
        mask_INV = cv2.bitwise_not(mask_new)
        img_Aug = cv2.bitwise_and(img_Aug, img_Aug, mask=mask_INV)
        img_Aug = cv2.bitwise_or(img_wrap, img_Aug)

        cv2.imshow("masknew", img_Aug)
        # cv2.imshow("imgwrap", img_wrap)
        frame_counter += 1
        # cv2.imshow("img2", img2)

    cv2.imshow("Image Features", imgFeatures)
    # cv2.imshow("Webcam", img_webcam)
    # cv2.imshow("Target Image", img_target)
    # cv2.imshow("Video", img_video)
    # 帧率26
    if cv2.waitKey(80) & 0xFF == ord('q'):
        break


