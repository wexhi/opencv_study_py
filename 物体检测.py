import cv2
import numpy as np

#  img = cv2.imread('./images/lena.jpg')
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

class_file = "./models/coco.names"
with open(class_file, 'rt') as f:
    class_names = f.read().rstrip('\n').split('\n')

config_path = "./models/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
weights_path = "./models/frozen_inference_graph.pb"

net = cv2.dnn_DetectionModel(weights_path, config_path)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

threshold = 0.6
while True:
    seccess, img = cap.read()
    class_ids, confs, bbox = net.detect(img, confThreshold=threshold)
    bbox = list(bbox)
    confs = list(np.array(confs).reshape(1, -1)[0])
    confs = list(map(float, confs))
    # print(class_ids, bbox)
    # print(type(confs[0]))


    indices = cv2.dnn.NMSBoxes(bbox, confs, threshold, nms_threshold=0.2)
    print(indices)
    for i in indices:
        print(i)
        box = bbox[i]
        x, y, w, h = box[0], box[1], box[2], box[3]
        cv2.rectangle(img, (x, y), (x + w, h + y), color=(0, 255, 0), thickness=2)
        cv2.putText(img, class_names[class_ids[i] - 1].upper(), (box[0] + 10, box[1] + 30),
                    cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(img, str(round(confs[i] * 100, 2)), (box[0] + 200, box[1] + 30),
                    cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 255), 1)


    # if len(class_ids) != 0:
    #     for class_ids, confidence, box in zip(class_ids.flatten(), confs.flatten(), bbox):
    #         cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)
    #         cv2.putText(img, class_names[class_ids - 1].upper(), (box[0] + 10, box[1] + 30),
    #                     cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
    #         cv2.putText(img, str(round(confidence * 100, 2)), (box[0] + 200, box[1] + 30),
    #                     cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 255), 1)

    cv2.imshow('image', img)
    if cv2.waitKey(1) == ord('q'):
        break
