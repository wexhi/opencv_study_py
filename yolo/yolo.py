import cv2
import numpy as np

confThreshold = 0.5
nnsThreshold = 0.3


def findObjects(outputs, img):
    hT, wT, cT = img.shape
    bbox = []
    class_ids = []
    confidences = []

    for output in outputs:
        for det in output:
            scores = det[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > confThreshold:
                w, h = int(det[2] * wT), int(det[3] * hT)
                x, y = int(det[0] * wT - w / 2), int(det[1] * hT - h / 2)
                bbox.append([x, y, w, h])
                class_ids.append(class_id)
                confidences.append(float(confidence))
    # print(len(bbox))
    indices = cv2.dnn.NMSBoxes(bbox, confidences, confThreshold, nnsThreshold)

    for i in indices:
        box = bbox[i]
        x, y, w, h = box[0], box[1], box[2], box[3]
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 2)
        cv2.putText(img, f"{class_names[class_ids[i]].upper()} {int(confidences[i] * 100)}%",
                    (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)


cap = cv2.VideoCapture(0)
whT = 320

classes_file = "G:/opencv_py/models/coco.names"

with open(classes_file, "r") as f:
    class_names = f.read().rstrip('\n').split('\n')
# print(class_names)

model_config = "G:/opencv_py/models/yolov3.cfg"
model_weights = "G:/opencv_py/models/yolov3.weights"

net = cv2.dnn.readNetFromDarknet(model_config, model_weights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

while True:
    success, img = cap.read()

    blob = cv2.dnn.blobFromImage(img, 1 / 255, (whT, whT), (0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)

    layer_names = net.getLayerNames()
    # print(layer_names)
    # print(net.getUnconnectedOutLayers())
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    outputs = net.forward(output_layers)
    # print(type(outputs))

    findObjects(outputs, img)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
