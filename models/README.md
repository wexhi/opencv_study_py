# Object-Detection-using-COCO-dataset

purpose: learning opencv<br>

It is a real time object detection project using pretrained dnn model named mobileNet SSD.<br>

MobileNet SSD is a single-shot multibox detection network intended to perform object detection <br>

model: ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt<br>
model input: 300x300x3x1 in BGR<br>
model output: vector containing tracked object data<br>
weights: frozen_inference_graph.pb <br>

Used NMS (non max suppression) to avoid multiple bounding boxes over single object<br>
COCO is a large-scale object detection, segmentation, and captioning dataset having 80 object categories.<br>

<img src="https://github.com/HarshitDolu/Object-Detection-using-COCO-dataset/blob/main/demo_img.jpg" width="900">
