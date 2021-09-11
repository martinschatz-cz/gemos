import numpy as np
import cv2
import os

right_top = (110, 395)
dx = (1000, 350)
scale_position = ((223, 584), (521, 577), (570, 617), (249, 625))

PROTOTXT = "MobileNetSSD_deploy.prototxt"
MODEL = "MobileNetSSD_deploy.caffemodel"
INP_VIDEO_PATH = 'cars.mp4'
OUT_VIDEO_PATH = 'cars_detection.mp4'
GPU_SUPPORT = 1
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
           "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

img_path = os.path.join("..", "..", "data", "VAHA2")  # path to processed directory
images = os.listdir(img_path)  # list of images in processed directory

net = cv2.dnn.readNetFromCaffe(PROTOTXT, MODEL)
if GPU_SUPPORT:
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

wierd_img = 1

#cap = cv2.VideoCapture(INP_VIDEO_PATH)
#while True:
for idx, image in enumerate(images[wierd_img:10000]):
    original_image = cv2.imread(os.path.join(img_path, image))
    #ret, frame = cap.read()
    # if not ret:
    #     break
    h, w = original_image.shape[:2]
    original_image = cv2.blur(original_image, (3, 3))  # 3,3
    blob = cv2.dnn.blobFromImage(original_image, 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()
    for i in np.arange(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.3:
            idx = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
            cv2.rectangle(original_image, (startX, startY), (endX, endY), COLORS[idx], 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(original_image, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
    # cv2.imshow("detectin", original_image)
    cv2.imshow("detection", original_image[right_top[1]:right_top[1] + dx[1], right_top[0]:right_top[0] + dx[0]])
    pressed_key = cv2.waitKey(5)
    if pressed_key == 27:
        break
