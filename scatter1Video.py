# Generates 1 density scatter plot in one video. For proof of concept purposes.
# Created December 2020
# Written by Justin Yu

from matplotlib import pyplot as plt
import numpy as np
import cv2
import sys
import csv
import os

net = cv2.dnn.readNet("YOLO/yolov3_training_final.weights", "YOLO/yolov3_testing.cfg")
classes=["bee"]
layer_names=net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

def YOLO():
    cap = cv2.VideoCapture(sys.argv[1])
    frames = 0
    while (frames < 30000):
        frames=frames+1
        ret, frame = cap.read()
        img=frame[180:450, 0:800]
        height, width, channels = img.shape
        blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > .3:
                    center_x = int(detection[0] * width)
                    #print(center_x)
        key = cv2.waitKey(1)

YOLO()
cv2.destroyAllWindows()
