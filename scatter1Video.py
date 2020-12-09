# Generates 1 density scatter plot in one video. For proof of concept purposes.
# Created December 2020
# Written by Justin Yu

from matplotlib import pyplot as plt
import numpy as np
import cv2
import sys
import csv
import os

# load path of video
name = sys.argv[1]

# used for later iterations of auto naming output txts
# txtName = name[27:37] + "--" + name[38:43]

#load video and YOLO weights into OpenCV
video = cv2.VideoCapture(name)
net = cv2.dnn.readNet("YOLO/yolov3_training_final.weights", "YOLO/yolov3_testing.cfg")
layer_names=net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

#Run inference by frame
def YOLO(cap):
    
    #keep track of frames so x coord time can be tracked in output graph
    frames = 0
    
    # Each video is 20 minutes long @ 25fps so total 30000 frames
    while (frames < 30000):
        frames=frames+1
        ret, frame = cap.read()
        
        # set region of interest to speed up inference times (fit around the tube)
        img=frame[180:450, 0:800]
        height, width, channels = img.shape
        
        # begins detection process with loaded weights
        blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)
        
        #iterate thru every detected object and check confidence to confirm detection. 
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > .3:
                    center_x = int(detection[0] * width)
                    #below divide by 8 to get percentage of distance down the tube
                    #os.system("echo '{}    {}' >> {}.txt".format(center_x/8, frames, txtName))
                    os.system("echo '{}    {}' >> high2.txt".format(center_x/8, frames))
        key = cv2.waitKey(1)

YOLO(video)
cv2.destroyAllWindows()
