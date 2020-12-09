# First argument is the path of the video
# Second argument is the path of the directory you wish to save pictures to
# Third argument is the number of frames between each selected picture

## Alter the upper bound of frames as needed (seconds of video times fps) 
## Alter video file type accordingly (.h264 is default)

import cv2
import os
import sys
import ntpath

video = cv2.VideoCapture(sys.argv[1])
frames = 0
interval = int(sys.argv[3])

while (frames < 29900):
    ret, frame = video.read()
    frame = frame[180:450, 0:1000]
    if (frames % interval == 0):
        pathOfImage = str(sys.argv[2]) + str(ntpath.basename(sys.argv[1])) + "f" + str(frames)
        pathOfImageFixed = pathOfImage.replace('.h264','') + ".jpg"
        cv2.imwrite(pathOfImageFixed, frame)
    frames = frames + 1
