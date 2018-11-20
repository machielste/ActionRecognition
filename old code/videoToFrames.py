import cv2
from imageToSingleVector import Extractor
import numpy as np
import os

cap = cv2.VideoCapture("data/vid.mp4")
check, vid = cap.read()
counter = 0
check = True
frame_list = []

while (check == True):
    cv2.imwrite("data/framesOfVideo/frame%d.jpg" % counter, vid)
    check, vid = cap.read()
    frame_list.append(vid)
    counter += 1

vectorlist = []

x = Extractor()
directory = "data/framesOfVideo"

i = 0
for i in range(counter - 1):
    vectorlist.append(x.extract(directory + "/frame" + str(i) + ".jpg"))
    i+=1

print(vectorlist[0])
