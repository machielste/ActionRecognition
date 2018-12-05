import glob
import os

import cv2
import numpy as np

from videoToVectorList import Process


class createVectors():

    def __init__(self):
        self.x = Process

    def doWork(self, folderName, max):

        fileCount = 0
        ##Loop over the first X videos in folder and convert them to inception v3'd vectors
        for pathAndFilename in sorted(glob.iglob(os.path.join(r"data/" + folderName, r'*.mp4'))):
            cap = cv2.VideoCapture(pathAndFilename)

            # Create the folder to store our vectors for this specific video class
            if not os.path.exists("vectors" + "/" + folderName):
                os.makedirs("vectors" + "/" + folderName)
            ##Make sure the file doesn't already exist
            if not os.path.exists(r"vectors/" + folderName + "/" + str(fileCount) + ".npy"):
                np.save(r"vectors/" + folderName + "/" + str(fileCount), self.x.convToVector(cap))
                print(pathAndFilename + "<< current video")
            else:
                print("file number " + str(fileCount) + " already exists, skipping")

            fileCount += 1
            ##Stop when max video ammount per class is reached
            if fileCount > max:
                break

    def createVectors(self):

        ##create array of folder names for all the clasess
        folderNameArray = [dI for dI in os.listdir('data') if os.path.isdir(os.path.join('data', dI))]

        for folder in folderNameArray:
            ##Second arg is maz videos per class
            self.doWork(folder, 2500)


x = createVectors()
x.createVectors()
