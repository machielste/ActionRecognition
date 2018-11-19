import os
from cv2 import VideoWriter_fourcc

import cv2
import numpy as np
from keras.models import load_model
from sklearn import preprocessing

from videoToVectorList import Process


class classifyAndMarkVideos:
    def __init__(self):
        self.x = Process()
        self.model = load_model("model.h5")

    def createVector(self, filePath):
        ##Pull all the frames of the video through the partial inceptionV3 network, returning a 2048 length vector
        cap = cv2.VideoCapture(filePath)
        return self.x.convToVector(cap)

    def procesVectorList(self, vectorList):
        ##Pull processed video through our custom lstm model to classify it
        data = np.array(vectorList)
        data = np.expand_dims(data, axis=0)
        return self.model.predict(data)

    def getResult(self, videoPath):
        ##Put all out video classes into a laben encoder, it will make the labels into one hot encoded integer arrays
        le = preprocessing.LabelEncoder()
        classes = [dI for dI in os.listdir('vectors') if os.path.isdir(os.path.join('vectors', dI))]
        le.fit(classes)

        ##Predict what our video is
        prediction = (self.procesVectorList(self.createVector(videoPath))[0])

        ##Get index of largest value of prediction array
        resultList = []

        ##Retrieve the results in descending order
        for i in range(0, len(prediction)):
            index_max = np.argmax(prediction)
            x = np.array([index_max])
            result = le.inverse_transform(np.array(x))
            resultList.append([(result.tolist())[0], prediction[index_max]])
            prediction[index_max] = 0
        return resultList
        ##Find out what class belongs to that index

        return result

    def gatherVideoClassifications(self):
        classificationList = []
        for path, subdirs, files in os.walk('videosToClassify'):
            for video in files:
                classificationList.append(self.getResult(os.path.join(path, video)))

        return classificationList

    def createVideoWithText(self, dataToWrite, video, count):
        ##Loop over every video and create a version of it with the right classification super imposed onto it.
        ext = 'mp4'
        codec = 'mp4v'
        savepath = "classifiedVideos/" + str(count) + "__%s.%s" % (codec, ext)

        cap = cv2.VideoCapture(video)

        fourcc = VideoWriter_fourcc(*codec)
        out = cv2.VideoWriter(savepath, fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))), 1)
        framecount = 0

        while True:
            ret, frame = cap.read()
            if ret:
                print("writing frame: " + str(framecount))
                framecount += 1
                ##Put text on our frame
                cv2.putText(frame, str(dataToWrite), (50, 50), cv2.FONT_HERSHEY_COMPLEX_SMALL, .7, (0, 0, 255))
                out.write(frame)
            else:
                ##Break the loop
                break

        ##When everything is done, release the video capture object
        cap.release()
        out.release()

    def createClassifiedVideos(self):
        ##Combine classification tags with their videos
        cList = self.gatherVideoClassifications()
        vList = []
        for path, subdirs, files in os.walk('videosToClassify'):
            for video in files:
                vList.append(os.path.join(path, video))

        for i in range(len(cList)):
            self.createVideoWithText(cList[i], vList[i], i)


x = classifyAndMarkVideos()
x.createClassifiedVideos()
