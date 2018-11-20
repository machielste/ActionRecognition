import os

import cv2
import numpy as np
from keras.models import load_model
from sklearn import preprocessing

from videoToVectorList import Process


class classifyVideo:

    def __init__(self):
        self.x = Process()
        self.model = load_model("model.h5")

    def createVector(self, filePath):
        ##Pull all the frames of the video through the partial inceptionV3 network, returning a 2048 length vector
        x = Process()
        cap = cv2.VideoCapture(filePath)
        return x.convToVector(cap)

    def procesVectorList(self, vectorList):
        ##Pull processed video through our custom lstm model to classify it

        data = np.array(vectorList)
        data = np.expand_dims(data, axis=0)
        return self.model.predict(data)

    def getResult(self):
        ##Put all out video classes into a laben encoder, it will make the labels into one hot encoded integer arrays
        le = preprocessing.LabelEncoder()
        classes = [dI for dI in os.listdir('vectors') if os.path.isdir(os.path.join('vectors', dI))]
        le.fit(classes)

        ##predict what our video is
        prediction = (self.procesVectorList(self.createVector("video3.mp4"))[0])

        ##get index of largest value of prediction array
        resultList = []

        ##Retrieve the results in descending order
        for i in range(0, len(prediction)):
            index_max = np.argmax(prediction)
            x = np.array([index_max])
            result = le.inverse_transform(np.array(x))
            resultList.append([(result.tolist())[0], prediction[index_max]])
            prediction[index_max] = 0;
        return resultList
        ## find out what class belongs to that index

        return result

    def returnAwnser(self):
        result = self.getResult()
        return result
