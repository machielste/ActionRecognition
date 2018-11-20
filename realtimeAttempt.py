import os
import time

import cv2
import numpy as np
from keras.models import load_model
from sklearn import preprocessing

from imageToSingleVector import Extractor


class realtime:
    def __init__(self):
        self.imToVec = Extractor()
        self.model = load_model("model.h5")
        self.le = preprocessing.LabelEncoder()
        self.classes = [dI for dI in os.listdir('vectors') if os.path.isdir(os.path.join('vectors', dI))]
        self.le.fit(self.classes)
        self.start_time = time.time()

        # current model has 72% val accuracy

    def function(self):
        framearray = []

        cap = cv2.VideoCapture("videoSmaller.mp4")

        while (True):
            # Capture frame-by-frame
            ret, frame = cap.read()

            inceptionFrame = self.imToVec.extract(frame)

            # Buffer of the last 87 frames for the lstm network to process
            if len(framearray) < 87:

                for i in range(87):
                    framearray.append(inceptionFrame)
            else:

                framearray.pop()
                framearray.append(inceptionFrame)

                ##Put text on our frame, containing the current framerate and the classification
                cv2.putText(frame, str(self.returnAwnser(framearray)), (20, 50),
                            cv2.FONT_HERSHEY_COMPLEX_SMALL, .7,
                            (0, 0, 255))

                cv2.putText(frame, "fps: " + str(round(1 / (time.time() - self.start_time), 3)), (20, 80),
                            cv2.FONT_HERSHEY_COMPLEX_SMALL, .7,
                            (0, 0, 255))
                self.start_time = time.time()

            # Our operations on the frame come here

            # Display the resulting frame
            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # When everything done, release the capture
        cap.release()
        cv2.destroyAllWindows()

    def procesVectorList(self, vectorList):
        ##Pull processed video through our custom lstm model to classify it

        data = np.array(vectorList)
        data = np.expand_dims(data, axis=0)
        return self.model.predict(data)

    def getResult(self, array):

        ##predict what our video is
        prediction = (self.procesVectorList(array)[0])

        ##get index of largest value of prediction array
        resultList = []

        ##Retrieve the results in descending order
        for i in range(0, len(prediction)):
            index_max = np.argmax(prediction)
            x = np.array([index_max])
            result = self.le.inverse_transform(np.array(x))
            resultList.append([(result.tolist())[0], prediction[index_max]])
            prediction[index_max] = 0
        return resultList
        ## find out what class belongs to that index

        return result

    def returnAwnser(self, array):
        result = self.getResult(array)
        return result[0]


x = realtime()
x.function()
