import os
import time

import cv2
import numpy as np
from keras.models import load_model
from sklearn import preprocessing

from imageToVector import Extractor

imToVec = Extractor()
model = load_model("model.h5")
le = preprocessing.LabelEncoder()
classes = [dI for dI in os.listdir('vectors') if os.path.isdir(os.path.join('vectors', dI))]
le.fit(classes)


def realtime():
    """
    Pick a video to display with real time classification results by our CNN + RNN network
    """
    extracted_frames_buffer = []
    cap = cv2.VideoCapture("realtimeVideos/drinking.mp4")
    start_time = time.time()

    while True:
        # Read the next frame
        ret, frame = cap.read()

        # If we are out of frames, exit
        if not ret:
            cap.release()
            cv2.destroyAllWindows()
            quit()

        extracted_frame = imToVec.extract(frame)

        # Buffer of the last 87 frames for the which the LSTM network will process every frame
        # Since the network allays needs a full buffer, artificially fill the buffer with the first extracted frame
        if len(extracted_frames_buffer) < 87:
            for i in range(87):
                extracted_frames_buffer.append(extracted_frame)
        else:
            extracted_frames_buffer.pop(0)
            extracted_frames_buffer.append(extracted_frame)

            put_text_on_frame(frame, get_prediction(extracted_frames_buffer), round(1 / (time.time() - start_time), 3))

            start_time = time.time()

        # Display the resulting frame
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


def get_prediction(frame_array):
    """
    Get the raw prediction for the network, and process it into a singular label with a confidence score
    :param frame_array: list of 87 vectors, which are extracted frame, to be fed to the LSTM network
    :return: singular label with a confidence score
    """
    # Predict what our video is
    prediction = (proces_vectorlist(frame_array)[0])

    # Get index of largest value of prediction array
    resultList = []

    # Retrieve the results in descending order
    for i in range(0, len(prediction)):
        index_max = np.argmax(prediction)
        x = np.array([index_max])
        result = le.inverse_transform(np.array(x))
        resultList.append([(result.tolist())[0], prediction[index_max]])
        prediction[index_max] = 0
    # Only return the most confident class
    return resultList[0]


def proces_vectorlist(vector_list):
    """
    Retrieve a raw prediction for a list of extracted vectors
    :param vectorList: list of 87 vectors, which are extracted frame, to be fed to the LSTM network
    :return: raw prediction
    """
    data = np.array(vector_list)
    data = np.expand_dims(data, axis=0)
    return model.predict(data)


def put_text_on_frame(frame, prediction, fps):
    """
    Draw framerate, the prediction, and the confidence score on the frame to be displayed
    :param frame: current frame object being processed
    :param prediction: class and confidence score
    :param fps: framerate, speed at which the realtime classification is running
    :return: nothing, since this function edits an existing object
    """
    cv2.putText(frame, str(prediction), (20, 50),
                cv2.FONT_HERSHEY_COMPLEX_SMALL, 2,
                (0, 0, 255))

    cv2.putText(frame, "fps: " + str(fps), (20, 80),
                cv2.FONT_HERSHEY_COMPLEX_SMALL, 2,
                (0, 0, 255))


if __name__ == '__main__':
    realtime()
