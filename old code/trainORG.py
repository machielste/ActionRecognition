import os
import pip
import tensorflow as tf
import numpy as np
from keras.optimizers import Adam
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras import models
from keras import layers
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D, Convolution2D
from keras.layers import Dense, Activation, Dropout, Flatten
from imutils import paths
import random
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator
import cv2

def create_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(640,490,3)))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(7, activation='softmax'))

    return model
#----VARS----#

def train():
    model = create_model()
    Epochs = 30
    BatchSize = 5
    data = []
    labels = []
    imagePaths = sorted(list(paths.list_images('dataset')))
    print(imagePaths)
    random.shuffle(imagePaths)
    for imagePath in imagePaths:
        image = cv2.imread(imagePath)
        image = img_to_array(image)
        data.append(image)
        label = imagePath.split(os.path.sep)[-2]
        if label == "anger":
            label = 0
        elif label == "disgust":
            label = 1
        elif label == "fear":
            label = 2
        elif label == "happy":
            label = 3
        elif label == "sadness":
            label = 4
        elif label == "surprise":
            label = 5
        elif label == "neutral":
            label = 6
        labels.append(label)
    data = np.array(data, dtype="float") / 255.0
    labels = np.array(labels)
    (trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25)
    adam = Adam(lr=0.0001)
    model.compile(optimizer=adam, loss="sparse_categorical_crossentropy", metrics=['accuracy'])
    model.fit(trainX, trainY, batch_size=BatchSize, epochs=Epochs, verbose=1, validation_data=(testX, testY), shuffle=True)
    model.save_weights('weightsORG.h5')

def load_model():
    model = create_model()
    model.load_weights('weightsORG.h5')
    return model
