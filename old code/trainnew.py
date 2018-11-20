import os
import pip
import tensorflow as tf
import numpy as np
from keras import Sequential
from keras.optimizers import Adam
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.applications.inception_v3 import InceptionV3
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, GlobalMaxPooling2D, Convolution2D
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from sklearn.model_selection import train_test_split
from keras.layers import Dense, Activation, Dropout, Flatten
import random
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator
import cv2

def create_model(weights='imagenet'):
    model = Sequential()
    # let's add a fully-connected layer
    model.add(Dense(2048, activation='relu'))
    # and a logistic layer
    model.add(Dense(2, activation='softmax'))

    # this is the model we will train
    return model


def train():
    model = create_model()
    Epochs = 3
    BatchSize = 5
    data = []
    labels = []
    imagePaths = []
    for path, subdirs, files in os.walk('vectors'):
        for name in files:
            imagePaths.append(os.path.join(path, name))
    random.shuffle(imagePaths)
    for imagePath in imagePaths:
        numpy = np.load(imagePath)
        data.append(numpy)
        label = imagePath.split('\\')[1]
        if label == "anger":
            label = 0
        elif label == "contempt":
            label = 1

        labels.append(label)
    data_gen = ImageDataGenerator()
    data = np.array(data, dtype="float") / 255.0
    #data = np.expand_dims(data, 2)
    #data = np.expand_dims(data, 3)
    #data = np.put(data,[2,3],[1,3])

    labels = np.array(labels)
    (trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25)

    adam = Adam(lr=0.001)
    model.compile(optimizer=adam, loss="sparse_categorical_crossentropy", metrics=['accuracy'])
    #model.fit_generator(data_gen.flow_from_directory('../dataset'),steps_per_epoch=100, epochs=50, verbose=1)
    model.fit(trainX, trainY, batch_size=BatchSize, epochs=Epochs, verbose=1, validation_data=(testX, testY), shuffle=True)
    model.save_weights('weights.h5')

def load_model():
    model = create_model()
    model.load_weights('weights.h5')
    return model

train()