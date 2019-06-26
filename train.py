import os
import random

import keras
import keras.utils.vis_utils
import numpy as np
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

from model import get_model

config = tf.ConfigProto()
config.gpu_options.allow_growth = True  ##Dynamically grow the memory used on the GPU
config.log_device_placement = True  ##To log device placement (on which device the operation ran)
sess = tf.Session(config=config)
set_session(sess)


def train():
    # init
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True  ##Dynamically grow the memory used on the GPU
    config.log_device_placement = True  ##To log device placement (on which device the operation ran)
    sess = tf.Session(config=config)
    set_session(sess)

    Epochs = 50
    BatchSize = 32

    encoder = preprocessing.LabelEncoder()
    classes = [dI for dI in os.listdir('vectors') if os.path.isdir(os.path.join('vectors', dI))]
    encoder.fit(classes)

    model = get_model()

    data = []
    labels = []
    imagePaths = []
    videosPerClass = 0

    # Get all the vector files and check their label, save all of it to the label and data lists
    for path, subdirs, files in os.walk('vectors'):
        for name in files:
            if videosPerClass > 2500:
                videosPerClass = 0
                break

            imagePaths.append(os.path.join(path, name))
            videosPerClass += 1
    random.shuffle(imagePaths)
    for imagePath in imagePaths:
        numpy = np.load(imagePath)
        data.append(numpy)
        label = imagePath.split('\\')[1]
        labels.append(label)

    labels = encoder.transform(np.asarray(labels))
    (X_train, X_test, y_train, y_test) = train_test_split(data, labels, test_size=0.25)

    # One hot encoding
    y_train = keras.utils.to_categorical(y_train, num_classes=10)
    y_test = keras.utils.to_categorical(y_test, num_classes=10)

    model.fit(np.array(X_train), np.array(y_train), batch_size=BatchSize, epochs=Epochs, verbose=1,
              validation_data=(np.array(X_test), np.array(y_test)), shuffle=True)

    model.save('model.h5')


if __name__ == '__main__':
    train()
