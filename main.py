import os
import random

import keras
import numpy as np
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.utils import compute_class_weight

from createVectors import createVectors
from model import LstmModel

config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.log_device_placement = True  # to log device placement (on which device the operation ran)
sess = tf.Session(config=config)
set_session(sess)


def createV():
    x = createVectors()
    x.createVectors()


def train():
    ##init
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
    config.log_device_placement = True  # to log device placement (on which device the operation ran)
    sess = tf.Session(config=config)
    set_session(sess)
    x = LstmModel()
    le = preprocessing.LabelEncoder()
    classes = [dI for dI in os.listdir('vectors') if os.path.isdir(os.path.join('vectors', dI))]
    le.fit(classes)
    model = x.model
    Epochs = 25
    BatchSize = 22
    data = []
    labels = []
    imagePaths = []
    videosPerClass = 0

    ##Get all the vector files and check their label, save all of it to the label and data lists
    for path, subdirs, files in os.walk('vectors'):
        for name in files:
            if videosPerClass > 1150:
                videosPerClass = 0
                break

            imagePaths.append(os.path.join(path, name))
            videosPerClass += 1
    random.shuffle(imagePaths)
    for imagePath in imagePaths:
        numpy = np.load(imagePath)
        # for val in numpy:
        data.append(numpy)
        label = imagePath.split('\\')[1]
        labels.append(label)

    ##Normalize labels using LabelEncoder
    labels = np.array(labels)
    labels = le.transform(labels)

    (X_train, X_test, y_train, y_test) = train_test_split(data, labels, test_size=0.25)

    ##Account for differences in file count from one class to the other
    class_weights = compute_class_weight('balanced', (np.unique(y_train)), y_train)

    ##One hot encoding
    y_train = keras.utils.to_categorical(y_train, num_classes=10)
    y_test = keras.utils.to_categorical(y_test, num_classes=10)

    ##Save checkpoint files to get the very best version of the model after a training session
    filepath = "weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
    # checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    # callbacks_list = [checkpoint]

    model.fit(np.array(X_train), np.array(y_train), batch_size=BatchSize, epochs=Epochs, verbose=1,
              validation_data=(np.array(X_test), np.array(y_test)), shuffle=True, class_weight=class_weights, )

    model.save('model.h5')


train()
