from collections import deque

from keras.layers import Dense, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential, load_model
from keras.optimizers import Adadelta


class LstmModel():
    def __init__(self, features_length=512):
        ##Init for custom lstm model
        self.load_model = load_model
        self.feature_queue = deque()

        metrics = ['accuracy']

        print("Loading LSTM model.")
        self.input_shape = (87, features_length)
        self.model = self.lstm()

        ##Now compile the network.
        optimizer = Adadelta()
        self.model.compile(loss='categorical_crossentropy', optimizer=optimizer,
                           metrics=metrics)

        print(self.model.summary())

    def lstm(self):
        ##Custom Lstm model.
        model = Sequential()
        model.add(LSTM(512, return_sequences=False,
                       input_shape=self.input_shape,
                       dropout=0.5, recurrent_dropout=0.4
                       ))
        model.add(Dense(1024, activation='relu', ))
        model.add(Dropout(0.5))
        model.add(Dense(1, activation='softmax'))

        return model
