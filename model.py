from keras.layers import Dense, Dropout, CuDNNLSTM, CuDNNGRU
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.optimizers import Adadelta


def get_model():
    input_shape = (87, 512)
    model = lstm(input_shape)

    # Compile the chosen network
    model.compile(loss='categorical_crossentropy', optimizer=Adadelta(), metrics=['accuracy'])

    return model


def lstm(input_shape):
    ##Custom Lstm model.
    model = Sequential()
    model.add(LSTM(512,
                   return_sequences=False,
                   input_shape=input_shape,
                   dropout=0.5,
                   recurrent_dropout=0.4,
                   unroll=True,
                   unit_forget_bias=True
                   ))
    model.add(Dense(256, activation='relu', ))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))

    return model


def CuDNNLSTM(self):
    ##Custom CuDNNLSTM model.
    model = Sequential()
    model.add(CuDNNLSTM(512))
    model.add(Dense(256, activation='relu', ))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))

    return model


def CuDNNGRU(self):
    ##Custom GruDNNLSTM model.
    model = Sequential()
    model.add(CuDNNGRU(512))
    model.add(Dense(256, activation='relu', ))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))

    return model
