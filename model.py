from keras.layers import Dense, Dropout, CuDNNLSTM, CuDNNGRU
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.optimizers import Adadelta


def get_model():
    """
    Choose and compile a model in preparation for training

    Used concepts:

    Categorical crossentropy will compare the distribution of the predictions (the activations in the output layer,
    one for each class)
    with the true distribution, where the probability of the true class is set to 1 and 0 for the other classes.

    Adadelta dynamically adapts over time using only first order information and has minimal computational overhead
    beyond vanilla stochastic gradient descent

    :return: Compiled model, ready to be trained
    """
    input_shape = (87, 512)
    model = lstm(input_shape)

    # Compile the chosen network
    model.compile(loss='categorical_crossentropy', optimizer=Adadelta(), metrics=['accuracy'])

    return model


def lstm(input_shape):
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


def CuDNNLSTM():
    model = Sequential()
    model.add(CuDNNLSTM(512))
    model.add(Dense(256, activation='relu', ))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))

    return model


def CuDNNGRU():
    model = Sequential()
    model.add(CuDNNGRU(512))
    model.add(Dense(256, activation='relu', ))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))

    return model
