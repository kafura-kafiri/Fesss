import numpy
from keras.models import Sequential
from keras.layers import Dense, LSTM
# fix random seed for reproducibility
numpy.random.seed(7)
max_types = 10


def config(model):
    model = Sequential()
    model.add(LSTM(5, input_shape=(1, max_types)))
    model.add(Dense(1))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
