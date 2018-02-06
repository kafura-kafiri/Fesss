# LSTM for international airline passengers problem with regression framing
import numpy
from pandas import read_csv
import datetime
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error



# fix random seed for reproducibility
numpy.random.seed(7)
# load the dataset


def parse(x):
    return datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')


dataframe = read_csv('fesss.csv', parse_dates=['Date'], engine='python', date_parser=parse)
dataset = dataframe.values
start = dataset[0, 0]

for i in range(len(dataset)):
    _start = dataset[i, 0]
    dataset[i, 0] = (dataset[i, 0] - start).total_seconds()
    start = _start


dataset = dataset.astype('float32')
# normalize the dataset
delta_scaler = MinMaxScaler(feature_range=(0, 1))
delay_scaler = MinMaxScaler(feature_range=(0, 1))


# print(dataset)
def scale(scaler, dataset, i):
    data = dataset[:, i]
    data = data.reshape(data.shape[0], 1)
    data = scaler.fit_transform(data)
    dataset[:, i] = data.reshape(data.shape[0])
    return dataset


dataset = scale(delta_scaler, dataset, 0)
dataset = scale(delay_scaler, dataset, 1)

# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back):
        l = [dataset[i + 1][0]]
        l.extend(dataset[i:(i + look_back), 1])
        l.append(dataset[i + 1][2])
        dataX.append(l)
        dataY.append(dataset[i + look_back, 0])
    return numpy.array(dataX), numpy.array(dataY)


look_back = 1
dataX, dataY = create_dataset(dataset, look_back)
# reshape input to be [samples, time steps, features]
dataX = numpy.reshape(dataX, (dataX.shape[0], 1, dataX.shape[1]))
print(dataset)
print(dataX)
print(dataY)

# create and fit the LSTM network
model = Sequential()
model.add(LSTM(4, input_shape=(1, look_back + 2)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(dataX, dataY, epochs=100, batch_size=1, verbose=2)
# make predictions
trainPredict = model.predict(dataX)
from math import sqrt
rmse = sqrt(mean_squared_error(dataY, trainPredict))
print('RMSE: %.3f' % rmse)
