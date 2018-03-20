import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt
import warnings

from tensorflow.contrib import learn
from tensorflow.contrib.learn.python import SKCompat
from sklearn.metrics import mean_squared_error

import lstm_predictior as predictor

from pandas import DataFrame
from pandas import Series
from pandas import concat

# from lstm_predictor import generate_data, lstm_model

warnings.filterwarnings("ignore")

LOG_DIR = 'resources/logs/'
TIMESTEPS = 1
RNN_LAYERS = [{'num_units': 400}]
DENSE_LAYERS = None
TRAINING_STEPS = 3000
PRINT_STEPS = TRAINING_STEPS  # / 10
BATCH_SIZE = 1

regressor = SKCompat(learn.Estimator(model_fn=predictor.lstm_model(TIMESTEPS, RNN_LAYERS, DENSE_LAYERS), ))
#   model_dir=LOG_DIR)
from pandas import read_csv
from pandas import Series

from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

series = read_csv('../CorpData/InventoryHistory/2010_2018_books_sortable inventory.csv',
                  header=0, parse_dates=[0], index_col=0, squeeze=True, usecols=[0, 4])


def difference(dataset, interval=1):
    diff = list()
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return Series(diff)


def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


n_lag = 1
n_seq = 1
n_test = 30
n_val = 30

raw_values = series.values
# transform data to be stationary
diff_series = difference(raw_values, 1)
diff_values = diff_series.values
diff_values = diff_values.reshape(len(diff_values), 1)
# rescale values to -1, 1
scaler = MinMaxScaler(feature_range=(-1, 1))
scaled_values = scaler.fit_transform(diff_values)
scaled_values = scaled_values.reshape(len(scaled_values), 1)
# transform into supervised learning problem X, y
supervised = series_to_supervised(scaled_values, n_lag, n_seq)
supervised_values = supervised.values
# split into train and test sets
train, test, val = supervised_values[0:-n_test - n_val], \
                   supervised_values[-n_test - n_val: -n_test], \
                   supervised_values[-n_val:]

# create a lstm instance and validation monitor
test_X, test_y = test[:, 0:n_lag], test[:, n_lag:]
test_X = test_X.reshape(test_X.shape[0], 1, test_X.shape[1])

val_X, val_y = val[:, 0:n_lag], val[:, n_lag:]
val_X = val_X.reshape(val_X.shape[0], 1, val_X.shape[1])
validation_monitor = learn.monitors.ValidationMonitor(val_X, val_y, )
# every_n_steps=PRINT_STEPS,)
# early_stopping_rounds=1000)
# print(X['train'])
# print(y['train'])
train_X, train_y = train[:, 0:n_lag], train[:, n_lag:]
train_X = train_X.reshape(train_X.shape[0], 1, train_X.shape[1])

SKCompat(regressor.fit(train_X, train_y,
                       monitors=[validation_monitor],
                       batch_size=BATCH_SIZE,
                       steps=TRAINING_STEPS))

print('X train shape', train_X.shape)
print('y train shape', train_y.shape)

print('X test shape', test_X.shape)
print('y test shape', test_y.shape)
predicted = np.asmatrix(regressor.predict(test_X), dtype=np.float32)  # ,as_iterable=False))
predicted = np.transpose(predicted)

rmse = np.sqrt((np.asarray((np.subtract(predicted, test_y))) ** 2).mean())
# this previous code for rmse was incorrect, array and not matricies was needed: rmse = np.sqrt(((predicted - y['test']) ** 2).mean())  
score = mean_squared_error(predicted, test_y)
nmse = score / np.var(
    test_y)  # should be variance of original data and not data from fitted model, worth to double check

print("RSME: %f" % rmse)
print("NSME: %f" % nmse)
print("MSE: %f" % score)

plot_test, = plt.plot(test_y, label='test')
plot_predicted, = plt.plot(predicted, label='predicted')
plt.legend(handles=[plot_predicted, plot_test])
plt.show()
