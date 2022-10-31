import pandas
from keras.layers import Dense
from keras.layers import LSTM as lstm
from keras.models import Sequential
from numpy import array

# split a univariate sequence into samples
from forecaster.Test_Models.TestModel import TestModel


def split_sequence(sequence, n_steps):
    X, y = list(), list()
    for i in range(len(sequence)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the sequence
        if end_ix > len(sequence) - 1:
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix], sequence.iloc[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)


class LSTM(TestModel):
    def __init__(self, n_steps, n_features):
        super().__init__("LSTM_with_n_steps_{}".format(n_steps))
        self.n_steps = n_steps
        self.n_features = n_features
        self.model = None

    def train_model(self, train, column_name):
        # define input sequence
        raw_seq = [10, 20, 30, 40, 50, 60, 70, 80, 90]
        # choose a number of time steps
        n_steps = self.n_steps
        # split into samples
        X, y = split_sequence(train[column_name], n_steps)
        # reshape from [samples, timesteps] into [samples, timesteps, features]
        n_features = self.n_features
        X = X.reshape((X.shape[0], X.shape[1], n_features))
        # define model
        model = Sequential()
        model.add(lstm(50, activation='relu', input_shape=(n_steps, n_features)))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')
        # fit model
        model.fit(X, y, epochs=200, verbose=0)
        self.model = model

    def get_prediction(self, train, test, column_name):
        # demonstrate prediction
        n_steps = self.n_steps
        n_features = self.n_features

        last_n_records_in_train = train.iloc[-n_steps:]
        new_test = pandas.concat([last_n_records_in_train, test], ignore_index=True)
        # new_test = test

        X_test, y_test = split_sequence(new_test[column_name], n_steps)
        x_input = X_test.reshape((X_test.shape[0], n_steps, n_features))

        yhat = pandas.DataFrame(self.model.predict(x_input, verbose=0))
        yhat.columns = ["{}_Predicted".format(column_name)]

        return yhat["{}_Predicted".format(column_name)]

    def run_model(self, train, test, column_name):
        # define input sequence
        raw_seq = [10, 20, 30, 40, 50, 60, 70, 80, 90]
        # choose a number of time steps
        n_steps = self.n_steps
        # split into samples
        X, y = split_sequence(train[column_name], n_steps)
        # reshape from [samples, timesteps] into [samples, timesteps, features]
        n_features = self.n_features
        X = X.reshape((X.shape[0], X.shape[1], n_features))
        # define model
        model = Sequential()
        model.add(lstm(50, activation='relu', input_shape=(n_steps, n_features)))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')
        # fit model
        model.fit(X, y, epochs=200, verbose=0)
        # demonstrate prediction
        last_n_records_in_train = train.iloc[-n_steps:]
        new_test = last_n_records_in_train.append(test, ignore_index=True)

        X_test, y_test = split_sequence(new_test[column_name], n_steps)
        x_input = X_test.reshape((X_test.shape[0], n_steps, n_features))

        yhat = pandas.DataFrame(model.predict(x_input, verbose=0))
        yhat.columns = ["{}_Predicted".format(column_name)]

        return yhat["{}_Predicted".format(column_name)]
