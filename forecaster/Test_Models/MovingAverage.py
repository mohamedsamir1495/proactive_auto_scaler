from forecaster.Test_Models.TestModel import TestModel


class MovingAverage(TestModel):
    def __init__(self, rolling_window):
        super().__init__("moving_average_with_{}_rolling_window".format(rolling_window))
        self.rolling_window = rolling_window

    def train_model(self, train, column_name):
        pass

    def get_prediction(self, train, test, column_name):
        train_copy = train.copy(deep=True)
        test_copy = test.copy(deep=True)
        rolling_window = self.rolling_window
        test_copy["{}_Predicted".format(column_name)] = test_copy[column_name].shift(1).rolling(
            window=rolling_window).mean()
        test_copy["{}_Predicted".format(column_name)][0] = train_copy[column_name][-rolling_window:].mean()

        for idx in range(1, self.rolling_window):
            test_copy["{}_Predicted".format(column_name)][idx] = (train_copy[column_name][
                                                                  -rolling_window + idx:].sum() +
                                                                  test_copy[column_name][0:idx].sum()
                                                                  ) / rolling_window

        return test_copy["{}_Predicted".format(column_name)]

    def run_model(self, train, test, column_name):
        rolling_window = self.rolling_window
        test["rolling_average"] = test[column_name].shift(1).rolling(window=rolling_window).mean()
        test["rolling_average"][0] = train[column_name][-rolling_window:].mean()

        for idx in range(1, self.rolling_window):
            test["rolling_average"][idx] = (train[column_name][-rolling_window + idx:].sum() +
                                            test[column_name][0:idx].sum()) / rolling_window

        return test["rolling_average"]

