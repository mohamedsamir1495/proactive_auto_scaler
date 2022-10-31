from statsmodels.tsa.ar_model import AutoReg

from forecaster.Test_Models.TestModel import TestModel


class LinearRegression(TestModel):
    def __init__(self):
        super().__init__("linear_regression")
        self.model = None
        self.train_copy = None

    def train_model(self, train, column_name):
        train_copy = train.copy(deep=True)
        model_fit = AutoReg(train_copy[column_name], lags=24).fit()
        self.model = model_fit
        self.train_copy = train_copy

    def get_prediction(self, train, test, column_name):
        test_copy = test.copy(deep=True)
        prediction = self.model.predict(start=len(self.train_copy), end=len(self.train_copy) + (len(test) - 1), dynamic=True)
        test_copy["{}_Predicted".format(column_name)] = prediction.reset_index(drop=True)

        return test_copy["{}_Predicted".format(column_name)]

    def run_model(self, train, test, column_name):
        model_fit = AutoReg(train[column_name], lags=24).fit()
        prediction = model_fit.predict(start=len(train), end=len(train) + (len(test) - 1), dynamic=True)
        test["{}_Predicted".format(column_name)] = prediction.reset_index(drop=True)

        return test["{}_Predicted".format(column_name)]
