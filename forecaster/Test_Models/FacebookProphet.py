from pandas import DataFrame
from pandas import to_datetime
from prophet import Prophet

from forecaster.Test_Models.TestModel import TestModel


class FacebookProphet(TestModel):
    def __init__(self):
        super().__init__("facebook_prophet")
        self.model = None

    def train_model(self, train, column_name):
        train_copy = train.copy(deep=True)
        train_copy.columns = ['ds', 'y']
        train_copy['ds'] = to_datetime(train_copy['ds'])

        # define the model
        # with suppress_stdout_stderr():
        model = Prophet().fit(train_copy)
        # fit the model
        self.model = model

    def get_prediction(self, train, test, column_name):
        test_copy = test.copy(deep=True)

        old_test_columns = test_copy.columns
        test_copy.columns = ['ds', 'y']
        test_copy['ds'] = to_datetime(test_copy['ds'])

        future = DataFrame(test_copy['ds'])
        future.columns = ['ds']

        # use the model to make a forecast
        # with suppress_stdout_stderr():
        forecast = self.model.predict(future)

        # calculate MAE between expected and predicted values for december
        y_pred = DataFrame(forecast['yhat'].values)
        y_pred.columns = ["{}_Predicted".format(column_name)]

        test_copy.columns = old_test_columns

        return y_pred["{}_Predicted".format(column_name)]

    def run_model(self, train, test, column_name):
        train.columns = ['ds', 'y']
        train['ds'] = to_datetime(train['ds'])

        old_test_columns = test.columns
        test.columns = ['ds', 'y']
        test['ds'] = to_datetime(test['ds'])

        # define the model
        # with suppress_stdout_stderr():
        model = Prophet().fit(train)
        # fit the model

        future = DataFrame(test['ds'])
        future.columns = ['ds']

        # use the model to make a forecast
        # with suppress_stdout_stderr():
        forecast = model.predict(future)

        # calculate MAE between expected and predicted values for december
        y_pred = DataFrame(forecast['yhat'].values)
        y_pred.columns = ["{}_Predicted".format(column_name)]

        test.columns = old_test_columns

        return y_pred["{}_Predicted".format(column_name)]
