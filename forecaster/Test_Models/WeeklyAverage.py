import pandas

from forecaster.Test_Models.TestModel import TestModel


class WeeklyAverage(TestModel):
    def __init__(self):
        super().__init__("weekly_average")

    def train_model(self, train, column_name):
        pass

    def get_prediction(self, train, test, column_name):
        train_copy = train.copy(deep=True)
        test_copy = test.copy(deep=True)

        train_copy["day"] = train_copy["timestamp_date_format"].dt.dayofweek
        train_copy['time'] = train_copy["timestamp_date_format"].apply(lambda x: x.time())
        prediction_per_day_and_hour = train_copy.groupby(["day", "time"]).mean()
        prediction_per_day_and_hour.columns = ["{}_Predicted".format(column_name)]

        test_copy["day"] = test_copy["timestamp_date_format"].dt.dayofweek
        test_copy['time'] = test_copy["timestamp_date_format"].apply(lambda x: x.time())

        prediction_on_test = pandas.merge(test_copy, prediction_per_day_and_hour, on=['day', 'time'],how='left')
        prediction_on_test = prediction_on_test.fillna(value=0)
        prediction_on_test.sort_values(by=['timestamp_date_format'], inplace=True)

        return prediction_on_test["{}_Predicted".format(column_name)]

    def run_model(self, train, test, column_name):
        train_copy = train.copy(deep=True)
        test_copy = test.copy(deep=True)

        train_copy["day"] = train_copy["timestamp_date_format"].dt.dayofweek
        train_copy['time'] = train_copy["timestamp_date_format"].apply(lambda x: x.time())
        prediction_per_day_and_hour = train_copy.groupby(["day", "time"]).mean()
        prediction_per_day_and_hour.columns = ["{}_Predicted".format(column_name)]

        test_copy["day"] = test_copy["timestamp_date_format"].dt.dayofweek
        test_copy['time'] = test_copy["timestamp_date_format"].apply(lambda x: x.time())

        prediction_on_test = pandas.merge(test_copy, prediction_per_day_and_hour, on=['day', 'time'],how='left')
        prediction_on_test = prediction_on_test.fillna(value=99999999)
        prediction_on_test.sort_values(by=['timestamp_date_format'], inplace=True)

        return prediction_on_test["{}_Predicted".format(column_name)]
