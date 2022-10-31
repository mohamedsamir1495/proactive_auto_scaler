import collections
from datetime import timedelta

import pandas as pd
from sklearn.metrics import mean_absolute_error

from forecaster.Test_Models.TestModel import TestModel
from forecaster.Test_Models.FacebookProphet import FacebookProphet
from forecaster.Test_Models.LSTM import LSTM
from forecaster.Test_Models.LinearRegression import LinearRegression
from forecaster.Test_Models.MovingAverage import MovingAverage
from forecaster.Test_Models.SVMPolynomialKernel import SVMPolynomialKernel
from forecaster.Test_Models.SVMRBFKernel import SVMRBFKernel
from forecaster.Test_Models.WeeklyAverage import WeeklyAverage


def train_models_on_data(train_data, column_name):
    weeklyAverage = WeeklyAverage()
    movingAverage = MovingAverage(2)
    lstm = LSTM(14, 1)
    linearRegression = LinearRegression()
    svmPolynomialKernel = SVMPolynomialKernel()
    svmRBFKernel = SVMRBFKernel()
    # facebookProphet = FacebookProphet()
    test_models = collections.ChainMap({
        lstm.name: lstm,
        weeklyAverage.name: weeklyAverage,
        movingAverage.name: movingAverage,
        # facebookProphet.name: facebookProphet,
        linearRegression.name: linearRegression,
        svmPolynomialKernel.name: svmPolynomialKernel,
        svmRBFKernel.name: svmRBFKernel
    })

    print('=========== Starting Model training process =========')
    for key, val in test_models.items():
        print('Training {} model'.format(key))
        val.train_model(train_data, column_name)
    print('=========== End Model training process =========')

    return test_models


def get_current_best_prediction_model(
        actual_result_of_last_record,
        previous_prediction,
        train,
        test_models,
        column_name,
        previous_best_model_name,
):
    best_mae = mean_absolute_error(actual_result_of_last_record[column_name], previous_prediction.iloc[:, -1:])
    current_best_model_name = previous_best_model_name
    for key, val in test_models.items():
        if (current_best_model_name == key):
            continue
        else:
            proposed_previous_prediction = val.get_prediction(train, actual_result_of_last_record, column_name) \
                .fillna(value=99999999)

            if best_mae == None:
                best_mae = mean_absolute_error(actual_result_of_last_record[column_name], proposed_previous_prediction)
            else:
                current_mae = mean_absolute_error(actual_result_of_last_record[column_name],
                                                  proposed_previous_prediction)
                if current_mae < best_mae:
                    best_mae = current_mae
                    current_best_model_name = key

    return test_models[current_best_model_name]


class EnsembleModel(TestModel):
    def __init__(self, backward_window, forward_window, period_scale):
        super().__init__("Ensemble_Model_with_bkWindow_{}_and_fdWindow_{}".format(backward_window, forward_window))
        self.backward_window = backward_window
        self.forward_window = forward_window
        self.period_scale = period_scale

    def run_model(self, train, test, column_name):
        backward_window = self.backward_window
        forward_window = self.forward_window
        period_scale = self.period_scale

        test_models = train_models_on_data(train, column_name)

        current_test_records = train[-forward_window - 1:-1].reset_index(drop=True)
        best_predictions = pd.DataFrame()

        current_best_prediction_model = list(test_models.values())[0].name

        step_end = None
        if len(test) == 0:  # Means we are not testing the model but using it for actual prediction
            step_end = forward_window
        else:  # Means we are testing the model against test data
            step_end = len(test)

        for idx in range(0, step_end, forward_window):
            if idx == 0:
                predicted = test_models[current_best_prediction_model].get_prediction(train, current_test_records,
                                                                                      column_name)
                timestamp_for_predicted = train[-forward_window:]['timestamp_date_format'] + \
                                          timedelta(minutes=int(period_scale[0:len(period_scale) - 3]))
                best_predictions = pd.concat(
                    [
                        best_predictions,
                        pd.concat(
                            [timestamp_for_predicted.reset_index(drop=True),
                             predicted.reset_index(drop=True)], axis=1)
                    ],
                    ignore_index=True)
                continue
            else:
                previous_predicted_record_count = min(len(best_predictions), backward_window)

                previous_prediction = best_predictions[-previous_predicted_record_count:].reset_index(drop=True)
                actual_result_of_last_record = test[idx - previous_predicted_record_count:idx].reset_index(
                    drop=True)

                current_best_test_model = get_current_best_prediction_model(
                    actual_result_of_last_record,
                    previous_prediction,
                    train,
                    test_models,
                    column_name,
                    current_best_prediction_model
                )
                train = pd.concat([train, actual_result_of_last_record], ignore_index=True)
                current_test_records = test[idx:min(len(test), idx + forward_window)].reset_index(drop=True)

                predicted = current_best_test_model.get_prediction(train, current_test_records, column_name)
                timestamp_for_predicted = test[idx:min(len(test), idx + forward_window)]['timestamp_date_format'] \
                                          + timedelta(minutes=int(period_scale[0:len(period_scale) - 3]))

                best_predictions = pd.concat(
                    [
                        best_predictions,
                        pd.concat(
                            [timestamp_for_predicted.reset_index(drop=True),
                             predicted.reset_index(drop=True)], axis=1)
                    ],
                    ignore_index=True)
                continue

        return best_predictions["{}_Predicted".format(column_name)]
