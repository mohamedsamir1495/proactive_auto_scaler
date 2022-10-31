from datetime import timedelta
from timeit import default_timer as timer

import numpy as np
from pandas import to_datetime
from sklearn.metrics import mean_squared_error, mean_absolute_error

import pandas as pd

from forecaster.utils.constants import GIGA_BYTE_SIZE_IN_BYTES

pd.options.mode.chained_assignment = None


def prepare_data(data, exp_details):
    period_scale = exp_details["period_scale"]
    experiment_type = exp_details["type"].lower()
    data["timestamp_date_format"] = pd.Series(to_datetime(data["timestamp_date_format"])).dt.floor(period_scale)
    for ind, column in enumerate(data.columns):
        if ind >= 2:
            if experiment_type == 'cpu':
                data[column] *= exp_details["coreCount"] / 100
                data[column] = data[column].astype(float).round(2)
            else:
                data[column] = data[column].astype(float) / GIGA_BYTE_SIZE_IN_BYTES
    return data


def run_model(train, test, test_model, experiment_type, period_scale):
    column_name = test.columns[1]
    actual = test[column_name]
    x_axis = test["timestamp_date_format"]

    start = timer()
    predicted = test_model.run_model(train, test, column_name)
    end = timer()

    runtime = timedelta(seconds=end - start).seconds
    MAE = None
    RMSE = None
    if len(test):
        MAE = mean_absolute_error(actual, predicted)
        RMSE = mean_squared_error(actual, predicted, squared=False)

    return pd.concat([actual, predicted], axis=1), MAE, RMSE, runtime


def run_model_on_data(data, train_percentage, test_model, experiment_type, period_scale):
    train_size = int(len(data) * train_percentage)
    train, test = data[0:train_size], (data[train_size:]).reset_index(drop=True)

    experiment_results = pd.concat([test[[test.columns[0], test.columns[1]]]], ignore_index=True)
    MAEs = []
    RMSEs = []
    runtimes = []
    for ind, column in enumerate(train.columns):
        if ind >= 2:
            results, MAE, RMSE, runtime = run_model(
                train[[train.columns[1], column]].copy(deep=True),
                test[[test.columns[1], column]].copy(deep=True),
                test_model,
                experiment_type,
                period_scale
            )
            MAEs.append(MAE)
            RMSEs.append(RMSE)
            runtimes.append(runtime)
            experiment_results = pd.concat([experiment_results, results], axis=1)

    MAEs_as_df = pd.DataFrame(np.array(MAEs), columns=['MAE'], index=np.arange(1, len(MAEs) + 1))
    RMSEs_as_df = pd.DataFrame(np.array(RMSEs), columns=['RMSE'], index=np.arange(1, len(RMSEs) + 1))
    runtime_as_df = pd.DataFrame(
        np.array(runtimes), columns=['Runtime In Seconds'], index=np.arange(1, len(runtimes) + 1)
    )

    return experiment_results, MAEs_as_df, RMSEs_as_df, runtime_as_df


def getPredictionForAppResource(app_id, data, test_model, exp_details):
    print('=========== Starting prediction on app with id : {} ========='.format(app_id))

    data = prepare_data(data, exp_details)
    exp_results, MAEs, RMSEs, runtime = run_model_on_data(
        data,
        1,
        test_model,
        exp_details["type"],
        exp_details["period_scale"]
    );

    max_resource_value = 0
    min_resource_value = 9999999999999
    for ind, column in enumerate(exp_results.columns):
        if ind >= 2 and "Predicted" in column:
            max_resource_value = max(max(exp_results[column]), max_resource_value)
            min_resource_value = min(min(exp_results[column]), min_resource_value)

    print('=========== Ending prediction on app with id : {} ========='.format(app_id))
    max_resource_value_as_array = [max_resource_value]
    min_resource_value_as_array = [min_resource_value]

    # Create the pandas DataFrame with column name is provided explicitly
    max_resource_value_df = pd.DataFrame(max_resource_value_as_array,
                                         columns=['max_{}_value'.format(exp_details["type"])])
    min_resource_value_df = pd.DataFrame(min_resource_value_as_array,
                                         columns=['min_{}_value'.format(exp_details["type"])])
    return exp_results, MAEs, RMSEs, runtime, min_resource_value_df, max_resource_value_df
