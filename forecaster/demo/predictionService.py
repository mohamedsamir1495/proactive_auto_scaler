import collections
from datetime import timedelta
from queue import Queue
from threading import Thread
from timeit import default_timer as timer

import numpy as np
from numpy import mean
from pandas import read_csv
from pandas import to_datetime
from sklearn.metrics import mean_squared_error, mean_absolute_error

from forecaster.utils.generic_functions import *

pandas.options.mode.chained_assignment = None


def scale_down_A_data(data, column_name, experiment_type):
    if experiment_type == 'cpu':
        data[column_name] *= A_CPU_CORE_COUNT / 100
        data[column_name] = data[column_name].round(2)
    else:
        data[column_name] /= GIGA_BYTE_SIZE_IN_BYTES


def scale_down_B_data(data, column_name, experiment_type):
    if experiment_type == 'cpu':
        data[column_name] *= B_CPU_CORE_COUNT / 100
        data[column_name] = data[column_name].round(2)
    else:
        data[column_name] *= B_RAM_SIZE_IN_GIGA / 100


def scale_down_C_data(data, column_name, experiment_type):
    if experiment_type == 'cpu':
        data[column_name] *= C_CPU_CORE_COUNT / 100
        data[column_name] = data[column_name].round(2)
    else:
        data[column_name] /= GIGA_BYTE_SIZE_IN_BYTES


def prepare_data(data, app_name, experiment_type, period_scale):
    data["timestamp_date_format"] = pandas.Series(to_datetime(data["timestamp_date_format"])).dt.floor(period_scale)
    for ind, column in enumerate(data.columns):
        if ind >= 2:
            if app_name.lower() in 'a':
                scale_down_A_data(data, column, experiment_type)

            elif app_name.lower() in 'b':
                scale_down_B_data(data, column, experiment_type)

            elif app_name.lower() in 'c':
                scale_down_C_data(data, column, experiment_type)
    return data


def run_model(train, test, test_model, experiment_type, period_scale):
    column_name = test.columns[1]
    actual = test[column_name]
    x_axis = test["timestamp_date_format"]

    start = timer()
    predicted = test_model.run_model(train, test, column_name)
    end = timer()

    runtime = timedelta(seconds=end - start).seconds
    MAE = mean_absolute_error(actual, predicted)
    RMSE = mean_squared_error(actual, predicted, squared=False)

    plot_actual_vs_predicted(
        actual,
        predicted,
        x_axis,
        test_model.name,
        period_scale,
        experiment_type,
        column_name
    )
    # print_experiment_result(column_name, MAE, RMSE, experiment_type)
    # predicted = pandas.DataFrame(predicted)
    return pandas.concat([actual, predicted], axis=1), MAE, RMSE, runtime


def run_model_on_data(data, train_percentage, test_model, experiment_type, period_scale):
    train_size = int(len(data) * train_percentage)
    train, test = data[0:train_size], (data[train_size:]).reset_index(drop=True)

    experiment_results = pandas.concat([test[[test.columns[0], test.columns[1]]]], ignore_index=True)
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
            experiment_results = pandas.concat([experiment_results, results], axis=1)

    MAEs_as_df = pandas.DataFrame(np.array(MAEs), columns=['MAE'], index=np.arange(1, len(MAEs) + 1))
    RMSEs_as_df = pandas.DataFrame(np.array(RMSEs), columns=['RMSE'], index=np.arange(1, len(RMSEs) + 1))
    runtime_as_df = pandas.DataFrame(
        np.array(runtimes), columns=['Runtime In Seconds'], index=np.arange(1, len(runtimes) + 1)
    )

    return experiment_results, MAEs_as_df, RMSEs_as_df, runtime_as_df


def run_one_month_exp(app_name, month, train_percentage, period_scale, test_model, experiment_type):
    input_file_name = deduce_input_file_name(app_name, month, experiment_type)
    data = prepare_data(read_csv(input_file_name, header=0, parse_dates=True), app_name, experiment_type, period_scale)
    month_name = data.iloc[0].timestamp_date_format.strftime("%B")

    print('=========== Starting {} App. {} experiment on file {} ========='.format(
        app_name,
        experiment_type,
        input_file_name
    ))

    exp_results, MAEs, RMSEs, runtime = run_model_on_data(
        data,
        train_percentage,
        test_model,
        experiment_type,
        period_scale
    )
    exp_output_dir = 'results/{}/{}/month/{}'.format(app_name, experiment_type, month_name)
    exp_output_path = exp_output_dir + '/{}_{}_{}_train_{}_{}_results.xlsx'.format(
        app_name,
        experiment_type,
        month_name,
        train_percentage,
        test_model.name
    )
    export_results_as_excel([exp_results, MAEs, RMSEs, runtime], 'sheet 1', exp_output_dir, exp_output_path, 1)
    print('=========== Ending {} App. {} experiment on file {} ========='.format(
        app_name,
        experiment_type,
        input_file_name
    ))





def run_exp_on_all_months_separately(app_name, train_percentage, period_scale, test_model, experiment_type):
    input_file_names = deduce_input_files_package(app_name, experiment_type)

    results = None
    for idx, input_file_name in enumerate(input_file_names):
        print('=========== Starting {} App. {} experiment on file {} ========='.format(
            app_name,
            experiment_type,
            input_file_name
        ))
        data = prepare_data(read_csv(input_file_name, header=0, parse_dates=True), app_name, experiment_type,
                            period_scale)
        month_name = data.iloc[0].timestamp_date_format.strftime("%B")

        exp_results, MAEs, RMSEs, runtime = run_model_on_data(
            data,
            train_percentage,
            test_model,
            experiment_type,
            period_scale
        )
        if idx == 0:
            results = collections.ChainMap({
                month_name: collections.ChainMap({
                    "experiment_results": exp_results,
                    "MAE": MAEs,
                    "RMSE": RMSEs,
                    "Runtime": runtime
                })
            })
        else:
            results = results.new_child({
                month_name: collections.ChainMap({
                    "experiment_results": exp_results,
                    "MAE": MAEs,
                    "RMSE": RMSEs,
                    "Runtime": runtime
                })
            })

        exp_output_dir = 'results/{}/{}/month/{}'.format(app_name, experiment_type, month_name)
        exp_output_path = exp_output_dir + '/{}_{}_{}_train_{}_{}_results.xlsx'.format(
            app_name,
            experiment_type,
            month_name,
            train_percentage,
            test_model.name
        )
        export_results_as_excel([exp_results, MAEs, RMSEs, runtime], 'sheet 1', exp_output_dir, exp_output_path, 1)
        print('=========== Ending {} App. {} experiment on file {} ========='.format(
            app_name,
            experiment_type,
            input_file_name
        ))
    return results


def run_one_month_exp_in_parallel(app_name, month, train_percentage, period_scale, test_model, experiment_type,queue):
    input_file_name = deduce_input_file_name(app_name, month, experiment_type)
    data = prepare_data(read_csv(input_file_name, header=0, parse_dates=True), app_name, experiment_type, period_scale)
    month_name = data.iloc[0].timestamp_date_format.strftime("%B")

    print('=========== Starting {} App. {} experiment on file {} ========='.format(
        app_name,
        experiment_type,
        input_file_name
    ))

    exp_results, MAEs, RMSEs, runtime = run_model_on_data(
        data,
        train_percentage,
        test_model,
        experiment_type,
        period_scale
    )
    exp_output_dir = 'results/{}/{}/month/{}'.format(app_name, experiment_type, month_name)
    exp_output_path = exp_output_dir + '/{}_{}_{}_train_{}_{}_results.xlsx'.format(
        app_name,
        experiment_type,
        month_name,
        train_percentage,
        test_model.name
    )
    export_results_as_excel([exp_results, MAEs, RMSEs, runtime], 'sheet 1', exp_output_dir, exp_output_path, 1)
    print('=========== Ending {} App. {} experiment on file {} ========='.format(
        app_name,
        experiment_type,
        input_file_name
    ))
    queue.put({
        month_name: collections.ChainMap({
            "experiment_results": exp_results,
            "MAE": MAEs,
            "RMSE": RMSEs,
            "Runtime": runtime
        })})

def run_exp_on_all_months_separately_in_parallel(app_name, train_percentage, period_scale, test_model, experiment_type):
    q = Queue()
    arguments = (
        [app_name, "_aug_", train_percentage, period_scale, test_model, experiment_type, q],
        [app_name, "_sep_", train_percentage, period_scale, test_model, experiment_type, q],
        [app_name, "_oct_", train_percentage, period_scale, test_model, experiment_type, q],
        [app_name, "_nov_", train_percentage, period_scale, test_model, experiment_type, q],
        [app_name, "_dec_", train_percentage, period_scale, test_model, experiment_type, q],
        [app_name, "_jan_", train_percentage, period_scale, test_model, experiment_type, q],
        [app_name, "_feb_", train_percentage, period_scale, test_model, experiment_type, q],
        [app_name, "_mar_", train_percentage, period_scale, test_model, experiment_type, q],
        [app_name, "_apr_", train_percentage, period_scale, test_model, experiment_type, q],
    )
    threads = []

    for argument in arguments:
        t = Thread(target=run_one_month_exp_in_parallel, args=argument)
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    # result = [q.get() for _ in range(len(arguments))]
    results = None
    for idx in range(len(arguments)):
        one_month_result = q.get()
        if idx == 0:
            results = collections.ChainMap(one_month_result)
        else:
            results = results.new_child(one_month_result)
    return results


def get_avg_mae_for_all_months(app_name, train_percentage, period_scale, test_model, experiment_type):
    exp_results = run_exp_on_all_months_separately_in_parallel(app_name, train_percentage, period_scale, test_model,
                                                               experiment_type)

    all_month_mae_sum = 0
    all_month_rmse_sum = 0

    for month, month_results in exp_results.items():
        all_month_mae_sum += mean(month_results["MAE"])[0]
        all_month_rmse_sum += mean(month_results["RMSE"])[0]

    return all_month_mae_sum / len(exp_results), all_month_rmse_sum / len(exp_results)


def run_exp_on_between_start_and_end_date(
        app_name,
        start_date_str,
        end_date_str,
        train_percentage,
        period_scale,
        test_model,
        experiment_type
):
    start_date = to_datetime(start_date_str)
    end_date = to_datetime(end_date_str)

    input_file_names = list(
        filter(
            lambda file_name: is_file_name_between_start_and_end_date(file_name, start_date, end_date),
            deduce_input_files_package(app_name, experiment_type)
        )
    )

    big_data = pandas.DataFrame()
    for idx, input_file_name in enumerate(input_file_names):
        big_data = pandas.concat([big_data, read_csv(input_file_name, header=0, parse_dates=True)], ignore_index=True)

    big_data = prepare_data(big_data, app_name, experiment_type, period_scale)
    print('=========== Starting {} App. {} experiment ========='.format(
        app_name,
        experiment_type,

    ))

    big_data = big_data[(start_date <= big_data.timestamp_date_format) & (big_data.timestamp_date_format <= end_date)]

    exp_results, MAEs, RMSEs, runtime = run_model_on_data(big_data, train_percentage, test_model, experiment_type,
                                                          period_scale)
    print('=========== Ending {} App. {} experiment ========='.format(
        app_name,
        experiment_type,
        input_file_name
    ))
    exp_output_dir = 'results/{}/{}/periods'.format(app_name, experiment_type)
    exp_output_path = exp_output_dir + '/{}_{}_from_{}_to_{}_{}_results.xlsx'.format(
        app_name,
        experiment_type,
        start_date.strftime("%d-%b-%Y_%H-%M"),
        end_date.strftime("%d-%b-%Y_%H-%M"),
        test_model.name
    )
    export_results_as_excel([exp_results, MAEs, RMSEs, runtime], 'sheet 1', exp_output_dir, exp_output_path, 1)
