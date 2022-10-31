from pathlib import Path

import pandas

from forecaster.utils.constants import *


def deduce_input_files_package(app_name, experiment_type):
    if app_name.lower() in 'a':
        if experiment_type.lower() == 'cpu':
            return A_CPU_INPUT_FILE_NAMES
        else:
            return A_RAM_INPUT_FILE_NAMES

    elif app_name.lower() in 'b':
        if experiment_type.lower() == 'cpu':
            return B_CPU_INPUT_FILE_NAMES
        else:
            return B_RAM_INPUT_FILE_NAMES

    elif app_name.lower() in 'c':
        if experiment_type.lower() == 'cpu':
            return C_CPU_INPUT_FILE_NAMES
        else:
            return C_RAM_INPUT_FILE_NAMES



def deduce_input_file_name(app_name, month, experiment_type):
    input_file_name = None
    if app_name.lower() in 'a':
        if experiment_type.lower() == 'cpu':
            input_file_name = list(filter(lambda file_name: month.lower() in file_name, A_CPU_INPUT_FILE_NAMES))[0]
        else:
            input_file_name = list(filter(lambda file_name: month.lower() in file_name, A_RAM_INPUT_FILE_NAMES))[0]

    elif app_name.lower() in 'b':
        if experiment_type.lower() == 'cpu':
            input_file_name = list(filter(lambda file_name: month.lower() in file_name, B_CPU_INPUT_FILE_NAMES))[0]
        else:
            input_file_name = list(filter(lambda file_name: month.lower() in file_name, B_RAM_INPUT_FILE_NAMES))[0]

    elif app_name.lower() in 'c':
        if experiment_type.lower() == 'cpu':
            input_file_name = list(filter(lambda file_name: month.lower() in file_name, C_CPU_INPUT_FILE_NAMES))[0]
        else:
            input_file_name = list(filter(lambda file_name: month.lower() in file_name, C_RAM_INPUT_FILE_NAMES))[0]

    return input_file_name


def print_experiment_result(column_name, MAE, RMSE, experiment_type):
    if experiment_type.lower() == 'cpu':
        print('CPU MAE & RMSE for column {} respectively'.format(column_name), 'is : %.3f  , %.3f' % (MAE, RMSE))
        # print('CPU RMSE for column {}'.format(column_name), 'is: %.3f' % RMSE)
    else:
        print('RAM MAE & RMSE for column {} respectively'.format(column_name), 'is : %.3f  , %.3f' % (MAE, RMSE))


def deduce_the_month_of_input_file(input_file_name):
    if '_jan_' in input_file_name.lower():
        return 'jan', 1
    elif '_feb_' in input_file_name.lower():
        return 'feb', 2
    elif '_mar_' in input_file_name.lower():
        return 'mar', 3
    elif '_apr_' in input_file_name.lower():
        return 'apr', 4
    elif '_may_' in input_file_name.lower():
        return 'may', 5
    elif '_jun_' in input_file_name.lower():
        return 'jun', 6
    elif '_jul_' in input_file_name.lower():
        return 'jul', 7
    elif '_aug_' in input_file_name.lower():
        return 'aug', 8
    elif '_sep_' in input_file_name.lower():
        return 'sep', 9
    elif '_oct_' in input_file_name.lower():
        return 'oct', 10
    elif '_nov_' in input_file_name.lower():
        return 'nov', 11
    elif '_dec_' in input_file_name.lower():
        return 'dec', 12


def is_file_name_between_start_and_end_date(input_file_name, start_date, end_date):
    month_name, month_num = deduce_the_month_of_input_file(input_file_name)
    year_of_file = int(input_file_name[-8:-4])
    if start_date.year == year_of_file == end_date.year:
        return start_date.month <= month_num <= end_date.month
    else:
        if start_date.year < year_of_file:
            return month_num <= end_date.month
        else:
            return start_date.month <= month_num


def plot_actual_vs_predicted(actual, predicted, x_axis_values, model_name, period_scale, experiment_type, app_name):
    return
    # fig, ax = pyplot.subplots()
    #
    # ax.plot(x_axis_values, actual, label='Actual')
    # ax.plot(x_axis_values, predicted, label='Predicted')

    # ax.xaxis.set_major_formatter(dates.DateFormatter('%b-%d'))
    #
    # pyplot.xlim(x_axis_values[0], x_axis_values[len(x_axis_values) - 1])
    # pyplot.xlabel("time (every {})".format(period_scale))
    # pyplot.title(model_name)
    # pyplot.suptitle(app_name)
    #
    # pyplot.ylabel(experiment_type)
    # pyplot.legend()
    #
    # fig.autofmt_xdate()
    #
    # pyplot.show()


def export_results_as_excel(df_list, sheets, file_dir, file_name, spaces):
    output_dir = Path(file_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    writer = pandas.ExcelWriter(file_name, engine='xlsxwriter')
    row = 0

    for dataframe in df_list:
        dataframe.to_excel(writer, sheet_name=sheets, startrow=row, startcol=0)
        row = row + len(dataframe.index) + spaces + 1
    writer.save()

