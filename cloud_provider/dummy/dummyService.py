import pandas as pd
from pandas import read_csv
from pandas import to_datetime

from forecaster.utils.constants import GIGA_BYTE_SIZE_IN_BYTES


def prepare_data(data, resource_type, period_scale):
    DUMMY_APP_CPU_CORE_COUNT = 2
    data["timestamp_date_format"] = pd.Series(to_datetime(data["timestamp_date_format"])).dt.floor(period_scale)
    for ind, column in enumerate(data.columns):
        if ind >= 2:
            if resource_type.lower() == 'cpu':
                data[column] *= DUMMY_APP_CPU_CORE_COUNT / 100
                data[column] = data[column].round(2)
            else:
                data[column] /= GIGA_BYTE_SIZE_IN_BYTES
    return data


"""
- This is a dummy function that simulates calling your cloud hosting provider with your
        app_id => your app id on the cloud provider side
        resource_type => resource you want to forecast
        start_date => the start date of history date you want to fetch
        end_date => the end date of history date you want to fetch

- The returned data is returned in a format that the forecasting service can work on

- You should implement a similar function to work with your cloud provider and return data in the proper format
"""
def getApplicationData(app_id, resource_type, start_date, end_date):
    input_file_name = None
    period_scale = None
    if resource_type.lower() in "cpu":
        input_file_name = "cloud_provider/dummy/data/CPU/dummy_cpu_usage.csv"
        period_scale = "13min"
    elif resource_type.lower() in "ram":
        input_file_name = "cloud_provider/dummy/data/RAM/dummy_ram_usage.csv"
        period_scale = "40min"

    # data = prepare_data(read_csv(input_file_name, header=0, parse_dates=True), resource_type, period_scale)
    data = read_csv(input_file_name, header=0, parse_dates=True)

    return data
