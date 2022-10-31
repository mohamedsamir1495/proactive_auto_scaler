import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

from apscheduler.schedulers.background import BackgroundScheduler
from flask import Flask, request

from cloud_provider.dummy.dummyService import getApplicationData
from forecaster.Test_Models.EnsembleModel import EnsembleModel
from forecaster.predictionService import getPredictionForAppResource
from forecaster.utils.generic_functions import export_results_as_excel
from mailer import sendEmail

app = Flask(__name__)

_threadpool_cpus = int(os.cpu_count() / 2)
EXECUTOR = ThreadPoolExecutor(max_workers=max(_threadpool_cpus, 2))


def preparePrediction(app_id, exp_details, email_recipients, start_date, points_to_predict):
    # getting the current timestamp
    end_date = int(datetime.timestamp(datetime.now()))

    # scrap the data from the cloud provider
    data = getApplicationData(app_id, exp_details["type"], start_date, end_date)

    # run the ensemble model and get the prediction results
    exp_results, MAEs, RMSEs, runtime, min_resource_value, max_resource_value = getPredictionForAppResource(
        app_id, data, EnsembleModel(14, points_to_predict, exp_details["period_scale"]), exp_details
    )

    # save the results to Excel file
    exp_output_dir = 'results/{}/{}'.format(app_id, exp_details["type"])
    exp_output_path = exp_output_dir + '/{}_{}_results.xlsx'.format(app_id, exp_details["type"])
    export_results_as_excel([exp_results, MAEs, RMSEs, runtime, min_resource_value, max_resource_value], 'sheet 1',
                            exp_output_dir, exp_output_path, 1)

    # Send results via Email
    sendEmail(email_recipients, app_id, exp_details["type"].upper(), exp_output_path)


def isRequestBodyValidForPredicition(body):
    if "emailRecipients" not in body.keys():
        return False
    if "startDate" not in body.keys():
        return False
    if "pointsToPredict" not in body.keys():
        return False
    if "expDetails" not in body.keys():
        return False

    return True


@app.route('/')
def index():
    return "Welcome to the course API"


@app.route('/predict/<int:app_id>', methods=['post'])
def predict(app_id):
    content_type = request.headers.get('Content-Type')
    if content_type == 'application/json':
        if not isRequestBodyValidForPredicition(request.json):
            return "Invalid JSON Body", 400
        email_recipients = request.json["emailRecipients"]
        start_date = request.json["startDate"]
        points_to_predict = request.json["pointsToPredict"]
        exp_details = request.json["expDetails"]
        preparePrediction(app_id, exp_details, email_recipients, start_date, points_to_predict)
        # EXECUTOR.submit(preparePrediction, app_id, exp_details, email_recipients, start_date, points_to_predict)
    else:
        return 'Content-Type not supported!'
    return "Your prediction request is processing and an email with the results will be sent to you :)"


def run_prediction_on_app_every_time_interval():
    app_id = 5009087
    email_recipients = ["icemaster71@gmail.com"]
    start_date = 1657051928
    points_to_predict = 14
    exp_details = {
        "type": 'cpu',
        "coreCount": 2,
        "period_scale": '13min'
    }
    preparePrediction(app_id, exp_details, email_recipients, start_date, points_to_predict)


# sched = BackgroundScheduler()
# sched.add_job(run_prediction_on_app_every_time_interval, 'interval', seconds=60)
# sched.start()

if __name__ == "__main__":
    app.run()