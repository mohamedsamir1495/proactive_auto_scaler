from flask import Flask
from flask_mail import Mail, Message

app = Flask(__name__)

app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 465
app.config['MAIL_USERNAME'] = '*****'  # replace with your gmail username
app.config['MAIL_PASSWORD'] = '*****'  # replace with your gmail password
app.config['MAIL_USE_TLS'] = False
app.config['MAIL_USE_SSL'] = True
mail = Mail(app)


def sendEmail(recipients, app_id, exp_type, exp_output_path):
    msg = Message('{} Prediction Results for App with ID {}'.format(exp_type, app_id),
                  sender='*****@gmail.com',  # replace with your gmail email
                  recipients=recipients)
    msg.body = "Please find the {} prediction results for app with ID {} attached to this email".format(
        exp_type.lower(), app_id
    )
    with app.open_resource(exp_output_path) as fp:
        msg.attach(exp_output_path, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", fp.read())
    mail.send(msg)
    return "Sent"
