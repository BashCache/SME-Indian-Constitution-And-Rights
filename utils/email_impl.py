import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication

def send_email(filepath, recipient_email):
    subject = "Report"
    body = "Please find the attached document. Thanks!"
    sender_email = "lmaproject123@gmail.com"
    recipient_email = recipient_email
    sender_password = "bsqldcrekusmaxqg"
    smtp_server = 'smtp.gmail.com'
    smtp_port = 465
    path_to_file = filepath

    message = MIMEMultipart()
    message['Subject'] = subject
    message['From'] = sender_email
    message['To'] = recipient_email
    body_part = MIMEText(body)
    message.attach(body_part)

    with open(path_to_file,'rb') as file:
        message.attach(MIMEApplication(file.read(), Name="example.csv"))

    with smtplib.SMTP_SSL(smtp_server, smtp_port) as server:
        server.login(sender_email, sender_password)
        server.sendmail(sender_email, recipient_email, message.as_string())