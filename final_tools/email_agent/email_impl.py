import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication
import os
from typing import List, Union

def send_email(filepaths: Union[str, List[str]], recipient_email: str, subject: str = "Report"):
    body = "Please find the attached document(s). Thanks!"
    sender_email = "lmaproject123@gmail.com"
    recipient_email = recipient_email
    sender_password = "bsqldcrekusmaxqg"
    smtp_server = 'smtp.gmail.com'
    smtp_port = 465
    
    # Convert single file to list for uniform processing
    if isinstance(filepaths, str):
        filepaths = [filepaths]

    message = MIMEMultipart()
    message['Subject'] = subject
    message['From'] = sender_email
    message['To'] = recipient_email
    body_part = MIMEText(body)
    message.attach(body_part)

    # Attach all files
    for filepath in filepaths:
        with open(filepath, 'rb') as file:
            filename = os.path.basename(filepath)
            message.attach(MIMEApplication(file.read(), Name=filename))

    with smtplib.SMTP_SSL(smtp_server, smtp_port) as server:
        server.login(sender_email, sender_password)
        server.sendmail(sender_email, recipient_email, message.as_string())