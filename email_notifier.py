import os, smtplib, ssl
from email.message import EmailMessage
from concurrent.futures import ThreadPoolExecutor

from email_config import EMAIL_HOST_PASSWORD_KEY, EMAIL_HOST_USER_NAME, EMAIL_RECIPIENT_USER_NAME

_executor = ThreadPoolExecutor(max_workers=2)

SENDER = EMAIL_HOST_USER_NAME
PASSWORD = EMAIL_HOST_PASSWORD_KEY
RECIPIENT = EMAIL_RECIPIENT_USER_NAME  # می‌توان چند گیرنده با جداکردن با کاما گذاشت

def _build_message(subject: str, body: str) -> EmailMessage:
    msg = EmailMessage()
    msg["From"] = SENDER
    msg["To"] = RECIPIENT
    msg["Subject"] = subject
    msg.set_content(body)
    return msg

def _send(subject: str, body: str):
    if not (SENDER and PASSWORD and RECIPIENT):
        print("Email env vars missing; skip sending.")
        return
    try:
        msg = _build_message(subject, body)
        context = ssl.create_default_context()
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as smtp:
            smtp.login(SENDER, PASSWORD)
            smtp.send_message(msg)
    except Exception as e:
        print(f"Email send error: {e}")

def send_trade_email_async(subject: str, body: str):
    _executor.submit(_send, subject, body)