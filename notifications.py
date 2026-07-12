import requests
import json
import threading
import smtplib
import os
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dotenv import load_dotenv

load_dotenv()

def send_webhook_alert(event_type: str, description: str, webhook_url: str):
    """
    Sends a generic HTTP POST webhook (e.g. to Slack, Teams, or a custom server)
    without blocking the main thread.
    """
    if not webhook_url:
        return
        
    def _post():
        try:
            payload = {
                "text": f"🚨 *SafeSight Alert: {event_type}*\n{description}"
            }
            requests.post(webhook_url, json=payload, timeout=5)
        except Exception as e:
            print(f"[SafeSight] Failed to send webhook: {e}")
            
    # Fire and forget
    threading.Thread(target=_post, daemon=True).start()

def send_email_alert(event_type: str, description: str, recipient_email: str):
    """
    Sends an email alert using SMTP.
    """
    if not recipient_email:
        return

    sender = os.getenv("EMAIL_SENDER")
    password = os.getenv("EMAIL_PASSWORD")

    if not sender or not password:
        print("[SafeSight] Email credentials not found in .env. Skipping email alert.")
        return

    def _send_email():
        try:
            msg = MIMEMultipart()
            msg['From'] = sender
            msg['To'] = recipient_email
            msg['Subject'] = f"🚨 SafeSight Alert: {event_type}"

            body = f"An incident was detected by SafeSight AI.\n\nEvent Type: {event_type}\nDescription: {description}\n\nPlease check the dashboard for more details."
            msg.attach(MIMEText(body, 'plain'))

            server = smtplib.SMTP('smtp.gmail.com', 587)
            server.starttls()
            server.login(sender, password)
            server.send_message(msg)
            server.quit()
        except Exception as e:
            print(f"[SafeSight] Failed to send email: {e}")

    # Fire and forget
    threading.Thread(target=_send_email, daemon=True).start()
