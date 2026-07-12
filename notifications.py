import requests
import json
import threading

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
