from twilio.rest import Client

from project_config import config

account_sid = config.TWILIO_SID
auth_token = config.TWILIO_TOKEN
from_config = config.TWILIO_WHATSAPP_FROM

client = Client(account_sid, auth_token)

def send_message(to: str, message: str) -> None:
    '''
    Send message to a Whatsapp user.
    Parameters:
        - to(str): sender whatsapp number in this whatsapp:+5218119759893 form
        - message(str): text message to send
    Returns:
        - None
    '''

    _ = client.messages.create(
        from_=from_config,
        body=message,
        to=to
    )