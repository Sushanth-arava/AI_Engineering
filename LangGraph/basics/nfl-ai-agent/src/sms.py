"""Send a plain SMS text message via Twilio."""

import os
from twilio.rest import Client

client = Client(os.environ["TWILIO_ACCOUNT_SID"], os.environ["TWILIO_AUTH_TOKEN"])


def send_sms(body: str) -> str:
    msg = client.messages.create(
        from_=os.environ["TWILIO_FROM"],   # your Twilio phone number, e.g. "+1XXXXXXXXXX"
        to=os.environ["TWILIO_TO"],        # destination number, e.g. "+1XXXXXXXXXX"
        body=body,
    )
    return msg.sid


if __name__ == "__main__":
    print(send_sms("NFL agent test"))