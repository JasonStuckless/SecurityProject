# to run: 
# pip install twilio
# pip install python-dotenv

import os
from dotenv import load_dotenv
from twilio.rest import Client

load_dotenv()  # Load environment variables from .env file

# Load Twilio credentials from environment variables
account_sid = os.getenv("TWILIO_ACCOUNT_SID")
auth_token = os.getenv("TWILIO_AUTH_TOKEN")
if not account_sid or not auth_token:
    raise ValueError("Twilio credentials not found. Check environment variables or .env file.")

client = Client(account_sid, auth_token)

# Send code via SMS to the specified phone number
verification = client.verify \
    .v2 \
    .services('VAe006659fb42245790567f41506584e51') \
    .verifications \
    .create(to='+14167218116', channel='sms')

print(verification.sid)