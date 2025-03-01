import os
from twilio.rest import Client

# Load Twilio credentials from environment variables
account_sid = os.getenv("TWILIO_ACCOUNT_SID")
auth_token = os.getenv("TWILIO_AUTH_TOKEN")

client = Client(account_sid, auth_token)

# User enters the code received via SMS
verification_code = input("Enter the verification code: ")

# Verify the code
verification_check = client.verify \
    .v2 \
    .services('VAe006659fb42245790567f41506584e51') \
    .verification_checks \
    .create(to='+14167218116', code=verification_code)

# Check verification status
if verification_check.status == "approved":
    print("✅ Verification successful!")
else:
    print("❌ Verification failed. Please try again.")