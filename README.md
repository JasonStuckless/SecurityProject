**Secure User Authentication System**  
**SOFE 4840U: Software & Computer Security**

Group 18 Members:  
Javier Chung  100785653  
Andy Dai  100726784  
Antonio Lio  100805668  
Jason Stuckless  100248154  

How to Run:
1. Download repository.
2. Install the following packages:
```bash
pip install numpy==2.2.4
pip install opencv-python==4.9.0.80
pip install pyaudio
pip install pyannote.audio
pip install scipy
pip install torch
pip install python-dotenv
pip install twilio
pip install bcrypt
```
3. Enter environment variables in IDs.env. (details in the report)
4. On line 42 of voiceDetection.py, and line 1406 of gui.py enter value for user_auth_token. (details in the report)
5. Run gui.py in IDE of your choice or in a terminal with:
```bash
python gui.py
```
6. Register as a user in program.
7. Login with that account.

Program runs through four types of authentication, username/password, voice, facial and SMS 2FA.
