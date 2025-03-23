import cv2
import numpy as np
import sqlite3
import os
import sys
import ctypes
import voiceDetection
from dotenv import load_dotenv
from twilio.rest import Client
import bcrypt

def suppress_opencv_warnings():
    """Redirect stderr to suppress OpenCV warning messages."""
    original_stderr = sys.stderr
    null_device = open(os.devnull, 'w')
    sys.stderr = null_device
    return original_stderr, null_device

def restore_stderr(original_stderr, null_device):
    """Restore original stderr."""
    sys.stderr = original_stderr
    null_device.close()

load_dotenv("IDs.env")  # Load environment variables from .env file

# Load Twilio credentials from environment variables
account_sid = os.getenv("TWILIO_ACCOUNT_SID")
auth_token = os.getenv("TWILIO_AUTH_TOKEN")
verify_sid = os.getenv("TWILIO_VERIFY_SID")

# Ensure TWILIO_VERIFY_SID is set before proceeding
if not verify_sid:
    raise ValueError("ERROR: TWILIO_VERIFY_SID is not set. Please check your environment variables.")

# Initialize Twilio Client
client = Client(account_sid, auth_token)

# Path to Haar Cascade for face detection
CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

# Database setup
DB_PATH = "user_auth.db"


def initialize_database():
    """Creates the SQLite database and user table if not exists."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE,
            password TEXT,
            voice BLOB,
            face BLOB,
            phone TEXT
        )
    ''')
    conn.commit()
    conn.close()


initialize_database()

# Load the Haar Cascade classifier
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)


def bring_capture_window_to_front():
    """Brings the OpenCV window to the front on Windows systems."""
    if os.name == "nt":
        hwnd = ctypes.windll.user32.FindWindowW(None, "Face Capture")
        if hwnd:
            ctypes.windll.user32.ShowWindow(hwnd, 6)  # Minimize
            ctypes.windll.user32.ShowWindow(hwnd, 9)  # Restore

            ctypes.windll.user32.SetFocus(hwnd)
            ctypes.windll.user32.ShowWindow(hwnd, 5)  # SW_SHOW


def capture_face_image():
    """Capture an image from the webcam and detect the face."""
    # Suppress warnings before camera initialization
    original_stderr, null_device = suppress_opencv_warnings()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        # Restore stderr before printing error
        restore_stderr(original_stderr, null_device)
        print("Error: Could not access the camera.")
        return None

    print("Press 'SPACE' to capture, 'ESC' to exit.")

    cv2.namedWindow("Face Capture", cv2.WINDOW_NORMAL)
    cv2.moveWindow("Face Capture", 100, 100)  # Position the window
    bring_capture_window_to_front()

    while True:
        ret, frame = cap.read()
        if not ret:
            # Restore stderr before printing error
            restore_stderr(original_stderr, null_device)
            print("Error: Failed to capture image.")
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow("Face Capture", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == 27:  # ESC key to exit
            # Restore stderr before printing
            restore_stderr(original_stderr, null_device)
            print("Exiting without saving.")
            break
        elif key == 32:  # SPACE key to capture
            if len(faces) == 1:
                (x, y, w, h) = faces[0]
                face_img = gray[y:y + h, x:x + w]
                cap.release()
                cv2.destroyAllWindows()
                # Restore stderr
                restore_stderr(original_stderr, null_device)
                return cv2.resize(face_img, (100, 100))  # Ensure uniform size
            else:
                # Restore stderr before printing
                restore_stderr(original_stderr, null_device)
                print("No face detected or multiple faces detected. Try again.")

    cap.release()
    cv2.destroyAllWindows()
    # Restore stderr
    restore_stderr(original_stderr, null_device)
    return None

def send_2fa_code(phone_number):
    """Send a verification code via Twilio SMS."""
    verification = client.verify.v2.services(verify_sid).verifications.create(to=phone_number, channel="sms")
    return verification.sid


def verify_2fa_code(phone_number, code):
    """Verify the user's entered 2FA code."""
    verification_check = client.verify.v2.services(verify_sid).verification_checks.create(to=phone_number, code=code)
    return verification_check.status == "approved"


def register_user():
    """Register a new user with facial data."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    while True:
        username = input("Enter username to register: ").strip()
        cursor.execute("SELECT username FROM users WHERE username = ?", (username,))
        if cursor.fetchone() is not None:
            print(f"Username '{username}' already exists. Please enter a different username.")
        else:
            break  # Username is unique, proceed with registration

    password = input("Enter a password for the new user: ").strip()
    confirm_password = input("Confirm password: ").strip()

    # if the both inputted password matches, hash the password with a salt
    if password != confirm_password:
        print("Passwords do not match. Please try again.")
        return

    salt = bcrypt.gensalt()
    hashPass = bcrypt.hashpw(password.encode('utf-8'), salt)

    phone_number = input("Enter your phone number (e.g., 9057214116): ").strip()
    phone_number = "+1" + phone_number

    # Registering the voice of the specified user
    print("Registering voice...")
    voiceAudioBLOB = voiceDetection.registerVoice()

    face_img = capture_face_image()
    if face_img is not None:
        face_data = np.array(face_img).tobytes()
        try:
            cursor.execute("INSERT INTO users (username, password, voice, face, phone) VALUES (?, ?, ?, ?, ?)",
                           (username, hashPass, voiceAudioBLOB, face_data, phone_number))
            conn.commit()
            print(f"User '{username}' registered successfully.")
        except sqlite3.IntegrityError:
            print(f"Unexpected error: Username '{username}' should have been checked before insertion.")
        finally:
            conn.close()
    else:
        print("Registration failed: No face captured.")


def authenticate_user():
    """Authenticate a user using multiple authentication methods."""

    # all authentication booleans
    authenticatedFace = False
    authenticatedVoice = False
    authenticated2FA = False

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    username = input("Enter username for authentication: ").strip()
    cursor.execute("SELECT password, voice, face, phone FROM users WHERE username = ?", (username,))
    user_data = cursor.fetchone()

    if user_data is None:
        print("Authentication failed: User not found.")
        conn.close()
        return

    stored_password, stored_voice, stored_face, phone_number = user_data

    print("Step 1: Password authentication")
    inputPass = input("Enter your password: ")
    inputPass = inputPass.strip()

    # hashes the password and checks if it matches the salted hashed password stored on the database
    checkPass = inputPass.encode('utf-8')

    # checks if password hashes matches
    if bcrypt.checkpw(checkPass, user_data[0]):
        print("Authentication successful.")
    else:
        print("Authentication failed: Incorrect password.")

    print("Step 2: Voice authentication")
    authenticatedVoice = voiceDetection.authenticateVoice(stored_voice)

    print("Step 3: Face authentication")
    face_img = capture_face_image()
    if face_img is None:
        print("Face authentication failed.")
        conn.close()
        return

    min_distance = float('inf')

    face_img_resized = cv2.resize(face_img, (100, 100))
    stored_face_array = np.frombuffer(stored_face, dtype=np.uint8)

    try:
        stored_face_resized = cv2.resize(stored_face_array.reshape(100, 100), (100, 100))
        distance = np.mean((stored_face_resized - face_img_resized) ** 2)
        if distance < 1000:  # Threshold
            authenticatedFace = True
    except ValueError:
        print("Face data size mismatch.")

    conn.close()

    print("Step 4: 2FA Verification")
    send_2fa_code(phone_number)
    code = input("Enter the 2FA verification code sent to your phone: ")
    if verify_2fa_code(phone_number, code):
        authenticated2FA = True;
    else:
        print("2FA Verification failed, incorrect input.")

    if (authenticatedFace and authenticatedVoice and authenticated2FA):
        print(f"User '{username}' authenticated successfully.")
    else:
        print("Authentication failed.")


if __name__ == "__main__":
    while True:
        print("\nOptions:")
        print("1. Register User")
        print("2. Authenticate User")
        print("3. Exit")
        choice = input("Enter your choice: ").strip()

        if choice == '1':
            register_user()
        elif choice == '2':
            authenticate_user()
        elif choice == '3':
            print("Exiting program.")
            break
        else:
            print("Invalid choice. Please try again.")