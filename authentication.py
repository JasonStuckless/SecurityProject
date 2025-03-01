import cv2
import numpy as np
import sqlite3
import os
import ctypes

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
            fingerprint BLOB,
            voice BLOB,
            face BLOB
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
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not access the camera.")
        return None

    print("Press 'SPACE' to capture, 'ESC' to exit.")

    cv2.namedWindow("Face Capture", cv2.WINDOW_NORMAL)
    cv2.moveWindow("Face Capture", 100, 100)  # Position the window
    bring_capture_window_to_front()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image.")
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow("Face Capture", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == 27:  # ESC key to exit
            print("Exiting without saving.")
            break
        elif key == 32:  # SPACE key to capture
            if len(faces) == 1:
                (x, y, w, h) = faces[0]
                face_img = gray[y:y + h, x:x + w]
                cap.release()
                cv2.destroyAllWindows()
                return cv2.resize(face_img, (100, 100))  # Ensure uniform size
            else:
                print("No face detected or multiple faces detected. Try again.")

    cap.release()
    cv2.destroyAllWindows()
    return None

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

    while True:
        password = input("Enter a password for the new user: ").strip()
        confirm_password = input("Confirm password: ").strip()
        if password != confirm_password:
            print("Passwords do not match. Please try again.")
        else:
            break

    print("Step 1: Fingerprint registration (Placeholder)")
    print("Step 2: Voice registration (Placeholder)")

    face_img = capture_face_image()
    if face_img is not None:
        face_data = np.array(face_img).tobytes()
        try:
            cursor.execute("INSERT INTO users (username, password, fingerprint, voice, face) VALUES (?, ?, ?, ?, ?)",
                           (username, password, None, None, face_data))
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
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    username = input("Enter username for authentication: ").strip()
    cursor.execute("SELECT password, fingerprint, voice, face FROM users WHERE username = ?", (username,))
    user_data = cursor.fetchone()

    if user_data is None:
        print("Authentication failed: User not found.")
        conn.close()
        return

    stored_password, stored_fingerprint, stored_voice, stored_face = user_data

    print("Step 1: Password authentication (Placeholder)")
    input("Enter your password: ")  # Placeholder for actual password validation

    print("Step 2: Fingerprint authentication (Placeholder)")
    print("Processing fingerprint authentication...")  # Placeholder for actual fingerprint verification

    print("Step 3: Voice authentication (Placeholder)")
    print("Processing voice authentication...")  # Placeholder for actual voice verification

    print("Step 4: Face authentication")
    face_img = capture_face_image()
    if face_img is None:
        print("Authentication failed.")
        conn.close()
        return

    min_distance = float('inf')
    authenticated = False

    face_img_resized = cv2.resize(face_img, (100, 100))
    stored_face_array = np.frombuffer(stored_face, dtype=np.uint8)

    try:
        stored_face_resized = cv2.resize(stored_face_array.reshape(100, 100), (100, 100))
        distance = np.mean((stored_face_resized - face_img_resized) ** 2)
        if distance < 1000:  # Threshold
            authenticated = True
    except ValueError:
        print("Face data size mismatch.")

    conn.close()

    if authenticated:
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