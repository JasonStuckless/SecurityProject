import sys
import os
import io
import threading
import time
import wave
import pyaudio
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QLabel,
                             QVBoxLayout, QHBoxLayout, QWidget, QLineEdit,
                             QStackedWidget, QMessageBox, QDialog, QProgressBar,
                             QInputDialog, QFrame, QSpacerItem, QSizePolicy, QStyle)
from PyQt5.QtGui import QPixmap, QImage, QFont, QIcon, QColor, QPalette, QBrush, QLinearGradient, QPainter, QPen, QPainterPath
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread, QSize, QRectF, QPointF
import cv2
import numpy as np
import authentication
import voiceDetection


# Redirect stderr to suppress OpenCV warnings
class OpenCVWarningSupressor:
    def __init__(self):
        self.devnull = None
        self.original_stderr = None

    def suppress(self):
        self.original_stderr = sys.stderr
        self.devnull = open(os.devnull, 'w')
        sys.stderr = self.devnull

    def restore(self):
        if self.original_stderr:
            sys.stderr = self.original_stderr
            self.original_stderr = None
        if self.devnull:
            self.devnull.close()
            self.devnull = None


# Create a global instance
cv_warning_suppressor = OpenCVWarningSupressor()

# Suppress OpenCV warnings globally
cv_warning_suppressor.suppress()

def get_app_stylesheet():
    """Returns the global stylesheet for the application."""
    return """
    QMainWindow, QDialog {
        background-color: #f0f4f8;
    }
    QWidget {
        font-family: 'Arial', sans-serif;
    }
    QPushButton {
        border-radius: 6px;
        padding: 10px 15px;
        font-weight: bold;
        font-size: 14px;
    }
    QPushButton.primary {
        background-color: #3b82f6;
        color: white;
        border: none;
    }
    QPushButton.primary:hover {
        background-color: #2563eb;
    }
    QPushButton.primary:pressed {
        background-color: #1d4ed8;
    }
    QPushButton.secondary {
        background-color: white;
        color: #3b82f6;
        border: 1px solid #93c5fd;
    }
    QPushButton.secondary:hover {
        background-color: #f0f9ff;
    }
    QPushButton.secondary:pressed {
        background-color: #e0f2fe;
    }
    QLineEdit {
        border: 1px solid #cbd5e1;
        border-radius: 4px;
        padding: 10px;
        background-color: white;
        selection-background-color: #3b82f6;
    }
    QLineEdit:focus {
        border: 1px solid #3b82f6;
    }
    QLabel.title {
        font-size: 24px;
        font-weight: bold;
        color: #1e293b;
    }
    QLabel.subtitle {
        font-size: 16px;
        color: #64748b;
    }
    QLabel.heading {
        font-size: 20px;
        font-weight: bold;
        color: #1e293b;
    }
    QProgressBar {
        border: none;
        border-radius: 4px;
        background-color: #e2e8f0;
        text-align: center;
        height: 10px;
    }
    QProgressBar::chunk {
        background-color: #3b82f6;
        border-radius: 4px;
    }
    .card {
        background-color: white;
        border-radius: 8px;
        padding: 20px;
        border: 1px solid #e2e8f0;
    }
    """

def suppress_opencv_warnings():
    """
    Redirects stderr to suppress OpenCV warning messages.
    Call this before initializing any camera capture.
    """
    # Save the original stderr so we can restore it if needed
    original_stderr = sys.stderr

    # Redirect stderr to null device
    null_device = open(os.devnull, 'w')
    sys.stderr = null_device

    return original_stderr, null_device

def create_icon_frame(icon_path, size=64):
    """Creates a circular frame with an icon."""
    frame = QFrame()
    frame.setFixedSize(size, size)
    frame.setStyleSheet(f"""
        background-color: #3b82f6;
        border-radius: {size // 2}px;
    """)

    icon_label = QLabel(frame)
    icon_label.setPixmap(QIcon(icon_path).pixmap(size // 1.5, size // 1.5))
    icon_label.setAlignment(Qt.AlignCenter)
    icon_label.setStyleSheet("background-color: transparent;")

    layout = QVBoxLayout(frame)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.addWidget(icon_label)

    return frame


def create_feature_card(icon_path, text):
    """Creates a feature card with an icon and text."""
    card = QFrame()
    card.setStyleSheet("""
        background-color: white;
        border-radius: 8px;
        border: 1px solid #e2e8f0;
        padding: 15px;
    """)

    layout = QVBoxLayout(card)
    layout.setAlignment(Qt.AlignCenter)
    layout.setSpacing(10)

    icon_label = QLabel()
    icon_label.setPixmap(QIcon(icon_path).pixmap(32, 32))
    icon_label.setAlignment(Qt.AlignCenter)

    text_label = QLabel(text)
    text_label.setAlignment(Qt.AlignCenter)
    text_label.setStyleSheet("color: #64748b; font-size: 13px;")
    text_label.setWordWrap(True)

    layout.addWidget(icon_label)
    layout.addWidget(text_label)

    return card


def create_styled_button(text, is_primary=True, icon_path=None):
    """Creates a styled button with consistent appearance."""
    btn = QPushButton(text)

    if is_primary:
        btn.setProperty("class", "primary")
        btn.setStyleSheet("""
            QPushButton {
                background-color: #3b82f6;
                color: white;
                border-radius: 6px;
                font-weight: bold;
                font-size: 15px;
                padding: 12px 15px;
            }
            QPushButton:hover {
                background-color: #2563eb;
            }
            QPushButton:pressed {
                background-color: #1d4ed8;
            }
        """)
    else:
        btn.setProperty("class", "secondary")
        btn.setStyleSheet("""
            QPushButton {
                background-color: white;
                color: #3b82f6;
                border: 1px solid #93c5fd;
                border-radius: 6px;
                font-weight: bold;
                font-size: 15px;
                padding: 12px 15px;
            }
            QPushButton:hover {
                background-color: #f0f9ff;
            }
            QPushButton:pressed {
                background-color: #e0f2fe;
            }
        """)

    if icon_path:
        btn.setIcon(QIcon(icon_path))
        btn.setIconSize(QSize(18, 18))

    btn.setFixedHeight(50)
    return btn

# Override the is_recording event for GUI control
voiceDetection.isRecording = threading.Event()


class VoiceRecordingThread(QThread):
    finished = pyqtSignal(bytes)

    def run(self):
        # Call the record function but handle the stopping via GUI
        audio_data = authentication.voiceDetection.registerVoice()
        self.finished.emit(audio_data)


class WebcamCaptureThread(QThread):
    update_frame = pyqtSignal(np.ndarray)
    face_captured = pyqtSignal(np.ndarray)

    def __init__(self):
        super().__init__()
        self.running = True
        self.face_cascade = cv2.CascadeClassifier(authentication.CASCADE_PATH)
        # Store original stderr
        self.original_stderr = None
        self.null_device = None

    def run(self):
        # Suppress OpenCV warnings before camera initialization
        self.original_stderr = sys.stderr
        self.null_device = open(os.devnull, 'w')
        sys.stderr = self.null_device

        # Initialize camera
        cap = cv2.VideoCapture(0)
        while self.running:
            ret, frame = cap.read()
            if ret:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))

                # Draw rectangle around detected faces
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                self.update_frame.emit(frame)

                # Save last detected face for capture button
                if len(faces) == 1:
                    self.last_face = gray[faces[0][1]:faces[0][1] + faces[0][3], faces[0][0]:faces[0][0] + faces[0][2]]
                    self.last_face_frame = frame

        # Release camera and restore stderr
        cap.release()
        sys.stderr = self.original_stderr
        if self.null_device:
            self.null_device.close()

    def capture_face(self):
        if hasattr(self, 'last_face'):
            resized_face = cv2.resize(self.last_face, (100, 100))
            self.face_captured.emit(resized_face)
            return True
        return False

    def stop(self):
        self.running = False
        self.wait()


class FaceDialog(QDialog):
    face_selected = pyqtSignal(np.ndarray)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Face Capture")
        self.setFixedSize(640, 550)
        self.setStyleSheet("""
            QDialog {
                background-color: #f8fafc;
            }
        """)

        layout = QVBoxLayout()
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)

        # Header
        header_layout = QHBoxLayout()
        title_label = QLabel("Face Capture")
        title_label.setStyleSheet("font-size: 18px; font-weight: bold; color: #1e293b;")
        header_layout.addWidget(title_label)
        header_layout.addStretch()
        layout.addLayout(header_layout)

        # Instructions
        instruction_label = QLabel("Position your face in the center of the frame and press 'Capture Face'")
        instruction_label.setStyleSheet("color: #64748b; font-size: 14px;")
        instruction_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(instruction_label)

        # Camera feed frame
        camera_frame = QFrame()
        camera_frame.setStyleSheet("""
            background-color: #0f172a;
            border-radius: 8px;
            padding: 2px;
        """)
        camera_layout = QVBoxLayout(camera_frame)
        camera_layout.setContentsMargins(1, 1, 1, 1)

        # Camera feed display
        self.camera_label = QLabel()
        self.camera_label.setFixedSize(600, 450)
        self.camera_label.setAlignment(Qt.AlignCenter)
        self.camera_label.setStyleSheet("background-color: black;")
        camera_layout.addWidget(self.camera_label)

        layout.addWidget(camera_frame)

        # Buttons
        button_layout = QHBoxLayout()
        button_layout.setSpacing(15)

        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setStyleSheet("""
            QPushButton {
                background-color: white;
                color: #64748b;
                border: 1px solid #cbd5e1;
                border-radius: 6px;
                padding: 10px 15px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #f1f5f9;
            }
        """)
        self.cancel_btn.setFixedSize(120, 40)

        self.capture_btn = QPushButton("Capture Face")
        self.capture_btn.setStyleSheet("""
            QPushButton {
                background-color: #3b82f6;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 10px 15px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #2563eb;
            }
        """)
        self.capture_btn.setFixedSize(120, 40)

        button_layout.addStretch()
        button_layout.addWidget(self.cancel_btn)
        button_layout.addWidget(self.capture_btn)
        layout.addLayout(button_layout)

        self.setLayout(layout)

        # Setup webcam thread
        self.webcam_thread = WebcamCaptureThread()
        self.webcam_thread.update_frame.connect(self.update_frame)
        self.webcam_thread.face_captured.connect(self.face_selected)

        self.capture_btn.clicked.connect(self.capture_face)
        self.cancel_btn.clicked.connect(self.reject)

        self.webcam_thread.start()

    def update_frame(self, frame):
        # Convert the OpenCV BGR image to QImage
        h, w, ch = frame.shape
        bytes_per_line = ch * w
        qt_image = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
        self.camera_label.setPixmap(QPixmap.fromImage(qt_image))

    def capture_face(self):
        if self.webcam_thread.capture_face():
            self.accept()
        else:
            QMessageBox.warning(
                self,
                "No Face Detected",
                "No face was detected. Please position yourself in the camera view.",
                QMessageBox.Ok
            )

    def closeEvent(self, event):
        self.webcam_thread.stop()
        event.accept()


class VoiceRecordingDialog(QDialog):
    recording_finished = pyqtSignal(bytes)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Voice Recording")
        self.setFixedSize(400, 250)

        layout = QVBoxLayout()
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)

        # Title
        title_label = QLabel("Voice Authentication")
        title_label.setFont(QFont("Arial", 16, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)

        # Instructions
        self.instruction_label = QLabel("Please say: \"Log me in to my device.\"")
        self.instruction_label.setAlignment(Qt.AlignCenter)
        self.instruction_label.setFont(QFont("Arial", 12))
        layout.addWidget(self.instruction_label)

        # Status
        self.status_label = QLabel("Press Start to begin recording")
        self.status_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.status_label)

        # Buttons - placed at fixed positions to prevent layout shifts
        button_layout = QHBoxLayout()
        button_layout.setSpacing(20)

        self.start_button = QPushButton("Start Recording")
        self.start_button.setFixedSize(150, 40)
        self.start_button.setStyleSheet("background-color: #90caf9; font-weight: bold;")

        self.stop_button = QPushButton("Stop Recording")
        self.stop_button.setFixedSize(150, 40)
        self.stop_button.setEnabled(False)
        self.stop_button.setStyleSheet("background-color: #ef9a9a; font-weight: bold;")

        button_layout.addStretch()
        button_layout.addWidget(self.start_button)
        button_layout.addWidget(self.stop_button)
        button_layout.addStretch()

        layout.addStretch()
        layout.addLayout(button_layout)
        layout.addStretch()

        self.setLayout(layout)

        # Connect signals
        self.start_button.clicked.connect(self.start_recording)
        self.stop_button.clicked.connect(self.stop_recording)

        # Audio recording setup
        self.chunk = 1024
        self.format = pyaudio.paInt16
        self.channels = 1
        self.rate = 16000
        self.frames = []
        self.is_recording = False
        self.audio = pyaudio.PyAudio()
        self.stream = None

        # Auto-close timer
        self.close_timer = QTimer()
        self.close_timer.setSingleShot(True)
        self.close_timer.timeout.connect(self.accept)

    def start_recording(self):
        self.status_label.setText("Recording in progress...")
        self.status_label.setStyleSheet("color: red; font-weight: bold;")
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)

        # Start recording directly (without using the original isDoneRecording function)
        self.frames = []
        self.is_recording = True

        # Setup and start audio stream
        self.stream = self.audio.open(format=self.format,
                                      channels=self.channels,
                                      rate=self.rate,
                                      input=True,
                                      frames_per_buffer=self.chunk)

        # Start recording in a thread
        self.record_thread = threading.Thread(target=self.record_audio)
        self.record_thread.daemon = True  # Thread will end when main program ends
        self.record_thread.start()

    def record_audio(self):
        print("Recording started...")
        while self.is_recording:
            try:
                data = self.stream.read(self.chunk)
                self.frames.append(data)
            except Exception as e:
                print(f"Error during recording: {e}")
                break

        # After recording stops
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()

        # Save the file and process it
        filename = "registerVoice.wav"
        self.save_audio(filename)

        # Read the file as a BLOB
        with open(filename, "rb") as file:
            audio_blob = file.read()

        # Remove the file
        if os.path.exists(filename):
            os.remove(filename)

        # Signal completion
        self.recording_finished.emit(audio_blob)

        # Auto-close the dialog after processing
        self.close_timer.start(1500)  # Close after 1.5 seconds

    def save_audio(self, filename):
        waveFile = wave.open(filename, 'wb')
        waveFile.setnchannels(self.channels)
        waveFile.setsampwidth(self.audio.get_sample_size(self.format))
        waveFile.setframerate(self.rate)
        waveFile.writeframes(b''.join(self.frames))
        waveFile.close()

    def stop_recording(self):
        self.is_recording = False  # Signal to stop recording
        self.status_label.setText("Processing audio...")
        self.status_label.setStyleSheet("color: blue;")
        self.stop_button.setEnabled(False)

        # Add a short timer to allow the audio processing to complete
        # and then automatically close the dialog
        QTimer.singleShot(1000, self.accept)  # Close after 2 seconds

    def closeEvent(self, event):
        self.is_recording = False
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.audio.terminate()
        event.accept()


class VerifyCodeDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("2FA Verification")
        self.setFixedSize(400, 220)  # Increased width to prevent text cutoff

        layout = QVBoxLayout()
        layout.setContentsMargins(25, 25, 25, 25)  # Increased horizontal margins
        layout.setSpacing(15)

        # Title
        title_label = QLabel("2FA Verification")
        title_label.setFont(QFont("Arial", 16, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)

        # Instructions with proper width
        self.instruction_label = QLabel("Enter 2FA code sent to your phone:")
        self.instruction_label.setAlignment(Qt.AlignCenter)
        self.instruction_label.setFont(QFont("Arial", 12))
        self.instruction_label.setWordWrap(True)  # Enable word wrapping
        layout.addWidget(self.instruction_label)

        # Code input with proper height
        self.code_input = QLineEdit()
        self.code_input.setMaxLength(6)
        self.code_input.setPlaceholderText("Verification Code")
        self.code_input.setAlignment(Qt.AlignCenter)
        self.code_input.setFixedHeight(40)  # Increased height
        self.code_input.setFont(QFont("Arial", 14))
        layout.addWidget(self.code_input)

        # Buttons
        button_layout = QHBoxLayout()
        button_layout.setSpacing(20)

        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setFixedSize(120, 40)  # Increased button height

        self.verify_button = QPushButton("Verify")
        self.verify_button.setFixedSize(120, 40)  # Increased button height
        self.verify_button.setStyleSheet("background-color: #3b82f6; color: white;")

        button_layout.addStretch()
        button_layout.addWidget(self.cancel_button)
        button_layout.addWidget(self.verify_button)
        button_layout.addStretch()

        layout.addSpacing(10)  # Extra space before buttons
        layout.addLayout(button_layout)

        self.setLayout(layout)

        # Connect signals
        self.verify_button.clicked.connect(self.accept)
        self.cancel_button.clicked.connect(self.reject)

        # Set focus on the input field
        self.code_input.setFocus()

    def get_code(self):
        return self.code_input.text()


class AuthenticationApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Biometric Authentication System")
        self.setMinimumSize(800, 600)

        # Apply global stylesheet
        self.setStyleSheet(get_app_stylesheet())

        # Set up the central widget and main layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        # Create a stacked widget to manage different screens
        self.stacked_widget = QStackedWidget()

        # Create different screens
        self.create_welcome_screen()
        self.create_register_screen()
        self.create_login_screen()

        # Add the success screen (add this line)
        self.stacked_widget.addWidget(self.create_success_screen())

        # Main layout
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        main_layout.addWidget(self.stacked_widget)
        self.central_widget.setLayout(main_layout)

        # Show the welcome screen initially
        self.stacked_widget.setCurrentIndex(0)

        # Initialize database
        authentication.initialize_database()

    def create_welcome_screen(self):
        welcome_widget = QWidget()
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(30, 30, 30, 30)
        main_layout.setSpacing(20)

        # Header with app name and version
        header_layout = QHBoxLayout()
        app_name_label = QLabel("SecureAuth")
        app_name_label.setStyleSheet("font-size: 16px; font-weight: bold; color: #3b82f6;")
        version_label = QLabel("v1.0")
        version_label.setStyleSheet("color: #94a3b8;")
        header_layout.addWidget(app_name_label)
        header_layout.addStretch()
        header_layout.addWidget(version_label)
        main_layout.addLayout(header_layout)

        # Center content - REORDERED to place icon above text
        content_layout = QVBoxLayout()
        content_layout.setSpacing(30)
        content_layout.setAlignment(Qt.AlignCenter)

        # Create vertical spacer to push content down a bit
        main_layout.addStretch(1)

        # Logo/Icon - now positioned before the title
        logo_frame = QFrame()
        logo_frame.setFixedSize(80, 80)
        logo_frame.setStyleSheet("""
            background-color: #3b82f6;
            border-radius: 40px;
        """)

        icon_label = QLabel()
        icon_path = "icons/shield-lock.png"
        if os.path.exists(icon_path):
            pixmap = QPixmap(icon_path)
            pixmap = pixmap.scaled(60, 60, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            icon_label.setPixmap(pixmap)
        else:
            icon_label.setText("✓")
            icon_label.setStyleSheet("color: white; font-size: 36px;")

        icon_label.setAlignment(Qt.AlignCenter)

        layout = QVBoxLayout(logo_frame)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(icon_label)

        logo_layout = QHBoxLayout()
        logo_layout.setAlignment(Qt.AlignCenter)
        logo_layout.addWidget(logo_frame)
        content_layout.addLayout(logo_layout)

        # Title - now positioned after the icon
        title_label = QLabel("Secure User Authentication System")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("font-size: 28px; font-weight: bold; color: #1e293b;")
        content_layout.addWidget(title_label)

        # Description
        desc_label = QLabel("Secure login with facial recognition, voice authentication, and 2FA")
        desc_label.setAlignment(Qt.AlignCenter)
        desc_label.setStyleSheet("font-size: 16px; color: #64748b;")
        content_layout.addWidget(desc_label)

        # Features cards
        features_layout = QHBoxLayout()
        features_layout.setSpacing(15)

        face_card = create_feature_card("", "Facial\nRecognition")
        voice_card = create_feature_card("", "Voice\nAuthentication")
        tfa_card = create_feature_card("", "2FA\nSecurity")

        features_layout.addWidget(face_card)
        features_layout.addWidget(voice_card)
        features_layout.addWidget(tfa_card)

        content_layout.addLayout(features_layout)

        # Buttons
        buttons_layout = QVBoxLayout()
        buttons_layout.setSpacing(15)

        register_btn = QPushButton("Register New User")
        register_btn.setStyleSheet("""
            background-color: #3b82f6;
            color: white;
            border-radius: 6px;
            font-weight: bold;
            font-size: 15px;
            padding: 12px;
        """)
        register_btn.setFixedHeight(50)
        register_btn.clicked.connect(lambda: self.stacked_widget.setCurrentIndex(1))

        login_btn = QPushButton("Login")
        login_btn.setStyleSheet("""
            background-color: white;
            color: #3b82f6;
            border: 1px solid #93c5fd;
            border-radius: 6px;
            font-weight: bold;
            font-size: 15px;
            padding: 12px;
        """)
        login_btn.setFixedHeight(50)
        login_btn.clicked.connect(lambda: self.stacked_widget.setCurrentIndex(2))

        buttons_layout.addWidget(register_btn)
        buttons_layout.addWidget(login_btn)

        content_layout.addLayout(buttons_layout)
        main_layout.addLayout(content_layout)

        # Footer
        footer_label = QLabel("© 2025 SecureAuth • Multi-factor Authentication")
        footer_label.setAlignment(Qt.AlignCenter)
        footer_label.setStyleSheet("color: #94a3b8; font-size: 12px;")
        main_layout.addStretch(1)
        main_layout.addWidget(footer_label)

        welcome_widget.setLayout(main_layout)
        self.stacked_widget.addWidget(welcome_widget)

    def create_register_screen(self):
        register_widget = QWidget()
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(30, 30, 30, 30)
        main_layout.setSpacing(20)

        # Header with back navigation
        header_layout = QHBoxLayout()

        try:
            back_btn = QPushButton()
            back_btn.setIcon(QIcon("icons/arrow-left.png"))
            back_btn.setIconSize(QSize(20, 20))
        except:
            back_btn = QPushButton("Back")

        back_btn.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                border: none;
                padding: 5px;
            }
            QPushButton:hover {
                background-color: #e2e8f0;
                border-radius: 4px;
            }
        """)
        back_btn.setFixedSize(36, 36)
        back_btn.clicked.connect(lambda: self.stacked_widget.setCurrentIndex(0))

        title_label = QLabel("Register New User")
        title_label.setStyleSheet("font-size: 22px; font-weight: bold; color: #1e293b;")

        header_layout.addWidget(back_btn)
        header_layout.addWidget(title_label)
        header_layout.addStretch()

        main_layout.addLayout(header_layout)

        # Main content
        content_widget = QFrame()
        content_widget.setStyleSheet("""
            background-color: white;
            border-radius: 10px;
            border: 1px solid #e2e8f0;
        """)

        content_layout = QVBoxLayout(content_widget)
        content_layout.setContentsMargins(25, 25, 25, 25)
        content_layout.setSpacing(20)

        # Registration form
        form_layout = QVBoxLayout()
        form_layout.setSpacing(15)

        # Username
        username_layout = QVBoxLayout()
        username_label = QLabel("Username")
        username_label.setStyleSheet("color: #475569; font-weight: bold;")
        self.username_input = QLineEdit()
        self.username_input.setPlaceholderText("Enter username")
        self.username_input.setFixedHeight(45)
        username_layout.addWidget(username_label)
        username_layout.addWidget(self.username_input)
        form_layout.addLayout(username_layout)

        # Password
        password_layout = QVBoxLayout()
        password_label = QLabel("Password")
        password_label.setStyleSheet("color: #475569; font-weight: bold;")
        self.password_input = QLineEdit()
        self.password_input.setEchoMode(QLineEdit.Password)
        self.password_input.setPlaceholderText("Enter password")
        self.password_input.setFixedHeight(45)
        password_layout.addWidget(password_label)
        password_layout.addWidget(self.password_input)
        form_layout.addLayout(password_layout)

        # Confirm Password
        confirm_layout = QVBoxLayout()
        confirm_label = QLabel("Confirm Password")
        confirm_label.setStyleSheet("color: #475569; font-weight: bold;")
        self.confirm_input = QLineEdit()
        self.confirm_input.setEchoMode(QLineEdit.Password)
        self.confirm_input.setPlaceholderText("Confirm your password")
        self.confirm_input.setFixedHeight(45)
        confirm_layout.addWidget(confirm_label)
        confirm_layout.addWidget(self.confirm_input)
        form_layout.addLayout(confirm_layout)

        # Phone
        phone_layout = QVBoxLayout()
        phone_label = QLabel("Phone Number")
        phone_label.setStyleSheet("color: #475569; font-weight: bold;")
        self.phone_input = QLineEdit()
        self.phone_input.setPlaceholderText("9051234567 (without country code)")
        self.phone_input.setFixedHeight(45)
        phone_layout.addWidget(phone_label)
        phone_layout.addWidget(self.phone_input)
        form_layout.addLayout(phone_layout)

        content_layout.addLayout(form_layout)

        # Biometric data section
        biometric_label = QLabel("Secure User Authentication")
        biometric_label.setStyleSheet("font-size: 16px; font-weight: bold; color: #1e293b; margin-top: 10px;")
        content_layout.addWidget(biometric_label)

        biometric_layout = QHBoxLayout()
        biometric_layout.setSpacing(15)

        # Face capture button
        face_frame = QFrame()
        face_frame.setStyleSheet("""
            background-color: #f8fafc;
            border-radius: 6px;
            border: 1px solid #e2e8f0;
        """)
        face_layout = QVBoxLayout(face_frame)

        self.capture_face_btn = QPushButton("Capture Face")

        self.capture_face_btn.setStyleSheet("""
            QPushButton {
                background-color: #f1f5f9;
                color: #475569;
                border: 1px solid #cbd5e1;
                border-radius: 4px;
                padding: 8px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #e2e8f0;
            }
        """)
        self.capture_face_btn.clicked.connect(self.capture_face)

        self.face_status = QLabel("Face: Not captured")
        self.face_status.setAlignment(Qt.AlignCenter)
        self.face_status.setStyleSheet("color: #94a3b8;")

        face_layout.addWidget(self.capture_face_btn)
        face_layout.addWidget(self.face_status)

        # Voice recording button
        voice_frame = QFrame()
        voice_frame.setStyleSheet("""
            background-color: #f8fafc;
            border-radius: 6px;
            border: 1px solid #e2e8f0;
        """)
        voice_layout = QVBoxLayout(voice_frame)

        self.record_voice_btn = QPushButton("Record Voice")

        self.record_voice_btn.setStyleSheet("""
            QPushButton {
                background-color: #f1f5f9;
                color: #475569;
                border: 1px solid #cbd5e1;
                border-radius: 4px;
                padding: 8px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #e2e8f0;
            }
        """)
        self.record_voice_btn.clicked.connect(self.record_voice)

        self.voice_status = QLabel("Voice: Not recorded")
        self.voice_status.setAlignment(Qt.AlignCenter)
        self.voice_status.setStyleSheet("color: #94a3b8;")

        voice_layout.addWidget(self.record_voice_btn)
        voice_layout.addWidget(self.voice_status)

        biometric_layout.addWidget(face_frame)
        biometric_layout.addWidget(voice_frame)
        content_layout.addLayout(biometric_layout)

        # Register button
        self.register_btn = create_styled_button("Register User", True)
        self.register_btn.clicked.connect(self.register_user)
        content_layout.addWidget(self.register_btn)

        main_layout.addWidget(content_widget)
        register_widget.setLayout(main_layout)
        self.stacked_widget.addWidget(register_widget)

        # Initialize storage for biometric data
        self.face_data = None
        self.voice_data = None

    def create_login_screen(self):
        login_widget = QWidget()
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(30, 30, 30, 30)
        main_layout.setSpacing(20)

        # Header with back navigation
        header_layout = QHBoxLayout()

        try:
            back_btn = QPushButton()
            back_btn.setIcon(QIcon("icons/arrow-left.png"))
            back_btn.setIconSize(QSize(20, 20))
        except:
            back_btn = QPushButton("Back")

        back_btn.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                border: none;
                padding: 5px;
            }
            QPushButton:hover {
                background-color: #e2e8f0;
                border-radius: 4px;
            }
        """)
        back_btn.setFixedSize(36, 36)
        back_btn.clicked.connect(self.reset_login)

        title_label = QLabel("User Authentication")
        title_label.setStyleSheet("font-size: 22px; font-weight: bold; color: #1e293b;")

        header_layout.addWidget(back_btn)
        header_layout.addWidget(title_label)
        header_layout.addStretch()

        main_layout.addLayout(header_layout)

        # Main content
        content_widget = QFrame()
        content_widget.setStyleSheet("""
            background-color: white;
            border-radius: 10px;
            border: 1px solid #e2e8f0;
        """)

        content_layout = QVBoxLayout(content_widget)
        content_layout.setContentsMargins(25, 25, 25, 25)
        content_layout.setSpacing(20)

        # Username input
        username_layout = QVBoxLayout()
        username_label = QLabel("Username")
        username_label.setStyleSheet("color: #475569; font-weight: bold;")
        self.login_username = QLineEdit()
        self.login_username.setPlaceholderText("Enter your username")
        self.login_username.setFixedHeight(45)
        username_layout.addWidget(username_label)
        username_layout.addWidget(self.login_username)
        content_layout.addLayout(username_layout)

        # Authentication steps section
        auth_title = QLabel("Authentication Process")
        auth_title.setStyleSheet("font-size: 16px; font-weight: bold; color: #1e293b; margin-top: 10px;")
        content_layout.addWidget(auth_title)

        # Status message
        self.auth_status = QLabel("Ready to authenticate")
        self.auth_status.setAlignment(Qt.AlignCenter)
        self.auth_status.setStyleSheet("""
            background-color: #f1f5f9;
            border-radius: 4px;
            padding: 10px;
            color: #475569;
        """)
        content_layout.addWidget(self.auth_status)

        # Progress bar
        progress_layout = QVBoxLayout()
        progress_label = QLabel("Authentication Progress")
        progress_label.setStyleSheet("color: #475569; font-size: 13px;")
        self.auth_progress = QProgressBar()
        self.auth_progress.setRange(0, 4)
        self.auth_progress.setValue(0)
        self.auth_progress.setFixedHeight(8)
        self.auth_progress.setTextVisible(False)
        progress_layout.addWidget(progress_label)
        progress_layout.addWidget(self.auth_progress)
        content_layout.addLayout(progress_layout)

        # Authentication steps
        steps_frame = QFrame()
        steps_frame.setStyleSheet("""
            background-color: #f8fafc;
            border-radius: 6px;
            border: 1px solid #e2e8f0;
            padding: 5px;
        """)

        steps_layout = QVBoxLayout(steps_frame)
        steps_layout.setSpacing(10)

        # Password authentication
        self.password_auth_btn = QPushButton("1. Password Authentication")

        self.password_auth_btn.setStyleSheet("""
            QPushButton {
                background-color: #3b82f6;
                color: white;
                border-radius: 4px;
                padding: 10px;
                text-align: left;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #2563eb;
            }
            QPushButton:disabled {
                background-color: #93c5fd;
            }
        """)
        self.password_auth_btn.clicked.connect(self.authenticate_password)
        steps_layout.addWidget(self.password_auth_btn)

        # Voice authentication
        self.voice_auth_btn = QPushButton("2. Voice Authentication")

        self.voice_auth_btn.setStyleSheet("""
            QPushButton {
                background-color: #3b82f6;
                color: white;
                border-radius: 4px;
                padding: 10px;
                text-align: left;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #2563eb;
            }
            QPushButton:disabled {
                background-color: #93c5fd;
            }
        """)
        self.voice_auth_btn.setEnabled(False)
        self.voice_auth_btn.clicked.connect(self.authenticate_voice)
        steps_layout.addWidget(self.voice_auth_btn)

        # Face authentication
        self.face_auth_btn = QPushButton("3. Face Authentication")

        self.face_auth_btn.setStyleSheet("""
            QPushButton {
                background-color: #3b82f6;
                color: white;
                border-radius: 4px;
                padding: 10px;
                text-align: left;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #2563eb;
            }
            QPushButton:disabled {
                background-color: #93c5fd;
            }
        """)
        self.face_auth_btn.setEnabled(False)
        self.face_auth_btn.clicked.connect(self.authenticate_face)
        steps_layout.addWidget(self.face_auth_btn)

        # 2FA authentication
        self.tfa_auth_btn = QPushButton("4. 2FA Verification")

        self.tfa_auth_btn.setStyleSheet("""
            QPushButton {
                background-color: #3b82f6;
                color: white;
                border-radius: 4px;
                padding: 10px;
                text-align: left;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #2563eb;
            }
            QPushButton:disabled {
                background-color: #93c5fd;
            }
        """)
        self.tfa_auth_btn.setEnabled(False)
        self.tfa_auth_btn.clicked.connect(self.authenticate_2fa)
        steps_layout.addWidget(self.tfa_auth_btn)

        content_layout.addWidget(steps_frame)

        main_layout.addWidget(content_widget)
        login_widget.setLayout(main_layout)
        self.stacked_widget.addWidget(login_widget)

        # Authentication state
        self.auth_state = {
            "password": False,
            "voice": False,
            "face": False,
            "2fa": False,
            "username": "",
            "phone": ""
        }

    def capture_face(self):
        dialog = FaceDialog(self)
        dialog.face_selected.connect(self.set_face_data)
        result = dialog.exec_()

        if result == QDialog.Accepted:
            self.face_status.setText("Face: Captured ✓")
            self.face_status.setStyleSheet("color: green")

    def set_face_data(self, face_data):
        """Stores and updates UI after face capture."""
        self.face_data = face_data
        self.update_face_status(True)

    def record_voice(self):
        dialog = VoiceRecordingDialog(self)
        dialog.recording_finished.connect(self.set_voice_data)
        dialog.exec_()

    def set_voice_data(self, voice_data):
        """Stores and updates UI after voice recording."""
        self.voice_data = voice_data
        self.update_voice_status(True)

    def register_user(self):
        """Register a new user with facial and voice data."""
        username = self.username_input.text().strip()
        password = self.password_input.text()
        confirm_pass = self.confirm_input.text()
        phone = self.phone_input.text().strip()

        # Validate inputs
        if not username:
            self.show_error_message("Input Error", "Username cannot be empty.")
            return

        if not password:
            self.show_error_message("Input Error", "Password cannot be empty.")
            return

        if password != confirm_pass:
            self.show_error_message("Input Error", "Passwords do not match.")
            return

        if not phone or not phone.isdigit() or len(phone) != 10:
            self.show_error_message("Input Error", "Please enter a valid 10-digit phone number.")
            return

        if self.face_data is None:
            self.show_error_message("Missing Data", "Please capture your face image.")
            return

        if self.voice_data is None:
            self.show_error_message("Missing Data", "Please record your voice.")
            return

        # Check if username already exists
        conn = authentication.sqlite3.connect(authentication.DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT username FROM users WHERE username = ?", (username,))
        if cursor.fetchone() is not None:
            self.show_error_message("Registration Error", f"Username '{username}' already exists.")
            conn.close()
            return

        # Process registration
        try:
            # Hash password
            salt = authentication.bcrypt.gensalt()
            hashed_pass = authentication.bcrypt.hashpw(password.encode('utf-8'), salt)

            # Prepare face data
            face_data_bytes = np.array(self.face_data).tobytes()

            # Format phone number
            phone_number = "+1" + phone

            # Insert into database
            cursor.execute("INSERT INTO users (username, password, voice, face, phone) VALUES (?, ?, ?, ?, ?)",
                           (username, hashed_pass, self.voice_data, face_data_bytes, phone_number))
            conn.commit()

            self.show_success_message("Registration Successful",
                                      f"User '{username}' has been registered successfully.")

            # Reset form
            self.username_input.clear()
            self.password_input.clear()
            self.confirm_input.clear()
            self.phone_input.clear()
            self.face_data = None
            self.voice_data = None
            self.update_face_status(False)
            self.update_voice_status(False)

            # Return to welcome screen
            self.stacked_widget.setCurrentIndex(0)

        except Exception as e:
            self.show_error_message("Registration Error", f"An error occurred: {str(e)}")
        finally:
            conn.close()

    def authenticate_password(self):
        """First authentication step: password verification."""
        username = self.login_username.text().strip()

        if not username:
            self.show_error_message("Authentication Error", "Please enter a username.")
            return

        # Check if user exists
        conn = authentication.sqlite3.connect(authentication.DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT password, phone FROM users WHERE username = ?", (username,))
        user_data = cursor.fetchone()
        conn.close()

        if user_data is None:
            self.show_error_message("Authentication Error", "User not found.")
            return

        # Store username and phone for later steps
        self.auth_state["username"] = username
        self.auth_state["phone"] = user_data[1]

        # Password dialog
        password, ok = QInputDialog.getText(self, "Password Authentication",
                                            "Enter your password:", QLineEdit.Password)
        if not ok:
            return

        # Verify password
        stored_password = user_data[0]
        if authentication.bcrypt.checkpw(password.encode('utf-8'), stored_password):
            self.auth_state["password"] = True
            self.auth_progress.setValue(1)
            self.update_auth_status("Password authentication successful", True)
            self.voice_auth_btn.setEnabled(True)

            # Update button styles to indicate completion
            self.password_auth_btn.setStyleSheet("""
                QPushButton {
                    background-color: #10b981;
                    color: white;
                    border-radius: 4px;
                    padding: 10px;
                    text-align: left;
                    font-weight: bold;
                }
                QPushButton:hover {
                    background-color: #059669;
                }
            """)
        else:
            self.update_auth_status("Incorrect password", False)
            self.show_error_message("Authentication Error", "Incorrect password.")

    def authenticate_voice(self):
        """Second authentication step: voice verification."""
        # Get stored voice data
        conn = authentication.sqlite3.connect(authentication.DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT voice FROM users WHERE username = ?", (self.auth_state["username"],))
        stored_voice = cursor.fetchone()[0]
        conn.close()

        # Save the stored voice to a temporary file for comparison
        with open("temp_stored_voice.wav", "wb") as file:
            file.write(stored_voice)

        # Update status
        self.update_auth_status("Voice authentication in progress...", warning=True)

        # Open voice recording dialog
        dialog = VoiceRecordingDialog(self)
        dialog.recording_finished.connect(lambda x: self.process_voice_auth(x, stored_voice))
        result = dialog.exec_()

    def process_voice_auth(self, recorded_voice, stored_voice):
        """Process the voice authentication result."""
        # Save the new recording for authentication
        with open("authenticateVoice.wav", "wb") as file:
            file.write(recorded_voice)

        # Manually perform voice comparison using the pyannote model
        try:
            model = authentication.voiceDetection.Model.from_pretrained(
                "pyannote/wespeaker-voxceleb-resnet34-LM",
                use_auth_token=""
            )
            inference = authentication.voiceDetection.Inference(model, window="whole")

            # Get embeddings
            embedding1 = inference("authenticateVoice.wav")

            # Save stored voice to a temporary file
            with open("temp_stored_voice.wav", "wb") as file:
                file.write(stored_voice)

            embedding2 = inference("temp_stored_voice.wav")

            # Reshape for distance calculation
            embedding1 = np.array(embedding1).reshape(1, -1)
            embedding2 = np.array(embedding2).reshape(1, -1)

            # Calculate distance
            distance = authentication.voiceDetection.cdist(embedding1, embedding2, metric="cosine")[0, 0]

            # Clean up temporary files
            if os.path.exists("authenticateVoice.wav"):
                os.remove("authenticateVoice.wav")
            if os.path.exists("temp_stored_voice.wav"):
                os.remove("temp_stored_voice.wav")

            # Check threshold
            if distance <= 0.40:
                self.auth_state["voice"] = True
                self.auth_progress.setValue(2)
                self.update_auth_status("Voice authentication successful", True)
                self.face_auth_btn.setEnabled(True)

                # Update button styles to indicate completion
                self.voice_auth_btn.setStyleSheet("""
                    QPushButton {
                        background-color: #10b981;
                        color: white;
                        border-radius: 4px;
                        padding: 10px;
                        text-align: left;
                        font-weight: bold;
                    }
                    QPushButton:hover {
                        background-color: #059669;
                    }
                """)
            else:
                self.update_auth_status("Voice authentication failed", False)
                self.show_error_message("Authentication Error",
                                        f"Voice authentication failed. Distance: {distance:.2f}")

        except Exception as e:
            self.update_auth_status("Voice authentication error", False)
            self.show_error_message("Authentication Error", f"Error during voice authentication: {str(e)}")
            # Clean up any temporary files
            for file in ["authenticateVoice.wav", "temp_stored_voice.wav"]:
                if os.path.exists(file):
                    os.remove(file)

    def authenticate_face(self):
        """Third authentication step: face verification."""
        # Get stored face data
        conn = authentication.sqlite3.connect(authentication.DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT face FROM users WHERE username = ?", (self.auth_state["username"],))
        stored_face = cursor.fetchone()[0]
        conn.close()

        # Update status
        self.update_auth_status("Face authentication in progress...", warning=True)

        # Open face capture dialog
        dialog = FaceDialog(self)
        result = dialog.exec_()

        if result == QDialog.Accepted and hasattr(dialog.webcam_thread, 'last_face'):
            face_img = cv2.resize(dialog.webcam_thread.last_face, (100, 100))

            # Compare faces
            stored_face_array = np.frombuffer(stored_face, dtype=np.uint8)

            try:
                stored_face_resized = stored_face_array.reshape(100, 100)
                distance = np.mean((stored_face_resized - face_img) ** 2)

                if distance < 1000:  # Same threshold as in authentication.py
                    self.auth_state["face"] = True
                    self.auth_progress.setValue(3)
                    self.update_auth_status("Face authentication successful", True)
                    self.tfa_auth_btn.setEnabled(True)

                    # Update button styles to indicate completion
                    self.face_auth_btn.setStyleSheet("""
                        QPushButton {
                            background-color: #10b981;
                            color: white;
                            border-radius: 4px;
                            padding: 10px;
                            text-align: left;
                            font-weight: bold;
                        }
                        QPushButton:hover {
                            background-color: #059669;
                        }
                    """)
                else:
                    self.update_auth_status("Face authentication failed", False)
                    self.show_error_message("Authentication Error", "Face authentication failed.")
            except ValueError:
                self.update_auth_status("Face data error", False)
                self.show_error_message("Authentication Error", "Face data size mismatch.")
        else:
            self.update_auth_status("Face authentication cancelled", warning=True)

    def authenticate_2fa(self):
        """Fourth authentication step: 2FA verification."""
        # Update status
        self.update_auth_status("Sending verification code...", warning=True)

        # Send verification code
        try:
            authentication.send_2fa_code(self.auth_state["phone"])
        except Exception as e:
            self.update_auth_status("Failed to send verification code", False)
            self.show_error_message("2FA Error", f"Failed to send verification code: {str(e)}")
            return

        # Open 2FA dialog
        dialog = VerifyCodeDialog(self)
        result = dialog.exec_()

        if result == QDialog.Accepted:
            code = dialog.get_code()
            if authentication.verify_2fa_code(self.auth_state["phone"], code):
                self.auth_state["2fa"] = True
                self.auth_progress.setValue(4)
                self.update_auth_status("Authentication successful!", True)

                # Update button styles to indicate completion
                self.tfa_auth_btn.setStyleSheet("""
                    QPushButton {
                        background-color: #10b981;
                        color: white;
                        border-radius: 4px;
                        padding: 10px;
                        text-align: left;
                        font-weight: bold;
                    }
                    QPushButton:hover {
                        background-color: #059669;
                    }
                """)

                # Check if all authentication methods passed
                if all([self.auth_state["password"], self.auth_state["voice"],
                        self.auth_state["face"], self.auth_state["2fa"]]):
                    # Show success message
                    self.show_success_message("Authentication Successful",
                                              f"User '{self.auth_state['username']}' authenticated successfully.")

                    # Navigate to success screen (index 3) - ADD THIS LINE
                    self.stacked_widget.setCurrentIndex(3)
            else:
                self.update_auth_status("Incorrect verification code", False)
                self.show_error_message("Authentication Error", "Incorrect verification code.")
        else:
            self.update_auth_status("2FA verification cancelled", warning=True)

    def reset_login(self):
        """Reset the authentication state and UI."""
        # Reset authentication state
        self.auth_state = {
            "password": False,
            "voice": False,
            "face": False,
            "2fa": False,
            "username": "",
            "phone": ""
        }

        # Reset UI elements
        self.login_username.clear()
        self.auth_progress.setValue(0)
        self.update_auth_status("Ready to authenticate", warning=False)
        self.voice_auth_btn.setEnabled(False)
        self.face_auth_btn.setEnabled(False)
        self.tfa_auth_btn.setEnabled(False)

        # Reset button styles
        auth_button_style = """
            QPushButton {
                background-color: #3b82f6;
                color: white;
                border-radius: 4px;
                padding: 10px;
                text-align: left;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #2563eb;
            }
            QPushButton:disabled {
                background-color: #93c5fd;
            }
        """

        self.password_auth_btn.setStyleSheet(auth_button_style)
        self.voice_auth_btn.setStyleSheet(auth_button_style)
        self.face_auth_btn.setStyleSheet(auth_button_style)
        self.tfa_auth_btn.setStyleSheet(auth_button_style)

        # Return to welcome screen
        self.stacked_widget.setCurrentIndex(0)

    def update_face_status(self, success=True):
        """Updates the face capture status indicator with success or failure styling."""
        if success:
            self.face_status.setText("Face: Captured ✓")
            self.face_status.setStyleSheet("color: #10b981; font-weight: bold;")
        else:
            self.face_status.setText("Face: Not captured")
            self.face_status.setStyleSheet("color: #94a3b8;")

    def update_voice_status(self, success=True):
        """Updates the voice recording status indicator with success or failure styling."""
        if success:
            self.voice_status.setText("Voice: Recorded ✓")
            self.voice_status.setStyleSheet("color: #10b981; font-weight: bold;")
        else:
            self.voice_status.setText("Voice: Not recorded")
            self.voice_status.setStyleSheet("color: #94a3b8;")

    def update_auth_status(self, message, success=True, warning=False):
        """Updates the authentication status message with appropriate styling."""
        if success:
            self.auth_status.setText(message)
            self.auth_status.setStyleSheet("""
                background-color: #ecfdf5;
                border: 1px solid #a7f3d0;
                border-radius: 4px;
                padding: 10px;
                color: #10b981;
                font-weight: bold;
            """)
        elif warning:
            self.auth_status.setText(message)
            self.auth_status.setStyleSheet("""
                background-color: #fffbeb;
                border: 1px solid #fde68a;
                border-radius: 4px;
                padding: 10px;
                color: #f59e0b;
                font-weight: bold;
            """)
        else:
            self.auth_status.setText(message)
            self.auth_status.setStyleSheet("""
                background-color: #fee2e2;
                border: 1px solid #fecaca;
                border-radius: 4px;
                padding: 10px;
                color: #ef4444;
                font-weight: bold;
            """)

    def show_success_message(self, title, message):
        """Displays a styled success message box."""
        msg_box = QMessageBox(self)
        msg_box.setWindowTitle(title)
        msg_box.setText(message)
        msg_box.setIcon(QMessageBox.Information)
        msg_box.setStandardButtons(QMessageBox.Ok)

        # Apply custom styling
        msg_box.setStyleSheet("""
            QMessageBox {
                background-color: white;
            }
            QLabel {
                color: #1e293b;
                font-size: 14px;
            }
            QPushButton {
                background-color: #3b82f6;
                color: white;
                border-radius: 4px;
                padding: 6px 16px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #2563eb;
            }
        """)

        msg_box.exec_()

    def show_error_message(self, title, message):
        """Displays a styled error message box."""
        msg_box = QMessageBox(self)
        msg_box.setWindowTitle(title)
        msg_box.setText(message)
        msg_box.setIcon(QMessageBox.Warning)
        msg_box.setStandardButtons(QMessageBox.Ok)

        # Apply custom styling
        msg_box.setStyleSheet("""
            QMessageBox {
                background-color: white;
            }
            QLabel {
                color: #1e293b;
                font-size: 14px;
            }
            QPushButton {
                background-color: #ef4444;
                color: white;
                border-radius: 4px;
                padding: 6px 16px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #dc2626;
            }
        """)

        msg_box.exec_()

    def create_success_screen(self):
        """Creates a success screen to show after successful authentication."""
        success_widget = QWidget()
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(30, 30, 30, 30)
        main_layout.setSpacing(20)

        # Header with back navigation
        header_layout = QHBoxLayout()

        back_btn = QPushButton("<- Back")
        back_btn.setFixedWidth(80)
        back_btn.setStyleSheet("border: none; font-weight: bold;")
        back_btn.clicked.connect(lambda: self.stacked_widget.setCurrentIndex(0))  # Go back to welcome screen

        title_label = QLabel("Authentication Complete")
        title_label.setStyleSheet("font-size: 22px; font-weight: bold; color: #1e293b;")

        header_layout.addWidget(back_btn)
        header_layout.addWidget(title_label)
        header_layout.addStretch()

        main_layout.addLayout(header_layout)

        # Main content
        content_layout = QVBoxLayout()
        content_layout.setAlignment(Qt.AlignCenter)
        content_layout.setSpacing(30)

        # Success icon
        success_icon = QLabel()
        success_icon.setText("✓")
        success_icon.setStyleSheet("""
            font-size: 64px;
            color: white;
            background-color: #10b981;
            border-radius: 50px;
            padding: 10px;
        """)
        success_icon.setFixedSize(100, 100)
        success_icon.setAlignment(Qt.AlignCenter)

        icon_layout = QHBoxLayout()
        icon_layout.setAlignment(Qt.AlignCenter)
        icon_layout.addWidget(success_icon)
        content_layout.addLayout(icon_layout)

        # Success message
        success_message = QLabel("Verification Successful!")
        success_message.setStyleSheet("font-size: 24px; font-weight: bold; color: #10b981;")
        success_message.setAlignment(Qt.AlignCenter)
        content_layout.addWidget(success_message)

        # Detail message
        detail_message = QLabel("You have successfully completed the multi-factor authentication process.")
        detail_message.setStyleSheet("font-size: 14px; color: #64748b;")
        detail_message.setAlignment(Qt.AlignCenter)
        detail_message.setWordWrap(True)
        content_layout.addWidget(detail_message)

        # Add username info
        if hasattr(self, 'auth_state') and self.auth_state.get("username"):
            username_message = QLabel(f"Logged in as: {self.auth_state['username']}")
            username_message.setStyleSheet("font-size: 14px; color: #1e293b; font-weight: bold;")
            username_message.setAlignment(Qt.AlignCenter)
            content_layout.addWidget(username_message)

        # Home button
        home_btn = QPushButton("Return to Home")
        home_btn.setStyleSheet("""
            background-color: #3b82f6;
            color: white;
            border-radius: 6px;
            font-weight: bold;
            font-size: 15px;
            padding: 12px;
        """)
        home_btn.setFixedHeight(50)
        home_btn.setFixedWidth(200)
        home_btn.clicked.connect(lambda: self.stacked_widget.setCurrentIndex(0))  # Go back to welcome screen

        button_layout = QHBoxLayout()
        button_layout.setAlignment(Qt.AlignCenter)
        button_layout.addWidget(home_btn)
        content_layout.addLayout(button_layout)

        main_layout.addStretch()
        main_layout.addLayout(content_layout)
        main_layout.addStretch()

        success_widget.setLayout(main_layout)
        return success_widget


if __name__ == "__main__":
    # Create application
    app = QApplication(sys.argv)

    # Set application-wide font
    font = QFont("Arial", 10)
    app.setFont(font)

    # Create the main window
    window = AuthenticationApp()

    # Center the window on screen
    screen_geometry = app.desktop().screenGeometry()
    x = (screen_geometry.width() - window.width()) // 2
    y = (screen_geometry.height() - window.height()) // 2
    window.move(x, y)

    # Show and execute the application
    window.show()
    sys.exit(app.exec_())
