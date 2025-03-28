# File: voiceDetection.py
# Author: Javier Chung
# Date: March 2, 2025
# Description: This code is used to register and authenticate user voices

# libraries needed 
# brew install portaudio
# pip install pyaudio 
# pip install pyannote.audio 
# pip install scipy 
# pip install torch

import wave
import pyaudio
import threading
import torch
from pyannote.audio import Model, Inference
from scipy.spatial.distance import cdist
import numpy as np
import os

def registerVoice():
    recordAudio("registerVoice.wav")

    # converting the audio clip to a BLOB to store into the sqlite server
    with open("registerVoice.wav", "rb") as file:
        voiceBLOB = file.read()

    removeAudioFile("registerVoice.wav")

    return voiceBLOB;

def removeAudioFile(filename):
    # deleting the created audio file
    os.remove(filename)

# this method is used to authenticate a speaker through their voice in audio the two files
def authenticateVoice(blob):
    recordAudio("authenticateVoice.wav")

    # loading the model from hugging face
    model = Model.from_pretrained("pyannote/wespeaker-voxceleb-resnet34-LM", use_auth_token="")
    inference = Inference(model, window="whole")

    embedding1 = inference("authenticateVoice.wav")
    removeAudioFile("authenticateVoice.wav")

    # creating the audio clip from a BLOB 
    with open("registerVoice.wav", "wb") as file:
        file.write(blob)

    embedding2 = inference("registerVoice.wav")
    removeAudioFile("registerVoice.wav")

    # reshapping the 1D arrays to 2D arrays to measure the distance
    embedding1 = np.array(embedding1).reshape(1, -1)
    embedding2 = np.array(embedding2).reshape(1, -1)

    # The method returns (float) how dissimilar the two speakers in the audio files are
    distance = cdist(embedding1, embedding2, metric="cosine")[0,0]

    # with a threshold of 60%, check if the user is the once registered to the user login
    if(distance <= 0.40):
        print("User Authenticated. Voice matched")
        return True
    elif(distance > 0.40):
        print("User Authentication failed. Voice did not match.")
        return False

    return distance

# This function is used to record audio
def recordAudio(filename):
    
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000
    voiceDetectionAudio = pyaudio.PyAudio()


    stream = voiceDetectionAudio.open(format=FORMAT,
                                      channels=CHANNELS,
                                      rate=RATE, 
                                      input=True, 
                                      frames_per_buffer=CHUNK)

    print("Currently Recording...")

    frames = []

    isRecording.set() # starting the recording proccess by setting the state to true which signals thread to continue running

    # creating a thread to track when a user wants to interrupt and terminate the recording
    intruptThread = threading.Thread(target=isDoneRecording)
    intruptThread.start()

    # Recording live audio
    while isRecording.is_set():
        data = stream.read(CHUNK)
        frames.append(data)

    print("DONE RECORDING...")

    stream.stop_stream()
    stream.close()
    voiceDetectionAudio.terminate()

    # Saving the recorded audio as a wave file
    waveFile = wave.open(filename, 'wb')
    waveFile.setnchannels(CHANNELS)
    waveFile.setsampwidth(voiceDetectionAudio.get_sample_size(FORMAT))
    waveFile.setframerate(RATE)
    waveFile.writeframes(b''.join(frames))
    waveFile.close()

'''
Now handled by GUI
# this function uses another thread to poll a keyboard input that will end the recording of a user
def isDoneRecording():
    while isRecording.is_set():
        recordAnswer = str(input("Say: Log me in to my device.\nPress n to stop recording: "))

        if (recordAnswer == "n"):
            isRecording.clear() # finishing execution of the thread and clearing state
'''

# Declaring a variable that creates an event object that a thread can track
isRecording = threading.Event() # set to cleared state

# SHOULD BE IN MAIN PROGRAM:
# print("Program Started...")
# print("Registering voice...")

# blob = registerVoice()


# print("Authenticating voice...")
# verifiedVoice = authenticateVoice(blob)
