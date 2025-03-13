import os
import pandas as pd
import speech_recognition as sr
from pydub import AudioSegment
from textblob import TextBlob
import librosa
import numpy as np
import wave
import ftplib
import streamlit as st
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Function to check if audio file is valid
def check_audio_validity(audio_file):
    try:
        with wave.open(audio_file, 'r') as wav:
            if wav.getnframes() == 0:
                return False  # No audio data
        return True
    except Exception:
        return False

# Function to perform speech-to-text sentiment analysis
def analyze_sentiment(audio_file):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_file) as source:
        audio_data = recognizer.record(source)
        try:
            text = recognizer.recognize_google(audio_data)  # Convert speech to text
            blob = TextBlob(text)
            sentiment_score = blob.sentiment.polarity  # Sentiment score (-1 to +1)
            sentiment = "P" if sentiment_score > 0 else "N" if sentiment_score < 0 else "NU"

            # Count positive and negative words
            words = text.split()
            positive_words = sum(1 for word in words if TextBlob(word).sentiment.polarity > 0)
            negative_words = sum(1 for word in words if TextBlob(word).sentiment.polarity < 0)

            return text, sentiment, sentiment_score, positive_words, negative_words
        except (sr.UnknownValueError, sr.RequestError):
            return "", "NU", 0, 0, 0  # Unable to transcribe

# Function to analyze voice tone
def analyze_tone(audio_file):
    y, sr = librosa.load(audio_file)

    # Calculate pitch (fundamental frequency)
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    pitch_values = pitches[pitches > 0]
    avg_pitch = np.mean(pitch_values) if len(pitch_values) > 0 else 0

    # Calculate intensity (loudness)
    energy = np.mean(librosa.feature.rms(y=y))

    # Classify Tone Based on Pitch & Intensity
    if avg_pitch > 200 and energy > 0.1:
        return "Happy", 1
    elif avg_pitch > 150 and energy > 0.08:
        return "Calm", 0.5
    elif avg_pitch < 100 and energy > 0.1:
        return "Angry", -1
    elif avg_pitch < 100 and energy < 0.05:
        return "Sad", -0.5
    else:
        return "Frustrated", -0.8

# Function to connect to FTP and get folder list
def get_ftp_folders(ftp_host, ftp_user, ftp_pass):
    try:
        ftp = ftplib.FTP(ftp_host)
        ftp.login(ftp_user, ftp_pass)
        folders = []
        ftp.retrlines("LIST", folders.append)  # List files and directories
        
        directories = []
        for folder in folders:
            if folder.startswith('d'):
                directories.append(folder.split()[-1])  # Extract folder names
        
        ftp.quit()
        return directories
    except ftplib.all_errors as e:
        st.error(f"FTP connection error: {e}")
        return []

# Function to connect to FTP and get file list from folder
def get_audio_files_from_ftp(ftp_host, ftp_user, ftp_pass, folder):
    try:
        ftp = ftplib.FTP(ftp_host)
        ftp.login(ftp_user, ftp_pass)
        ftp.cwd(folder)
        
        files = ftp.nlst()  # List files in the directory
        mp3_files = [file for file in files if file.endswith(".mp3")]
        ftp.quit()
        
        return mp3_files
    except ftplib.all_errors as e:
        st.error(f"Error while fetching files: {e}")
        return []

# Streamlit UI setup
st.title("Audio File Analysis App")

# User input fields for FTP connection
ftp_host = st.text_input("FTP Host", "ftp.example.com")
ftp_user = st.text_input("FTP Username", "username")
ftp_pass = st.text_input("FTP Password", "password", type="password")

# Connect to FTP and get folder list after FTP login
if st.button("Connect to FTP"):
    directories = get_ftp_folders(ftp_host, ftp_user, ftp_pass)

    # Show the list of folders if available
    if directories:
        folder_path = st.selectbox("Select Folder", directories)
        
        if folder_path:
            st.write(f"Selected folder: {folder_path}")
            
            if st.button("Process Audio Files"):
                # Fetch audio files from selected folder
                mp3_files = get_audio_files_from_ftp(ftp_host, ftp_user, ftp_pass, folder_path)
                
                # Data storage for analysis
                results = []

                for i, mp3_file in enumerate(mp3_files):
                    st.write(f"Processing ({i+1}/{len(mp3_files)}): {mp3_file}")

                    # Download MP3 file from FTP
                    audio_path = f"temp_{i}.mp3"
                    ftp = ftplib.FTP(ftp_host)
                    ftp.login(ftp_user, ftp_pass)
                    with open(audio_path, "wb") as f:
                        ftp.retrbinary(f"RETR {mp3_file}", f.write)
                    ftp.quit()

                    # Convert MP3 to WAV (16kHz, Mono)
                    audio = AudioSegment.from_mp3(audio_path)
                    audio = audio.set_frame_rate(16000).set_channels(1)[:30000]  # 30s limit
                    wav_file = f"temp_{i}.wav"
                    audio.export(wav_file, format="wav")

                    if check_audio_validity(wav_file):
                        agent_text, agent_sentiment, agent_sentiment_score, apw, anw = analyze_sentiment(wav_file)
                        customer_text, customer_sentiment, customer_sentiment_score, cpw, cnw = analyze_sentiment(wav_file)  # Assuming same file
                        agent_tone, agent_tone_score = analyze_tone(wav_file)
                    else:
                        agent_text, agent_sentiment, agent_sentiment_score, apw, anw = "", "NU", 0, 0, 0
                        customer_text, customer_sentiment, customer_sentiment_score, cpw, cnw = "", "NU", 0, 0, 0
                        agent_tone, agent_tone_score = "Unknown", 0

                    # Calculate Overall Call Score (scaled between 0-1)
                    overall_score = (
                        (0.3 * (agent_sentiment_score + 1) / 2) +
                        (0.3 * (customer_sentiment_score + 1) / 2) +
                        (0.2 * (agent_tone_score + 1) / 2) +
                        (0.2 * (apw - anw + cpw - cnw + 5) / 10)
                    )

                    overall_score = max(0, min(1, overall_score))  # Ensure it stays in [0,1]

                    # Save results in list
                    results.append([
                        mp3_file, agent_sentiment, customer_sentiment, agent_tone, overall_score,
                        agent_sentiment_score, customer_sentiment_score, agent_tone_score, apw, anw, cpw, cnw
                    ])

                    os.remove(wav_file)  # Remove temp file

                # Create DataFrame
                df = pd.DataFrame(results, columns=[
                    "File", "Agent Sentiment", "Customer Sentiment", "Agent Tone", "Overall Score",
                    "Agent Sentiment Score", "Customer Sentiment Score", "Agent Tone Score", "APW", "ANW", "CPW", "CNW"
                ])

                # Save to CSV
                df.to_csv("call_analysis_results.csv", index=False)
                st.write("Analysis completed. Results saved to `call_analysis_results.csv`")

                # ML Model and visualization steps (same as in your original code)
                # ML Model for predictions and visualization
                df = pd.read_csv('call_analysis_results.csv')
                df = df.drop(columns=['File', 'Agent Tone'])

                le = LabelEncoder()
                df['Agent Sentiment'] = le.fit_transform(df['Agent Sentiment'])
                df['Customer Sentiment'] = le.fit_transform(df['Customer Sentiment'])

                features = ['Agent Sentiment', 'Customer Sentiment', 'Agent Sentiment Score',
                            'Customer Sentiment Score', 'Agent Tone Score', 'APW', 'ANW', 'CPW', 'CNW']
                target = 'Overall Score'

                X = df[features].values
                y = df[target].values

                # Normalize features
                scaler = MinMaxScaler()
                X = scaler.fit_transform(X)

                # Reshape for RNN (samples, timesteps, features)
                X = X.reshape((X.shape[0], 1, X.shape[1]))

                # Split the data (60% train, 40% test)
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

                # Build RNN model using LSTM
                model = Sequential([
                    LSTM(50, activation='relu', return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
                    LSTM(50, activation='relu'),
                    Dense(25, activation='relu'),
                    Dense(1)  # Output layer for regression
                ])

                model.compile(optimizer='adam', loss='mse')

                # Train model
                model.fit(X_train, y_train, epochs=50, batch_size=16, validation_data=(X_test, y_test), verbose=1)

                # Predict on test data
                y_pred = model.predict(X_test)

                # Plot actual vs predicted values
                plt.figure(figsize=(12, 6))
                plt.plot(np.arange(len(y_test)), y_test, label='Actual Values', marker='o', linestyle='-', color='blue')
                plt.plot(np.arange(len(y_pred)), y_pred, label='Predicted Values (RNN)', marker='x', linestyle='--', color='red')

                plt.xlabel('Sample Index')
                plt.ylabel('Overall Score')
                plt.title('Actual vs Predicted Values (RNN with LSTM)')
                plt.legend()
                st.pyplot()
    else:
        st.warning("No directories found on the FTP server.")
