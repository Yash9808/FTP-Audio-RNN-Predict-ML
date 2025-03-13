import os
import ftplib
import pandas as pd
import streamlit as st
import assemblyai as aai
from pydub import AudioSegment
from textblob import TextBlob
import librosa
import numpy as np

# Function to get directories from FTP
def get_ftp_folders(ftp_host, ftp_user, ftp_pass):
    try:
        # Connect to FTP server
        ftp = ftplib.FTP(ftp_host)
        ftp.login(ftp_user, ftp_pass)
        
        # List all files and directories in the root
        files = ftp.nlst()
        directories = [file for file in files if is_ftp_directory(ftp, file)]
        
        ftp.quit()
        return directories
    except ftplib.all_errors as e:
        st.error(f"Error while fetching directories: {e}")
        return []

# Function to check if the FTP item is a directory
def is_ftp_directory(ftp, item):
    try:
        # Try to change to the directory
        ftp.cwd(item)
        ftp.cwd('..')  # Go back to the parent directory
        return True
    except:
        return False

# Function to load audio files from FTP
def get_audio_files_from_ftp(ftp_host, ftp_user, ftp_pass, folder):
    try:
        # Connect to FTP server
        ftp = ftplib.FTP(ftp_host)
        ftp.login(ftp_user, ftp_pass)
        
        # Change to the specified folder
        ftp.cwd(folder)
        
        # List all files in the folder
        files = ftp.nlst()
        ftp.quit()
        
        return files
    except ftplib.all_errors as e:
        st.error(f"Error while fetching files: {e}")
        return []

# Function to transcribe audio file using AssemblyAI
def transcribe_audio_aai(audio_file):
    # Get the API Key from Streamlit secrets or manual input
    aai_key = st.text_input("Enter AssemblyAI API Key", type="password")
    if aai_key:
        aai.settings.api_key = aai_key
    else:
        st.warning("API Key is required")
        return ""

    transcriber = aai.Transcriber()

    # If it's a local file, you can directly pass the file path, otherwise pass a URL
    if audio_file.startswith('http'):
        transcript = transcriber.transcribe(audio_file)  # For URL
    else:
        transcript = transcriber.transcribe(audio_file)  # For local file
    
    # Wait until the transcription is complete
    while transcript.status != 'completed':
        st.write("Transcription in progress...")
        transcript = transcriber.get(transcript.id)

    return transcript.text

# Function to perform speech-to-text sentiment analysis
def analyze_sentiment(text):
    blob = TextBlob(text)
    sentiment_score = blob.sentiment.polarity  # Sentiment score (-1 to +1)
    sentiment = "P" if sentiment_score > 0 else "N" if sentiment_score < 0 else "NU"

    # Count positive and negative words
    words = text.split()
    positive_words = sum(1 for word in words if TextBlob(word).sentiment.polarity > 0)
    negative_words = sum(1 for word in words if TextBlob(word).sentiment.polarity < 0)

    return sentiment, sentiment_score, positive_words, negative_words

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

# Function to process the audio and save results in CSV
def process_audio_files(ftp_host, ftp_user, ftp_pass, folder):
    files_in_folder = get_audio_files_from_ftp(ftp_host, ftp_user, ftp_pass, folder)
    
    # Store the results in a list
    results = []
    
    for file in files_in_folder:
        if file.endswith(".mp3"):
            st.write(f"Processing file: {file}")
            
            # Download the MP3 file from FTP
            audio_path = f"temp_{file}"
            ftp = ftplib.FTP(ftp_host)
            ftp.login(ftp_user, ftp_pass)
            with open(audio_path, "wb") as f:
                ftp.retrbinary(f"RETR {file}", f.write)
            ftp.quit()
            
            # Convert MP3 to WAV for analysis
            audio = AudioSegment.from_mp3(audio_path)
            audio = audio.set_frame_rate(16000).set_channels(1)  # Convert to mono, 16kHz
            wav_file = f"{file}.wav"
            audio.export(wav_file, format="wav")
            
            # Perform transcription using AssemblyAI API
            transcribed_text = transcribe_audio_aai(wav_file)
            
            if transcribed_text:
                # Analyze sentiment and tone
                sentiment, sentiment_score, positive_words, negative_words = analyze_sentiment(transcribed_text)
                tone, tone_score = analyze_tone(wav_file)
                
                # Save the results
                results.append([file, sentiment, sentiment_score, positive_words, negative_words, tone, tone_score])
            
            # Remove temp files
            os.remove(audio_path)
            os.remove(wav_file)
    
    # Create DataFrame and save to CSV
    if results:
        df = pd.DataFrame(results, columns=["File", "Sentiment", "Sentiment Score", "Positive Words", "Negative Words", "Tone", "Tone Score"])
        df.to_csv("audio_analysis_results.csv", index=False)
        st.write("Analysis completed. Results saved to `audio_analysis_results.csv`")
    else:
        st.warning("No audio files were processed.")

# Streamlit UI setup
st.title("Audio Analysis from FTP")

# Ask user for FTP password
ftp_pass = st.text_input("Enter FTP Password", type="password")

# FTP credentials (hardcoded for security)
ftp_host = "cph.v4one.co.uk"
ftp_user = "yash.sharma"

# Handle FTP connection when password is entered
if ftp_pass:
    if st.button("Connect to FTP"):
        # Get directories from FTP server
        directories = get_ftp_folders(ftp_host, ftp_user, ftp_pass)
        
        if directories:
            folder_path = st.selectbox("Select Folder", directories)
            
            if folder_path:
                st.write(f"Processing audio files from folder `{folder_path}`")
                if st.button("Start Analysis"):
                    process_audio_files(ftp_host, ftp_user, ftp_pass, folder_path)
        else:
            st.warning("No directories found on the FTP server.")
else:
    st.warning("Please enter the FTP password.")
