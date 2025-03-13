import os
import ftplib
import streamlit as st
import assemblyai as aai
from pydub import AudioSegment
from textblob import TextBlob
import librosa
import numpy as np
import pandas as pd

# Function to check if an item is a directory
def is_ftp_directory(ftp, item):
    try:
        ftp.cwd(item)
        ftp.cwd('..')  # Go back to the parent directory
        return True
    except:
        return False

# Function to get directories from FTP server
def get_ftp_folders(ftp_host, ftp_user, ftp_pass):
    try:
        ftp = ftplib.FTP(ftp_host)
        ftp.login(ftp_user, ftp_pass)
        items = ftp.nlst()
        directories = [item for item in items if is_ftp_directory(ftp, item)]
        ftp.quit()
        print("Directories found:", directories)  # Debugging line
        return directories
    except Exception as e:
        st.error(f"Error while fetching directories: {e}")
        return []

# Function to fetch audio files from a selected FTP folder
def get_audio_files_from_ftp(ftp_host, ftp_user, ftp_pass, folder):
    try:
        ftp = ftplib.FTP(ftp_host)
        ftp.login(ftp_user, ftp_pass)
        ftp.cwd(folder)
        files = ftp.nlst()
        ftp.quit()
        print("Files found:", files)  # Debugging line
        return files
    except Exception as e:
        st.error(f"Error while fetching files: {e}")
        return []

# Function to transcribe audio file using AssemblyAI
def transcribe_audio_aai(audio_file, api_key):
    aai.settings.api_key = api_key  # Set API Key for AssemblyAI
    transcriber = aai.Transcriber()
    transcript = transcriber.transcribe(audio_file)

    while transcript.status != 'completed':
        st.write("Transcription in progress...")
        transcript = transcriber.get(transcript.id)
    
    return transcript.text

# Perform sentiment analysis on text
def analyze_sentiment(text):
    blob = TextBlob(text)
    sentiment_score = blob.sentiment.polarity
    sentiment = "Positive" if sentiment_score > 0 else "Negative" if sentiment_score < 0 else "Neutral"
    return sentiment, sentiment_score

# Perform tone analysis on audio file using librosa
def analyze_tone(audio_file):
    y, sr = librosa.load(audio_file)
    pitches, _ = librosa.piptrack(y=y, sr=sr)
    pitch_values = pitches[pitches > 0]
    avg_pitch = np.mean(pitch_values) if len(pitch_values) > 0 else 0
    energy = np.mean(librosa.feature.rms(y=y))

    if avg_pitch > 200 and energy > 0.1:
        return "Happy"
    elif avg_pitch < 100 and energy < 0.05:
        return "Sad"
    else:
        return "Neutral"

# Main process to download audio, transcribe, and analyze
def process_audio_files(ftp_host, ftp_user, ftp_pass, folder, api_key):
    # Fetch audio files from FTP
    files_in_folder = get_audio_files_from_ftp(ftp_host, ftp_user, ftp_pass, folder)
    
    # List to store results
    results = []

    for file in files_in_folder:
        if file.endswith(".mp3"):
            st.write(f"Processing {file}...")

            # Download the audio file
            audio_path = f"temp_{file}"
            ftp = ftplib.FTP(ftp_host)
            ftp.login(ftp_user, ftp_pass)
            with open(audio_path, "wb") as f:
                ftp.retrbinary(f"RETR {file}", f.write)
            ftp.quit()

            # Convert MP3 to WAV
            audio = AudioSegment.from_mp3(audio_path)
            audio = audio.set_frame_rate(16000).set_channels(1)
            wav_file = f"{file}.wav"
            audio.export(wav_file, format="wav")
            
            # Perform transcription using AssemblyAI
            transcribed_text = transcribe_audio_aai(wav_file, api_key)
            
            # Analyze sentiment and tone
            sentiment, sentiment_score = analyze_sentiment(transcribed_text)
            tone = analyze_tone(wav_file)

            # Store the results
            results.append([file, sentiment, sentiment_score, tone])

            # Remove temporary files
            os.remove(audio_path)
            os.remove(wav_file)
    
    # Create DataFrame and save to CSV
    if results:
        df = pd.DataFrame(results, columns=["File", "Sentiment", "Sentiment Score", "Tone"])
        df.to_csv("audio_analysis_results.csv", index=False)
        st.write("Analysis completed. Results saved to `audio_analysis_results.csv`")
    else:
        st.warning("No audio files were processed.")

# Streamlit UI setup
st.title("FTP Audio Analysis")

# Ask user for FTP password
ftp_pass = st.text_input("Enter FTP Password", type="password")

# Input for AssemblyAI API key
api_key = st.text_input("Enter AssemblyAI API Key", type="password")

# FTP credentials
ftp_host = "cph.v4one.co.uk"
ftp_user = "yash.sharma"

# When user enters FTP password and API Key
if ftp_pass and api_key:
    # Connect to FTP and list directories
    if st.button("Connect to FTP"):
        directories = get_ftp_folders(ftp_host, ftp_user, ftp_pass)
        
        if directories:
            folder_path = st.selectbox("Select Folder", directories)
            
            if folder_path:
                st.write(f"Processing audio files from folder `{folder_path}`")
                if st.button("Start Analysis"):
                    st.write("Starting the analysis...")  # Debugging line
                    process_audio_files(ftp_host, ftp_user, ftp_pass, folder_path, api_key)
        else:
            st.warning("No directories found on the FTP server.")
else:
    st.warning("Please enter FTP password and AssemblyAI API Key.")
