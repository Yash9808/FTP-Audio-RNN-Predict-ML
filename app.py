import os
import ftplib
import assemblyai as aai
import pandas as pd
from pydub import AudioSegment
from textblob import TextBlob
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from datetime import datetime
import streamlit as st
from pytz import timezone

# FTP Server details
HOST = "cph.v4one.co.uk"
USERNAME = "yash.sharma"
PASSWORD = "8x]EnG/Z%DXvcjt[%<;S"

# AssemblyAI API Key
aai.settings.api_key = "507f082334f2416d9784aa3829477738"

# Email Settings
sender_email = "yash.sharma@chartwellprivatehospital.co.uk"
email_password = "CWPh-03002"
receiver_emails = [
    "karuna.p@chartwellprivatehospital.co.uk",
    "anika.misra@guidinglights.co.uk",
    "yash.sharma@chartwellprivatehospital.co.uk"
]

# Function to list available folders
def list_folders(ftp):
    ftp.cwd('/')
    folders = [f for f in ftp.nlst() if '.' not in f]  # Assuming folders don't have file extensions
    return folders

# Function to download MP3 files from FTP
def download_mp3_files(ftp, remote_folder, local_folder):
    try:
        os.makedirs(local_folder, exist_ok=True)
        ftp.cwd(remote_folder)
        files = ftp.nlst()
        mp3_files = [file for file in files if file.endswith('.mp3')]

        for file in mp3_files:
            local_file_path = os.path.join(local_folder, os.path.basename(file))
            with open(local_file_path, "wb") as local_file:
                ftp.retrbinary(f"RETR {file}", local_file.write)
            print(f"Downloaded: {file}")
        return mp3_files
    except Exception as e:
        print(f"Error: {e}")
        return []

# Function to check if audio is valid
def check_audio_validity(audio_file):
    try:
        audio = AudioSegment.from_file(audio_file)
        if len(audio.get_array_of_samples()) == 0:
            return False
        return True
    except Exception:
        return False

# Sentiment Analysis for agent and customer
def analyze_sentiment(text):
    sentiment_score = TextBlob(text).sentiment.polarity  # Using TextBlob for sentiment polarity score
    apw = text.lower().count("good")  # Example: Counting positive words
    anw = text.lower().count("bad")  # Number of negative words
    return text, sentiment_score, apw, anw

# Improved tone analysis using TextBlob's polarity
def analyze_tone(text):
    sentiment = TextBlob(text).sentiment
    if sentiment.polarity > 0.1:
        tone = "Positive"
        tone_score = sentiment.polarity
    elif sentiment.polarity < -0.1:
        tone = "Negative"
        tone_score = sentiment.polarity
    else:
        tone = "Neutral"
        tone_score = 0.5  # Neutral tone

    return tone, tone_score

# Transcribe audio using AssemblyAI
def transcribe_audio(audio_file):
    transcriber = aai.Transcriber()
    transcript = transcriber.transcribe(audio_file)

    if transcript.status == aai.TranscriptStatus.completed:
        return transcript.text
    return ""

# Send the email with the report attached
def send_email(report_file):
    # Set up the server and sender email
    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['Subject'] = "Daily Report: Call Analysis"

    # Attach the body of the email
    body = "Please find attached the daily call analysis report."
    msg.attach(MIMEText(body, 'plain'))

    # Attach the report file
    attachment = open(report_file, "rb")
    part = MIMEBase('application', 'octet-stream')
    part.set_payload(attachment.read())
    encoders.encode_base64(part)
    part.add_header('Content-Disposition', f"attachment; filename={os.path.basename(report_file)}")
    msg.attach(part)

    # Send email to all recipients
    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(sender_email, email_password)
        for receiver_email in receiver_emails:
            msg['To'] = receiver_email
            server.sendmail(sender_email, receiver_email, msg.as_string())
        server.quit()
        print("Email sent successfully!")
    except Exception as e:
        print(f"Error sending email: {e}")

# Main function to process and analyze data
def analyze_and_generate_report(remote_folder):
    # Get the current date in YYYY-MM-DD format
    today = datetime.now(timezone('Europe/London')).strftime('%Y-%m-%d')
    
    # FTP connection
    ftp = ftplib.FTP(HOST)
    ftp.login(USERNAME, PASSWORD)

    # Use the selected folder
    local_folder = f"download_audio/{remote_folder}"
    os.makedirs(local_folder, exist_ok=True)

    # Download files
    mp3_files = download_mp3_files(ftp, remote_folder, local_folder)

    # Process and analyze each file
    results = []
    for i, mp3_file in enumerate(mp3_files):
        print(f"Processing ({i+1}/{len(mp3_files)}): {mp3_file}")

        audio_path = os.path.join(local_folder, mp3_file)

        # Convert MP3 to WAV (16kHz, Mono)
        audio = AudioSegment.from_mp3(audio_path)
        audio = audio.set_frame_rate(16000).set_channels(1)[:30000]  # 30s limit
        wav_file = f"{local_folder}/temp_{i}.wav"
        audio.export(wav_file, format="wav")

        if check_audio_validity(wav_file):
            transcriber = aai.Transcriber()
            transcript = transcriber.transcribe(wav_file)

            if transcript.status == aai.TranscriptStatus.completed:
                text = transcript.text
                print(f"Transcription: {text}")

                # Sentiment Analysis for agent and customer
                agent_text = text
                customer_text = text  # Assuming same text for demo
                agent_sentiment, agent_sentiment_score, apw, anw = analyze_sentiment(agent_text)
                customer_sentiment, customer_sentiment_score, cpw, cnw = analyze_sentiment(customer_text)  

                agent_tone, agent_tone_score = analyze_tone(agent_text)  # Pass 'text' to 'analyze_tone'

                # Compute Overall Score
                overall_score = (
                    (0.3 * (agent_sentiment_score + 1) / 2) +
                    (0.3 * (customer_sentiment_score + 1) / 2) +
                    (0.2 * (agent_tone_score + 1) / 2) +
                    (0.2 * (apw - anw + cpw - cnw + 5) / 10)
                )
                overall_score = max(0, min(1, overall_score))

                results.append([  # Append results without unnecessary columns
                    mp3_file, agent_sentiment, customer_sentiment, agent_tone, overall_score,
                    agent_sentiment_score, customer_sentiment_score, agent_tone_score, apw, anw, cpw, cnw
                ])
            os.remove(wav_file)

    # Save results to CSV
    df = pd.DataFrame(results, columns=[  # No unnecessary columns, include only what is required for training
        "File", "Agent Sentiment", "Customer Sentiment", "Agent Tone", "Overall Score",
        "Agent Sentiment Score", "Customer Sentiment Score", "Agent Tone Score", "APW", "ANW", "CPW", "CNW"
    ])
    csv_path = f"{local_folder}/call_analysis_results.csv"
    df.to_csv(csv_path, index=False)
    print("Analysis completed! Results saved to CSV.")

    # Load the CSV
    df_predictions = pd.read_csv(csv_path)

    # Drop unnecessary columns
    df_predictions.drop(["File", "Agent Sentiment", "Customer Sentiment"], axis=1, inplace=True)

    # Initialize LabelEncoder for 'Agent Tone'
    le = LabelEncoder()
    df_predictions["Agent Tone"] = le.fit_transform(df_predictions["Agent Tone"])

    # Generate the plots and save them as a report
    report_file = f"{local_folder}/daily_report.png"
    plt.figure(figsize=(12, 6))
    
    # Plot 1: Distribution of 'Agent Tone'
    plt.subplot(3, 2, 1)
    df_predictions["Agent Tone"].value_counts().plot(kind='bar', color='skyblue', edgecolor='black')
    plt.title('Distribution of Agent Tone')
    plt.xlabel('Agent Tone')
    plt.ylabel('Frequency')
    plt.xticks(rotation=0)

    # Plot 2: Scatter plot of 'Agent Sentiment Score' vs 'Customer Sentiment Score'
    plt.subplot(3, 2, 2)
    plt.scatter(df_predictions['Agent Sentiment Score'], df_predictions['Customer Sentiment Score'], color='orange')
    plt.title('Agent Sentiment vs Customer Sentiment')
    plt.xlabel('Agent Sentiment Score')
    plt.ylabel('Customer Sentiment Score')
    plt.grid(True)

    # Plot 3: Scatter plot of 'Agent Tone Score' vs 'Overall Score'
    plt.subplot(3, 2, 3)
    plt.scatter(df_predictions['Agent Tone Score'], df_predictions['Overall Score'], color='green')
    plt.title('Agent Tone Score vs Overall Score')
    plt.xlabel('Agent Tone Score')
    plt.ylabel('Overall Score')
    plt.grid(True)

    # Plot 4: Correlation heatmap
    plt.subplot(3, 2, 4)
    correlation_matrix = df_predictions[['Agent Sentiment Score', 'Customer Sentiment Score', 'Agent Tone Score', 'Overall Score']].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Correlation Heatmap')

    # Plot 5: Sample index vs Overall Score
    plt.subplot(3, 2, 5)
    plt.plot(np.arange(len(df_predictions)), df_predictions['Overall Score'], marker='o', color='b', linestyle='-', markersize=5)
    plt.title('Sample Index vs Overall Score')
    plt.xlabel('Sample Index')
    plt.ylabel('Overall Score')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(report_file)

    # Send the email with the report
    send_email(report_file)

    # Clean up - delete the audio files after analysis
    for file in mp3_files:
        os.remove(os.path.join(local_folder, file))

    ftp.quit()

# Streamlit app interface
st.title("Daily Call Analysis Report Generator")
st.write("This app generates and sends the daily call analysis report.")

# Folder selection based on today's date
today = datetime.now(timezone('Europe/London')).strftime('%Y-%m-%d')
folder = st.selectbox("Select the FTP folder (e.g., 2025-03-13)", [today])

if st.button("Generate and Send Report"):
    analyze_and_generate_report(folder)
    st.success("Report generated and sent successfully!")
