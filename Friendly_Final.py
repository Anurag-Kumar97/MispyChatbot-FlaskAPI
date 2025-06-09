import os
import asyncio
import sys
import time
import glob
import pandas as pd
import numpy as np
import edge_tts
import whisper
import sounddevice as sd
from scipy.io.wavfile import write
from sentence_transformers import SentenceTransformer, util
from pydub import AudioSegment
from pydub.playback import play
from transformers import pipeline
from flask import Flask, request, jsonify
from flask_cors import CORS
import warnings

# Fix tokenizer parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Suppress FP16 warning from Whisper
warnings.filterwarnings("ignore", message="FP16 is not supported on CPU*")

# Optional: suppress all HuggingFace fork warnings
warnings.filterwarnings("ignore", message="The current process just got forked*")

app = Flask(__name__, static_url_path='/static')
CORS(app)

# Global variables
whisper_model = None
model = None
sentiment_pipeline = None
emotion_classifier = None
user_queries = None
responses = None
query_embeddings = None

# -------- AUDIO FILE MANAGEMENT --------
def cleanup_old_audio_files(max_files=5):
    """Keep only the most recent audio files to prevent storage bloat"""
    try:
        files = sorted(glob.glob('static/response_*.mp3'), key=os.path.getmtime)
        while len(files) > max_files:
            os.remove(files.pop(0))
    except Exception as e:
        print(f"Error cleaning up audio files: {e}")

def get_unique_filename():
    """Generate unique filename with timestamp"""
    return f"response_{int(time.time())}.mp3"

# -------- SPEAK FUNCTION --------
async def speak(text, play_audio=False):
    """Convert text to speech and optionally play it immediately"""
    try:
        tts = edge_tts.Communicate(text, "en-US-AvaNeural")
        
        # Ensure the static directory exists
        if not os.path.exists('static'):
            os.makedirs('static')
        
        filename = get_unique_filename()
        path = os.path.join('static', filename)
        
        await tts.save(path)
        
        if play_audio:
            try:
                audio = AudioSegment.from_file(path, format="mp3")
                play(audio)
            except Exception as e:
                print(f"Error playing audio: {e}")
        
        cleanup_old_audio_files()
        return path
        
    except Exception as e:
        print(f"Error in speak function: {e}")
        raise

# -------- RECORD AUDIO --------
def record_audio(filename="user_input.wav", duration=7, fs=16000):
    """Record audio from microphone"""
    try:
        print("Recording...")
        recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
        sd.wait()  # Wait until recording is finished
        write(filename, fs, recording)  # Save as WAV file
        return filename
    except Exception as e:
        print(f"Error recording audio: {e}")
        raise

# -------- TRANSCRIBE AUDIO --------
def transcribe_audio(filename):
    """Convert speech to text using Whisper"""
    try:
        result = whisper_model.transcribe(filename)
        return result["text"].strip()
    except Exception as e:
        print(f"Error transcribing audio: {e}")
        raise

# -------- LOAD MODELS AND DATA --------
def init_models_and_data():
    """Initialize all models and load conversation data"""
    global whisper_model, model, sentiment_pipeline, emotion_classifier, user_queries, responses, query_embeddings
    
    print("📦 Loading models and data...")
    
    try:
        # Load Whisper model
        whisper_model = whisper.load_model("base")
        
        # Load sentence transformer model
        model = SentenceTransformer('all-MiniLM-L12-v2')
        
        # Load sentiment analysis model
        sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
            revision="714eb0f"
        )
        
        # Load emotion classification model
        emotion_classifier = pipeline(
            "text-classification",
            model="bhadresh-savani/distilbert-base-uncased-emotion",
            return_all_scores=True
        )
        
        # Load conversation dataset
        # file_path = "/usr/local/bin/Friendly_Conversation_Pie.xlsx"
        file_path = os.path.join(os.path.dirname(__file__), "data", "Friendly_Conversation_Pie.xlsx")
        data = pd.read_excel(file_path)
        data = data.dropna(subset=["Users", "Conversations"]).reset_index(drop=True)
        
        # Prepare user queries and assistant responses
        user_queries = data['Users'].tolist()
        responses = data['Conversations'].tolist()
        query_embeddings = model.encode(user_queries, convert_to_tensor=True)
        
        print("✅ Models and data loaded successfully.")
    except Exception as e:
        print(f"❌ Error loading models/data: {e}")
        raise

# -------- FIND BEST MATCH --------
def find_best_match(user_input):
    """Find the most relevant response from the dataset"""
    try:
        input_embedding = model.encode([user_input], convert_to_tensor=True)
        similarity_scores = util.pytorch_cos_sim(input_embedding, query_embeddings).cpu().numpy().flatten()
        best_idx = np.argmax(similarity_scores)
        return responses[best_idx]
    except Exception as e:
        print(f"Error finding best match: {e}")
        return "I'm sorry, I couldn't process that request."

# -------- SENTIMENT & EMOTION ANALYSIS --------
def get_sentiment_emotion(text):
    """Analyze text sentiment and emotion"""
    try:
        sentiment_result = sentiment_pipeline(text)[0]
        sentiment = "neutral" if sentiment_result['score'] < 0.6 else sentiment_result['label'].lower()
        
        emotion_scores = emotion_classifier(text)[0]
        top_emotion = max(emotion_scores, key=lambda x: x['score'])['label']
        
        emotion_map = {"sadness": "sad", "joy": "happy", "anger": "angry"}
        emotion = emotion_map.get(top_emotion, "neutral")
        
        return sentiment, emotion
    except Exception as e:
        print(f"Error in sentiment/emotion analysis: {e}")
        return "neutral", "neutral"

# -------- API ROUTES --------
@app.route("/ask", methods=["POST"])
def ask():
    """Handle text input requests"""
    try:
        data = request.get_json()
        query = data.get("query", "").strip()

        if not query:
            return jsonify({"error": "Invalid input", "audio_url": ""}), 400

        # Check for exit conditions
        if query.lower() in ["exit", "goodbye", "bye", "thank you", "babye", "thankyou", "good bye"]:
            response = "Goodbye! Hope My Spy was able to help you. Give Pie a genuine feedback about our conversations. Thank you!"
        else:
            response = find_best_match(query)

        # Generate audio file (don't play it here - let frontend handle playback)
        audio_path = asyncio.run(speak(response, play_audio=False))

        return jsonify({
            "response": response,
            "audio_url": f"/static/{os.path.basename(audio_path)}"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/listen", methods=["POST"])
def listen():
    """Handle voice input requests"""
    try:
        # Record audio
        audio_file = record_audio()
        
        # Transcribe audio
        user_input = transcribe_audio(audio_file)
        
        if not user_input:
            return jsonify({"error": "No speech detected"}), 400
        
        # Get response
        if user_input.lower() in ["exit", "goodbye", "bye", "thank you", "babye", "thankyou", "good bye"]:
            response = "Goodbye! Hope My Spy was able to help you. Give Pie a genuine feedback about our conversations. Thank you!"
        else:
            response = find_best_match(user_input)
        
        # Generate audio file (don't play it here)
        audio_path = asyncio.run(speak(response, play_audio=False))
        
        return jsonify({
            "user_input": user_input,
            "response": response,
            "audio_url": f"/static/{os.path.basename(audio_path)}"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/welcome", methods=["GET"])
def welcome():
    """Provide welcome message"""
    try:
        welcome_message = "Welcome to My Spy! Hi, I am Pie. I am your friend. How can I help you today?"
        audio_path = asyncio.run(speak(welcome_message, play_audio=False))
        return jsonify({
            "response": welcome_message,
            "audio_url": f"/static/{os.path.basename(audio_path)}"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# -------- STARTUP --------
if __name__ == "__main__":
    try:
        init_models_and_data()  # Load models and data once on startup
        # Ensure static directory exists
        if not os.path.exists('static'):
            os.makedirs('static')
        app.run(host="0.0.0.0", port=8000)
    except Exception as e:
        print(f"Failed to start application: {e}")
        sys.exit(1)