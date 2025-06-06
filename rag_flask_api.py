import os
import asyncio
import numpy as np
import faiss
from flask import Flask, request, jsonify
from flask_cors import CORS
from sentence_transformers import SentenceTransformer, CrossEncoder
from pydub import AudioSegment
from pydub.playback import play
import edge_tts

app = Flask(__name__, static_url_path='/static')
CORS(app)

# Globals
bi_encoder = None
cross_encoder = None
chunk_map = {}
index = None

# -------- SPEAK FUNCTION --------
async def speak(text):
    tts = edge_tts.Communicate(text, "en-US-AvaNeural")
    
    # Ensure the static directory exists
    if not os.path.exists('static'):
        os.makedirs('static')
    
    filename = "response.mp3"
    path = os.path.join('static', filename)
    
    await tts.save(path)
    
    audio = AudioSegment.from_file(path, format="mp3")
    play(audio)  # Play audio immediately if needed
    
    return path  # Return the path to the audio file

# -------- LOAD MODELS AND FAISS INDEX --------
def init_models_and_index():
    global bi_encoder, cross_encoder, chunk_map, index

    print("📦 Loading models and creating index...")
    bi_encoder = SentenceTransformer("intfloat/e5-large-v2")
    cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    with open("/usr/local/bin/Newdata_cleaned.txt", "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]

    chunks = []
    for i in range(len(lines)):
        if lines[i].startswith("User"):
            user_query = lines[i].replace("User", "").strip()
            ai_response = []
            j = i + 1
            while j < len(lines) and lines[j].startswith("AI"):
                ai_response.append(lines[j].replace("AI", "").strip())
                j += 1
            full_chunk = f"User: {user_query}\nAI: {' '.join(ai_response)}"
            chunks.append(full_chunk)

    chunk_embeddings = bi_encoder.encode(chunks, convert_to_tensor=False, show_progress_bar=True)
    chunk_embeddings = np.array(chunk_embeddings).astype("float32")
    dimension = chunk_embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(chunk_embeddings)
    chunk_map.update({i: chunks[i] for i in range(len(chunks))})
    print("✅ Model and index loaded.")

# -------- RETRIEVE RESPONSE --------
def retrieve_top_chunk(query):
    query_embedding = bi_encoder.encode(query, convert_to_tensor=False).astype("float32")
    D, I = index.search(np.array([query_embedding]), 10)
    candidate_chunks = [chunk_map[i] for i in I[0]]
    scores = cross_encoder.predict([[query, c] for c in candidate_chunks])
    sorted_chunks = sorted(zip(candidate_chunks, scores), key=lambda x: x[1], reverse=True)
    return sorted_chunks[0][0]

def get_ai_response(user_input):
    top_chunk = retrieve_top_chunk(user_input)
    lines = top_chunk.split("\n")
    ai_line = next((line.replace("AI:", "").strip() for line in lines if line.startswith("AI:")), "Let me help you.")
    return ai_line

# -------- API ROUTE --------
@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    query = data.get("query", "")

    if not query:
        return jsonify({"response": "Invalid input", "audio_url": ""}), 400

    response = get_ai_response(query)

    # Generate audio file (asynchronously)
    audio_path = asyncio.run(speak(response))

    # Return text response + path to the audio file
    return jsonify({
        "response": response,
        "audio_url": f"/static/{os.path.basename(audio_path)}"
    })

# -------- STARTUP --------
if __name__ == "__main__":
    init_models_and_index()  # Load once on startup
    app.run(host="0.0.0.0", port=8002)
