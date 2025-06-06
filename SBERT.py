from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import asyncio
import edge_tts
from sentence_transformers import SentenceTransformer, util
from pydub import AudioSegment
from pydub.playback import play
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)

# Load the model
model = SentenceTransformer('all-MiniLM-L12-v2')

# Load your data from Excel
data = pd.read_excel("/usr/local/bin/excel.xlsx")
data['Combined_Text'] = data.apply(lambda row: f"{row['Backstory']} {row['User Location']}", axis=1)

# Generate embeddings for the combined text data
provider_embeddings = model.encode(data['Combined_Text'].tolist(), convert_to_tensor=True)

# Clean text function
def clean_text(text):
    return ''.join(e for e in str(text) if e.isalnum() or e.isspace())

# Text-to-Speech function
async def speak(text):
    tts = edge_tts.Communicate(text, "en-US-AvaNeural")
    
    # Ensure the static directory exists
    if not os.path.exists('static'):
        os.makedirs('static')
    
    filename = "response.mp3"
    path = os.path.join('static', filename)
    
    await tts.save(path)
    
    # Play the generated audio
    play(AudioSegment.from_file(path, format="mp3"))
    
    return filename  # Return only the filename for the URL

# Function to find the best match from the data
def find_best_match(user_backstory, user_location):
    user_input = f"{user_backstory} {user_location}"
    user_embedding = model.encode([user_input], convert_to_tensor=True)
    similarity_scores = util.pytorch_cos_sim(user_embedding, provider_embeddings).cpu().numpy().flatten()
    best_match_idx = np.argmax(similarity_scores)
    best_provider = data.iloc[best_match_idx]
    
    # Format the response text
    response_text = (
        f"The best provider for your case is {clean_text(best_provider['Matched Provider'])}, "
        f"located in {clean_text(best_provider['Provider Location'])}. "
        f"Their specialties include {clean_text(best_provider['Specialties'])}."
    )
    
    return response_text

# Define the route to get recommendations
@app.route("/recommend", methods=["POST"])
def recommend():
    content = request.get_json()
    backstory = content.get("backstory", "")
    location = content.get("location", "")

    if not backstory or not location:
        return jsonify({"error": "Missing input fields"}), 400

    result_text = find_best_match(backstory, location)
    audio_filename = asyncio.run(speak(result_text))

    return jsonify({
        "response": result_text,
        "audio_url": f"/static/{audio_filename}"
    })

# Run the Flask app
if __name__ == "__main__":
 app.run(host="0.0.0.0", port=8003, debug=False)
