from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import asyncio
import edge_tts
from sentence_transformers import SentenceTransformer, util
import os
from transformers import pipeline
from flask_cors import CORS
import logging
import time
 
# Disable tokenizers parallelism for consistent model behavior
os.environ["TOKENIZERS_PARALLELISM"] = "false"
 
# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
 
app = Flask(__name__)
CORS(app)
 
# Load models
try:
    model = SentenceTransformer('all-MiniLM-L12-v2')
    sentiment_pipeline = pipeline(
        "sentiment-analysis",
        model="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
        revision="714eb0f"
    )
except Exception as e:
    logger.error(f"Error loading models: {e}")
    raise RuntimeError("Failed to load models")
 
# Load dataset
file_path = "/usr/local/bin/excel2.xlsx"
try:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Excel file not found at {file_path}")
    data = pd.read_excel(file_path)
    data['Combined_Text'] = data.apply(lambda row: f"{row['Backstory']} {row['User Location']}", axis=1)
    provider_embeddings = model.encode(data['Combined_Text'].tolist(), convert_to_tensor=True)
except Exception as e:
    logger.error(f"Error loading dataset: {e}")
    raise RuntimeError("Failed to load dataset")
 
# PI-related sample prompts (full list from original script)
pi_samples = [
    "Can you find out who scratched my car last night?",
    "I think someone is following me. Can you look into it?",
    "My coworker is spreading rumors about me. I want to know why.",
    "I need help verifying if this person actually served in the military.",
    "Someone keeps stealing my mail. I want to catch them.",
    "Can you help me locate a tenant who fled without paying rent?",
    "There’s been a lot of strange noise from the neighbor’s garage. I want to know what’s going on.",
    "My child’s friend seems dangerous. Can you do a background check?",
    "I think someone is using my identity. Can you find out who?",
    "I suspect corporate espionage within my startup. I need someone to investigate.",
    "I want to know if this online seller is legit before I send money.",
    "Can you follow my husband and report where he goes after work?",
    "A former business partner scammed me. I need help building a case.",
    "Help me check if this nanny has a criminal background.",
    "I found a listening device in my office. I want to know who put it there.",
    "Can you track down my stolen drone?",
    "Someone used my social security number. I need help tracking the culprit.",
    "I think my neighbor is running an illegal business from their basement.",
    "Please help me verify if this person is lying on their resume.",
    "I want to confirm whether my daughter’s boyfriend has a record.",
    "Help me figure out who vandalized my storefront last weekend.",
    "There’s been embezzlement at my company. Can you investigate the finances?",
    "My landlord may be spying on me. I need proof.",
    "Can you help find the person who broke into my car?",
    "A former employee may be leaking confidential info. Can you look into it?",
    "I need to find a long-lost family member for a health-related issue.",
    "Someone is impersonating me online. Can you track them down?",
    "I think I’m being watched. Can you sweep for hidden cameras?",
    "My ex is stalking me. I need proof for a restraining order.",
    "I heard my neighbor is abusing animals. I want this investigated.",
    "A strange man has been loitering around my daughter’s school. Can you identify him?",
    "Can you verify the legitimacy of a charity before I donate?",
    "I found a box of old letters hinting at a family scandal. Can you uncover the truth?",
    "Someone scratched threatening messages on my car. I need help.",
    "My wife’s behavior has changed suddenly. I want to know what’s going on.",
    "I saw someone trying to break into my house. I need help finding them.",
    "I’ve been getting strange calls at work. Can you trace them?",
    "Can you investigate the sudden disappearance of my pet?",
    "A scam artist tricked my elderly mother. Can you recover any info?",
    "Can you investigate property records to see if there’s fraud involved?",
    "My boss is making shady business deals. I want to blow the whistle.",
    "My daughter met a man online. I want to make sure she’s safe.",
    "I need to verify if my tenant lied on the lease application.",
    "Help me prove harassment by a coworker that’s been off the radar.",
    "Can you track stolen property being sold online?",
    "I think someone in my apartment complex is stealing my deliveries.",
    "My friend's boyfriend has a mysterious past. Can you look into it?",
    "There’s been unexplained cash transfers from my joint account.",
    "I want to verify if my babysitter is who she says she is.",
    "Someone tried to blackmail me. I need to know who it was.",
    "My late father had a storage unit. Can you help me locate it?",
    "Can you verify if this person is married before I go on a date?",
    "I want to check if my new business partner has legal issues.",
    "I’m receiving threats from a blocked number. Can you trace it?",
    "I suspect someone is sabotaging my vehicle. I need proof.",
    "I think I saw someone from a cold case in my neighborhood.",
    "Can you look into the past of this private school before I enroll my kid?",
    "Help me find out if my brother is involved in a gang.",
    "Can you investigate if a teacher at my child’s school is abusive?",
    "There’s been a spike in break-ins nearby. I think I know who’s behind it.",
    "Someone forged my signature on a contract. I need evidence.",
    "Can you locate the owner of an abandoned vehicle near my property?",
    "Can you confirm whether a nonprofit is real before I volunteer?",
    "My fiancé refuses to talk about his ex. I want to know why.",
    "I need to find the person who hit my parked car and drove away.",
    "A suspicious person keeps showing up at my workplace. I want to know who they are.",
    "Can you help me find out if my boyfriend is cheating when he says he's working late?",
    "I need help identifying a woman who's been messaging my husband anonymously.",
    "Someone left a note on my windshield with a threat. I need to know who it was.",
    "Can you find the owner of a license plate I wrote down from a hit-and-run?",
    "My daughter’s acting distant and keeps sneaking out. Can you look into who she's seeing?",
    "I think a coworker is faking a disability for time off. Can you verify it?",
    "There’s someone new hanging around our street every night. Can you check them out?",
    "My elderly mom is suddenly short on money. I suspect a caregiver might be taking advantage.",
    "Can you investigate a guy who says he’s a veteran before I donate to his cause?",
    "I just got served divorce papers out of nowhere. I want to know if my spouse planned this.",
    "I want to check if someone changed my father’s will before he died.",
    "My ex moved states with our child without telling me. Can you track them?",
    "I think my landlord has hidden cameras in the house. Can someone sweep the place?",
    "A guy at work has been making threats off the clock. I want to know if he has a record.",
    "My teenage son won’t say where he’s getting extra money from. I want to know the source.",
    "Can you check if this wedding photographer actually exists? I already paid a deposit.",
    "My sister’s new boyfriend seems sketchy. Can you find out if he’s been in jail?",
    "There’s a strange car parked outside my house every morning. Can you figure out who owns it?",
    "My roommate has been lying about rent payments. I want proof.",
    "I found a suspicious tracking device under my car. Can you trace it back to someone?",
    "Can you track someone who sold me fake event tickets?",
    "My best friend’s fiancé might be hiding something. I want to know the truth before their wedding.",
    "A guy offered to invest in my startup. I want to verify he’s legit.",
    "I’m being accused of something I didn’t do. Can you gather evidence to clear my name?",
    "My daughter’s grades dropped suddenly and she seems scared. Can you check what’s going on at school?",
    "I suspect someone is forging signatures at my nonprofit.",
    "My credit card keeps getting hacked. Can you trace who's behind it?",
    "A contractor took my money and vanished. I want to track him down.",
    "I got a letter from a lawyer I’ve never heard of. Can you check if it’s real?",
    "I just found an old photo with someone my spouse claims to have never met. Can you investigate?",
    "My business partner is making large purchases without approval. I need evidence for legal action.",
    "I saw my personal medical info online. Can you find out who leaked it?",
    "I think someone is impersonating me on dating apps. Can you look into this?",
    "I’ve seen someone sneaking into the back of the office after hours. I want to know what they’re doing.",
    "My brother is missing and police aren’t helping. I need someone to find him.",
    "Can you investigate whether a charity fundraiser was a scam?",
    "My kid is getting bullied, but the school won’t act. I need video proof.",
    "I need to check if an accident claim against me is legit or staged.",
    "My ex is using a fake profile to stalk me online. Can you trace it?",
    "I’m getting packages I didn’t order. I think someone is using my address.",
    "Can you help verify if this offshore job offer is a scam?",
    "My husband keeps deleting texts. I need to know who he's talking to.",
    "I found a receipt for a hotel I never went to. I need answers.",
    "Someone keyed my car with a slur. I need proof to file charges.",
    "A rival store might be copying our business plan. Can you investigate?",
    "There are unusual charges on my joint account. Can you track them?",
    "My child’s coach is acting inappropriately. I need a background check.",
    "My identity was used to rent an apartment. Can you find out by who?",
    "I think I was followed home last night. Can you check for surveillance footage?",
    "A mystery person keeps liking all my social posts and showing up where I go. Can you identify them?",
    # Extended samples for missing persons
    "My sister has been missing for two days and no one knows her whereabouts.",
    "I’m trying to locate an old friend who suddenly disappeared from social media and work.",
    "Can you find my dad? He left without telling anyone and we haven’t heard from him since.",
    "I need to know if my elderly uncle is okay—he’s not answering calls and lives alone.",
    "We’ve lost contact with a relative after a family dispute. Can you help reconnect us?",
    "A former tenant disappeared owing several months’ rent—can you track them down?",
    "I’m trying to find my biological father but only have his first name and a city.",
    "Can you help locate someone who owes me money and moved out of state?",
    "I want to reconnect with a childhood friend who vanished after high school.",
    "I need to serve legal papers but the person has moved and left no forwarding address.",
]
pi_sample_embeddings = model.encode(pi_samples, convert_to_tensor=True)
 
# PI context samples (including relevant missing person entry)
pi_context_samples = [
    ("My dog ran away and I can’t find her anywhere.", "pet recovery"),
    ("I suspect my spouse of cheating after seeing strange messages.", "infidelity"),
    ("My sister has been missing for two days and no one knows her whereabouts.", "missing person"),
    ("I want to verify someone's criminal record before hiring them.", "background check"),
    ("I believe someone is hacking into my email account.", "cyber investigation"),
    ("My car was hit in a hit-and-run and the driver vanished.", "accident reconstruction"),
    ("Our warehouse was set ablaze under suspicious circumstances.", "arson investigation"),
    ("I think a coworker is abusing FMLA while working another job.", "leave abuse"),
    ("Someone forged my signature on a legal document.", "document forgery"),
    ("I need to find hidden assets for a divorce case.", "asset tracing"),
    ("Lisa has had her car scratched four times in the past two months. She suspects a jealous coworker and needs proof before confronting them.",
     "vandalism investigation"),
    ("Margaret has spent years searching for her younger brother who vanished in the 1970s, with only a torn postcard address as a clue.",
     "cold case missing person"),
    # Add remaining pi_context_samples from your original script as needed
]
context_texts = [t for t, lbl in pi_context_samples]
context_labels = [lbl for t, lbl in pi_context_samples]
context_embeddings = model.encode(context_texts, convert_to_tensor=True)
 
# Clean text helper
def clean_text(text):
    return ''.join(e for e in str(text) if e.isalnum() or e.isspace())
 
# Speak response using edge TTS
async def speak(text):
    try:
        if not os.path.exists('static'):
            os.makedirs('static')
        filename = f"response_{int(time.time() * 1000)}.mp3"
        path = os.path.join('static', filename)
        tts = edge_tts.Communicate(text, "en-US-JennyNeural")
        await tts.save(path)
        return filename
    except Exception as e:
        logger.error(f"Error generating audio: {e}")
        raise
 
# Run async tasks in Flask
def run_async(coro):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()
 
# Enhanced semantic guardrail check with debug logging
def is_pi_related_semantic(text):
    try:
        user_embedding = model.encode([text], convert_to_tensor=True)
        similarity_scores = util.pytorch_cos_sim(user_embedding, pi_sample_embeddings).cpu().numpy().flatten()
        max_score = np.max(similarity_scores)
        top_indices = np.argsort(similarity_scores)[-5:][::-1]
        logger.info(f"Input: {text[:50]}...")
        logger.info(f"Max similarity score: {max_score}")
        for idx in top_indices:
            logger.info(f"Score: {similarity_scores[idx]}, Prompt: {pi_samples[idx]}")
        return max_score > 0.3  # Lowered threshold to ensure missing person cases pass
    except Exception as e:
        logger.error(f"Error in semantic check: {e}")
        return False
 
def extract_context_label(backstory: str) -> str:
    try:
        user_emb = model.encode([backstory], convert_to_tensor=True)
        sims = util.pytorch_cos_sim(user_emb, context_embeddings).squeeze().cpu().numpy()
        best_idx = int(np.argmax(sims))
        return context_labels[best_idx]
    except Exception as e:
        logger.error(f"Error extracting context label: {e}")
        return "unknown"
 
# Sentiment detection
def get_sentiment(text):
    try:
        result = sentiment_pipeline(text)[0]
        return result['label'], result['score']
    except Exception as e:
        logger.error(f"Error in sentiment analysis: {e}")
        return "UNKNOWN", 0.0
 
# Find best provider match
def find_best_match(user_backstory, user_location):
    try:
        user_input = f"{user_backstory} {user_location}"
        user_embedding = model.encode([user_input], convert_to_tensor=True)
        similarity_scores = util.pytorch_cos_sim(user_embedding, provider_embeddings).cpu().numpy().flatten()
        best_match_idx = np.argmax(similarity_scores)
        best_provider = data.iloc[best_match_idx]
 
        specialties = clean_text(best_provider['Specialties'])
        location = clean_text(best_provider['Provider Location'])
        context = extract_context_label(user_backstory)
 
        response_text = (
            f"Based on your request, I can connect you with a private investigator "
            f"located in {location} who specializes in {specialties}. "
            f"This private investigator can help you with issues related to {context}."
        )
 
        return {
            "response": response_text,
            "specialties": best_provider['Specialties'],
            "provider_location": best_provider['Provider Location'],
            "context": context
        }
    except Exception as e:
        logger.error(f"Error finding best match: {e}")
        raise
 
@app.route("/recommend", methods=["POST"])
def recommend():
    try:
        content = request.get_json()
        if not content:
            return jsonify({"error": "Invalid JSON payload"}), 400
 
        backstory = content.get("backstory", "").strip()
        location = content.get("location", "").strip()
 
        if not backstory or not location:
            return jsonify({"error": "Missing backstory or location fields"}), 400
 
        # Guardrail check
        if not is_pi_related_semantic(backstory):
            error_msg = "Sorry, I can only assist with private investigation services."
            audio_filename = run_async(speak(error_msg))
            return jsonify({
                "error": error_msg,
                "audio_url": f"/static/{audio_filename}"
            }), 400
 
        # Sentiment analysis
        sentiment, _ = get_sentiment(backstory)
        sentiment_msg = f"I understand you're feeling {sentiment.lower()}."
        sentiment_audio = run_async(speak(sentiment_msg))
 
        # Find best match
        match_result = find_best_match(backstory, location)
        audio_filename = run_async(speak(match_result["response"]))
 
        return jsonify({
            "sentiment": sentiment_msg,
            "sentiment_audio_url": f"/static/{sentiment_audio}",
            "response": match_result["response"],
            "specialties": match_result["specialties"],
            "provider_location": match_result["provider_location"],
            "context": match_result["context"],
            "audio_url": f"/static/{audio_filename}"
        }), 200
    except Exception as e:
        logger.error(f"Error in /recommend: {e}")
        error_msg = "Internal server error"
        audio_filename = run_async(speak(error_msg))
        return jsonify({
            "error": error_msg,
            "audio_url": f"/static/{audio_filename}"
        }), 500
 
@app.route("/welcome", methods=["GET"])
def welcome():
    try:
        welcome_msg = "Welcome to My Spy! Hi, I am Pie. How can I help you today?"
        audio_filename = run_async(speak(welcome_msg))
        return jsonify({
            "response": welcome_msg,
            "audio_url": f"/static/{audio_filename}"
        }), 200
    except Exception as e:
        logger.error(f"Error in /welcome: {e}")
        error_msg = "Internal server error"
        audio_filename = run_async(speak(error_msg))
        return jsonify({
            "error": error_msg,
            "audio_url": f"/static/{audio_filename}"
        }), 500
 
if __name__ == "__main__":
    if not os.path.exists('static'):
        os.makedirs('static')
    app.run(host="0.0.0.0", port=8003, debug=False)