import sys
import pandas as pd
import numpy as np
import asyncio
import edge_tts
from sentence_transformers import SentenceTransformer, util
from pydub import AudioSegment
from pydub.playback import play
from transformers import pipeline
from flask import Flask, request, jsonify
from flask_cors import CORS  # ✅ Import CORS

# ✅ Initialize Flask app
app = Flask(__name__)
CORS(app)  # ✅ Enable CORS for all routes
# Load models
model = SentenceTransformer('all-MiniLM-L12-v2')
#sentiment_pipeline = pipeline("sentiment-analysis")

sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
    revision="714eb0f"
)

# Load dataset
file_path = "/usr/local/bin/excel2.xlsx"
data = pd.read_excel(file_path)
data['Combined_Text'] = data.apply(lambda row: f"{row['Backstory']} {row['User Location']}", axis=1)
provider_embeddings = model.encode(data['Combined_Text'].tolist(), convert_to_tensor=True)

# Clean text helper
def clean_text(text):
    return ''.join(e for e in str(text) if e.isalnum() or e.isspace())

# Speak response using edge TTS
async def speak(text):
    tts = edge_tts.Communicate(text, "en-US-JennyNeural")
    filename = "response.mp3"
    await tts.save(filename)
    audio = AudioSegment.from_file(filename, format="mp3")
    play(audio)

# PI-related sample prompts
pi_samples= [
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
     # Cheating Spouse
    "Can you help me find out if my boyfriend is cheating when he says he's working late?",
    "I need help identifying a woman who's been messaging my husband anonymously.",
    "My wife has been very secretive lately and I suspect she's seeing someone else.",
    "I want to know who my husband is meeting during his 'business trips'.",
    "Can you follow my partner this weekend to see if they're being faithful?",

    # Missing Person / Alive & Well Check
    "My sister hasn't answered her phone in a week and I'm getting worried—can you check on her?",
    "I’m trying to locate an old friend who suddenly disappeared from social media and work.",
    "Can you find my dad? He left without telling anyone and we haven't heard from him since.",
    "I need to know if my elderly uncle is okay—he’s not answering calls and lives alone.",
    "We’ve lost contact with a relative after a family dispute. Can you help reconnect us?",

    # Child Custody
    "I believe my ex is leaving our kids unsupervised during his custody weekends.",
    "Can you monitor my child’s mother to see if she’s living with someone dangerous?",
    "I want proof that my ex is exposing our children to unsafe environments.",
    "I’m worried that my daughter isn’t being taken care of properly by her father—can you help?",
    "Can you help me prove that my ex is violating the custody agreement?",

    # Background Checks
    "I’m hiring a new nanny and want to make sure she has no criminal history.",
    "Can you run a background check on someone I met online before I meet them?",
    "We’re considering a new investor, and I want to verify his financial and legal background.",
    "I want to know if my daughter's boyfriend has a criminal past.",
    "Before we rent out our property, can you check if the applicant has a history of evictions or fraud?",

    # Disability/FMLA Abuse Investigations
    "We think an employee on FMLA is working another job while on leave—can you investigate?",
    "One of our workers claims a back injury but is posting gym videos online—can you verify this?",
    "Can you check if an employee collecting disability is actually injured?",
    "Someone at our company is suspected of faking migraines for time off—can you help prove it?",
    "I need evidence that my co-worker is abusing the FMLA system for extended vacations.",

    # Criminal Investigations / Defense
    "My brother is facing assault charges but we believe it's a case of mistaken identity—can you help?",
    "Can you investigate to find witnesses that might help clear my name?",
    "We suspect the police overlooked evidence in our case—can you re-investigate the scene?",
    "I need someone to review the case files and uncover anything that can support my defense.",
    "My cousin is being framed, and we need help gathering proof of his innocence.",

    # Asset Checks
    "I believe my ex-husband is hiding assets during our divorce—can you find out?",
    "Can you run an asset search on someone I’m about to sue?",
    "I need to know if the person who owes me money has anything worth going after.",
    "We’re trying to collect a judgment—can you locate their hidden assets?",
    "Before I file for divorce, I want to know what assets my wife may be concealing.",

    # Surveillance
    "I want someone to watch my tenant and confirm if they’re illegally subletting the property.",
    "Can you conduct surveillance on my neighbor? I suspect drug activity.",
    "I need footage of my ex violating our restraining order.",
    "Can you monitor an employee who might be leaking information to competitors?",
    "Someone’s been trespassing on my property at night—can you set up covert surveillance?",

    # Skip Trace / Locate
    "A former tenant disappeared owing several months’ rent—can you track them down?",
    "I’m trying to find my biological father but only have his first name and a city.",
    "Can you help locate someone who owes me money and moved out of state?",
    "I want to reconnect with a childhood friend who vanished after high school.",
    "I need to serve legal papers but the person has moved and left no forwarding address.",
    # Accident Reconstruction (ACC)
    "Can you recreate the scene of my car accident to prove it wasn’t my fault?",
    "We need a reconstruction expert to determine who caused the collision on the highway.",
    "Can you analyze dashcam and road footage to support my insurance claim from the crash?",
    "My insurance company is denying fault—can you review the accident scene and give us proof?",
    "I need help proving the other driver ran the light before hitting me.",

    # Arson (ARS)
    "My business burned down and I suspect it was intentional—can you investigate?",
    "Can you determine if my neighbor’s recent fire was arson?",
    "The fire marshal ruled our house fire accidental, but I think it was set on purpose.",
    "I suspect someone is targeting us—can you check if there’s a pattern behind recent fires?",
    "Our competitor had a suspicious fire at their warehouse—can you dig into it?",

    # Asset Checks (AST)
    "I’m about to sue someone and want to know if they have assets worth pursuing.",
    "Can you check if my former partner moved money into hidden accounts?",
    "I want to verify if the person I’m dating actually owns the businesses they claim to.",
    "We need to confirm asset holdings before settling a business dispute.",
    "Can you find offshore or hidden accounts tied to a fraudulent investment scheme?",

    # Audio/Video Enhancement (AVE)
    "Can you enhance this blurry security video to identify a license plate?",
    "We have a recording from a nanny cam—can you clean up the audio for court?",
    "I need help making a phone recording clearer for legal purposes.",
    "Can you enhance this doorbell camera footage to identify the person in it?",
    "Is it possible to extract clearer audio from a noisy surveillance recording?",

    # Computer Forensics (CRF)
    "Can you recover deleted emails from my employee’s work laptop?",
    "My ex accessed my private photos—can you prove it using computer forensics?",
    "We suspect data theft by a former IT worker—can you analyze the devices involved?",
    "Can you check a USB drive for deleted files linked to insider trading?",
    "I need forensic evidence from a computer that may have been used for cyberbullying.",

    # Internet/OSINT/Social Media (IOS)
    "Can you find out who’s behind this anonymous Instagram account harassing me?",
    "I need to investigate a potential catfish before I get too involved.",
    "Can you help me find someone’s real identity through their online activity?",
    "Someone is posting private info about me online—can you trace their IP?",
    "We want to verify a suspect’s timeline using their social media posts.",

    # Fraud (FRD)
    "My mother was scammed out of her savings—can you trace the person behind it?",
    "We suspect one of our employees is manipulating invoices—can you investigate?",
    "Can you find out who used my credit card for unauthorized purchases?",
    "I invested in a company that vanished overnight—can you find out what really happened?",
    "My insurance claim was denied for fraud—can you help prove my case is legitimate?",

    # Insurance Investigations (INS)
    "Can you verify if this injury claim is legitimate before we approve it?",
    "A policyholder claimed their car was stolen, but we suspect they staged it.",
    "Can you investigate a suspicious home burglary claim that seems exaggerated?",
    "We believe someone is inflating damages from a minor accident—can you confirm?",
    "I think someone faked an injury on our property—can you gather proof?",

    # Workplace Investigations (WIN)
    "We received a tip about drug use among warehouse staff—can you verify discreetly?",
    "Someone is leaking company secrets—can you identify the source?",
    "There’s a harassment complaint in our office—we need an outside investigator.",
    "I want to confirm if our night shift supervisor is actually showing up to work.",
    "An employee is suspected of stealing—can you monitor their activity without alerting them?",

    # Adoption (ADP)
    "Can you help me find my birth parents? I was adopted in the 80s.",
    "I’m trying to locate my biological daughter who was adopted years ago.",
    "Can you trace the adoption records to identify my real family?",
    "I have non-identifying info from the agency—can you use that to find my sibling?",
    "We suspect illegal adoption procedures in a family member’s case—can you investigate?",

    # Civil Rights (CVL)
    "I believe my rights were violated during a protest—can you gather video and witness evidence?",
    "Can you help me investigate police misconduct during my arrest?",
    "We want to prove a pattern of racial discrimination at my workplace.",
    "I was profiled at a store and have partial video—can you help document the incident?",
    "My son was harassed by school staff—can you collect independent proof for legal use?",

    # Malpractice Medical/Legal (MAL)
    "We think a doctor misdiagnosed my mother—can you review the case?",
    "I need help proving a surgeon’s negligence caused permanent damage.",
    "Can you investigate an attorney who mishandled my case on purpose?",
    "A dentist gave my child a procedure we never approved—can you help us document it?",
    "We suspect that multiple patients have suffered under the same physician—can you confirm patterns?",

    # Intellectual Property (IPR)
    "Can you find out who’s copying my brand’s logo and selling fake products?",
    "We need to investigate a competitor using our patented software illegally.",
    "Someone is selling bootleg versions of our product online—can you track them down?",
    "Can you help me prove that my copyrighted content was stolen for a major ad campaign?",
    "I need to know who leaked our internal designs before our product launch.",

    # Personal Injury (PIN)
    "Can you verify the truth behind a slip and fall claim at our store?",
    "I was hit by a delivery truck and need independent evidence for my injury claim.",
    "My neighbor’s dog attacked me—can you collect witness statements and surveillance?",
    "Someone staged a car accident to scam me—can you help prove it?",
    "We need photos and interviews to support a premises liability case.",

    # Skip Trace (SKT)
    "I need to locate a tenant who fled without paying rent.",
    "Can you track down a witness who moved and didn’t leave any contact info?",
    "I’m trying to find someone who owes me money and has disappeared.",
    "A former business partner vanished with company funds—can you help locate them?",
    "Can you help find my estranged father? We haven’t heard from him in years.",

    # Surveillance (SUR)
    "I want surveillance on a business partner I suspect is breaching our contract.",
    "Can you follow my ex to see if they’re violating our custody agreement?",
    "We think an employee is faking injuries—can you monitor them discreetly?",
    "My elderly mother’s caregiver might be mistreating her—can you gather video proof?",
    "Can you do overnight surveillance on a vacant property that’s been vandalized recently?",

    # Social Media Investigation (SMI)
    "Can you track down someone spreading false info about me online?",
    "We need to know if a job candidate has posted anything inappropriate on social media.",
    "I think my partner is using a fake profile—can you find out who it really is?",
    "Can you collect evidence from Facebook posts for a harassment case?",
    "An anonymous account is threatening our business—can you trace it?",

    # Maritime and Cargo Handling (MRT)
    "Can you investigate who’s been stealing shipments from our port facility?",
    "We had a cargo container go missing—can you help trace it?",
    "I suspect smuggling activity on one of our boats—can you find proof?",
    "There was damage to our cargo during shipping—can you find out who’s responsible?",
    "Can you look into suspected sabotage involving one of our dock employees?",

    # Oil Field Accidents (OIL)
    "Can you investigate a recent oil rig explosion and determine fault?",
    "We believe safety violations led to my husband’s injury at an oil field—can you verify?",
    "Can you collect evidence to prove a contractor’s negligence on the rig?",
    "There was a chemical spill at the drilling site—can you document compliance failures?",
    "I need help investigating labor abuse at a remote oil operation.",

    # Children’s Rights/Abuse (CHL)
    "Can you help document neglect at a foster home where my niece stays?",
    "We suspect our daughter’s school is failing to report bullying—can you investigate?",
    "I believe my child’s other parent is exposing them to unsafe conditions—can you verify?",
    "My son came home with bruises and won’t talk—can you discreetly look into it?",
    "Can you monitor a daycare we’re concerned about for potential mistreatment?",

    # Due Diligence (DDI)
    "I need a background check on someone I'm about to go into business with.",
    "Can you look into the financials of a company we’re considering acquiring?",
    "I’m considering investing in a startup—can you verify the founders and legal filings?",
    "Can you check if this potential supplier has any history of fraud or lawsuits?",
    "We want to confirm an executive’s credentials before hiring—can you verify everything?",

    # Polygraph and PSE (POL)
    "Can you administer a lie detector test to an employee suspected of theft?",
    "My partner agreed to a polygraph—can you arrange and conduct it?",
    "We want to use voice stress analysis on a witness who keeps changing their story.",
    "Is it possible to verify the truthfulness of a harassment claim with a polygraph?",
    "Can you conduct a polygraph on a nanny we suspect of hiding something?",


     # Expert Witness (EXW)
    "Can you provide an expert witness to testify in a civil fraud case?",
    "We need a digital forensics expert to testify about tampered emails.",
    "Can someone testify about industry standards in our construction dispute?",
    "Our attorney is looking for a PI who can serve as an expert in surveillance protocols.",
    "Do you have a forensic accounting expert who can testify about embezzlement?",

    # Fraud (FRD)
    "I suspect a former employee is committing workers' comp fraud—can you investigate?",
    "Can you look into a business partner who may be falsifying revenue reports?",
    "My insurance claim was denied—can you prove fraud on the insurer’s part?",
    "We think someone is using our company’s name to run a scam online—can you trace them?",
    "I think my elderly parent is being financially exploited—can you look into it?",

    # Arson (ARS)
    "Our warehouse caught fire under suspicious circumstances—can you investigate for arson?",
    "Can you determine if our tenant set the fire intentionally to claim insurance?",
    "The police ruled our fire accidental, but we suspect foul play—can you help?",
    "My neighbor's garage fire spread to our home—can you check if it was intentional?",
    "We had a series of small fires at our site—can you figure out who might be behind it?",

    # Financial (FIN)
    "I need help tracking suspicious wire transfers in our company account.",
    "Can you investigate whether our CFO is hiding assets or cooking the books?",
    "My business partner suddenly disappeared with investor money—can you find out what happened?",
    "Can you verify if a financial advisor has a history of fraud or regulatory actions?",
    "My ex is claiming bankruptcy to avoid payments—can you uncover hidden financials?",

    # Equine Injuries (EQU)
    "My horse was injured during boarding—can you find out who’s responsible?",
    "We suspect sabotage during a recent horse show—can you investigate?",
    "Can you gather evidence on a trainer who’s been mistreating horses?",
    "There was an injury at our stable—can you determine if negligence was involved?",
    "Our horse was sold under false pretenses—can you track down the seller?"
]

pi_sample_embeddings = model.encode(pi_samples, convert_to_tensor=True)

# Enhanced semantic guardrail check
def is_pi_related_semantic(text):
    user_embedding = model.encode([text], convert_to_tensor=True)
    similarity_scores = util.pytorch_cos_sim(user_embedding, pi_sample_embeddings).cpu().numpy().flatten()
    return np.max(similarity_scores) > 0.4  # threshold can be adjusted



pi_context_samples = [
    # Existing examples
    ("My dog ran away and I can’t find her anywhere.",                  "pet recovery"),
    ("I suspect my spouse of cheating after seeing strange messages.",   "infidelity"),
    ("My sister has been missing for two days and no one knows her whereabouts.", "missing person"),
    ("I want to verify someone's criminal record before hiring them.",    "background check"),
    ("I believe someone is hacking into my email account.",              "cyber investigation"),
    ("My car was hit in a hit-and-run and the driver vanished.",          "accident reconstruction"),
    ("Our warehouse was set ablaze under suspicious circumstances.",     "arson investigation"),
    ("I think a coworker is abusing FMLA while working another job.",     "leave abuse"),
    ("Someone forged my signature on a legal document.",                 "document forgery"),
    ("I need to find hidden assets for a divorce case.",                 "asset tracing"),

    # New realistic backstories
    ("Lisa has had her car scratched four times in the past two months. She suspects a jealous coworker and needs proof before confronting them.", 
     "vandalism investigation"),
    ("Margaret has spent years searching for her younger brother who vanished in the 1970s, with only a torn postcard address as a clue.", 
     "cold case missing person"),
    ("Every night at 2 AM Jason hears soft footsteps above him; his landlord insists no one lives there. He needs someone to check it out.", 
     "unauthorized entry investigation"),
    ("A woman is worried her ex’s new girlfriend may be a con artist and wants to know if her child is safe during visits.", 
     "child safety / infidelity"),
    ("I need to find out what my husband is doing at night; I suspect he’s hiding something.", 
     "infidelity surveillance"),
    ("Look into a decades-old murder case and uncover evidence long buried.", 
     "homicide cold case"),
    ("Investigate a potential whistleblower inside my company who may expose fraud.", 
     "whistleblower investigation"),
    ("A Victorian home we moved into has whispers in the halls and objects moving—find a logical explanation or prove the supernatural.", 
     "paranormal investigation"),
    ("Find out if my friend’s inheritance claim is real before they sign the papers.", 
     "inheritance verification"),
    ("Investigate corruption at my workplace and gather evidence of misconduct.", 
     "corporate fraud investigation"),
    ("Investigate strange behavior by my spouse and determine if they’re hiding anything.", 
     "infidelity / surveillance"),
    ("Help me track down a street artist who spray-painted my storefront.", 
     "vandalism / graffiti investigation"),
    ("A man discovered a hidden bank account under his wife's name and needs to know why.", 
     "financial investigation"),
    ("Look into my favorite celebrity’s private life for a magazine exposé.", 
     "celebrity background check"),
    ("Investigate my father’s wartime past and verify details from old records.", 
     "historical / genealogical investigation"),
    ("Find out who left flowers on my porch several nights in a row.", 
     "anonymous gift investigation"),
    ("A rare 18th-century watch was stolen at an exclusive event—recover it before it’s lost forever.", 
     "theft / high-value recovery"),
    ("Discover if my father had a secret family I never knew about.", 
     "genealogy / secret family"),
    ("Locate the owner of an abandoned house I inherited through a will.", 
     "property ownership investigation"),
    ("Investigate who has been leaving threatening notes on my door.", 
     "harassment / stalking"),
    ("Help me locate my missing manuscript that disappeared from my studio.", 
     "asset recovery"),
    ("Help me track an heirloom stolen 30 years ago from my grandparents’ home.", 
     "cold case theft recovery"),
    ("Confirm if my new online boyfriend is a real person or a catfish.", 
     "online identity verification"),
    ("Find out what happened to my stolen bike and who took it.", 
     "vehicle theft investigation"),
    ("A fan believes they saw a wedding band on a star’s finger despite their single status—prove it.", 
     "celebrity infidelity investigation"),
    ("Locate the sender of anonymous threats sent via mail and email.", 
     "threat source identification"),
    ("Help me track an estranged sibling who disappeared after our parents’ divorce.", 
     "family reunification"),
    ("I’m being sued for fraud but I insist I’ve been framed—find the real perpetrator.", 
     "legal defense investigation"),
    ("Find the source of a local conspiracy theory affecting my neighborhood.", 
     "community / conspiracy investigation"),
    ("Unmask the person sending me anonymous social media threats referencing personal details.", 
     "cyber harassment investigation"),
    ("My rival’s new fashion line looks exactly like mine—gather proof of design theft.", 
     "intellectual property investigation"),
    ("Check if my new house has a shady past including liens, crimes, or code violations.", 
     "real estate due diligence"),
    ("Investigate a stalker at my college who’s been following me and sending notes.", 
     "campus stalking investigation"),
    ("Reconnect me with a best friend from high school using only a yearbook photo and old address.", 
     "missing friend search"),
    ("Book suspicious last-minute hotel rooms for my boss—confirm if it’s an affair or business.", 
     "infidelity surveillance"),
    ("Confirm whether the twin hinted at in my adoption records actually exists.", 
     "adoption / sibling search"),
    ("Investigate who keeps sabotaging my car with strange dents and scratches.", 
     "vandalism investigation"),
    ("Check if a local politician is corrupt using public records and leaks.", 
     "political corruption investigation"),
    ("Investigate a series of strange disappearances in my area over the past month.", 
     "pattern disappearance investigation"),
    ("Find out if my coworker is sabotaging my work by deleting important files.", 
     "workplace sabotage investigation"),
    ("My startup’s mysterious investor vanished overnight—find out if it was a scam.", 
     "investment fraud investigation"),
    ("Help identify my stalker who’s leaving creepy gifts at my door.", 
     "stalking investigation"),
    ("Look into whether my employee has a hidden criminal record.", 
     "employment background check"),
    ("Uncover an old family secret hinted at in my grandmother’s diary.", 
     "family history investigation"),
    ("Investigate my brother’s new girlfriend—she seems to have a hidden agenda.", 
     "infidelity / personal investigation"),
    ("Find out why my emails are being hacked and who’s behind it.", 
     "cyber intrusion investigation"),
    ("Discover who hacked my business network and stole customer data.", 
     "corporate cyber forensics"),
    ("Investigate a fake charity scam targeting elderly donors in my town.", 
     "charity fraud investigation"),
    ("Packages marked 'delivered' keep disappearing from my doorstep—catch the thief.", 
     "package theft investigation"),
    ("Investigate a suspicious tenant who never signed a lease but lives in my property.", 
     "tenant fraud investigation"),
    ("Prove my landlord is breaking housing laws by refusing repairs and charging illegal fees.", 
     "tenant rights investigation"),
    ("Find my biological parents using only limited adoption documents.", 
     "genealogical investigation"),
    ("My golden retriever went missing three days ago after a suspicious van was seen—help track Max.", 
     "pet recovery"),
    ("Help me prove my neighbor is dealing illegal substances on our street.", 
     "narcotics surveillance investigation"),
    ("Investigate my landlord’s criminal past before signing a new lease.", 
     "tenant background check"),

      # Aviation Accident Investigation
    ("A small private plane I was onboard experienced engine failure; I need to determine cause and liability.", 
     "aviation accident investigation"),

    # Maritime & Boating Accidents
    ("My yacht was rammed in the marina and the other vessel fled the scene; I need to recover it and find the culprit.", 
     "maritime accident investigation"),

    # Cell Phone Forensics
    ("I lost crucial messages when my phone was wiped remotely; I need those deleted texts recovered.", 
     "cell phone forensics"),

    # Computer Crimes / Cyber Investigation
    ("Our company server was breached and client data stolen; we need to identify the hacker.", 
     "computer crimes investigation"),

    # Construction Site Investigation
    ("Workers on my building site may be cutting corners and using substandard materials; I need proof.", 
     "construction site investigation"),

    # Corporate Investigations
    ("I suspect a senior executive of embezzling funds through shell companies; gather evidence.", 
     "corporate investigation"),

    # Criminal Defense Investigation
    ("I’ve been falsely accused of burglary; find alibi witnesses and analyze the scene for inconsistencies.", 
     "criminal defense investigation"),

    # Domestic Dispute Investigation
    ("My roommate has threatened me multiple times; I need video proof to obtain a restraining order.", 
     "domestic dispute investigation"),

    # Dram Shop / Liquor Liability
    ("A drunk patron assaulted me after leaving that bar—investigate whether they were over-served.", 
     "dram shop investigation"),

    # Drunk Driving Defense
    ("I was arrested for DUI but the breathalyzer seemed faulty; review logs and calibration records.", 
     "drunk driving defense"),

    # Due Diligence
    ("We are acquiring a supplier overseas; verify their corporate filings, lawsuits, and reputation.", 
     "due diligence"),

    # Electronic Surveillance
    ("I want covert cameras installed to monitor theft at my retail store; ensure it’s legally sound.", 
     "electronic surveillance"),

    # Executive Protection
    ("I’m a high-net-worth individual attending a public gala; arrange for a discreet bodyguard.", 
     "executive protection"),

    # Equine Injury Investigation
    ("My champion horse was injured in the stable under mysterious circumstances; determine negligence.", 
     "equine injury investigation"),

    # Expert Witness Services
    ("Our legal team needs a ballistics expert to testify about gunshot trajectories in our trial.", 
     "expert witness"),

    # Forensic Photography & Enhancement
    ("Enhance these low-light security camera images so we can read the license plate.", 
     "forensic photography"),

    # Genealogy / Heir Search
    ("My grandmother died intestate; I need to locate all potential heirs for probate.", 
     "heir search"),

    # Insurance Fraud Investigation
    ("A claimant reports a broken wrist but was spotted playing tennis; verify legitimacy.", 
     "insurance fraud investigation"),

    # Judgment Enforcement & Asset Recovery
    ("We won a lawsuit but can’t collect; locate hidden assets to satisfy the judgment.", 
     "judgment enforcement"),

    # Medical Malpractice Investigation
    ("My father suffered complications in surgery; gather hospital records and expert opinions.", 
     "medical malpractice investigation"),

    # Maritime Cargo Handling Investigation
    ("Multiple cargo containers vanished from our port; trace their last known locations.", 
     "maritime cargo investigation"),

    # Oil Field Accident Investigation
    ("There was an explosion on the rig that injured several workers; determine cause and fault.", 
     "oil field accident investigation"),

    # Personal Process Service / Locate for Service
    ("I need to serve divorce papers to someone who moved but left no forwarding address; find them.", 
     "process service location"),

    # Products Liability Investigation
    ("My power tool exploded and injured me; collect evidence on design defects.", 
     "product liability investigation"),

    # Railroad / FELA Investigation
    ("A coworker was critically injured on the railroad tracks; gather evidence for a FELA claim.", 
     "railroad accident investigation"),

    # Records Retrieval
    ("I need sealed juvenile court records from 2003 for a sensitive legal matter.", 
     "records retrieval"),

    # Real Estate Fraud Investigation
    ("I suspect the deed on my inherited property was forged; verify the chain of title.", 
     "real estate fraud investigation"),

    # Social Media / OSINT Investigation
    ("An anonymous account is defaming me on Twitter; trace it back to the real person.", 
     "social media investigation"),

    # Standards Research & Compliance
    ("Confirm that our factory’s chemical handling meets all federal safety regulations.", 
     "standards compliance investigation"),

    # Trial Preparation & Exhibits
    ("We need photo, video, and witness statements organized into exhibits for trial.", 
     "trial preparation"),

    # Workplace / Internal Investigations
    ("Several employees report harassment by HR; investigate and document all incidents.", 
     "workplace investigation"),

     # Kidnapping / Abduction Investigation
    ("My niece was taken by her non-custodial parent without warning; please locate her safely.", 
     "kidnapping / abduction investigation"),

    # Human Trafficking / Exploitation
    ("I suspect workers at a local factory are being trafficked; need proof to alert authorities.", 
     "human trafficking investigation"),

    # Environmental / Pollution Violations
    ("A nearby plant is dumping toxic waste into the river; gather evidence of illegal dumping.", 
     "environmental violation investigation"),

    # Counterfeit Goods / Anti‐Counterfeiting
    ("My designer handbags are being copied and sold as fakes; find the source.", 
     "counterfeit goods investigation"),

    # Lease / Rental Fraud
    ("The landlord claims I broke the lease but evidence suggests fraud; investigate the lease documents.", 
     "lease fraud investigation"),

    # Drone / Aerial Surveillance
    ("There’s illegal construction on protected land; perform drone surveillance to document it.", 
     "drone surveillance investigation"),

    # Digital Deepfake Detection
    ("Someone posted a deepfake video of me online; verify its authenticity and track the source.", 
     "deepfake detection investigation"),

    # Phone Forensics / Data Extraction
    ("My phone was destroyed in an accident; I need to recover deleted contacts and messages.", 
     "mobile forensics investigation"),

    # Drone / UAV Asset Recovery
    ("A high-value drone was stolen from our research facility; locate and recover it.", 
     "drone theft investigation"),

    # Threat Assessment / Security Risk
    ("I received an anonymous email threatening me; assess the risk and identify the sender.", 
     "threat assessment investigation"),

    # Executory Compliance / Contract Enforcement
    ("My business partner isn’t fulfilling contract obligations; gather proof of breach.", 
     "contract enforcement investigation"),

    # Intellectual Property / Trade Secret Theft
    ("My proprietary formula was leaked to a competitor; trace how it was stolen.", 
     "trade secret theft investigation"),

    # Undercover / Entrapment Operations
    ("I need someone to pose as a buyer to expose a black-market dealer in stolen art.", 
     "undercover operation investigation"),

    # Clandestine Stakeout
    ("We believe a suspect meets nightly at a remote location; conduct a covert stakeout.", 
     "stakeout surveillance"),

    # Asset Freezing & Recovery
    ("A verdict was won but assets moved offshore; trace and freeze their accounts.", 
     "asset recovery investigation"),

     # Forensic Accounting Investigation
    ("Our CFO’s books don't add up; I need detailed financial tracing to uncover embezzlement.", 
     "forensic accounting investigation"),
    
    # DNA / Paternity / Kinship Testing Support
    ("I want to confirm paternity using DNA evidence; coordinate discreet sample collection and analysis.", 
     "paternity/DNA testing investigation"),
    
    # VIP / Celebrity Protection Assessment
    ("A high-profile client is receiving threat emails; assess security risks and recommend protection measures.", 
     "VIP protection assessment"),
    
    # Food / Product Contamination Investigation
    ("Customers fell ill after eating at my restaurant; investigate potential food contamination sources.", 
     "food safety investigation"),
    
    # Pharmaceutical / Clinical Trial Fraud
    ("I suspect data manipulation in a drug trial; gather evidence of clinical fraud before FDA review.", 
     "clinical trial fraud investigation"),
    
    # Industrial Espionage / Trade Secret
    ("Our competitor hired former employees then launched our secret design; trace how the data was leaked.", 
     "industrial espionage investigation"),
    
    # Health Insurance / Medicaid Fraud
    ("A provider is billing for services never rendered; verify patient records to prove insurance fraud.", 
     "health insurance fraud investigation"),
    
    # Manufacturing Defect / Product Liability
    ("My machine exploded due to an apparent defect; document evidence to support a liability claim.", 
     "product liability investigation"),
    
    # Vehicle Defect / Automotive Recall Support
    ("Multiple brake failures in my model car; investigate if there’s a systemic manufacturing defect.", 
     "automotive defect investigation"),
    
    # Political Campaign Opposition Research
    ("We need background and public record digging on an opponent before the next election.", 
     "political opposition research"),
    
    # Workplace Drug Testing & Compliance
    ("I suspect on-site drug use among employees; coordinate covert testing and evidence gathering.", 
     "workplace drug investigation"),
    
    # Child Exploitation / Abuse Reporting
    ("I have evidence my child’s coach is grooming minors; verify and document before alerting authorities.", 
     "child exploitation investigation"),
    
    # Regulatory / Compliance Audit Support
    ("We need to ensure our factory meets all EPA regulations; audit and document any violations.", 
     "regulatory compliance investigation"),
    
    # Identity Theft Recovery & Resolution
    ("My identity was stolen and used for loans; trace fraudulent accounts and clear my records.", 
     "identity theft resolution"),
    
    # Anonymous Source Verification
    ("An anonymous informant provided tips on corruption; verify their credibility before acting.", 
     "source verification investigation"),

     # Art & Cultural Heritage Investigations
    ("Several valuable paintings disappeared from my gallery; track down the thief and recover the art.", 
     "art theft investigation"),
    ("I inherited an old manuscript that may be a forgery; verify its authenticity and provenance.", 
     "forensic manuscript verification"),

    # Jewelry & High-Value Asset Recovery
    ("My wedding ring slipped off during a beach walk and got buried in the sand; help me locate it.", 
     "lost jewelry recovery"),
    ("A diamond necklace was stolen from my safe; I need it traced and recovered.", 
     "high-value asset recovery"),

    # Cross-Border & International Cases
    ("I invested overseas and suspect the local partner hid profits; trace the flow of funds across borders.", 
     "cross-border financial investigation"),
    ("A child was taken by their other parent to another country; locate and bring them home safely.", 
     "international child abduction investigation"),

    # Threat Intelligence & Security Assessments
    ("I received extremist propaganda targeting me online; assess the threat and identify the source.", 
     "threat intelligence investigation"),
    ("A high-profile executive is attending a summit abroad; perform a security risk assessment.", 
     "executive security assessment"),

    # OSINT & Adverse Media Screening
    ("Before we hire a new C-suite executive, run an adverse media check on their public footprint.", 
     "adverse media screening"),
    ("We need open-source intel on this activist group’s leadership structure.", 
     "OSINT leadership profiling"),

    # Vendor & Partner Due Diligence
    ("We’re onboarding a supplier in a high-risk region; vet their ownership and legal history.", 
     "vendor due diligence"),
    ("Can you confirm if this NGO partner has any undisclosed legal liabilities?", 
     "NGO compliance investigation"),

    # IT & Cybersecurity Investigations
    ("Our email server was hacked and confidential files leaked; identify the breach origin.", 
     "cyber breach investigation"),
    ("I believe someone planted malware in our network; perform a digital forensics deep scan.", 
     "network forensics investigation"),

    # Regulatory & Compliance Audits
    ("My medical clinic may be billing for services not provided; investigate insurance compliance.", 
     "healthcare compliance investigation"),
    ("We need to confirm GDPR adherence for our European customer data practices.", 
     "data privacy compliance investigation"),

    # Maritime & Vessel Matters
    ("A derelict yacht is anchored off our coast illegally; find its owner and arrange removal.", 
     "vessel abandonment investigation"),
    ("Cargo was jettisoned from our freighter mid-voyage; recover the lost shipment.", 
     "maritime cargo salvage"),

    # Wildlife & Environmental Crimes
    ("There’s illegal poaching on my ranch; identify the poachers and gather evidence.", 
     "wildlife poaching investigation"),
    ("A factory upstream is discharging toxic waste into our river; document and report violations.", 
     "environmental crime investigation"),

    # Personal & Family Security
    ("I’m about to marry someone I met online; conduct a thorough background and character check.", 
     "pre-marital background check"),
    ("My child’s tutor is acting suspiciously; verify their credentials and history.", 
     "child tutor vetting"),

    # Miscellaneous Specialty Cases
    ("An investor’s office was bugged with hidden microphones; locate and remove the devices.", 
     "technical surveillance counter-measures"),
    ("I’m organizing a corporate retreat; need covert agents to test our venue’s security.", 
     "red team security assessment"),

     ("Find the owner of a lost diary and uncover the story behind its missing pages.", 
     "lost item recovery"),
    ("Kyle was tricked into investing in a startup that vanished overnight; he wants justice and his money back.", 
     "investment fraud investigation"),
    ("An independent journalist believes they’ve found a government cover-up and needs to verify the facts before publishing.", 
     "investigative journalism support"),
    ("Someone opened multiple credit cards in Nathan’s name; the police were unhelpful—he needs an investigator to track the identity thief.", 
     "identity theft investigation"),
    ("Find the source of strange prank calls that have been harassing me for weeks.", 
     "harassment / prank call investigation"),
    ("A high-end restaurant owner suspects a senior staff member is pocketing cash and disappearing inventory.", 
     "internal theft investigation"),
    ("An author received an ominous letter warning them to stop writing their book; they want to know who sent it and why.", 
     "threat investigation"),
    ("Find out who is framing me at work and uncover the evidence before I lose my job.", 
     "false accusation investigation"),
    ("Investigate whether my house is built on a burial ground before we renovate.", 
     "property history investigation"),
    ("Investigate a powerful businessman’s shady deals and gather proof of corruption.", 
     "white collar investigation"),
    ("Investigate a rumored cult in my town and determine if there’s any real activity.", 
     "cult investigation"),
    ("Look into my wife’s hidden bank accounts and uncover where the money is going.", 
     "asset tracing investigation"),
    ("Look into my therapist’s questionable practices and verify their credentials.", 
     "professional misconduct investigation"),
    ("Find out what happened to my childhood home and why it was suddenly sold.", 
     "real estate history investigation"),
    ("Since moving into my new apartment, I feel watched and my Wi-Fi shows unknown devices—find out why.", 
     "surveillance detection"),
    ("Investigate a hidden room in my house and discover what’s behind the locked door.", 
     "hidden room investigation"),
    ("Prove my spouse’s infidelity to strengthen my custody case.", 
     "custody / infidelity investigation"),
    ("Investigate a possible blackmail attempt after receiving threatening demands.", 
     "blackmail investigation"),
    ("Find out why my packages keep being stolen from my doorstep.", 
     "package theft investigation"),
    ("Sarah found a hidden camera in her smoke detector—she needs proof to take legal action.", 
     "illegal surveillance detection"),
    ("A grieving daughter suspects her father’s sudden ‘heart attack’ wasn’t natural; she needs answers.", 
     "suspicious death investigation"),
    ("A local politician is targeted by a fake smear campaign online—identify who’s behind it.", 
     "reputation management investigation"),
    ("Look into my family’s missing fortune and recover any hidden inheritance.", 
     "inheritance investigation"),
    ("Help me reconnect with my long-lost twin using only an old adoption record.", 
     "family reunion investigation"),
    ("Sarah believes someone is entering their shared apartment when no one’s home—nothing is missing but things are moved.", 
     "unauthorized entry investigation"),
    ("Emily’s father disappeared during a road trip a month ago; his phone is off and credit card shows last use at a remote gas station.", 
     "missing person investigation"),
    ("My husband, a schoolteacher, claims late-night work; odd perfume smells suggest otherwise—confirm the truth.", 
     "infidelity investigation"),
    ("Maya’s new boyfriend won’t talk about his past; her brother wants to ensure she isn’t dating a criminal.", 
     "personal background investigation"),
    ("A startup CEO suspects a mole after confidential documents leaked to competitors.", 
     "corporate espionage investigation"),
    ("Her 16-year-old left home after an argument and hasn’t returned; she fears he’s in danger.", 
     "runaway teen investigation"),
    ("Before accepting funding, Amit wants a background check on a flashy investor who seems too good to be true.", 
     "investment due diligence"),
    ("Her ex-boyfriend suddenly moved to a posh apartment and travels frequently; she suspects something illegal.", 
     "suspicious behavior investigation"),
    ("A strange chemical smell and locked doors raise suspicions of illegal activity in the rented property.", 
     "illegal activity investigation"),
    ("Every night at 3 AM, loud knocks disturb her sleep—she lives alone and feels unsafe, needing clarity.", 
     "disturbance investigation"),
    ("A teen feels followed from school and has received anonymous notes; she needs proof.", 
     "stalking investigation"),
    ("Strange visitors and late-night noises suggest a drug den; confirm if it’s illegal activity.", 
     "drug activity investigation"),
    ("He’s worried his kids are unsafe around his new boyfriend, who has no visible job or stable behavior.", 
     "child safety investigation"),
    ("Someone spread false rumors that cost him his job—he wants evidence to clear his name.", 
     "defamation investigation"),
    ("Days before the wedding, she learns of inconsistencies in his stories; she wants to ensure he isn’t hiding a secret family.", 
     "pre-marital investigation"),
    ("A remote construction site is missing materials and vandalized; the developer suspects squatters or internal theft.", 
     "construction theft investigation"),
    ("A social media star is receiving threats and creepy messages from someone she knows.", 
     "cyber harassment investigation"),
    ("His 14-year-old daughter has a secret online boyfriend; he fears she’s being groomed by an adult.", 
     "child exploitation investigation"),
    ("Devices in her home behave oddly—she suspects her partner installed surveillance equipment without consent.", 
     "technical surveillance investigation"),
    ("A local historian believes deaths in the 1960s were covered up—he wants modern evidence.", 
     "historical investigation"),
    ("A rare 18th-century watch disappeared at a private party—no sign of break-in; recover the stolen item.", 
     "high-value theft recovery"),
    ("A journalist investigating corruption is now anonymously targeted—determine if it’s retaliation or a hoax.", 
     "retaliation investigation"),

      # Academic Credential Verification
    ("I’m hiring a tutor for my child and need to verify their claimed PhD in physics.", 
     "academic credential verification"),

    # Cryptocurrency Fraud Investigation
    ("I invested in a crypto ICO that disappeared; I need to trace where the funds went.", 
     "crypto fraud investigation"),

    # Wildfire Origin Investigation
    ("A wildfire started near my property under suspicious circumstances; determine the cause.", 
     "wildfire cause investigation"),

    # Medical Device Failure Inquiry
    ("My implanted device malfunctioned and injured me; gather evidence of manufacturing defects.", 
     "medical device failure investigation"),

    # Philanthropic Fund Misuse
    ("I donated to a relief fund but suspect the money isn’t reaching victims; audit the charity.", 
     "charity fund misuse investigation"),

    # PPP Loan Fraud Check
    ("Our competitor received a PPP loan under false pretenses; verify the legitimacy of their application.", 
     "PPP loan fraud investigation"),

    # Museum Artifact Theft
    ("A priceless vase went missing from the local museum—track down the thief and recover it.", 
     "museum theft investigation"),

    # Autograph Authentication
    ("I bought a signed baseball that might be forged; confirm if the signature is real.", 
     "memorabilia authentication investigation"),

    # Land Title Dispute
    ("My neighbor claims ownership of part of my backyard; verify the historical deeds and title chain.", 
     "land title investigation"),

    # Trust Fund Distribution Audit
    ("I suspect the trustee diverted funds from my family’s trust; audit the transaction history.", 
     "trust fund audit investigation"),

    # Charity Governance Review
    ("Board members may be self-dealing at a nonprofit I’m involved with; investigate governance violations.", 
     "nonprofit governance investigation"),

    # Manufacturing Line Sabotage
    ("Our assembly line is mysteriously failing tests overnight; find out if someone’s sabotaging it.", 
     "industrial sabotage investigation"),

    # Cyberbullying / School Harassment
    ("My teen is being bullied online by classmates; identify the perpetrators and gather proof.", 
     "cyberbullying investigation"),

    # Remote Employee Time-Log Verification
    ("I suspect a remote worker is falsifying hours; verify computer usage and login records.", 
     "time-log fraud investigation"),

    # Insider Trading Probe
    ("A colleague made huge stock trades before a merger announcement; investigate possible insider trading.", 
     "insider trading investigation"),

    # Chemical Plant Compliance Check
    ("Smoke and odors from the local plant are violating EPA guidelines; document environmental breaches.", 
     "environmental compliance investigation"),

    # Counter-Surveillance Sweep
    ("I think my phone line is tapped; perform a technical sweep for wiretaps and bugs.", 
     "counter-surveillance investigation"),

    # Foreign Agent / Counter-Intelligence
    ("I work in defense contracting and suspect a foreign agent is leaking documents; identify the culprit.", 
     "counter-intelligence investigation"),

     ("My credit card details were skimmed at an ATM and fraudulent charges are showing up; find the perpetrator.", 
     "credit card skimming investigation"),
    ("Rare WWII artifacts from my family museum have gone missing; recover them and identify the thief.", 
     "antiquities theft investigation"),
    ("I donated to a charity but suspect it’s a front for embezzlement; uncover the scheme.", 
     "charity fraud investigation"),
    ("I received death threats via email from an unknown sender; trace and stop the harasser.", 
     "cyber threat investigation"),
    ("My neighbor’s drone hovers over my backyard every night; determine if it’s spying on me.", 
     "drone surveillance investigation"),
    ("Someone is illegally dumping chemicals on my farmland after dark; gather proof for authorities.", 
     "environmental crime investigation"),
    ("Confidential company files were leaked online; find who distributed our internal documents.", 
     "data leak investigation"),
    ("A stolen car I reported was spotted at a chop shop; locate and recover my vehicle.", 
     "vehicle recovery investigation"),
    ("Local rumors claim my house is haunted by a ghost; document or debunk paranormal activity.", 
     "paranormal investigation"),
    ("A music producer’s contract was forged, costing them royalties; trace the document forger.", 
     "contract forgery investigation"),
    ("My online business account was hacked and sold; identify who took it over.", 
     "account takeover investigation"),
    ("An informant has tips on political corruption but fears retaliation; verify the intel source.", 
     "informant verification investigation"),
    ("A neighbor repeatedly trespasses and tags my walls with graffiti; catch them in the act.", 
     "trespassing surveillance investigation"),
    ("I received a blackmail email threatening to leak my private photos; identify the extortionist.", 
     "blackmail investigation"),
    ("Our campaign was sabotaged with counterfeit flyers accusing our candidate of wrongdoing; find who printed them.", 
     "political sabotage investigation"),

      # Tenant & Landlord Issues
    ("Prove my landlord is breaking housing laws and refusing necessary repairs.", 
     "tenant rights investigation"),
    ("Investigate my landlord’s criminal past before signing a new lease.", 
     "landlord background investigation"),

    # Genealogical & Family Searches
    ("Find my biological parents using only limited adoption records.", 
     "genealogical investigation"),
    ("Help me uncover my family's medical history for potential hereditary conditions.", 
     "family health history investigation"),

    # Pet & Asset Recovery
    ("Sarah's golden retriever, Max, went missing after a suspicious van appeared at her home.", 
     "pet recovery investigation"),
    ("Find the owner of a lost diary and discover the story behind it before it’s forever lost.", 
     "lost item recovery"),

    # Neighborhood & Personal Safety
    ("Help me prove my neighbor is a criminal after witnessing suspicious activity.", 
     "neighbor surveillance investigation"),
    ("Check if my teenager is involved in illegal activities and ensure their safety.", 
     "juvenile welfare investigation"),
    ("Jessica’s teenage son sneaks out every night—find out where he’s going.", 
     "runaway teen investigation"),
    ("Investigate a hidden camera found in my smoke detector—gather proof for legal action.", 
     "illegal surveillance investigation"),

    # Child & Caregiver Vetting
    ("Check if my child’s nanny is trustworthy and has no hidden criminal records.", 
     "nanny vetting investigation"),
    ("Investigate a babysitter's odd behavior and report any misconduct.", 
     "childcare background check"),

    # Vehicular & Financial Crimes
    ("Find the person who hit my parked car and fled the scene.", 
     "hit-and-run investigation"),
    ("Help me get my money back from a scammer who conned me online.", 
     "scammer recovery investigation"),
    ("Kyle was tricked into investing in a fake startup—trace the fraudsters.", 
     "investment fraud investigation"),

    # Workplace & Corporate
    ("A corporate executive suspects a rival has skeletons in their closet—uncover the truth.", 
     "executive vetting investigation"),
    ("Investigate whether I’m being framed at work to protect my career.", 
     "false accusation investigation"),
    ("Investigate corruption at my workplace and gather evidence of misconduct.", 
     "corporate fraud investigation"),
    ("Find out who reported me to my employer under false pretenses.", 
     "whistleblower investigation"),

    # Threats, Harassment & Stalking
    ("An author received an ominous letter warning them to stop writing; identify the sender.", 
     "threat investigation"),
    ("Find the source of strange prank calls harassing me for weeks.", 
     "harassment investigation"),
    ("Investigate a possible blackmail attempt after receiving threatening demands.", 
     "blackmail investigation"),
    ("A social media profile is harassing me with false posts; trace the account owner.", 
     "social media harassment investigation"),
    ("A journalist investigating corruption is now anonymously targeted—determine if it’s retaliation.", 
     "retaliation investigation"),

    # Real Estate & Property History
    ("Investigate whether my house is built on a burial ground before I renovate.", 
     "property history investigation"),
    ("Find out what happened to my childhood home and why it changed hands so suddenly.", 
     "real estate history investigation"),
    ("Find my biological family’s missing fortune and recover any hidden inheritance.", 
     "inheritance investigation"),

    # Missing Persons & Disappearances
    ("My sister has been missing for two days with no phone or social media trace—locate her.", 
     "missing person investigation"),
    ("Emily’s father disappeared during a road trip; last known location was a remote gas station.", 
     "missing person investigation"),

    # Technical & Cyber Forensics
    ("Nathan discovered someone opened credit cards in his name; trace the identity thief.", 
     "identity theft investigation"),
    ("I believe someone is hacking into my Wi-Fi and monitoring my activity—find the intruder.", 
     "cyber intrusion investigation"),

    # Specialty & Niche Cases
    ("A powerful businessman may be hiding shady deals—gather proof for a legal case.", 
     "white collar investigation"),
    ("Investigate a secret underground club rumored to be involved in illicit rites.", 
     "cult investigation"),
    ("An independent journalist suspects a government cover-up—verify facts before publishing.", 
     "investigative journalism support"),
    ("A high-end restaurant owner suspects a staff member of pocketing cash—catch the culprit.", 
     "employee theft investigation"),

     # Environmental & Pollution Violations
    ("Neighbors complain of foul odors from the factory; gather evidence of illegal dumping.", 
     "environmental crime investigation"),
    ("Someone is spraying pesticides in my organic farm; identify the culprit and motive.", 
     "agricultural sabotage investigation"),

    # Aviation & Drone Incidents
    ("A drone crashed into my backyard after midnight; recover data and identify the operator.", 
     "drone incident investigation"),
    ("A small plane made an emergency landing on my property; determine what went wrong.", 
     "aviation accident investigation"),

    # Polygraph & Voice Stress Testing
    ("I want to confirm my employee isn’t stealing; arrange a polygraph or voice stress test.", 
     "lie detector investigation"),

    # Undercover & Entrapment Operations
    ("I need someone to pose as a buyer to expose an underground arms dealer.", 
     "undercover operation investigation"),
    ("Plant an undercover agent at the club suspected of drug sales; document illicit activity.", 
     "covert infiltration investigation"),

    # Asset & Intellectual Property Recovery
    ("Rare manuscripts were stolen from my study; trace them and recover before they disappear.", 
     "artifact recovery investigation"),
    ("My tech startup’s source code was leaked; find who stole and distributed it.", 
     "IP theft investigation"),

    # Food & Product Contamination
    ("Several patrons fell ill after tasting our new menu; identify if there was deliberate poisoning.", 
     "food tampering investigation"),
    ("My children’s toy turned up with toxic paint; trace manufacturing defects.", 
     "product safety investigation"),

    # Wildfire & Hazardous Incident Probes
    ("A wildfire started suspiciously near my home; determine if there was foul play.", 
     "arson / wildfire investigation"),
    ("A chemical spill occurred on the highway; identify the responsible party and collect samples.", 
     "hazardous spill investigation"),

    # Whistleblower & Corporate Espionage
    ("An insider is leaking our trade secrets to competitors; uncover their identity and methods.", 
     "corporate espionage investigation"),
    ("A whistleblower needs protection after exposing CFO fraud; verify credibility and secure evidence.", 
     "whistleblower support investigation"),

    # Maritime & Vessel Salvage
    ("My vintage boat went missing during a regatta; locate and recover it from unknown waters.", 
     "maritime recovery investigation"),

    # Threat & Risk Assessment
    ("My CEO received a cryptic threat via voicemail; analyze voice and assess security risk.", 
     "threat assessment investigation"),

    # Records & Data Retrieval
    ("I need deleted chat logs from my former partner’s phone for evidence in court.", 
     "digital forensics investigation"),
    ("Obtain sealed legal documents from the county archive dating back 30 years.", 
     "records retrieval investigation")
]

context_texts  = [t for t, lbl in pi_context_samples]
context_labels = [lbl for t, lbl in pi_context_samples]
context_embeddings = model.encode(context_texts, convert_to_tensor=True)

def extract_context_label(backstory: str) -> str:
    user_emb = model.encode([backstory], convert_to_tensor=True)
    sims = util.pytorch_cos_sim(user_emb, context_embeddings).squeeze().cpu().numpy()
    best_idx = int(np.argmax(sims))
    return context_labels[best_idx]



# Sentiment detection
def get_sentiment(text):
    result = sentiment_pipeline(text)[0]
    return result['label'], result['score']

# Find best provider match
def find_best_match(user_backstory, user_location):
    user_input = f"{user_backstory} {user_location}"
    user_embedding = model.encode([user_input], convert_to_tensor=True)
    similarity_scores = util.pytorch_cos_sim(user_embedding, provider_embeddings).cpu().numpy().flatten()
    best_match_idx = np.argmax(similarity_scores)
    best_provider = data.iloc[best_match_idx]

    specialties = clean_text(best_provider['Specialties'])
    location    = clean_text(best_provider['Provider Location'])
    context     = extract_context_label(backstory)

    # 🔄 Updated courteous response (name removed)
    response_text = (
        f"Based on your request, I can connect you with a private investigator "
        f"located in {location} who specializes in {specialties}."
        
        f"This private investigator can help you with issues related to {context}."
    )
    asyncio.run(speak(response_text))

    return {
        #"Matched Provider":      best_provider['Matched Provider'],
        "Specialties":           best_provider['Specialties'],
        "Provider Location":     best_provider['Provider Location']
    }

# Guardrail logic with semantic detection
def guardrail_ai(user_input):
    if not is_pi_related_semantic(user_input):
        warning = "Sorry, I can only assist with private investigation services. Good Bye."
        asyncio.run(speak(warning))
        sys.exit()
        return False

    sentiment, _ = get_sentiment(user_input)
    sentiment_msg = f"I understand you're feeling {sentiment.lower()}."
    asyncio.run(speak(sentiment_msg))
    return True

# Main loop
if __name__ == "__main__":
    welcome = "Welcome to My Spy! Hi, I am Pie. How can I help you today?"
    asyncio.run(speak(welcome))

    while True:
        backstory = input("\nTell me your backstory: ").strip()
        if backstory.lower() in ["exit", "goodbye", "bye", "thank you"]:
            bye = "Good Bye! Hope My Spy was able to help you."
            asyncio.run(speak(bye))
            break

        if not guardrail_ai(backstory):
            continue

        location = input("Enter your location: ").strip()
        if location.lower() in ["exit", "goodbye", "thank you"]:
            bye = "Good Bye! Hope My Spy was able to help you."
            asyncio.run(speak(bye))
            break

        result = find_best_match(backstory, location)
        print("\nBest Match Found:")
        for key, value in result.items():
            print(f"{key}: {value}")

