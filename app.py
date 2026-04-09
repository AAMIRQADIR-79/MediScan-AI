import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import json
from datetime import datetime
import io

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(page_title="MediScan AI", page_icon="🩺", layout="wide")

# ─────────────────────────────────────────────
#  GLOBAL CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=Syne:wght@700;800&display=swap');

*, *::before, *::after { box-sizing: border-box; }
html, body, .stApp {
    background-color: #060d1f !important;
    color: #e8f0fe;
    font-family: 'Space Grotesk', sans-serif;
}
.stApp::before {
    content: '';
    position: fixed; top: -50%; left: -50%;
    width: 200%; height: 200%;
    background:
        radial-gradient(ellipse at 20% 20%, rgba(0,212,255,0.07) 0%, transparent 50%),
        radial-gradient(ellipse at 80% 80%, rgba(99,102,241,0.07) 0%, transparent 50%),
        radial-gradient(ellipse at 50% 50%, rgba(16,185,129,0.04) 0%, transparent 60%);
    animation: bgShift 12s ease-in-out infinite alternate;
    pointer-events: none; z-index: 0;
}
@keyframes bgShift {
    0%   { transform: translate(0,0) rotate(0deg); }
    100% { transform: translate(2%,2%) rotate(1deg); }
}
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 1.5rem 2rem 4rem; position: relative; z-index: 1; }

/* HERO */
.hero-banner {
    background: linear-gradient(135deg, #0a1628 0%, #0d2040 50%, #0a1628 100%);
    border: 1px solid rgba(0,212,255,0.2);
    border-radius: 20px;
    padding: 2rem 2rem 1.6rem;
    margin-bottom: 1.5rem;
    text-align: center;
    position: relative; overflow: hidden;
    box-shadow: 0 0 60px rgba(0,212,255,0.08), inset 0 1px 0 rgba(255,255,255,0.05);
}
.hero-banner::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, transparent, #00d4ff, #6366f1, #10b981, transparent);
    animation: scanLine 3s ease-in-out infinite;
}
@keyframes scanLine { 0%,100%{opacity:.3} 50%{opacity:1} }
.hero-title {
    font-family: 'Syne', sans-serif; font-size: 2.6rem; font-weight: 800;
    background: linear-gradient(135deg, #00d4ff 0%, #6366f1 50%, #10b981 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text; letter-spacing: -1px; line-height: 1.1;
}
.hero-sub {
    font-size: 0.82rem; color: #7dd3fc; margin-top: 0.4rem;
    letter-spacing: 2px; text-transform: uppercase; opacity: 0.8;
}
.badge {
    display: inline-block;
    background: rgba(16,185,129,0.15); border: 1px solid rgba(16,185,129,0.4);
    color: #10b981; font-size: 0.68rem; letter-spacing: 1.5px;
    text-transform: uppercase; padding: 3px 10px; border-radius: 20px; margin-top: 0.7rem;
}

/* NAV TABS */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(255,255,255,0.03) !important;
    border-radius: 14px !important; padding: 4px !important;
    border: 1px solid rgba(255,255,255,0.06) !important;
    gap: 4px !important;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 10px !important; color: #64748b !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 0.85rem !important; font-weight: 500 !important;
    padding: 0.5rem 1.2rem !important;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, rgba(0,212,255,0.15), rgba(99,102,241,0.15)) !important;
    color: #00d4ff !important; border: 1px solid rgba(0,212,255,0.25) !important;
}
.stTabs [data-baseweb="tab-border"] { display: none !important; }

/* SECTION LABEL */
.section-label {
    font-size: 0.68rem; letter-spacing: 2.5px; text-transform: uppercase;
    color: #00d4ff; font-weight: 600; margin: 1.4rem 0 0.5rem;
    display: flex; align-items: center; gap: 8px;
}
.section-label::after {
    content: ''; flex: 1; height: 1px;
    background: linear-gradient(90deg, rgba(0,212,255,0.3), transparent);
}

/* INPUTS */
.stSelectbox label, .stTextInput label, .stTextArea label,
.stNumberInput label, .stSlider label { color: #64748b !important; font-size: 0.8rem !important; }
.stSelectbox > div > div, .stTextInput > div > div > input,
.stTextArea > div > div > textarea, .stNumberInput > div > div > input {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(0,212,255,0.15) !important;
    border-radius: 10px !important;
    color: #e8f0fe !important;
    font-family: 'Space Grotesk', sans-serif !important;
}
.stMultiSelect > div > div {
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid rgba(0,212,255,0.15) !important;
    border-radius: 12px !important;
}
.stMultiSelect > div > div:focus-within {
    border-color: rgba(0,212,255,0.5) !important;
    box-shadow: 0 0 0 3px rgba(0,212,255,0.08) !important;
}
span[data-baseweb="tag"] {
    background: linear-gradient(135deg, rgba(0,212,255,0.2), rgba(99,102,241,0.2)) !important;
    border: 1px solid rgba(0,212,255,0.35) !important;
    border-radius: 8px !important; color: #7dd3fc !important; font-size: 0.8rem !important;
}

/* BUTTONS */
.stButton > button {
    width: 100% !important;
    background: linear-gradient(135deg, #00d4ff, #6366f1) !important;
    color: #fff !important; border: none !important;
    border-radius: 14px !important; padding: 0.8rem 2rem !important;
    font-family: 'Syne', sans-serif !important; font-size: 0.95rem !important;
    font-weight: 700 !important; letter-spacing: 1px !important;
    text-transform: uppercase !important;
    box-shadow: 0 4px 20px rgba(0,212,255,0.25) !important;
    transition: all 0.25s ease !important; margin-top: 0.3rem !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 30px rgba(0,212,255,0.4) !important;
}

/* CARDS */
.card {
    background: linear-gradient(135deg, rgba(10,22,40,0.95), rgba(13,32,64,0.95));
    border: 1px solid rgba(0,212,255,0.18);
    border-radius: 18px; padding: 1.4rem; margin-bottom: 1rem;
    position: relative; overflow: hidden;
    box-shadow: 0 0 30px rgba(0,212,255,0.04);
}
.card::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, #00d4ff, #6366f1, #10b981);
    border-radius: 18px 18px 0 0;
}

/* RESULT CARD */
.result-card {
    background: linear-gradient(135deg, rgba(10,22,40,0.97), rgba(13,32,64,0.97));
    border: 1px solid rgba(0,212,255,0.3); border-radius: 20px;
    padding: 2rem; margin-top: 1.2rem; position: relative; overflow: hidden;
    box-shadow: 0 0 50px rgba(0,212,255,0.08);
    animation: fadeUp 0.4s ease;
}
@keyframes fadeUp {
    from { opacity: 0; transform: translateY(16px); }
    to   { opacity: 1; transform: translateY(0); }
}
.result-card::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 3px;
    background: linear-gradient(90deg, #00d4ff, #6366f1, #10b981);
    border-radius: 20px 20px 0 0;
}
.result-disease {
    font-family: 'Syne', sans-serif; font-size: 2.2rem; font-weight: 800;
    background: linear-gradient(135deg, #fff 0%, #7dd3fc 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text; margin-bottom: 0.2rem;
}
.result-meta { font-size: 0.8rem; color: #64748b; margin-bottom: 1rem; }

/* STAT BOXES */
.stats-row { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 10px; margin: 1rem 0; }
.stat-box {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 12px; padding: 0.85rem 1rem;
}
.stat-label { font-size: 0.65rem; letter-spacing: 1.5px; text-transform: uppercase; color: #64748b; margin-bottom: 4px; }
.stat-value { font-family: 'Syne', sans-serif; font-size: 1.3rem; font-weight: 700; color: #e8f0fe; }
.stat-value.low { color: #10b981; }
.stat-value.medium { color: #f59e0b; }
.stat-value.high { color: #ef4444; }
.confidence-bar { height: 5px; background: rgba(255,255,255,0.07); border-radius: 10px; margin-top: 6px; overflow: hidden; }
.confidence-fill { height: 100%; border-radius: 10px; background: linear-gradient(90deg, #00d4ff, #6366f1); }

/* PRECAUTION / ALERTS */
.precaution-box {
    background: rgba(0,212,255,0.05); border: 1px solid rgba(0,212,255,0.12);
    border-radius: 12px; padding: 1rem 1.2rem;
    font-size: 0.85rem; color: #bae6fd; margin-top: 1rem; line-height: 1.7;
}
.alert-critical {
    background: rgba(239,68,68,0.1); border: 1px solid rgba(239,68,68,0.35);
    border-radius: 12px; padding: 0.8rem 1.2rem;
    color: #fca5a5; font-weight: 600; font-size: 0.85rem; margin-top: 0.8rem;
}
.alert-ok {
    background: rgba(16,185,129,0.08); border: 1px solid rgba(16,185,129,0.25);
    border-radius: 12px; padding: 0.8rem 1.2rem;
    color: #6ee7b7; font-size: 0.85rem; margin-top: 0.8rem;
}
.feature-row {
    display: flex; flex-wrap: wrap; gap: 8px; margin-top: 0.8rem;
}
.feature-chip {
    background: rgba(99,102,241,0.12); border: 1px solid rgba(99,102,241,0.3);
    border-radius: 20px; padding: 3px 12px;
    font-size: 0.75rem; color: #a5b4fc;
}

/* HISTORY */
.history-item {
    background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.07);
    border-radius: 12px; padding: 0.9rem 1.1rem; margin-bottom: 0.6rem;
    display: flex; justify-content: space-between; align-items: center;
}
.history-disease { font-family: 'Syne', sans-serif; font-size: 1rem; font-weight: 700; color: #7dd3fc; }
.history-meta { font-size: 0.72rem; color: #475569; margin-top: 2px; }

/* AI CHAT */
.chat-msg-user {
    background: linear-gradient(135deg, rgba(0,212,255,0.1), rgba(99,102,241,0.1));
    border: 1px solid rgba(0,212,255,0.2);
    border-radius: 14px 14px 4px 14px;
    padding: 0.8rem 1rem; margin: 0.5rem 0; margin-left: 20%;
    font-size: 0.88rem; color: #e8f0fe;
}
.chat-msg-ai {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 4px 14px 14px 14px;
    padding: 0.8rem 1rem; margin: 0.5rem 0; margin-right: 20%;
    font-size: 0.88rem; color: #bae6fd; line-height: 1.6;
}

/* MISC */
.selected-count {
    font-size: 0.76rem; color: #00d4ff;
    background: rgba(0,212,255,0.06); border: 1px solid rgba(0,212,255,0.15);
    border-radius: 8px; padding: 4px 12px; display: inline-block; margin: 0.3rem 0 0.7rem;
}
.disclaimer {
    text-align: center; font-size: 0.7rem; color: #334155;
    margin-top: 2rem; border-top: 1px solid rgba(255,255,255,0.05); padding-top: 1rem;
}
h3 { color: #94a3b8 !important; font-family: 'Space Grotesk', sans-serif !important;
     font-weight: 500 !important; font-size: 0.88rem !important; }
.stRadio label { color: #94a3b8 !important; font-size: 0.85rem !important; }
.stTextArea textarea { min-height: 80px !important; }

/* Preset buttons — compact pill style */
div[data-testid="stHorizontalBlock"] .stButton > button {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(0,212,255,0.2) !important;
    border-radius: 20px !important;
    color: #cbd5e1 !important;
    font-size: 0.72rem !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 500 !important;
    padding: 0.3rem 0.5rem !important;
    letter-spacing: 0 !important;
    text-transform: none !important;
    box-shadow: none !important;
    margin-top: 0 !important;
    height: auto !important;
    white-space: nowrap !important;
    overflow: hidden !important;
    text-overflow: ellipsis !important;
}
div[data-testid="stHorizontalBlock"] .stButton > button:hover {
    background: rgba(0,212,255,0.1) !important;
    border-color: rgba(0,212,255,0.45) !important;
    color: #00d4ff !important;
    transform: none !important;
    box-shadow: none !important;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  LANGUAGE
# ─────────────────────────────────────────────
def t(en, hi):
    return hi if st.session_state.get("lang") == "Hindi" else en

# ─────────────────────────────────────────────
#  DATASET — Kaggle Disease-Symptom (41 diseases, 55 symptoms)
#  Each disease: 300 augmented rows = 12,300 total training samples
# ─────────────────────────────────────────────

SYMPTOM_COLS = [
    "itching","skin_rash","continuous_sneezing","shivering","chills",
    "joint_pain","stomach_pain","acidity","vomiting","fatigue",
    "weight_gain","anxiety","mood_swings","weight_loss","lethargy",
    "cough","high_fever","breathlessness","sweating","dehydration",
    "indigestion","headache","yellowish_skin","dark_urine","nausea",
    "loss_of_appetite","back_pain","constipation","abdominal_pain","diarrhoea",
    "mild_fever","yellowish_eyes","swelled_lymph_nodes","phlegm","throat_irritation",
    "redness_of_eyes","sinus_pressure","runny_nose","chest_pain","fast_heart_rate",
    "neck_stiffness","depression","muscle_pain","swollen_joints","loss_of_balance",
    "bladder_discomfort","foul_smell_of_urine","passage_of_gases","knee_pain","muscle_weakness",
    "palpitations","skin_peeling","inflammatory_nails","blister","patches_in_throat"
]

# ── Canonical symptom sets per disease (from Kaggle dataset) ──
DISEASE_SYMPTOMS = {
    "Fungal infection":     ["itching","skin_rash","skin_peeling","patches_in_throat","inflammatory_nails"],
    "Allergy":              ["continuous_sneezing","shivering","chills","redness_of_eyes","runny_nose","sinus_pressure","throat_irritation"],
    "GERD":                 ["stomach_pain","acidity","vomiting","indigestion","chest_pain","nausea","cough"],
    "Chronic cholestasis":  ["itching","vomiting","yellowish_skin","dark_urine","loss_of_appetite","abdominal_pain","yellowish_eyes"],
    "Drug Reaction":        ["itching","skin_rash","vomiting","nausea","fatigue","joint_pain"],
    "Peptic ulcer disease": ["stomach_pain","acidity","indigestion","vomiting","loss_of_appetite","nausea","abdominal_pain"],
    "AIDS":                 ["muscle_pain","fatigue","weight_loss","skin_rash","diarrhoea","swelled_lymph_nodes","high_fever"],
    "Diabetes":             ["fatigue","weight_loss","lethargy","nausea","bladder_discomfort","foul_smell_of_urine","mood_swings","weight_gain"],
    "Gastroenteritis":      ["vomiting","diarrhoea","stomach_pain","dehydration","nausea","mild_fever","abdominal_pain"],
    "Bronchial Asthma":     ["cough","breathlessness","phlegm","chest_pain","fatigue","throat_irritation"],
    "Hypertension":         ["headache","chest_pain","fatigue","palpitations","back_pain","sweating","nausea"],
    "Migraine":             ["headache","nausea","vomiting","fatigue","loss_of_balance","acidity"],
    "Cervical spondylosis": ["back_pain","neck_stiffness","headache","loss_of_balance","muscle_weakness","joint_pain"],
    "Paralysis (brain hemorrhage)": ["headache","loss_of_balance","neck_stiffness","vomiting","fatigue","muscle_weakness"],
    "Jaundice":             ["itching","vomiting","fatigue","yellowish_skin","dark_urine","abdominal_pain","loss_of_appetite","yellowish_eyes"],
    "Malaria":              ["chills","high_fever","sweating","headache","nausea","vomiting","fatigue","muscle_pain","diarrhoea"],
    "Chicken pox":          ["itching","skin_rash","blister","high_fever","fatigue","swelled_lymph_nodes","loss_of_appetite"],
    "Dengue":               ["skin_rash","chills","joint_pain","high_fever","headache","loss_of_appetite","fatigue","back_pain","nausea"],
    "Typhoid":              ["high_fever","headache","nausea","constipation","abdominal_pain","chills","vomiting","diarrhoea","fatigue"],
    "Hepatitis A":          ["joint_pain","vomiting","yellowish_skin","dark_urine","nausea","loss_of_appetite","abdominal_pain","mild_fever","fatigue"],
    "Hepatitis B":          ["itching","fatigue","vomiting","yellowish_skin","dark_urine","loss_of_appetite","abdominal_pain","yellowish_eyes","joint_pain"],
    "Hepatitis C":          ["fatigue","yellowish_skin","dark_urine","nausea","loss_of_appetite","yellowish_eyes","abdominal_pain","vomiting"],
    "Hepatitis D":          ["joint_pain","vomiting","fatigue","yellowish_skin","dark_urine","nausea","loss_of_appetite","yellowish_eyes"],
    "Hepatitis E":          ["joint_pain","vomiting","fatigue","yellowish_skin","dark_urine","nausea","loss_of_appetite","mild_fever"],
    "Alcoholic hepatitis":  ["vomiting","yellowish_skin","dark_urine","swelled_lymph_nodes","loss_of_appetite","abdominal_pain","fatigue"],
    "Tuberculosis":         ["cough","high_fever","fatigue","weight_loss","sweating","chills","breathlessness","chest_pain","phlegm"],
    "Common Cold":          ["continuous_sneezing","chills","fatigue","cough","runny_nose","sinus_pressure","throat_irritation","mild_fever","headache"],
    "Pneumonia":            ["cough","high_fever","breathlessness","chest_pain","fatigue","phlegm","sweating","chills","nausea"],
    "Dimorphic hemorrhoids (piles)": ["constipation","abdominal_pain","passage_of_gases","fatigue","bleeding"],
    "Heart attack":         ["chest_pain","fast_heart_rate","sweating","breathlessness","vomiting","back_pain","fatigue","nausea"],
    "Varicose veins":       ["fatigue","swollen_joints","knee_pain","muscle_weakness","palpitations","back_pain"],
    "Hypothyroidism":       ["fatigue","weight_gain","mood_swings","lethargy","depression","constipation","muscle_weakness","headache"],
    "Hyperthyroidism":      ["fatigue","mood_swings","weight_loss","fast_heart_rate","sweating","diarrhoea","anxiety","palpitations"],
    "Hypoglycemia":         ["fatigue","headache","sweating","nausea","vomiting","anxiety","palpitations","blurred_vision"],
    "Osteoarthritis":       ["joint_pain","knee_pain","swollen_joints","muscle_weakness","back_pain","neck_stiffness","fatigue"],
    "Arthritis":            ["joint_pain","swollen_joints","muscle_pain","knee_pain","fatigue","loss_of_balance","stiffness"],
    "(vertigo) Paroxysmal Positional Vertigo": ["headache","nausea","loss_of_balance","vomiting","fatigue"],
    "Acne":                 ["skin_rash","skin_peeling","inflammatory_nails","itching","pus_filled_pimples"],
    "Urinary tract infection": ["bladder_discomfort","foul_smell_of_urine","fatigue","abdominal_pain","nausea","chills","mild_fever","burning_micturition"],
    "Psoriasis":            ["skin_rash","itching","joint_pain","skin_peeling","inflammatory_nails","back_pain"],
    "Impetigo":             ["itching","skin_rash","blister","redness_of_eyes","fatigue","high_fever"],
}

def _make_rows():
    import random
    random.seed(42)
    sc  = SYMPTOM_COLS
    idx = {s: i for i, s in enumerate(sc)}
    n   = len(sc)

    def make_row(syms, noise_add=0, noise_remove=0.0):
        v = [0] * n
        for s in syms:
            if s in idx:
                v[idx[s]] = 1
        # add random noise symptoms
        for _ in range(noise_add):
            v[random.randint(0, n-1)] = 1
        # randomly drop some symptoms (simulate incomplete reporting)
        for i in range(n):
            if v[i] == 1 and random.random() < noise_remove:
                v[i] = 0
        return v

    rows, labels = [], []
    for disease, base_syms in DISEASE_SYMPTOMS.items():
        for trial in range(300):
            # Vary noise by trial cohort for realistic distribution
            if trial < 100:       # clean cases
                na, nr = 0, 0.05
            elif trial < 200:     # mild noise
                na, nr = random.choice([0,1]), 0.12
            else:                 # more variation
                na, nr = random.choice([0,1,2]), 0.20
            rows.append(make_row(base_syms, noise_add=na, noise_remove=nr))
            labels.append(disease)
    return rows, labels


# ─────────────────────────────────────────────
#  SYMPTOM ICONS & LOOKUPS
# ─────────────────────────────────────────────
SYMPTOM_ICONS = {
    "itching":              "🤜 Itching",
    "skin_rash":            "🔴 Skin Rash",
    "continuous_sneezing":  "🤧 Sneezing",
    "shivering":            "🥶 Shivering",
    "chills":               "❄️ Chills",
    "joint_pain":           "🦴 Joint Pain",
    "stomach_pain":         "🫃 Stomach Pain",
    "acidity":              "🔥 Acidity",
    "vomiting":             "🤢 Vomiting",
    "fatigue":              "😴 Fatigue",
    "weight_gain":          "⚖️ Weight Gain",
    "anxiety":              "😰 Anxiety",
    "mood_swings":          "🎭 Mood Swings",
    "weight_loss":          "📉 Weight Loss",
    "lethargy":             "🛌 Lethargy",
    "cough":                "😮‍💨 Cough",
    "high_fever":           "🌡️ High Fever",
    "breathlessness":       "😮 Breathlessness",
    "sweating":             "💧 Sweating",
    "dehydration":          "🚱 Dehydration",
    "indigestion":          "💊 Indigestion",
    "headache":             "🤕 Headache",
    "yellowish_skin":       "🟡 Yellowish Skin",
    "dark_urine":           "🟤 Dark Urine",
    "nausea":               "😵 Nausea",
    "loss_of_appetite":     "🍽️ Loss of Appetite",
    "back_pain":            "🔙 Back Pain",
    "constipation":         "🚫 Constipation",
    "abdominal_pain":       "🫁 Abdominal Pain",
    "diarrhoea":            "🚽 Diarrhoea",
    "mild_fever":           "🌡️ Mild Fever",
    "yellowish_eyes":       "👁️ Yellowish Eyes",
    "swelled_lymph_nodes":  "🔵 Swollen Lymph Nodes",
    "phlegm":               "🫧 Phlegm",
    "throat_irritation":    "🗣️ Throat Irritation",
    "redness_of_eyes":      "👁️ Red Eyes",
    "sinus_pressure":       "👃 Sinus Pressure",
    "runny_nose":           "👃 Runny Nose",
    "chest_pain":           "❤️ Chest Pain",
    "fast_heart_rate":      "💓 Fast Heart Rate",
    "neck_stiffness":       "🦒 Neck Stiffness",
    "depression":           "😔 Depression",
    "muscle_pain":          "💪 Muscle Pain",
    "swollen_joints":       "🦵 Swollen Joints",
    "loss_of_balance":      "⚖️ Loss of Balance",
    "bladder_discomfort":   "🔵 Bladder Discomfort",
    "foul_smell_of_urine":  "🟡 Foul Urine Smell",
    "passage_of_gases":     "💨 Gas / Bloating",
    "knee_pain":            "🦵 Knee Pain",
    "muscle_weakness":      "💪 Muscle Weakness",
    "palpitations":         "💗 Palpitations",
    "skin_peeling":         "🩹 Skin Peeling",
    "inflammatory_nails":   "💅 Inflamed Nails",
    "blister":              "🔴 Blisters",
    "patches_in_throat":    "🗣️ Throat Patches",
}
REVERSE_MAP = {v: k for k, v in SYMPTOM_ICONS.items()}

SEVERITY = {
    "Fungal infection": "Low",
    "Allergy": "Low",
    "GERD": "Low",
    "Chronic cholestasis": "Medium",
    "Drug Reaction": "Medium",
    "Peptic ulcer disease": "Medium",
    "AIDS": "High",
    "Diabetes": "Medium",
    "Gastroenteritis": "Medium",
    "Bronchial Asthma": "Medium",
    "Hypertension": "Medium",
    "Migraine": "Low",
    "Cervical spondylosis": "Low",
    "Paralysis (brain hemorrhage)": "High",
    "Jaundice": "Medium",
    "Malaria": "High",
    "Chicken pox": "Medium",
    "Dengue": "High",
    "Typhoid": "High",
    "Hepatitis A": "Medium",
    "Hepatitis B": "High",
    "Hepatitis C": "High",
    "Hepatitis D": "High",
    "Hepatitis E": "Medium",
    "Alcoholic hepatitis": "High",
    "Tuberculosis": "High",
    "Common Cold": "Low",
    "Pneumonia": "High",
    "Dimorphic hemorrhoids (piles)": "Low",
    "Heart attack": "High",
    "Varicose veins": "Low",
    "Hypothyroidism": "Medium",
    "Hyperthyroidism": "Medium",
    "Hypoglycemia": "Medium",
    "Osteoarthritis": "Low",
    "Arthritis": "Medium",
    "(vertigo) Paroxysmal Positional Vertigo": "Low",
    "Acne": "Low",
    "Urinary tract infection": "Medium",
    "Psoriasis": "Low",
    "Impetigo": "Low",
}

PRECAUTIONS = {
    "Fungal infection": "Keep skin clean and dry. Use antifungal cream. Avoid sharing personal items.",
    "Allergy": "Avoid allergens, take antihistamines, consult a doctor if severe.",
    "GERD": "Elevate head while sleeping, avoid fatty/spicy food, take antacids.",
    "Chronic cholestasis": "Consult a gastroenterologist. Avoid alcohol. Follow low-fat diet.",
    "Drug Reaction": "Stop the suspected drug immediately and consult a doctor.",
    "Peptic ulcer disease": "Avoid NSAIDs, alcohol, spicy food. Take prescribed antacids.",
    "AIDS": "ART therapy required. Consult an infectious disease specialist urgently.",
    "Diabetes": "Monitor blood sugar regularly. Low-sugar diet, exercise, medication as prescribed.",
    "Gastroenteritis": "ORS for rehydration. Light bland diet. Rest. See doctor if > 2 days.",
    "Bronchial Asthma": "Use prescribed inhaler. Avoid triggers. Keep rescue inhaler handy.",
    "Hypertension": "Low-salt diet, avoid stress, regular BP monitoring, medication if needed.",
    "Migraine": "Rest in a dark quiet room. Avoid triggers. Pain relief medication.",
    "Cervical spondylosis": "Physiotherapy, neck exercises, avoid prolonged screen use.",
    "Paralysis (brain hemorrhage)": "EMERGENCY: Call ambulance immediately. Do not move the patient.",
    "Jaundice": "Rest, hydration, avoid alcohol and fatty foods. Consult doctor urgently.",
    "Malaria": "Anti-malarial medication urgently needed. See a doctor immediately.",
    "Chicken pox": "Calamine lotion, avoid scratching, isolate from others. Rest.",
    "Dengue": "Hospital admission likely. Avoid mosquito bites. Monitor platelet count.",
    "Typhoid": "Antibiotics required. Stay hydrated. Avoid raw food and water.",
    "Hepatitis A": "Rest, avoid alcohol, eat a balanced low-fat diet. Vaccine available.",
    "Hepatitis B": "Antiviral medication. Avoid alcohol. Regular liver monitoring.",
    "Hepatitis C": "Antiviral therapy required. Avoid alcohol and sharing needles.",
    "Hepatitis D": "Requires Hepatitis B treatment first. Consult hepatologist.",
    "Hepatitis E": "Rest, hydration, avoid alcohol. Usually self-limiting.",
    "Alcoholic hepatitis": "Stop alcohol immediately. Medical supervision required.",
    "Tuberculosis": "Long-term antibiotics under medical supervision. Isolate initially.",
    "Common Cold": "Stay warm, drink hot fluids, steam inhalation helps. Rest.",
    "Pneumonia": "Antibiotics as prescribed. Rest, hydration. Hospitalise if severe.",
    "Dimorphic hemorrhoids (piles)": "High-fibre diet, drink water, sitz baths. See doctor for severe cases.",
    "Heart attack": "EMERGENCY: Call ambulance immediately. Aspirin if not allergic.",
    "Varicose veins": "Elevate legs, compression stockings, regular walks, avoid prolonged standing.",
    "Hypothyroidism": "Thyroid hormone replacement therapy. Regular TSH monitoring.",
    "Hyperthyroidism": "Anti-thyroid medication. Avoid iodine-rich foods. Regular monitoring.",
    "Hypoglycemia": "Eat sugar/glucose immediately. Identify and treat underlying cause.",
    "Osteoarthritis": "Low-impact exercise, physiotherapy, pain relief. Weight management.",
    "Arthritis": "Anti-inflammatory medication, physiotherapy, joint-friendly exercise.",
    "(vertigo) Paroxysmal Positional Vertigo": "Epley manoeuvre, avoid sudden head movements. See ENT specialist.",
    "Acne": "Keep skin clean, use prescribed topical cream, avoid touching face.",
    "Urinary tract infection": "Drink 3+ litres water daily. Antibiotics as prescribed. Cranberry juice.",
    "Psoriasis": "Topical steroids, phototherapy, avoid stress and skin trauma.",
    "Impetigo": "Antibiotic cream or oral antibiotics. Keep affected area clean.",
}

SOURCES = {
    "Malaria":      "https://www.who.int/news-room/fact-sheets/detail/malaria",
    "Dengue":       "https://www.who.int/news-room/fact-sheets/detail/dengue-and-severe-dengue",
    "Tuberculosis": "https://www.who.int/news-room/fact-sheets/detail/tuberculosis",
    "Pneumonia":    "https://www.who.int/news-room/fact-sheets/detail/pneumonia",
    "Hepatitis B":  "https://www.who.int/news-room/fact-sheets/detail/hepatitis-b",
    "Hepatitis C":  "https://www.who.int/news-room/fact-sheets/detail/hepatitis-c",
    "AIDS":         "https://www.who.int/news-room/fact-sheets/detail/hiv-aids",
    "Diabetes":     "https://www.who.int/news-room/fact-sheets/detail/diabetes",
    "Hypertension": "https://www.who.int/news-room/fact-sheets/detail/hypertension",
    "Heart attack": "https://www.who.int/news-room/fact-sheets/detail/cardiovascular-diseases-(cvds)",
    "Typhoid":      "https://www.who.int/news-room/fact-sheets/detail/typhoid",
}


@st.cache_resource
def build_model():
    rows, labels = _make_rows()          # 41 diseases × 300 rows = 12,300 training samples
    X = pd.DataFrame(rows, columns=SYMPTOM_COLS)
    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    clf.fit(X, labels)
    return clf

model = build_model()

# ─────────────────────────────────────────────
#  SESSION STATE
# ─────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []
if "lang" not in st.session_state:
    st.session_state.lang = "English"
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []
if "last_result" not in st.session_state:
    st.session_state.last_result = None

# ─────────────────────────────────────────────
#  HERO
# ─────────────────────────────────────────────
st.markdown("""
<div class="hero-banner">
    <div class="hero-title">🩺 MediScan AI</div>
    <div class="hero-sub">Advanced Symptom Analysis · 41 Diseases · 12,300 Training Samples · Random Forest</div>
    <div class="badge">● Powered by Machine Learning</div>
</div>
""", unsafe_allow_html=True)

# Language in top-right style
lang_col1, lang_col2 = st.columns([6, 1])
with lang_col2:
    lang_choice = st.selectbox("", ["English", "Hindi"], key="lang_selector")
    st.session_state.lang = lang_choice

# ─────────────────────────────────────────────
#  TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "🔍 " + t("Symptom Checker", "लक्षण जांच"),
    "🤖 " + t("AI Assistant", "AI सहायक"),
    "📊 " + t("Dashboard", "डैशबोर्ड"),
    "📋 " + t("History", "इतिहास"),
])

# ═══════════════════════════════════════════════
#  TAB 1 — SYMPTOM CHECKER
# ═══════════════════════════════════════════════
with tab1:
    left_col, right_col = st.columns([3, 2], gap="large")

    with left_col:
        # Patient profile
        st.markdown(f'<div class="section-label">👤 {t("Patient Profile", "रोगी प्रोफाइल")}</div>', unsafe_allow_html=True)
        pname_col, _ = st.columns([3,1])
        patient_name = pname_col.text_input(t("Patient Name", "रोगी का नाम"), placeholder=t("Enter full name…","पूरा नाम दर्ज करें…"))
        p1, p2, p3 = st.columns(3)
        age      = p1.number_input(t("Age", "आयु"), 1, 120, 25)
        gender   = p2.selectbox(t("Gender", "लिंग"), [t("Male","पुरुष"), t("Female","महिला"), t("Other","अन्य")])
        duration = p3.selectbox(t("Duration", "अवधि"), ["< 1 day","1–3 days","4–7 days","> 1 week"])

        # Symptom selector
        st.markdown(f'<div class="section-label">🧬 {t("Select Symptoms", "लक्षण चुनें")}</div>', unsafe_allow_html=True)
        display_names = [SYMPTOM_ICONS[s] for s in SYMPTOM_COLS]
        selected_display = st.multiselect(
            t("Search or select symptoms", "लक्षण खोजें या चुनें"),
            display_names,
            placeholder=t("Click to add symptoms…", "लक्षण जोड़ने के लिए क्लिक करें…")
        )
        selected_symptoms = [REVERSE_MAP[s] for s in selected_display if s in REVERSE_MAP]



        analyze = st.button(f"🔍 {t('Analyze Symptoms','लक्षण विश्लेषण करें')}")

    # ── RIGHT COL: RESULT ──
    with right_col:
        if analyze:
            if not selected_symptoms:
                st.warning(t("Please select at least one symptom.", "कृपया कम से कम एक लक्षण चुनें।"))
            elif len(selected_symptoms) < 3:
                st.markdown("""
                <div style="background:rgba(245,158,11,0.1);border:1px solid rgba(245,158,11,0.35);
                            border-radius:14px;padding:1.2rem 1.4rem;margin-top:1rem;">
                    <div style="font-size:1rem;font-weight:700;color:#f59e0b;margin-bottom:0.4rem;">
                        ⚠️ Too Few Symptoms
                    </div>
                    <div style="font-size:0.88rem;color:#fcd34d;line-height:1.6;">
                        Please select <strong>at least 3 symptoms</strong> for a reliable prediction.
                        Adding more specific symptoms significantly improves accuracy.
                    </div>
                    <div style="margin-top:0.8rem;font-size:0.8rem;color:#94a3b8;">
                        💡 <em>Tip: Most diseases have 4–6 key symptoms. Try to include all that apply.</em>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                input_vec = [1 if s in selected_symptoms else 0 for s in SYMPTOM_COLS]
                input_df  = pd.DataFrame([input_vec], columns=SYMPTOM_COLS)

                proba     = model.predict_proba(input_df)[0]
                classes   = model.classes_
                top3_idx  = np.argsort(proba)[::-1][:3]

                prediction  = classes[top3_idx[0]]
                confidence  = proba[top3_idx[0]] * 100

                # ── Low confidence guard ──────────────────────────────────
                if confidence < 40:
                    # Find which symptoms each top disease expects
                    def get_suggested(disease):
                        known = DISEASE_SYMPTOMS.get(disease, [])
                        missing = [s for s in known if s not in selected_symptoms][:3]
                        return [SYMPTOM_ICONS.get(s, s) for s in missing]

                    top1_name  = classes[top3_idx[0]]
                    top2_name  = classes[top3_idx[1]]
                    top1_miss  = get_suggested(top1_name)
                    top2_miss  = get_suggested(top2_name)
                    top1_conf  = str(round(proba[top3_idx[0]] * 100))
                    top2_conf  = str(round(proba[top3_idx[1]] * 100))
                    top1_sev   = SEVERITY.get(top1_name, "Medium")
                    top2_sev   = SEVERITY.get(top2_name, "Medium")
                    top1_icon  = "🟢" if top1_sev=="Low" else "🟡" if top1_sev=="Medium" else "🔴"
                    top2_icon  = "🟢" if top2_sev=="Low" else "🟡" if top2_sev=="Medium" else "🔴"
                    miss1_html = "".join('<span style="background:rgba(0,212,255,0.1);border:1px solid rgba(0,212,255,0.25);border-radius:20px;padding:2px 10px;font-size:0.78rem;color:#7dd3fc;margin:2px;">' + s + '</span>' for s in top1_miss)
                    miss2_html = "".join('<span style="background:rgba(99,102,241,0.1);border:1px solid rgba(99,102,241,0.25);border-radius:20px;padding:2px 10px;font-size:0.78rem;color:#a5b4fc;margin:2px;">' + s + '</span>' for s in top2_miss)

                    st.markdown(
                        "<div style=\"background:rgba(245,158,11,0.08);border:1px solid rgba(245,158,11,0.3);"
                        "border-radius:16px;padding:1.4rem 1.6rem;margin-top:1rem;\">"
                        "<div style=\"font-size:1rem;font-weight:700;color:#f59e0b;margin-bottom:0.5rem;\">🔍 Low Confidence — Add More Symptoms</div>"
                        "<div style=\"font-size:0.85rem;color:#94a3b8;margin-bottom:1rem;line-height:1.6;\">"
                        "Your symptom combination matches multiple diseases weakly. "
                        "The model needs more specific symptoms to give a confident result. "
                        "Here are the top candidates — try adding their missing symptoms:</div>"
                        "<div style=\"display:grid;grid-template-columns:1fr 1fr;gap:12px;\">"
                        "<div style=\"background:rgba(255,255,255,0.03);border:1px solid rgba(0,212,255,0.2);border-radius:12px;padding:1rem;\">"
                        "<div style=\"font-family:Syne,sans-serif;font-size:1.1rem;font-weight:700;color:#e8f0fe;\">" + top1_name + "</div>"
                        "<div style=\"font-size:0.75rem;color:#64748b;margin:2px 0 8px;\">" + top1_icon + " " + top1_sev + " &nbsp;·&nbsp; " + top1_conf + "% match</div>"
                        "<div style=\"font-size:0.72rem;color:#64748b;margin-bottom:4px;\">Try adding:</div>"
                        "<div style=\"display:flex;flex-wrap:wrap;gap:4px;\">" + (miss1_html if miss1_html else "<span style=\"color:#10b981;font-size:0.8rem;\">✓ Good symptom coverage</span>") + "</div>"
                        "</div>"
                        "<div style=\"background:rgba(255,255,255,0.03);border:1px solid rgba(99,102,241,0.2);border-radius:12px;padding:1rem;\">"
                        "<div style=\"font-family:Syne,sans-serif;font-size:1.1rem;font-weight:700;color:#e8f0fe;\">" + top2_name + "</div>"
                        "<div style=\"font-size:0.75rem;color:#64748b;margin:2px 0 8px;\">" + top2_icon + " " + top2_sev + " &nbsp;·&nbsp; " + top2_conf + "% match</div>"
                        "<div style=\"font-size:0.72rem;color:#64748b;margin-bottom:4px;\">Try adding:</div>"
                        "<div style=\"display:flex;flex-wrap:wrap;gap:4px;\">" + (miss2_html if miss2_html else "<span style=\"color:#10b981;font-size:0.8rem;\">✓ Good symptom coverage</span>") + "</div>"
                        "</div>"
                        "</div>"
                        "<div style=\"margin-top:1rem;font-size:0.78rem;color:#475569;\">"
                        "⚠ This is a preliminary estimate only. Always consult a qualified doctor.</div>"
                        "</div>",
                        unsafe_allow_html=True
                    )
                    st.stop()
                # ─────────────────────────────────────────────────────────

                sev       = SEVERITY.get(prediction, "Medium")
                sev_class = sev.lower()
                sev_icon  = "🟢" if sev == "Low" else "🟡" if sev == "Medium" else "🔴"

                # Feature importance for selected symptoms
                fi = model.feature_importances_
                selected_fi = {s: fi[SYMPTOM_COLS.index(s)] for s in selected_symptoms}
                top_features = sorted(selected_fi.items(), key=lambda x: x[1], reverse=True)[:4]

                alert_html = (
                    f'<div class="alert-critical">🚑 {t("Seek immediate medical attention!","तुरंत डॉक्टर से मिलें!")}</div>'
                    if sev == "High" else
                    f'<div class="alert-ok">👨‍⚕️ {t("Monitor condition. See doctor if worsens.","स्थिति देखें। बिगड़ने पर डॉक्टर से मिलें।")}</div>'
                )

                feat_chips = "".join([
                    '<span class="feature-chip">' + SYMPTOM_ICONS.get(f, f) + ' · ' + str(round(v*100)) + '%</span>'
                    for f, v in top_features
                ])

                source_link = ""
                if prediction in SOURCES:
                    source_link = '<a href="' + SOURCES[prediction] + '" target="_blank" style="color:#00d4ff;font-size:0.75rem;">🔗 WHO Reference</a>'

                conf_str       = str(round(confidence))
                sym_count      = str(len(selected_symptoms))
                label_primary  = t("PRIMARY DIAGNOSIS", "मुख्य निदान")
                label_age      = t("Age", "आयु")
                label_conf     = t("Confidence", "आत्मविश्वास")
                label_sev_lbl  = t("Severity", "गंभीरता")
                label_syms     = t("Symptoms", "लक्षण")
                label_key      = t("KEY INDICATORS", "मुख्य संकेतक")
                label_prec     = t("Recommended Precautions", "अनुशंसित सावधानियां")
                precaution_txt = PRECAUTIONS.get(prediction, "Consult a doctor.")
                pname_display  = (patient_name.strip() + " · ") if patient_name.strip() else ""
                meta_str       = pname_display + label_age + ": " + str(age) + " · " + str(gender) + " · " + str(duration)

                result_html = (
                    "<div class=\"result-card\">"
                    "<div style=\"font-size:0.65rem;letter-spacing:2px;text-transform:uppercase;color:#64748b;margin-bottom:2px\">" + label_primary + "</div>"
                    "<div class=\"result-disease\">" + prediction + "</div>"
                    "<div class=\"result-meta\">" + meta_str + "</div>"
                    "<div class=\"stats-row\">"
                    "<div class=\"stat-box\">"
                    "<div class=\"stat-label\">" + label_conf + "</div>"
                    "<div class=\"stat-value\">" + conf_str + "%</div>"
                    "<div class=\"confidence-bar\"><div class=\"confidence-fill\" style=\"width:" + conf_str + "%\"></div></div>"
                    "</div>"
                    "<div class=\"stat-box\">"
                    "<div class=\"stat-label\">" + label_sev_lbl + "</div>"
                    "<div class=\"stat-value " + sev_class + "\">" + sev_icon + " " + sev + "</div>"
                    "</div>"
                    "<div class=\"stat-box\">"
                    "<div class=\"stat-label\">" + label_syms + "</div>"
                    "<div class=\"stat-value\">" + sym_count + "</div>"
                    "</div>"
                    "</div>"
                    "<div style=\"font-size:0.65rem;letter-spacing:1.5px;text-transform:uppercase;color:#64748b;margin-top:0.8rem;margin-bottom:4px\">" + label_key + "</div>"
                    "<div class=\"feature-row\">" + feat_chips + "</div>"
                    "<div class=\"precaution-box\">"
                    "<strong style=\"color:#00d4ff;font-size:0.65rem;letter-spacing:1.5px;text-transform:uppercase;\">🛡 " + label_prec + "</strong><br><br>"
                    + precaution_txt +
                    "</div>"
                    + alert_html +
                    "<div style=\"margin-top:0.8rem\">" + source_link + "</div>"
                    "</div>"
                )
                st.markdown(result_html, unsafe_allow_html=True)

                # Differential diagnosis
                st.markdown(f'<div class="section-label" style="margin-top:1.2rem">🔬 {t("Differential Diagnosis","विभेदक निदान")}</div>', unsafe_allow_html=True)
                for i in top3_idx[1:3]:
                    d_name = classes[i]
                    d_conf = proba[i] * 100
                    d_sev  = SEVERITY.get(d_name,"Medium")
                    d_icon = "🟢" if d_sev=="Low" else "🟡" if d_sev=="Medium" else "🔴"
                    st.markdown(f"""
                    <div style="background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.07);
                                border-radius:10px;padding:0.7rem 1rem;margin-bottom:0.5rem;
                                display:flex;justify-content:space-between;align-items:center;">
                        <span style="font-family:'Syne',sans-serif;font-weight:700;color:#94a3b8;">{d_name}</span>
                        <span style="font-size:0.78rem;color:#64748b;">{d_icon} {d_sev} &nbsp;|&nbsp; {d_conf:.0f}%</span>
                    </div>
                    """, unsafe_allow_html=True)

                # Save to history
                st.session_state.history.append({
                    "disease": prediction,
                    "confidence": confidence,
                    "severity": sev,
                    "symptoms": selected_symptoms,
                    "age": age, "gender": gender,
                    "name": patient_name.strip() if patient_name.strip() else "Unknown",
                    "timestamp": datetime.now().strftime("%d %b %Y, %H:%M"),
                })
                st.session_state.last_result = {
                    "disease": prediction, "confidence": confidence,
                    "severity": sev, "symptoms": selected_symptoms,
                    "precaution": PRECAUTIONS.get(prediction,""),
                    "age": age, "gender": gender, "duration": duration,
                    "name": patient_name.strip() if patient_name.strip() else "Unknown",
                    "timestamp": datetime.now().strftime("%d %b %Y, %H:%M"),
                }

                # PDF-style download report
                report_text = f"""
MEDISCAN AI — HEALTH REPORT
============================
Date       : {datetime.now().strftime("%d %b %Y, %H:%M")}
Patient    : {patient_name.strip() if patient_name.strip() else "Unknown"}, Age {age}, {gender}
Duration   : {duration}

PRIMARY DIAGNOSIS : {prediction}
Confidence        : {confidence:.1f}%
Severity          : {sev}

SYMPTOMS REPORTED:
{chr(10).join("  • " + SYMPTOM_ICONS.get(s,s) for s in selected_symptoms)}

PRECAUTIONS:
{PRECAUTIONS.get(prediction,"")}

DIFFERENTIAL DIAGNOSES:
{chr(10).join(f"  • {classes[i]} ({proba[i]*100:.1f}%)" for i in top3_idx[1:3])}

---
⚠ This report is NOT a medical diagnosis.
   Always consult a qualified healthcare provider.
                """.strip()

                st.download_button(
                    label=f"📄 {t('Download Report','रिपोर्ट डाउनलोड करें')}",
                    data=report_text,
                    file_name=f"MediScan_Report_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                    mime="text/plain"
                )

        elif st.session_state.last_result:
            r = st.session_state.last_result
            st.markdown(f"""
            <div class="card" style="text-align:center;padding:2rem;">
                <div style="font-size:0.7rem;letter-spacing:2px;text-transform:uppercase;color:#475569">LAST RESULT</div>
                <div style="font-size:0.82rem;color:#00d4ff;margin-top:0.3rem;">👤 {r.get("name","Unknown")}</div>
                <div style="font-family:'Syne',sans-serif;font-size:1.6rem;font-weight:800;
                            background:linear-gradient(135deg,#fff,#7dd3fc);
                            -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                            background-clip:text;margin:0.4rem 0;">{r["disease"]}</div>
                <div style="font-size:0.8rem;color:#64748b;">{r["confidence"]:.0f}% confidence · {r["timestamp"]}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="card" style="text-align:center;padding:3rem 2rem;">
                <div style="font-size:3rem;margin-bottom:1rem;">🩺</div>
                <div style="font-family:'Syne',sans-serif;font-size:1.1rem;font-weight:700;color:#94a3b8;">
                    {t("Select symptoms and click Analyze","लक्षण चुनें और विश्लेषण करें")}
                </div>
                <div style="font-size:0.8rem;color:#475569;margin-top:0.5rem;">
                    {t("Results appear here","परिणाम यहाँ दिखेंगे")}
                </div>
            </div>
            """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════
#  TAB 2 — AI ASSISTANT (local NLP, no API key)
# ═══════════════════════════════════════════════

# Keyword → symptom key mapping for text parsing
KEYWORD_MAP = {
    "itching":             ["itch","itching","itchy","खुजली"],
    "skin_rash":           ["rash","skin rash","eruption","दाने"],
    "continuous_sneezing": ["sneez","sneezing","छींक"],
    "shivering":           ["shiver","shivering","कंपकंपी"],
    "chills":              ["chill","chills","cold feeling","ठंड लगना"],
    "joint_pain":          ["joint pain","joints","जोड़ों में दर्द"],
    "stomach_pain":        ["stomach pain","stomach ache","पेट दर्द"],
    "acidity":             ["acidity","acid","heartburn","एसिडिटी"],
    "vomiting":            ["vomit","vomiting","throwing up","उल्टी"],
    "fatigue":             ["tired","fatigue","exhausted","थकान","कमज़ोरी"],
    "weight_gain":         ["weight gain","gained weight","वजन बढ़ना"],
    "anxiety":             ["anxiety","anxious","worried","चिंता"],
    "mood_swings":         ["mood swing","irritable","mood","मूड"],
    "weight_loss":         ["weight loss","losing weight","वजन कम"],
    "lethargy":            ["letharg","sluggish","slow","सुस्ती"],
    "cough":               ["cough","coughing","खांसी"],
    "high_fever":          ["high fever","fever","temperature","febrile","तेज बुखार","बुखार"],
    "breathlessness":      ["breath","breathing","breathless","shortness","सांस"],
    "sweating":            ["sweat","sweating","night sweat","पसीना"],
    "dehydration":         ["dehydrat","dry mouth","thirsty","निर्जलीकरण"],
    "indigestion":         ["indigestion","digest","बदहजमी"],
    "headache":            ["headache","head pain","सिरदर्द"],
    "yellowish_skin":      ["yellow skin","jaundice","पीली त्वचा"],
    "dark_urine":          ["dark urine","brown urine","गहरे रंग का पेशाब"],
    "nausea":              ["nausea","nauseated","queasy","मतली"],
    "loss_of_appetite":    ["appetite","not eating","no hunger","भूख नहीं"],
    "back_pain":           ["back pain","backache","पीठ दर्द"],
    "constipation":        ["constipat","no bowel","कब्ज"],
    "abdominal_pain":      ["abdominal","belly pain","abdomen","पेट में दर्द"],
    "diarrhoea":           ["diarrhea","diarrhoea","loose motion","दस्त"],
    "mild_fever":          ["mild fever","low fever","हल्का बुखार"],
    "yellowish_eyes":      ["yellow eyes","jaundice eyes","पीली आंखें"],
    "swelled_lymph_nodes": ["lymph","lymph node","swollen gland","लिम्फ नोड"],
    "phlegm":              ["phlegm","mucus","sputum","बलगम"],
    "throat_irritation":   ["throat irritation","sore throat","throat","गले में दर्द"],
    "redness_of_eyes":     ["red eyes","redness eyes","आँखें लाल"],
    "sinus_pressure":      ["sinus","sinus pressure","साइनस"],
    "runny_nose":          ["runny nose","nose","nasal","नाक बहना"],
    "chest_pain":          ["chest pain","chest","सीने में दर्द"],
    "fast_heart_rate":     ["fast heart","palpitation","heart rate","धड़कन"],
    "neck_stiffness":      ["neck stiff","stiff neck","गर्दन अकड़न"],
    "depression":          ["depress","sad","hopeless","उदासी"],
    "muscle_pain":         ["muscle pain","body ache","myalgia","मांसपेशी दर्द"],
    "swollen_joints":      ["swollen joint","joint swelling","सूजे जोड़"],
    "loss_of_balance":     ["balance","dizzy","vertigo","चक्कर"],
    "bladder_discomfort":  ["bladder","urine discomfort","पेशाब में तकलीफ"],
    "foul_smell_of_urine": ["foul urine","smell urine","बदबूदार पेशाब"],
    "passage_of_gases":    ["gas","bloating","flatulence","गैस"],
    "knee_pain":           ["knee pain","knee","घुटने में दर्द"],
    "muscle_weakness":     ["muscle weak","weakness","कमजोरी"],
    "palpitations":        ["palpitation","heart flutter","धड़कन"],
    "skin_peeling":        ["skin peel","peeling skin","त्वचा छिलना"],
    "inflammatory_nails":  ["nail","nail inflammation","नाखून"],
    "blister":             ["blister","blisters","फफोले"],
    "patches_in_throat":   ["throat patch","white patch throat","गले में धब्बे"],
}

def parse_symptoms_from_text(text: str) -> list:
    """Extract symptom keys from free-text using keyword matching."""
    text_lower = text.lower()
    found = []
    for sym_key, keywords in KEYWORD_MAP.items():
        if any(kw in text_lower for kw in keywords):
            found.append(sym_key)
    return found

def generate_local_response(user_text: str) -> str:
    """Generate a rule-based AI response without any external API."""
    detected = parse_symptoms_from_text(user_text)

    greetings = ["hi","hello","hey","namaste","नमस्ते","हेलो"]
    if any(g in user_text.lower() for g in greetings) and len(detected) == 0:
        return (
            "👋 Hello! I'm **MediScan AI**, your health assistant.\n\n"
            "Please describe your symptoms in plain language, for example:\n"
            "*\"I have fever, cough and body ache since 2 days\"*\n\n"
            "I'll analyze them and suggest possible conditions."
        )

    if len(detected) == 0:
        return (
            "🤔 I couldn't detect specific symptoms from your message.\n\n"
            "Try describing symptoms like:\n"
            "• **fever, cough, headache, body ache**\n"
            "• **nausea, vomiting, diarrhea**\n"
            "• **sore throat, runny nose, fatigue**\n\n"
            "Or use the **Symptom Checker** tab for a structured analysis."
        )

    # Run the ML model on detected symptoms
    input_vec = [1 if s in detected else 0 for s in SYMPTOM_COLS]
    input_df  = pd.DataFrame([input_vec], columns=SYMPTOM_COLS)
    proba     = model.predict_proba(input_df)[0]
    classes   = model.classes_
    top3_idx  = np.argsort(proba)[::-1][:3]

    sym_display = ", ".join(SYMPTOM_ICONS.get(s, s) for s in detected)

    lines = [
        f"🔍 **Detected symptoms:** {sym_display}\n",
        "---",
        "### 🩺 Possible Conditions\n",
    ]

    for rank, idx in enumerate(top3_idx):
        d     = classes[idx]
        conf  = proba[idx] * 100
        sev   = SEVERITY.get(d, "Medium")
        icon  = "🟢" if sev == "Low" else "🟡" if sev == "Medium" else "🔴"
        prec  = PRECAUTIONS.get(d, "Consult a doctor.")
        lines.append(
            f"**{rank+1}. {d}** — {conf:.0f}% match {icon} *{sev} severity*\n"
            f"&nbsp;&nbsp;&nbsp;💊 {prec}\n"
        )

    top_disease = classes[top3_idx[0]]
    sev_top = SEVERITY.get(top_disease, "Medium")
    if sev_top == "High":
        lines.append("\n🚑 **One or more conditions detected are HIGH severity. Please consult a doctor immediately.**")
    else:
        lines.append("\n👨‍⚕️ Monitor your condition. Visit a doctor if symptoms worsen or persist beyond 3 days.")

    lines.append("\n---\n⚠ *This is not a medical diagnosis. Always consult a qualified healthcare provider.*")
    return "\n".join(lines)


with tab2:
    st.markdown(f'<div class="section-label">🤖 {t("Chat with AI Health Assistant","AI स्वास्थ्य सहायक से बात करें")}</div>', unsafe_allow_html=True)
    st.caption(t(
        "Describe your symptoms in plain language — the AI will identify possible conditions instantly (works offline, no API needed).",
        "अपने लक्षण सामान्य भाषा में बताएं — AI तुरंत संभावित स्थितियों की पहचान करेगा।"
    ))

    # Chat history display
    chat_container = st.container()
    with chat_container:
        if not st.session_state.chat_messages:
            st.markdown("""
            <div class="chat-msg-ai">
                <strong style="font-size:0.7rem;opacity:0.6;">🤖 MediScan AI</strong><br>
                👋 Hello! Describe your symptoms in plain language and I'll suggest possible conditions.<br><br>
                <em>Example: "I have fever and body ache since yesterday, also feeling very tired"</em>
            </div>
            """, unsafe_allow_html=True)
        for msg in st.session_state.chat_messages:
            role_class = "chat-msg-user" if msg["role"] == "user" else "chat-msg-ai"
            prefix = "🧑 You" if msg["role"] == "user" else "🤖 MediScan AI"
            # Convert markdown bold to html for rendering
            content = msg["content"].replace("**","<strong>").replace("*","<em>")
            st.markdown(
                f'<div class="{role_class}"><strong style="font-size:0.7rem;opacity:0.6;">{prefix}</strong><br>{content}</div>',
                unsafe_allow_html=True
            )

    # Input row
    user_input = st.text_area(
        t("Type your symptoms here…", "यहाँ अपने लक्षण लिखें…"),
        placeholder=t(
            "e.g. I have had fever and body ache for 2 days, also feeling very tired…",
            "जैसे: मुझे 2 दिनों से बुखार और शरीर दर्द है, बहुत थकान भी है…"
        ),
        key="ai_input", height=90
    )

    ai_col1, ai_col2 = st.columns([3, 1])
    send_btn  = ai_col1.button(f"💬 {t('Send','भेजें')}", key="send_chat")
    clear_btn = ai_col2.button(f"🗑 {t('Clear Chat','चैट साफ़ करें')}", key="clear_chat")

    if clear_btn:
        st.session_state.chat_messages = []
        st.rerun()

    if send_btn and user_input.strip():
        st.session_state.chat_messages.append({"role": "user", "content": user_input})
        reply = generate_local_response(user_input)
        st.session_state.chat_messages.append({"role": "assistant", "content": reply})
        st.rerun()

    # Quick prompt suggestions
    st.markdown(f'<div class="section-label" style="margin-top:1rem">⚡ {t("Try these…","ये आज़माएं…")}</div>', unsafe_allow_html=True)
    qc1, qc2, qc3 = st.columns(3)
    suggestions = [
        "I have fever and cough since 2 days",
        "Stomach pain, vomiting and diarrhea",
        "Headache, stress and can't sleep",
    ]
    for col, sug in zip([qc1, qc2, qc3], suggestions):
        if col.button(f"💬 {sug[:28]}…", key=f"sug_{sug[:10]}"):
            st.session_state.chat_messages.append({"role": "user", "content": sug})
            st.session_state.chat_messages.append({"role": "assistant", "content": generate_local_response(sug)})
            st.rerun()

# ═══════════════════════════════════════════════
#  TAB 3 — DASHBOARD
# ═══════════════════════════════════════════════
with tab3:
    st.markdown(f'<div class="section-label">📊 {t("Disease Insights Dashboard","रोग अंतर्दृष्टि डैशबोर्ड")}</div>', unsafe_allow_html=True)

    # Severity distribution of all 40 diseases
    sev_counts = {"Low": 0, "Medium": 0, "High": 0}
    for v in SEVERITY.values():
        sev_counts[v] += 1

    col_a, col_b = st.columns(2)

    with col_a:
        fig_sev = go.Figure(data=[go.Pie(
            labels=list(sev_counts.keys()),
            values=list(sev_counts.values()),
            hole=0.55,
            marker_colors=["#10b981","#f59e0b","#ef4444"],
            textinfo="label+percent",
            textfont=dict(color="white", size=12),
        )])
        fig_sev.update_layout(
            title=dict(text=t("Severity Distribution (40 Diseases)","गंभीरता वितरण"), font=dict(color="#94a3b8",size=13)),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#94a3b8", showlegend=True,
            legend=dict(font=dict(color="#94a3b8")),
            margin=dict(t=40,b=10,l=10,r=10), height=280,
        )
        st.plotly_chart(fig_sev, use_container_width=True)

    with col_b:
        # Symptom frequency across all diseases
        _rows, _lbls = _make_rows()
        _X = pd.DataFrame(_rows, columns=SYMPTOM_COLS)
        sym_freq = {s: int(_X[s].sum()) for s in SYMPTOM_COLS}
        sym_sorted = sorted(sym_freq.items(), key=lambda x: x[1], reverse=True)[:10]
        fig_sym = go.Figure(go.Bar(
            x=[SYMPTOM_ICONS.get(s,s).split(" ",1)[-1] for s,_ in sym_sorted],
            y=[v for _,v in sym_sorted],
            marker=dict(
                color=[v for _,v in sym_sorted],
                colorscale=[[0,"#6366f1"],[1,"#00d4ff"]],
                showscale=False,
            ),
            text=[str(v) for _,v in sym_sorted],
            textposition="outside",
            textfont=dict(color="#94a3b8", size=11),
        ))
        fig_sym.update_layout(
            title=dict(text=t("Most Common Symptoms (across diseases)","सबसे सामान्य लक्षण"), font=dict(color="#94a3b8",size=13)),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#94a3b8", xaxis=dict(tickfont=dict(color="#64748b",size=10)),
            yaxis=dict(gridcolor="rgba(255,255,255,0.05)", tickfont=dict(color="#64748b")),
            margin=dict(t=40,b=10,l=10,r=10), height=280,
        )
        st.plotly_chart(fig_sym, use_container_width=True)

    # Feature importance from RF model
    st.markdown(f'<div class="section-label">🧠 {t("Model Feature Importance","मॉडल फीचर महत्व")}</div>', unsafe_allow_html=True)
    fi_vals = model.feature_importances_
    fi_sorted = sorted(zip(SYMPTOM_COLS, fi_vals), key=lambda x: x[1], reverse=True)
    fig_fi = go.Figure(go.Bar(
        x=[v*100 for _,v in fi_sorted],
        y=[SYMPTOM_ICONS.get(s,s).split(" ",1)[-1] for s,_ in fi_sorted],
        orientation="h",
        marker=dict(
            color=[v*100 for _,v in fi_sorted],
            colorscale=[[0,"#1e3a5f"],[0.5,"#6366f1"],[1,"#00d4ff"]],
            showscale=False,
        ),
    ))
    fig_fi.update_layout(
        title=dict(text=t("Which symptoms matter most to the model?","मॉडल के लिए कौन से लक्षण सबसे महत्वपूर्ण हैं?"), font=dict(color="#94a3b8",size=13)),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#94a3b8",
        xaxis=dict(title="Importance (%)", gridcolor="rgba(255,255,255,0.05)", tickfont=dict(color="#64748b")),
        yaxis=dict(tickfont=dict(color="#94a3b8", size=11)),
        margin=dict(t=40,b=30,l=10,r=10), height=420,
    )
    st.plotly_chart(fig_fi, use_container_width=True)

    # Session history chart
    if st.session_state.history:
        st.markdown(f'<div class="section-label">📈 {t("Your Session History","आपका सत्र इतिहास")}</div>', unsafe_allow_html=True)
        hist_df = pd.DataFrame(st.session_state.history)
        fig_hist = px.bar(
            hist_df, x=hist_df.index, y="confidence", color="severity",
            color_discrete_map={"Low":"#10b981","Medium":"#f59e0b","High":"#ef4444"},
            text="disease",
            labels={"index":"Check #","confidence":"Confidence (%)"},
        )
        fig_hist.update_traces(textposition="outside", textfont=dict(color="white",size=10))
        fig_hist.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#94a3b8",
            xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
            yaxis=dict(gridcolor="rgba(255,255,255,0.05)", range=[0,115]),
            legend=dict(font=dict(color="#94a3b8")),
            margin=dict(t=20,b=10,l=10,r=10), height=260,
        )
        st.plotly_chart(fig_hist, use_container_width=True)

# ═══════════════════════════════════════════════
#  TAB 4 — HISTORY
# ═══════════════════════════════════════════════
with tab4:
    st.markdown(f'<div class="section-label">📋 {t("Symptom Check History","लक्षण जांच इतिहास")}</div>', unsafe_allow_html=True)

    if not st.session_state.history:
        st.markdown(f"""
        <div class="card" style="text-align:center;padding:2.5rem;">
            <div style="font-size:2.5rem">📋</div>
            <div style="color:#475569;margin-top:0.6rem;font-size:0.88rem;">
                {t("No checks yet. Run a symptom check first.","अभी तक कोई जांच नहीं। पहले लक्षण जांच चलाएं।")}
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        h1, h2 = st.columns([4,1])
        h2.metric(t("Total Checks","कुल जांच"), len(st.session_state.history))

        for i, entry in enumerate(reversed(st.session_state.history)):
            sev_icon = "🟢" if entry["severity"]=="Low" else "🟡" if entry["severity"]=="Medium" else "🔴"
            syms_str = ", ".join(SYMPTOM_ICONS.get(s,s) for s in entry["symptoms"][:4])
            if len(entry["symptoms"]) > 4:
                syms_str += f" +{len(entry['symptoms'])-4} more"
            st.markdown(f"""
            <div class="history-item">
                <div>
                    <div class="history-disease">{entry['disease']}</div>
                    <div class="history-meta">
                        {entry.get('name', 'Unknown')} · {entry['timestamp']} · {entry['age']} yrs, {entry['gender']}
                    </div>
                    <div style="font-size:0.72rem;color:#334155;margin-top:4px;">{syms_str}</div>
                </div>
                <div style="text-align:right;">
                    <div style="font-family:'Syne',sans-serif;font-size:1.1rem;font-weight:700;color:#7dd3fc;">
                        {entry['confidence']:.0f}%
                    </div>
                    <div style="font-size:0.75rem;color:#64748b;margin-top:2px;">{sev_icon} {entry['severity']}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        if st.button(f"🗑 {t('Clear History','इतिहास साफ़ करें')}"):
            st.session_state.history = []
            st.rerun()

# ─────────────────────────────────────────────
#  FOOTER
# ─────────────────────────────────────────────
st.markdown(f"""
<div class="disclaimer">
    ⚠ {t(
        "MediScan AI is not a substitute for professional medical diagnosis. Always consult a qualified healthcare provider. · Random Forest · 40 Conditions · 17 Symptoms",
        "MediScan AI पेशेवर चिकित्सा निदान का विकल्प नहीं है। हमेशा योग्य स्वास्थ्य सेवा प्रदाता से सलाह लें।"
    )}
</div>
""", unsafe_allow_html=True)