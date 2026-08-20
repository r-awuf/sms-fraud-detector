import streamlit as st
import joblib
import re
import nltk
import pandas as pd
import os
from datetime import datetime
from nltk.corpus import stopwords

nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))

# Load model and vectorizer
model      = joblib.load('src/model.pkl')
vectorizer = joblib.load('src/vectorizer.pkl')

# Ghanaian MoMo fraud keywords
FRAUD_KEYWORDS = [
    'pin', 'claim', 'winner', 'won', 'prize', 'reward', 'momo',
    'mtn', 'vodafone', 'airteltigo', 'suspended', 'verify', 'urgent',
    'congratulations', 'activate', 'locked', 'confirm', 'free', 'cash'
]

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = ' '.join(w for w in text.split() if w not in stop_words)
    return text

def get_fraud_keywords_found(message):
    words = message.lower().split()
    return [kw for kw in FRAUD_KEYWORDS if kw in words]

def predict(message):
    cleaned  = clean_text(message)
    features = vectorizer.transform([cleaned])
    
    # SVM doesn't have predict_proba, use decision function for confidence
    decision = model.decision_function(features)[0]
    
    # Convert decision score to a 0-1 probability-like confidence
    import numpy as np
    confidence = 1 / (1 + np.exp(-decision))  # sigmoid
    
    prediction = model.predict(features)[0]
    keywords   = get_fraud_keywords_found(message)
    return prediction, confidence, keywords

def log_report(message, prediction):
    log_path = 'data/reported_messages.csv'
    entry = pd.DataFrame([{
        'timestamp':  datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'message':    message,
        'prediction': 'Fraud/Spam' if prediction == 1 else 'Legitimate'
    }])
    if os.path.exists(log_path):
        entry.to_csv(log_path, mode='a', header=False, index=False)
    else:
        entry.to_csv(log_path, index=False)

# ── UI ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="SMS Fraud Detector", page_icon="🛡️", layout="centered")

if 'dark_mode' not in st.session_state:
    st.session_state['dark_mode'] = False

dark_mode = st.session_state['dark_mode']
theme = {
    'page': '#101716',
    'surface': '#18211f',
    'surface_alt': '#202c29',
    'text': '#f2f6f3',
    'muted': '#aabbb4',
    'border': '#31423d',
    'accent': '#55d6a4',
    'accent_dark': '#123d30',
} if dark_mode else {
    'page': '#f4f7f4',
    'surface': '#ffffff',
    'surface_alt': '#edf5f0',
    'text': '#17231f',
    'muted': '#5f7169',
    'border': '#d7e4dd',
    'accent': '#087f5b',
    'accent_dark': '#d8f3e8',
}

st.markdown(f"""
<style>
    :root {{
        --page: {theme['page']}; --surface: {theme['surface']};
        --surface-alt: {theme['surface_alt']}; --text: {theme['text']};
        --muted: {theme['muted']}; --border: {theme['border']};
        --accent: {theme['accent']}; --accent-dark: {theme['accent_dark']};
    }}
    .stApp {{ background: var(--page); color: var(--text); }}
    [data-testid="stHeader"] {{ background: transparent; }}
    [data-testid="stSidebar"] {{ background: var(--surface); border-right: 1px solid var(--border); }}
    [data-testid="stSidebar"] * {{ color: var(--text); }}
    h1 {{ letter-spacing: -0.03em; font-size: clamp(2rem, 5vw, 3.2rem); margin-bottom: 0.25rem; }}
    h2, h3, p, label {{ color: var(--text) !important; }}
    .eyebrow {{ color: var(--accent); font-size: 0.78rem; font-weight: 700; letter-spacing: 0.14em; text-transform: uppercase; }}
    .subtitle {{ color: var(--muted) !important; font-size: 1.05rem; margin-bottom: 1.5rem; }}
    [data-testid="stTextArea"] textarea {{ background: var(--surface); color: var(--text); border: 1px solid var(--border); border-radius: 12px; font-size: 1rem; }}
    [data-testid="stTextArea"] textarea:focus {{ border-color: var(--accent); box-shadow: 0 0 0 1px var(--accent); }}
    .example-label {{ color: var(--muted); font-size: 0.85rem; font-weight: 600; margin: 0.5rem 0 0.35rem; }}
    div.stButton > button {{ border: 1px solid var(--border); border-radius: 10px; min-height: 2.7rem; background: var(--surface); color: var(--text); }}
    div.stButton > button:hover {{ border-color: var(--accent); color: var(--accent); }}
    div.stButton > button[kind="primary"] {{ background: var(--accent); border-color: var(--accent); color: #ffffff; font-weight: 700; }}
    .result-card {{ background: var(--surface); border: 1px solid var(--border); border-radius: 14px; padding: 1.2rem 1.35rem; margin: 1.25rem 0; }}
    .result-kicker {{ color: var(--muted); font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.1em; font-weight: 700; }}
    .result-title {{ color: var(--text); font-size: 1.35rem; font-weight: 750; margin-top: 0.25rem; }}
    .footer-note {{ color: var(--muted); font-size: 0.82rem; text-align: center; margin-top: 2.5rem; }}
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### Preferences")
    st.toggle("Dark mode", key='dark_mode')
    st.caption("Your theme choice stays active while you use the detector.")
    st.divider()
    st.markdown("**How it works**")
    st.caption("The model looks for patterns in the message and highlights common MoMo fraud signals.")

st.markdown('<div class="eyebrow">Ghanaian SMS safety</div>', unsafe_allow_html=True)
st.title("🛡️ SMS Fraud Detector")
st.markdown('<p class="subtitle">A quick second opinion before you reply, click, or send money.</p>', unsafe_allow_html=True)

example_fraud = "Congratulations! Your MTN MoMo account has won GHS 500. Send your PIN to 024XXXXXXX to claim your reward."
example_legit = "Hey, are we still meeting at 3pm today? Let me know if anything changes."
example_momo = "ALERT: Your Vodafone Cash account will be suspended. Call 055XXXXXXX and provide your PIN to verify."

st.markdown('<div class="example-label">Start with an example</div>', unsafe_allow_html=True)
col1, col2, col3 = st.columns(3)
if col1.button("🚨 MoMo scam", use_container_width=True):
    st.session_state['sms_input'] = example_fraud
if col2.button("✅ Legit SMS", use_container_width=True):
    st.session_state['sms_input'] = example_legit
if col3.button("⚠️ Vodafone scam", use_container_width=True):
    st.session_state['sms_input'] = example_momo

sms_input = st.text_area(
    "Message to check",
    key='sms_input',
    height=150,
    placeholder="Paste or type an SMS message here...",
    label_visibility="visible"
)

if st.button("🔍 Analyse message", type="primary", use_container_width=True):
    if not (sms_input and sms_input.strip()):
        st.warning("Please enter a message to analyse.")
    else:
        prediction, confidence, keywords = predict(sms_input)
        score = confidence * 100
        st.session_state['last_message'] = sms_input
        st.session_state['last_prediction'] = prediction

        if score >= 70:
            status, icon, tone = "Likely fraud / spam", "🔴", "error"
        elif score >= 40:
            status, icon, tone = "Suspicious message", "🟡", "warning"
        else:
            status, icon, tone = "Looks legitimate", "🟢", "success"

        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        st.markdown('<div class="result-kicker">Analysis result</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="result-title">{icon} {status}</div>', unsafe_allow_html=True)
        metric_col, meter_col = st.columns([1, 2])
        metric_col.metric("Risk score", f"{score:.1f}%")
        meter_col.progress(int(score), text="Estimated risk")
        st.markdown('</div>', unsafe_allow_html=True)

        if keywords:
            st.markdown("**Fraud indicators found**")
            st.markdown(" ".join([f"`{kw}`" for kw in keywords]))
        else:
            st.success("No common fraud keywords detected.")

if st.session_state.get('last_message'):
    st.divider()
    st.markdown("**Help improve the detector**")
    st.caption("Was this result wrong? Report the message for review.")
    if st.button("🚩 Report as fraud", use_container_width=True):
        log_report(st.session_state['last_message'], st.session_state['last_prediction'])
        st.success("Thank you. This message has been logged for review.")

st.markdown('<div class="footer-note">Never share your PIN or one-time password by SMS or phone.</div>', unsafe_allow_html=True)