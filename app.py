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

st.title("🛡️ SMS Fraud & Spam Detector")
st.caption("Detects mobile money scams and spam SMS — built for the Ghanaian context.")
st.divider()

# Example buttons
st.markdown("**Try an example:**")
col1, col2, col3 = st.columns(3)

example_fraud = "Congratulations! Your MTN MoMo account has won GHS 500. Send your PIN to 024XXXXXXX to claim your reward."
example_legit = "Hey, are we still meeting at 3pm today? Let me know if anything changes."
example_momo  = "ALERT: Your Vodafone Cash account will be suspended. Call 055XXXXXXX and provide your PIN to verify."

if col1.button("🚨 MoMo Scam"):
    st.session_state['sms_input'] = example_fraud
if col2.button("✅ Legit SMS"):
    st.session_state['sms_input'] = example_legit
if col3.button("⚠️ Vodafone Scam"):
    st.session_state['sms_input'] = example_momo

# Text input
sms_input = st.text_area(
    "Paste or type an SMS message below:",
    value=st.session_state.get('sms_input', ''),
    height=140,
    placeholder="e.g. You have won a MoMo prize! Send your PIN to claim..."
)

# Analyse button
if st.button("🔍 Analyse Message", type="primary", use_container_width=True):
    if not sms_input.strip():
        st.warning("Please enter a message to analyse.")
    else:
        prediction, confidence, keywords = predict(sms_input)
        score = confidence * 100

        st.divider()

        # Risk meter
        if score >= 70:
            st.error(f"🔴 **Likely Fraud / Spam** — Risk Score: {score:.1f}%")
        elif score >= 40:
            st.warning(f"🟡 **Suspicious** — Risk Score: {score:.1f}%")
        else:
            st.success(f"🟢 **Looks Legitimate** — Risk Score: {score:.1f}%")

        # Progress bar as risk meter
        st.progress(int(score))

        # Keyword explainability
        if keywords:
            st.markdown("**⚠️ Fraud indicators found in this message:**")
            st.markdown(" ".join([f"`{kw}`" for kw in keywords]))
        else:
            st.markdown("**✅ No common fraud keywords detected.**")

        # Community report button
        st.divider()
        st.markdown("**Was this result wrong? Help improve the model:**")
        if st.button("🚩 Report this message as fraud"):
            log_report(sms_input, prediction)
            st.success("Thank you! This message has been logged for review.")