from pathlib import Path
import re
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = Path(__file__).resolve().parent / "model.pkl"
VECTORIZER_PATH = Path(__file__).resolve().parent / "vectorizer.pkl"


def _load_stop_words() -> set[str]:
    return set(ENGLISH_STOP_WORDS)


stop_words = _load_stop_words()
model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)

FRAUD_KEYWORDS = [
    "pin",
    "claim",
    "winner",
    "won",
    "prize",
    "reward",
    "momo",
    "mtn",
    "vodafone",
    "airteltigo",
    "suspended",
    "verify",
    "urgent",
    "congratulations",
    "activate",
    "locked",
    "confirm",
    "free",
    "cash",
]


def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = " ".join(word for word in text.split() if word not in stop_words)
    return text.strip()


def get_fraud_keywords_found(message: str) -> list[str]:
    words = set(message.lower().split())
    return [keyword for keyword in FRAUD_KEYWORDS if keyword in words]


def _normalize_prediction(raw_prediction) -> str:
    if isinstance(raw_prediction, (int, np.integer)):
        return "spam" if int(raw_prediction) == 1 else "ham"

    label = str(raw_prediction).strip().lower()
    if label in {"spam", "fraud", "fraud/spam", "1", "true"}:
        return "spam"
    if label in {"ham", "legitimate", "not spam", "0", "false"}:
        return "ham"
    return label


def predict(message: str) -> tuple[str, float, list[str]]:
    cleaned = clean_text(message)
    features = vectorizer.transform([cleaned])

    raw_prediction = model.predict(features)[0]

    if hasattr(model, "decision_function"):
        decision_score = float(model.decision_function(features)[0])
        confidence = float(1 / (1 + np.exp(-decision_score)))
    elif hasattr(model, "predict_proba"):
        confidence = float(max(model.predict_proba(features)[0]))
    else:
        confidence = 0.5

    prediction = _normalize_prediction(raw_prediction)
    keywords = get_fraud_keywords_found(message)
    return prediction, confidence, keywords


def log_report(message: str, prediction: str) -> None:
    log_path = PROJECT_ROOT / "data" / "reported_messages.csv"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    entry = pd.DataFrame(
        [
            {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "message": message,
                "prediction": "Fraud/Spam" if prediction == "spam" else "Legitimate",
            }
        ]
    )

    if log_path.exists():
        entry.to_csv(log_path, mode="a", header=False, index=False)
    else:
        entry.to_csv(log_path, index=False)
