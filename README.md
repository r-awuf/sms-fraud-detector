# 🛡️ SMS Fraud & Spam Detector

A machine learning web app that detects mobile money scams and spam SMS,
built with a focus on the Ghanaian MoMo context.

## Problem

SMS-based fraud is a growing threat to mobile money users in Ghana.
Scammers impersonate MTN, Vodafone, and AirtelTigo to steal PINs and funds.

## Solution

A trained SVM classifier that flags suspicious messages in real time,
with a risk meter, keyword explainability, and a community reporting feature.

## Results

| Model               | Accuracy | Precision | Recall | F1 Score |
| ------------------- | -------- | --------- | ------ | -------- |
| Naive Bayes         | 0.9704   | 1.0000    | 0.7815 | 0.8773   |
| Logistic Regression | 0.9704   | 0.8688    | 0.9205 | 0.8939   |
| SVM (chosen)        | 0.9794   | 0.9267    | 0.9205 | 0.9236   |

SVM was selected for its best balance of recall and F1 score.

## How to Run

```bash
# 1. Clone the repo
git clone <your-repo-link>
cd sms-fraud-detector

# 2. Create and activate virtual environment
python -m venv venv
venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run app.py
```

## Project Structure

sms-fraud-detector/
├── data/
│ ├── spam.csv
│ ├── cleaned_sms.csv
│ └── reported_messages.csv
├── notebooks/
│ ├── exploration.ipynb
│ └── model_training.ipynb
├── src/
│ ├── model.pkl
│ └── vectorizer.pkl
├── app.py
├── requirements.txt
└── README.md
