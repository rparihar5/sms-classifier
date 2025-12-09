# SMS / Email Spam Classifier 📩

## Overview

This project builds a machine learning model to classify text messages
(SMS or email-style) as **Spam** or **Not Spam**.

It includes:

- A scikit-learn **pipeline model** (TF-IDF + Logistic Regression)
- An NLTK-based model (custom preprocessing + TF-IDF + MultinomialNB)
- A **Streamlit web app** for interactive predictions

---

## Project Structure

sms-classifier/
│
├── app/                    → Streamlit app
│   └── app.py
│
├── data/
│   └── spam.csv            → original dataset
│
├── logs/                   → all logs saved here
│   ├── predictions.log
│   ├── train_nltk_model.log
│   ├── train_pipeline_model.log
│   └── app.log (optional)
│
├── models/                 → saved models and metrics
│   ├── model.pkl           → NLTK Naive Bayes
│   ├── vectorizer.pkl      → NLTK TF-IDF vectorizer
│   ├── sms_model.joblib    → Pipeline model
│   ├── nltk_metrics.json
│   └── pipeline_metrics.json
│
├── src/
│   ├── config.py           → all paths, hyperparameters
│   ├── preprocess.py       → transform_text()
│   ├── predict.py          → predictions + confidence + logging
│   ├── train_nltk_model.py → training script #1
│   ├── train_pipeline_model.py → training script #2
│   ├── utils/
│   │    └── logger.py      → custom file-only logger
│   └── models/
│        └── evaluate.py    → evaluate + save metrics JSON
│
├── venv/
└── requirements.txt

