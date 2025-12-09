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

```text
sms-classifier/
│
├── app/                          # Streamlit web application (UI)
│   └── app.py                    # Main app entrypoint
│
├── data/                         # Datasets
│   └── spam.csv                  # Original SMS Spam Collection dataset
│
├── logs/                         # Application & model logs
│   ├── predictions.log           # Logs every prediction made in the app
│   ├── train_nltk_model.log      # Logs for NLTK model training
│   ├── train_pipeline_model.log  # Logs for Pipeline model training
│   └── app.log                   # App-level logging (optional)
│
├── models/                       # Saved trained ML models + metrics
│   ├── model.pkl                 # NLTK Naive Bayes classifier
│   ├── vectorizer.pkl            # Vectorizer for NLTK model
│   ├── sms_model.joblib          # Logistic Regression Pipeline model
│   ├── nltk_metrics.json         # Evaluation metrics for NLTK model
│   └── pipeline_metrics.json     # Evaluation metrics for Pipeline model
│
├── src/                          # Core source code (modular architecture)
│   ├── config.py                 # Global configuration & file paths
│   ├── preprocess.py             # Text preprocessing (tokenization, stemming)
│   ├── predict.py                # Prediction logic + confidence scoring + logging
│   ├── train_nltk_model.py       # Training script: NLTK + Naive Bayes
│   ├── train_pipeline_model.py   # Training script: TF-IDF Pipeline + Logistic Regression
│   ├── utils/
│   │    └── logger.py            # Custom logger (file-only logging)
│   └── models/
│        └── evaluate.py          # Model evaluation metrics generator
│
├── venv/                         # (Optional) Python virtual environment
│
└── requirements.txt              # Project dependencies
