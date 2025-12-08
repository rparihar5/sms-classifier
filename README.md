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
├── data/
│   └── sms_spam_data.csv
├── models/
│   ├── sms_model.joblib
│   ├── vectorizer.pkl
│   └── model.pkl
├── src/
│   ├── __init__.py
│   ├── preprocess.py
│   ├── train_pipeline_model.py
│   ├── train_nltk_model.py
│   └── predict.py
├── app/
│   └── app.py
├── notebooks/
│   └── EDA.ipynb
├── requirements.txt
└── README.md
