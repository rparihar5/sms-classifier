📱 SMS Spam Classifier

An end-to-end Machine Learning project for classifying SMS messages as Spam or Ham.
Built with Python, Scikit-Learn, NLTK, Streamlit, and deployed on Streamlit Cloud.

🚀 Live Demo

🔗 App URL: [SMS-Spam-Classifier](https://sms-classifier-c7xhpszpnh23a8xf53drxz.streamlit.app/)
Try with example SMS messages and switch between two ML backends!

🧠 Project Overview

This project builds a complete end-to-end machine learning system:

Data preprocessing (tokenization, stopword removal, stemming)

Feature engineering (TF-IDF vectorization)

Two machine learning backends:

NLTK + Naive Bayes

TF-IDF + Logistic Regression Pipeline

Unified prediction interface with confidence scores

Evaluation reports (accuracy, precision, recall, F1)

Logging system for training and prediction

Streamlit app for real-time SMS classification

Deployment on Streamlit Community Cloud

It is structured as a production-quality ML project, not just a notebook.

🎯 Features
✔ Two ML Models

Naive Bayes (NLTK preprocessing)

Logistic Regression (Pipeline TF-IDF)
Easily switch between them in the UI.

✔ Clean text preprocessing

Lowercasing

Tokenization

Remove punctuation

Stopword removal

Stemming

✔ Confidence Scores

Every prediction returns:

Spam or Ham

Probability (0–1)

✔ Logs every prediction

Stored in logs/predictions.log

✔ Metrics stored as JSON

Each model has its own metrics file.

✔ Streamlit Frontend

User-friendly, fast, cloud-deployable.

📁 Project Structure
sms-classifier/
│
├── app/                          # Streamlit web application
│   └── app.py                    # Main UI entrypoint
│
├── data/                         # Datasets
│   └── spam.csv                  # SMS Spam Collection dataset
│
├── logs/                         # Application & model logs
│   ├── predictions.log
│   ├── train_nltk_model.log
│   ├── train_pipeline_model.log
│   └── app.log
│
├── models/                       # Trained models & evaluation metrics
│   ├── model.pkl                 # NLTK Naive Bayes model
│   ├── vectorizer.pkl            # TF-IDF vectorizer for NLTK model
│   ├── sms_model.joblib          # Logistic Regression Pipeline
│   ├── nltk_metrics.json
│   └── pipeline_metrics.json
│
├── src/                          # Source code (core ML pipeline)
│   ├── config.py                 # Global configuration
│   ├── preprocess.py             # Text cleaning + stemming
│   ├── predict.py                # Prediction + confidence + logging
│   ├── train_nltk_model.py       # Train NLTK + Naive Bayes model
│   ├── train_pipeline_model.py   # Train Pipeline LR model
│   ├── utils/
│   │    └── logger.py            # Custom logger
│   └── models/
│        └── evaluate.py          # Evaluation metrics generator
│
├── venv/                         # Python virtual environment (optional)
│
└── requirements.txt              # Python dependencies

🛠️ Installation

Clone the repo:

git clone https://github.com/rparihar5/sms-classifier.git
cd sms-classifier


Create and activate a virtual environment:

python3 -m venv venv
source venv/bin/activate    # macOS / Linux
venv\Scripts\activate       # Windows


Install dependencies:

pip install -r requirements.txt

🧪 Training Models
1. Train NLTK + Naive Bayes
python -m src.train_nltk_model


Generates:

models/model.pkl

models/vectorizer.pkl

models/nltk_metrics.json

logs → logs/train_nltk_model.log

2. Train Pipeline (TF-IDF + Logistic Regression)
python -m src.train_pipeline_model


Generates:

models/sms_model.joblib

models/pipeline_metrics.json

logs → logs/train_pipeline_model.log

🔍 Running the App Locally
streamlit run app/app.py


App opens at:

http://localhost:8501

🌐 Deployment (Streamlit Cloud)

The app is deployed at:

👉 Add your deployment URL here

To deploy yourself:

Push code to GitHub

Go to Streamlit Cloud → New app

Set:

Repository: sms-classifier

Main file: app/app.py

Deploy 🎉

🧪 Testing Examples
Spam
Congratulations! You have won a free iPhone. Click the link to claim your prize now.

Ham
Hey, are we still meeting at 7pm today?

Borderline
Your subscription is expiring soon. Renew to avoid interruption.

📊 Evaluation

Each model generates performance metrics like:

Accuracy

Precision

Recall

F1 Score

Saved in:

models/nltk_metrics.json
models/pipeline_metrics.json

📝 Logging

Logs are stored in:

logs/
    predictions.log
    train_nltk_model.log
    train_pipeline_model.log


Every prediction is logged with:

backend used

predicted label

confidence score

input text sample

🧩 Technologies Used

Python

NLTK

Scikit-Learn

Pandas

Streamlit

Joblib / Pickle

Logging module

JSON metrics

🏆 Acknowledgements

Dataset:
SMS Spam Collection Dataset
https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection

👨‍💻 Author
Rohit Parihar
UMass Dartmouth — MS in Data Science