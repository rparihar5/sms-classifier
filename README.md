
---

# 📱 SMS Spam Classifier

An **end-to-end Machine Learning project** for classifying SMS messages as **Spam** or **Ham**.
Built with **Python, Scikit-Learn, NLTK, Streamlit**, and deployed on **Streamlit Community Cloud**.

---

## 🚀 Live Demo

🔗 **App URL:**
👉 [Spam Classifier](https://sms-classifier-c7xhpszpnh23a8xf53drxz.streamlit.app/)

Try real SMS examples and **switch between two ML backends** in real time.

---

## 🧠 Project Overview

This project implements a **production-quality ML pipeline**, not just a notebook.

### What it covers end-to-end:

* Text preprocessing (tokenization, stopword removal, stemming)
* Feature engineering using **TF-IDF**
* **Two Machine Learning models**
* Unified prediction interface with **confidence scores**
* Model evaluation & metrics persistence
* Logging for training and predictions
* Interactive **Streamlit web app**
* Cloud deployment

---

## 🎯 Features

### ✅ Dual ML Backends

* **NLTK + Naive Bayes**
* **TF-IDF + Logistic Regression (Pipeline)**
  ✔ Switch models directly from the UI

---

### ✅ Clean Text Preprocessing

* Lowercasing
* Tokenization
* Punctuation removal
* Stopword removal
* Stemming

---

### ✅ Confidence Scores

Each prediction returns:

* **Spam / Ham**
* **Probability score (0–1)**

---

### ✅ Logging System

* Logs **every prediction**
* Logs **training activity**
* Stored under `logs/`

---

### ✅ Metrics Persistence

* Accuracy
* Precision
* Recall
* F1-Score
  Saved as **JSON files** per model

---

### ✅ Streamlit Frontend

* Fast
* Simple UI
* Cloud-deployable

---

## 📁 Project Structure

```
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
│   ├── vectorizer.pkl            # TF-IDF vectorizer (NLTK model)
│   ├── sms_model.joblib          # Logistic Regression Pipeline
│   ├── nltk_metrics.json
│   └── pipeline_metrics.json
│
├── src/                          # Core ML source code
│   ├── config.py                 # Global configuration
│   ├── preprocess.py             # Text preprocessing logic
│   ├── predict.py                # Prediction + confidence + logging
│   ├── train_nltk_model.py       # NLTK + Naive Bayes training
│   ├── train_pipeline_model.py   # TF-IDF + LR pipeline training
│   ├── utils/
│   │    └── logger.py             # Custom logging utility
│   └── models/
│        └── evaluate.py           # Evaluation metrics generator
│
├── venv/                         # Virtual environment (optional)
└── requirements.txt              # Python dependencies
```

---

## 🛠️ Installation

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/rparihar5/sms-classifier.git
cd sms-classifier
```

---

### 2️⃣ Create & Activate Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate     # macOS / Linux
venv\Scripts\activate        # Windows
```

---

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🧪 Training the Models

### 🔹 1. Train NLTK + Naive Bayes

```bash
python -m src.train_nltk_model
```

**Outputs:**

* `models/model.pkl`
* `models/vectorizer.pkl`
* `models/nltk_metrics.json`
* Logs → `logs/train_nltk_model.log`

---

### 🔹 2. Train TF-IDF + Logistic Regression Pipeline

```bash
python -m src.train_pipeline_model
```

**Outputs:**

* `models/sms_model.joblib`
* `models/pipeline_metrics.json`
* Logs → `logs/train_pipeline_model.log`

---

## 🔍 Run the App Locally

```bash
streamlit run app/app.py
```

Open in browser:

```
http://localhost:8501
```

---

## 🌐 Deployment (Streamlit Cloud)

The app is deployed at:

🔗 [Spam Classifier](https://sms-classifier-c7xhpszpnh23a8xf53drxz.streamlit.app/)

---

## 🧪 Sample Test Messages

### 📩 Spam

> Congratulations! You have won a free iPhone. Click the link to claim your prize now.

### 📩 Ham

> Hey, are we still meeting at 7pm today?

### 📩 Borderline

> Your subscription is expiring soon. Renew to avoid interruption.

---

## 📊 Model Evaluation

Each model generates:

* Accuracy
* Precision
* Recall
* F1-Score

Saved to:

```
models/nltk_metrics.json
models/pipeline_metrics.json
```

---

## 📝 Logging

All logs are stored in:

```
logs/
 ├── predictions.log
 ├── train_nltk_model.log
 ├── train_pipeline_model.log
 └── app.log
```

Each prediction logs:

* Model backend
* Predicted label
* Confidence score
* Text sample

---

## 🧩 Technologies Used

* Python
* NLTK
* Scikit-Learn
* Pandas
* Streamlit
* Joblib / Pickle
* Logging module
* JSON metrics

---

## 🏆 Acknowledgements

**Dataset:**
SMS Spam Collection Dataset
[https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection](https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection)

---

## 👨‍💻 Author

**Rohit Parihar**
MS in Data Science — **University of Massachusetts Dartmouth**

---

