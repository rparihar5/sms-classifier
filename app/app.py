import os
import sys
import json
import streamlit as st

# Make sure project root is importable
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from src.predict import predict_with_confidence

from src.utils.logger import get_logger

log = get_logger("app", "app.log")



def load_metrics(backend_key: str):
    """
    Loads the JSON file containing accuracy, precision, recall, f1 
    for the selected backend.
    """
    filename = (
        "pipeline_metrics.json" if backend_key == "pipeline" else "nltk_metrics.json"
    )
    filepath = os.path.join(ROOT_DIR, "models", filename)

    if not os.path.exists(filepath):
        return None

    with open(filepath, "r") as f:
        return json.load(f)


# ----------------------- STREAMLIT UI -----------------------

st.set_page_config(page_title="SMS Spam Classifier", page_icon="📩")

st.title("SMS / Email Spam Classifier")

backend = st.radio(
    "Choose model backend:",
    ["Logistic Regression", "NLTK (Preprocessing + Naive Bayes)"],
)

backend_key = "pipeline" if "Pipeline" in backend else "nltk"

# ----------------------- SPAM PREDICTION -----------------------

input_sms = st.text_area("Enter your message here:")

if st.button("Predict"):
    if input_sms.strip() == "":
        st.warning("Please enter a valid message.")
    else:
        label, spam_prob = predict_with_confidence(input_sms, backend=backend_key)
        confidence = spam_prob * 100  # convert to percentage

        if label == "spam":
            st.error(f"🚨 This message is SPAM ({confidence:.1f}% confidence).")
        else:
            not_spam_conf = 100 - confidence
            st.success(
                f"✅ This message is NOT SPAM ({not_spam_conf:.1f}% confidence)."
            )


