import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(ROOT_DIR, "data")
RAW_DATA_PATH = os.path.join(DATA_DIR, "spam.csv")
PROCESSED_DATA_PATH = os.path.join(DATA_DIR, "processed_spam.csv")

MODELS_DIR = os.path.join(ROOT_DIR, "models")
NLTK_MODEL_PATH = os.path.join(MODELS_DIR, "model.pkl")
NLTK_VECTORIZER_PATH = os.path.join(MODELS_DIR, "vectorizer.pkl")
PIPELINE_MODEL_PATH = os.path.join(MODELS_DIR, "sms_model.joblib")

PIPELINE_METRICS_PATH = os.path.join(MODELS_DIR, "pipeline_metrics.json")
NLTK_METRICS_PATH = os.path.join(MODELS_DIR, "nltk_metrics.json")

TEST_SIZE = 0.2
RANDOM_STATE = 2
TFIDF_MAX_FEATURES = 3000
