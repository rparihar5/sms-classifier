import pickle
import joblib
from typing import Literal, Tuple, Any

from src.preprocess import transform_text
from src.config import (
    PIPELINE_MODEL_PATH,
    NLTK_MODEL_PATH,
    NLTK_VECTORIZER_PATH,
)

from src.utils.logger import get_logger
log = get_logger(__name__, "predictions.log")

# Alias for readability
VECTORIZER_PATH = NLTK_VECTORIZER_PATH


def load_pipeline_model():
    return joblib.load(PIPELINE_MODEL_PATH)


def load_nltk_model():
    vectorizer = pickle.load(open(VECTORIZER_PATH, "rb"))
    model = pickle.load(open(NLTK_MODEL_PATH, "rb"))
    return vectorizer, model


def _map_label(pred: Any) -> str:
    """
    Convert whatever model outputs into 'spam' or 'ham'.
    Handles 0/1 or 'spam'/'ham' strings.
    """
    s = str(pred).lower()
    if s in ("1", "spam"):
        return "spam"
    return "ham"


def _get_spam_index(classes) -> int:
    """
    Given model.classes_, find index corresponding to spam class.
    Works whether classes are [0,1] or ['ham','spam'].
    """
    classes_list = list(classes)
    if "spam" in classes_list:
        return classes_list.index("spam")
    if 1 in classes_list:
        return classes_list.index(1)
    # Fallback: assume the "larger" class is spam (rarely needed)
    return classes_list.index(max(classes_list))


# ----------------- BASIC LABEL PREDICTION -----------------


def predict_with_pipeline(text: str) -> str:
    model = load_pipeline_model()
    pred = model.predict([text])[0]
    return _map_label(pred)


def predict_with_nltk(text: str) -> str:
    vectorizer, model = load_nltk_model()
    transformed = transform_text(text)
    vec = vectorizer.transform([transformed])
    pred = model.predict(vec)[0]  # 0 or 1
    return _map_label(pred)


def predict(text: str, backend: Literal["pipeline", "nltk"] = "nltk") -> str:
    """
    Unified entrypoint for just the label.
    """
    if backend == "pipeline":
        return predict_with_pipeline(text)
    return predict_with_nltk(text)


# ------------- LABEL + PROBABILITY (CONFIDENCE) -------------


def predict_with_confidence(
    text: str, backend: Literal["pipeline", "nltk"] = "nltk"
) -> Tuple[str, float]:
    """
    Returns:
        label: 'spam' or 'ham'
        spam_probability: float in [0, 1]
    """
    if backend == "pipeline":
        model = load_pipeline_model()
        proba = model.predict_proba([text])[0]
        spam_idx = _get_spam_index(model.classes_)
        spam_prob = float(proba[spam_idx])
        label = "spam" if spam_prob >= 0.5 else "ham"
    else:
        vectorizer, model = load_nltk_model()
        transformed = transform_text(text)
        vec = vectorizer.transform([transformed])
        proba = model.predict_proba(vec)[0]  # [p(ham), p(spam)] usually
        spam_idx = _get_spam_index(model.classes_)
        spam_prob = float(proba[spam_idx])
        label = "spam" if spam_prob >= 0.5 else "ham"

    # --- Logging the prediction ---
    safe_text = text[:100].replace("\n", " ")
    log.info(
        f"backend={backend}, label={label}, spam_prob={spam_prob:.4f}, text='{safe_text}'"
    )

    return label, spam_prob
