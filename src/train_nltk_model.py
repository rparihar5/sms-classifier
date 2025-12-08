import ssl
import pickle
import pandas as pd
import nltk

from nltk.corpus import stopwords
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

from src.config import (
    RAW_DATA_PATH,
    NLTK_MODEL_PATH,
    NLTK_VECTORIZER_PATH,
    NLTK_METRICS_PATH,
    TEST_SIZE,
    RANDOM_STATE,
    TFIDF_MAX_FEATURES,
)
from src.preprocess import transform_text
from src.models.evaluate import evaluate_and_save
from src.utils.logger import get_logger


# ---- SSL workaround for NLTK downloads on macOS ----
try:
    _create_unverified_https_context = ssl._create_unverified_context
    ssl._create_default_https_context = _create_unverified_https_context
except AttributeError:
    pass


from src.utils.logger import get_logger

log = get_logger("train_nltk_model", "train_nltk_model.log")



def ensure_nltk_resources():
    nltk.download("punkt", quiet=True)
    nltk.download("stopwords", quiet=True)
    _ = stopwords.words("english")


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="latin-1")

    # Drop extra unused columns (Kaggle SMS dataset)
    df.drop(columns=["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], inplace=True, errors="ignore")

    # Rename Kaggle SMS Spam columns → target & text
    df.rename(columns={"v1": "target", "v2": "text"}, inplace=True)

    # Encode ham/spam → 0/1
    encoder = LabelEncoder()
    df["target"] = encoder.fit_transform(df["target"])

    return df[["text", "target"]]


def main():
    DATA_PATH = RAW_DATA_PATH
    VECTORIZER_PATH = NLTK_VECTORIZER_PATH
    MODEL_PATH = NLTK_MODEL_PATH
    METRICS_PATH = NLTK_METRICS_PATH

    log.info("Ensuring NLTK resources...")
    ensure_nltk_resources()

    log.info(f"Loading data from: {DATA_PATH}")
    df = load_data(DATA_PATH)

    log.info("Transforming text...")
    df["transformed_text"] = df["text"].apply(transform_text)

    log.info("Vectorizing TF-IDF...")
    tfidf = TfidfVectorizer(max_features=TFIDF_MAX_FEATURES)
    X = tfidf.fit_transform(df["transformed_text"]).toarray()
    y = df["target"].values

    log.info("Splitting into train/test...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    log.info("Training Multinomial Naive Bayes...")
    mnb = MultinomialNB()
    mnb.fit(X_train, y_train)

    log.info("Evaluating model...")
    y_pred = mnb.predict(X_test)

    # spam is encoded as 1
    evaluate_and_save(
        y_true=y_test,
        y_pred=y_pred,
        metrics_path=METRICS_PATH,
        backend_name="nltk",
        positive_label=1,
        logger=log,
    )

    log.info("Saving vectorizer and model...")
    pickle.dump(tfidf, open(VECTORIZER_PATH, "wb"))
    pickle.dump(mnb, open(MODEL_PATH, "wb"))
    log.info(f"Saved:\n- {VECTORIZER_PATH}\n- {MODEL_PATH}")


if __name__ == "__main__":
    main()
