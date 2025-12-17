import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.pipeline import Pipeline
import joblib

from src.config import (
    RAW_DATA_PATH,
    PIPELINE_MODEL_PATH,
    PIPELINE_METRICS_PATH,
    TEST_SIZE,
    RANDOM_STATE,
)
from src.models.evaluate import evaluate_and_save
from src.utils.logger import get_logger


from src.utils.logger import get_logger

log = get_logger("train_pipeline_model", "train_pipeline_model.log")



def load_data(path: str):
    df = pd.read_csv(path, encoding="latin-1")

    # Match NLTK training cleanup (Kaggle SMS dataset)
    df.drop(columns=["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], inplace=True, errors="ignore")

    df = df.rename(columns={"v1": "label", "v2": "message"})
    df["label"] = df["label"].astype(str).str.strip().str.lower()

    return df[["label", "message"]]


def build_model():
    model = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    stop_words="english",
                    ngram_range=(1, 2),
                    max_df=0.9,
                    min_df=5,
                ),
            ),
            ("clf", LogisticRegression(max_iter=1000)),
        ]
    )
    return model


def main():
    DATA_PATH = RAW_DATA_PATH
    MODEL_PATH = PIPELINE_MODEL_PATH
    METRICS_PATH = PIPELINE_METRICS_PATH

    log.info("Loading data...")
    df = load_data(DATA_PATH)
    X = df["message"]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    model = build_model()

    log.info("Training pipeline model (TF-IDF + Logistic Regression)...")
    model.fit(X_train, y_train)

    log.info("Evaluating on test set...")
    y_pred = model.predict(X_test)

    # For this model, labels are strings: 'ham' / 'spam'
    evaluate_and_save(
        y_true=y_test,
        y_pred=y_pred,
        metrics_path=METRICS_PATH,
        backend_name="pipeline",
        positive_label="spam",
        logger=log,
    )

    log.info("Saving trained pipeline model...")
    joblib.dump(model, MODEL_PATH)
    log.info(f"Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
