import json
from typing import Any, Sequence

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


def evaluate_and_save(
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
    metrics_path: str,
    backend_name: str,
    positive_label: Any,
    logger=None,
):
    """
    Compute metrics, log them, and save to a JSON file.

    Args:
        y_true: true labels
        y_pred: predicted labels
        metrics_path: where to save metrics JSON
        backend_name: name of backend (e.g. "nltk", "pipeline")
        positive_label: which class is treated as "spam" (1 or "spam")
        logger: optional logger instance; if None, falls back to print
    """
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, pos_label=positive_label)
    rec = recall_score(y_true, y_pred, pos_label=positive_label)
    f1 = f1_score(y_true, y_pred, pos_label=positive_label)
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred)

    def _log(msg: str):
        if logger is not None:
            logger.info(msg)
        else:
            print(msg)

    _log(f"=== Evaluation for backend: {backend_name} ===")
    _log(f"Accuracy: {acc}")
    _log(f"Precision: {prec}")
    _log(f"Recall: {rec}")
    _log(f"F1-score: {f1}")
    _log(f"Confusion matrix:\n{cm}")
    _log(f"Classification report:\n{report}")

    metrics = {
        "backend": backend_name,
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "confusion_matrix": cm.tolist(),
    }

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)

    _log(f"Metrics saved to {metrics_path}")
