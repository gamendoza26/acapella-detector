from typing import Dict, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)

def evaluate_classifier(
    clf,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Tuple[Dict[str, float], np.ndarray]:
    """
    Compute accuracy, precision, recall, F1, and confusion matrix.
    Returns (metrics_dict, confusion_matrix_array).
    """
    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="macro"
    )
    cm = confusion_matrix(y_test, y_pred)

    metrics = {
        "accuracy": acc,
        "precision_macro": prec,
        "recall_macro": rec,
        "f1_macro": f1,
    }
    return metrics, cm
