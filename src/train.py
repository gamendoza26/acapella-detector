from typing import Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def train_logistic_regression(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.3,
    random_state: int = 42,
) -> Tuple[LogisticRegression, np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """
    Split data, scale features, and train a logistic regression classifier.
    Returns (clf, X_train_scaled, X_test_scaled, y_train, y_test, scaler).
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train_scaled, y_train)

    return clf, X_train_scaled, X_test_scaled, y_train, y_test, scaler
