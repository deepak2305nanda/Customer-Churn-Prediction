from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class TrainResult:
    accuracy: float
    n_train: int
    n_test: int


def train(random_state: int = 42) -> TrainResult:
    iris = load_iris()
    X = iris.data
    y = iris.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )

    model = LogisticRegression(max_iter=500)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = float(accuracy_score(y_test, preds))

    return TrainResult(accuracy=acc, n_train=int(X_train.shape[0]), n_test=int(X_test.shape[0]))


def main() -> None:
    result = train()
    print(
        "train.py OK",
        {"accuracy": np.round(result.accuracy, 4), "n_train": result.n_train, "n_test": result.n_test},
    )


if __name__ == "__main__":
    main()

