from pathlib import Path
import sys

import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.classification.preprocessing import (
    prepare_classification_data,
    linear_preprocessor,
)

ARTIFACT_DIR = ROOT / "artifacts" / "classification"
OUTPUT_DIR = ROOT / "outputs" / "classification" / "knn"

ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def train_knn():

    # load the data, split into train and test
    X_train, X_test, y_train, y_test = prepare_classification_data()

    # model pipelining
    model = Pipeline([
        ("preprocessor", linear_preprocessor()),
        ("classifier", KNeighborsClassifier(
            n_jobs=-1  # use all CPU cores to run faster
        )),
    ])

    # train the model on the training data
    print("\nTraining K-Nearest Neighbors...")
    model.fit(X_train, y_train)

    # predict on the test data 
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # check how well it did
    results = {
        "Model": "K-Nearest Neighbors",
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1-score": f1_score(y_test, y_pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, y_prob),
    }

    print("\n===== K-Nearest Neighbors Baseline =====")

    for metric, value in results.items():
        if metric != "Model":
            print(f"{metric}: {value:.4f}")

    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    print("\nClassification Report:")
    report = classification_report(y_test, y_pred, zero_division=0)
    print(report)

    # save the model so we don't retrain it
    model_path = ARTIFACT_DIR / "knn_baseline.joblib"
    joblib.dump(model, model_path)

    # save the scores to compare with the other models
    results_path = OUTPUT_DIR / "results.csv"
    pd.DataFrame([results]).to_csv(results_path, index=False)

    cm_df = pd.DataFrame(
        cm,
        index=["Actual 0", "Actual 1"],
        columns=["Predicted 0", "Predicted 1"]
    )
    cm_df.to_csv(OUTPUT_DIR / "confusion_matrix.csv")

    report_path = OUTPUT_DIR / "classification_report.txt"
    with open(report_path, "w") as file:
        file.write(report)

    print("\nFiles saved successfully:")
    print("Model:", model_path)
    print("Results:", results_path)
    print("Confusion matrix:", OUTPUT_DIR / "confusion_matrix.csv")
    print("Classification report:", report_path)

    return model, results


if __name__ == "__main__":
    train_knn()
