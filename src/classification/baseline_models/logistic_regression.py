from pathlib import Path

import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

# from ..preprocessing import (
#     prepare_classification_data,
#     linear_preprocessor,
# )
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.classification.preprocessing import (
    prepare_classification_data,
    linear_preprocessor,
)

# ROOT = Path(__file__).resolve().parents[2]

ARTIFACT_DIR = ROOT / "artifacts" / "classification"
OUTPUT_DIR = ROOT / "outputs" / "classification"  / "logistic_regression"

ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def train_logistic_regression():

    # 1. Load and split the data
    X_train, X_test, y_train, y_test = prepare_classification_data()

    # 2. Build the model pipeline
    model = Pipeline([
        ("preprocessor", linear_preprocessor()),
        ("classifier", LogisticRegression(
            max_iter=1000,
            random_state=42
        )),
    ])

    # 3. Train the model
    print("\nTraining Logistic Regression...")
    model.fit(X_train, y_train)

    # 4. Generate predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # 5. Calculate evaluation metrics
    results = {
        "Model": "Logistic Regression",
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1-score": f1_score(y_test, y_pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, y_prob),
    }

    # 6. Display results
    print("\n===== Logistic Regression Baseline =====")

    for metric, value in results.items():
        if metric != "Model":
            print(f"{metric}: {value:.4f}")

    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    print("\nClassification Report:")
    report_result = classification_report(
    y_test,
    y_pred,
    zero_division=0,
    output_dict=False
    )
    report = str(report_result)
    print(report)

    # 7. Save the model
    model_path = ARTIFACT_DIR / "logistic_regression_baseline.joblib"
    joblib.dump(model, model_path)

    # 8. Save evaluation results
    results_path = OUTPUT_DIR / "results.csv"
    results_df = pd.DataFrame([results])

    results_df.to_csv(
        results_path,
        index=False
    )

    # Save confusion matrix
    cm_df = pd.DataFrame(
        cm,
        index=["Actual 0", "Actual 1"],
        columns=["Predicted 0", "Predicted 1"]
    )

    cm_df.to_csv(OUTPUT_DIR / "confusion_matrix.csv")

    # Save classification report
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
    train_logistic_regression()
