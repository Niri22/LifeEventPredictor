"""Model evaluation: per-persona metrics, threshold tuning."""

import numpy as np
from sklearn.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_model(
    model, X_test: np.ndarray, y_test: np.ndarray, persona: str
) -> dict:
    """Evaluate a binary classifier and print metrics. Returns metrics dict."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    try:
        auc = roc_auc_score(y_test, y_proba)
    except ValueError:
        auc = 0.0

    print(f"\n  [{persona}] Evaluation on test set:")
    print(f"  Precision: {precision:.3f} | Recall: {recall:.3f} | F1: {f1:.3f} | AUC: {auc:.3f}")
    report_str = classification_report(
        y_test, y_pred, target_names=["none", persona], zero_division=0
    )
    print(f"\n  Classification Report:")
    print(report_str)

    # Find optimal threshold for target precision >= 0.80
    best_threshold = _find_precision_threshold(y_test, y_proba, target_precision=0.80)
    print(f"  Optimal threshold (precision >= 0.80): {best_threshold:.3f}")

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc,
        "optimal_threshold": best_threshold,
        "classification_report": report_str,
    }


def _find_precision_threshold(
    y_true: np.ndarray, y_proba: np.ndarray, target_precision: float = 0.80
) -> float:
    """Find the lowest threshold that achieves the target precision."""
    best_threshold = 0.5
    best_recall = 0.0

    for threshold in np.arange(0.1, 0.95, 0.05):
        preds = (y_proba >= threshold).astype(int)
        if preds.sum() == 0:
            continue
        prec = precision_score(y_true, preds, zero_division=0)
        rec = recall_score(y_true, preds, zero_division=0)
        if prec >= target_precision and rec > best_recall:
            best_recall = rec
            best_threshold = threshold

    return best_threshold
