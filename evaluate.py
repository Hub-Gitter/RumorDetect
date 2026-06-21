"""评测脚本：Accuracy, Macro F1, Per-class F1, AUC, 混淆矩阵, Per-event 指标。"""

import os
import sys
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix, roc_auc_score,
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def evaluate(predictions_csv: str, ground_truth_csv: str = None):
    """输入 predictions.csv 和 ground truth，输出所有指标。"""
    pred_df = pd.read_csv(predictions_csv)
    gt_df = None
    if ground_truth_csv:
        gt_df = pd.read_csv(ground_truth_csv)
        if len(gt_df) != len(pred_df):
            raise ValueError(
                f"Prediction rows ({len(pred_df)}) and ground truth rows "
                f"({len(gt_df)}) do not match."
            )

    # 如果有 label 列直接用，否则用 ground_truth_csv
    if 'label' in pred_df.columns:
        y_true = pred_df['label'].values
    elif 'label_true' in pred_df.columns:
        y_true = pred_df['label_true'].values
    elif gt_df is not None:
        y_true = gt_df['label'].values
    else:
        print("No ground truth labels found. Showing prediction distribution only.")
        print(pred_df['label_pred'].value_counts().to_string())
        return

    y_pred = pred_df['label_pred'].values

    # 总体指标
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    f1_0 = f1_score(y_true, y_pred, pos_label=0, zero_division=0)
    f1_1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)

    # AUC（优先使用显式 P(rumor)，兼容旧的 confidence 格式）
    if 'prob_rumor' in pred_df.columns:
        probs = pred_df['prob_rumor'].values
        try:
            auc = roc_auc_score(y_true, probs)
        except ValueError:
            auc = float('nan')
    elif 'confidence' in pred_df.columns:
        probs = []
        for _, row in pred_df.iterrows():
            if row['label_pred'] == 1:
                probs.append(row['confidence'])
            else:
                probs.append(1.0 - row['confidence'])
        try:
            auc = roc_auc_score(y_true, probs)
        except ValueError:
            auc = float('nan')
    else:
        auc = float('nan')

    print("=" * 50)
    print("Overall Metrics")
    print("=" * 50)
    print(f"  Accuracy:           {acc:.4f}")
    print(f"  Macro F1:           {macro_f1:.4f}")
    print(f"  F1 (Non-rumor, 0):  {f1_0:.4f}")
    print(f"  F1 (Rumor, 1):      {f1_1:.4f}")
    if not np.isnan(auc):
        print(f"  AUC:                {auc:.4f}")

    # 混淆矩阵
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    print(f"\n{'='*50}")
    print("Confusion Matrix")
    print(f"{'='*50}")
    print(f"{'':>12}  Predicted 0  Predicted 1")
    print(f"  Actual 0      {cm[0][0]:>8}      {cm[0][1]:>8}")
    print(f"  Actual 1      {cm[1][0]:>8}      {cm[1][1]:>8}")

    # Classification report
    print(f"\n{'='*50}")
    print("Classification Report")
    print(f"{'='*50}")
    print(classification_report(
        y_true, y_pred, labels=[0, 1], target_names=['Non-rumor', 'Rumor'],
        digits=4, zero_division=0
    ))

    # Per-event 指标
    if 'event' in pred_df.columns:
        event_values = pred_df['event'].values
    elif gt_df is not None and 'event' in gt_df.columns:
        event_values = gt_df['event'].values
    else:
        event_values = None

    if event_values is not None:
        print(f"{'='*50}")
        print("Per-Event Breakdown")
        print(f"{'='*50}")
        for event_id in sorted(pd.Series(event_values).dropna().unique()):
            mask = event_values == event_id
            ev_acc = accuracy_score(y_true[mask], y_pred[mask])
            ev_labels = y_true[mask]
            if len(set(ev_labels)) > 1:
                ev_f1 = f1_score(
                    y_true[mask], y_pred[mask], average='macro',
                    zero_division=0
                )
            else:
                ev_f1 = float('nan')
            print(f"  Event {event_id}: Acc={ev_acc:.4f}, F1_macro={ev_f1:.4f}, "
                  f"N={mask.sum()}")

    return {
        'accuracy': acc,
        'macro_f1': macro_f1,
        'f1_class0': f1_0,
        'f1_class1': f1_1,
        'auc': auc,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate rumor detection results")
    parser.add_argument('--predictions', type=str, required=True,
                        help='Path to predictions CSV')
    parser.add_argument('--ground_truth', type=str, default=None,
                        help='Path to ground truth CSV (if not in predictions)')
    args = parser.parse_args()
    evaluate(args.predictions, args.ground_truth)


if __name__ == '__main__':
    main()
