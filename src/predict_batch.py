"""Batch classification without LLM explanations."""

import argparse
import os
import sys

import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.model import DEFAULT_MODEL_NAME, get_tokenizer, load_trained_model, predict
from src.preprocess import preprocess


def main():
    parser = argparse.ArgumentParser(description="Run batch rumor classification")
    parser.add_argument("--input", default="data/val.csv", help="Input CSV path")
    parser.add_argument("--output", required=True, help="Output predictions CSV path")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME,
                        help="HuggingFace base model name")
    parser.add_argument("--checkpoint-dir", default="checkpoints",
                        help="Directory containing lora_adapter/ and classifier.pt")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Decision threshold for P(rumor)")
    parser.add_argument("--tuning-mode", choices=["lora", "full"], default="lora",
                        help="Checkpoint type to load")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Base model: {args.model_name}")
    print(f"Checkpoint: {args.checkpoint_dir}")
    print(f"Rumor threshold: {args.threshold}")
    print(f"Tuning mode: {args.tuning_mode}")

    adapter_path = os.path.join(args.checkpoint_dir, "lora_adapter")
    full_model_path = os.path.join(args.checkpoint_dir, "full_model")
    classifier_path = os.path.join(args.checkpoint_dir, "classifier.pt")
    model, classifier = load_trained_model(
        device=device,
        model_name=args.model_name,
        adapter_path=adapter_path,
        classifier_path=classifier_path,
        tuning_mode=args.tuning_mode,
        full_model_path=full_model_path,
    )
    tokenizer = get_tokenizer(args.model_name)

    df = pd.read_csv(args.input)
    rows = []
    for idx, row in df.iterrows():
        text = str(row["text"])
        pred = predict(model, classifier, tokenizer, preprocess(text), device=device)
        prob_rumor = pred["prob_rumor"]
        label_pred = int(prob_rumor >= args.threshold)
        confidence = prob_rumor if label_pred == 1 else pred["prob_non_rumor"]
        out = {
            "text": text,
            "label_pred": label_pred,
            "confidence": confidence,
            "prob_non_rumor": pred["prob_non_rumor"],
            "prob_rumor": prob_rumor,
        }
        for col in ["id", "label", "event"]:
            if col in df.columns:
                out[col] = row[col]
        rows.append(out)
        if (idx + 1) % 100 == 0:
            print(f"  Predicted {idx + 1}/{len(df)}")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)
    print(f"Saved {len(rows)} predictions to {args.output}")


if __name__ == "__main__":
    main()
