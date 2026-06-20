"""Two-phase batch pipeline: classify all first, then LLM explanations."""
import os, sys, time, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import pandas as pd

from src.preprocess import preprocess
from src.model import DEFAULT_MODEL_NAME, get_tokenizer, load_trained_model, predict
from src.explain_tokens import get_key_tokens
from src.llm_explain import batch_generate_explanations


def main(model_name=DEFAULT_MODEL_NAME):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    print(f"Base model: {model_name}")

    model, classifier = load_trained_model(device=device, model_name=model_name)
    tokenizer = get_tokenizer(model_name)

    df = pd.read_csv('data/val.csv')
    n = len(df)
    print(f"Total samples: {n}")

    # Phase 1: classify + key tokens (no API)
    print("\n=== Phase 1: Classification + Key Tokens ===")
    results = []
    t0 = time.time()
    for idx, row in df.iterrows():
        text = str(row['text'])
        clean = preprocess(text)
        pred = predict(model, classifier, tokenizer, clean, device=device)
        keys = get_key_tokens(
            model, classifier, tokenizer, clean,
            target_label=pred['label'], top_k=6, device=device
        )
        results.append({
            'id': row.get('id', idx),
            'text': text,
            'label_pred': pred['label'],
            'confidence': pred['confidence'],
            'label_true': int(row.get('label', -1)),
            'event': row.get('event', -1),
            'key_tokens': [list(t) for t in keys],
        })
        if (idx + 1) % 100 == 0:
            elapsed = time.time() - t0
            print(f"  [{idx+1}/{n}] {elapsed:.1f}s")

    elapsed = time.time() - t0
    print(f"  Done. {n} tweets in {elapsed:.1f}s ({n/elapsed:.0f} tweets/s)")

    # Save intermediate results
    os.makedirs('outputs', exist_ok=True)
    with open('outputs/phase1_results.json', 'w') as f:
        json.dump(results, f, ensure_ascii=False)
    print("  Saved to outputs/phase1_results.json")

    # Phase 2: batch LLM explanations
    print("\n=== Phase 2: LLM Explanations (batch mode) ===")
    items = [(r['text'], r['label_pred'], r['confidence'], r['key_tokens'])
             for r in results]
    explanations = batch_generate_explanations(items, batch_size=5)

    for r, exp in zip(results, explanations):
        r['explanation'] = exp

    # Save final
    out_df = pd.DataFrame(results)
    out_df['key_tokens'] = out_df['key_tokens'].apply(str)
    out_df.to_csv('outputs/predictions.csv', index=False)
    out_df[['text', 'label_pred', 'confidence', 'explanation']].to_csv(
        'outputs/explanations.csv', index=False
    )
    print(f"  Saved to outputs/predictions.csv and outputs/explanations.csv")

    # Quick stats if labels available
    if 'label' in df.columns:
        correct = (out_df['label_pred'] == out_df['label_true']).sum()
        acc = correct / n * 100
        print(f"\nAccuracy: {acc:.2f}% ({correct}/{n})")
    print("Done.")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Two-phase batch inference")
    parser.add_argument('--model-name', default=DEFAULT_MODEL_NAME,
                        help='HuggingFace base model name')
    args = parser.parse_args()
    main(args.model_name)
