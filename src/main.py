"""端到端入口：单条推理 + 批量推理。"""

import os
import sys
import time
import argparse
import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.preprocess import preprocess
from src.model import (
    DEFAULT_MODEL_NAME,
    get_tokenizer,
    load_trained_model,
    predict,
)
from src.llm_explain import generate_explanation, batch_generate_explanations

# 关键词提取（Saliency 方法）
try:
    from src.explain_tokens import get_key_tokens as _get_key_tokens_real

    def get_key_tokens(model, classifier, tokenizer, text,
                       target_label, top_k=6, device="cpu"):
        return _get_key_tokens_real(model, classifier, tokenizer, text,
                                     target_label, top_k, device)
except ImportError:
    def get_key_tokens(model, classifier, tokenizer, text,
                       target_label, top_k=6, device="cpu"):
        return [("(token extraction not yet available)", 0.0)]


def analyze_single(text: str, model, classifier, tokenizer,
                   device: str = "cpu") -> dict:
    """单条推文的完整分析流程。"""
    clean = preprocess(text)
    pred = predict(model, classifier, tokenizer, clean, device=device)
    key_tokens = get_key_tokens(
        model, classifier, tokenizer, clean,
        target_label=pred['label'], top_k=6, device=device
    )
    # LLM 解释用原始 text，分类和关键词用 clean text
    explanation = generate_explanation(
        text=text,
        prediction_label=pred['label'],
        confidence=pred['confidence'],
        key_tokens=key_tokens,
    )
    return {
        "text": text,
        "label_pred": pred['label'],
        "confidence": pred['confidence'],
        "key_tokens": str(key_tokens),
        "explanation": explanation,
    }


def print_single_result(result: dict):
    """格式化输出单条结果。"""
    label_name = "RUMOR" if result['label_pred'] == 1 else "NON-RUMOR"
    print("=" * 60)
    print(f"Input:      {result['text']}")
    print(f"Prediction: {label_name} (confidence: {result['confidence']*100:.1f}%)")
    print(f"Key tokens: {result['key_tokens']}")
    print(f"Explanation: {result['explanation']}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Rumor Detection with Explainable AI"
    )
    parser.add_argument('--text', type=str, help='Single tweet to analyze')
    parser.add_argument('--batch', type=str, help='Path to CSV for batch inference')
    parser.add_argument('--output', type=str, default='outputs/',
                        help='Output directory for batch results')
    parser.add_argument('--model-name', type=str, default=DEFAULT_MODEL_NAME,
                        help='HuggingFace base model name')
    args = parser.parse_args()

    if not args.text and not args.batch:
        parser.print_help()
        return

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model, classifier = load_trained_model(
        device=device,
        model_name=args.model_name,
    )
    tokenizer = get_tokenizer(args.model_name)

    if args.text:
        result = analyze_single(args.text, model, classifier, tokenizer, device)
        print_single_result(result)

    elif args.batch:
        df = pd.read_csv(args.batch)
        os.makedirs(args.output, exist_ok=True)

        # Phase 1: classify all tweets (no API, GPU fast)
        print(f"Phase 1/2: Classifying {len(df)} tweets...")
        results = []
        for i, row in df.iterrows():
            clean = preprocess(str(row['text']))
            pred = predict(model, classifier, tokenizer, clean, device=device)
            key_tokens = get_key_tokens(
                model, classifier, tokenizer, clean,
                target_label=pred['label'], top_k=6, device=device
            )
            result = {
                "text": str(row['text']),
                "label_pred": pred['label'],
                "confidence": pred['confidence'],
                "key_tokens": key_tokens,
            }
            for col in ['id', 'label', 'event']:
                if col in df.columns:
                    result[col] = row[col]
            results.append(result)
            if (i + 1) % 100 == 0:
                print(f"  Classified {i+1}/{len(df)}")

        # Phase 2: batch LLM explanations
        print(f"Phase 2/2: Generating explanations (batch mode)...")
        items_for_llm = [
            (r['text'], r['label_pred'], r['confidence'], r['key_tokens'])
            for r in results
        ]
        explanations = batch_generate_explanations(items_for_llm, batch_size=5)

        for r, exp in zip(results, explanations):
            r['explanation'] = exp
            r['key_tokens'] = str(r['key_tokens'])

        out_df = pd.DataFrame(results)
        out_df.to_csv(f'{args.output}/predictions.csv', index=False)

        # 保存解释单独文件
        exp_cols = [
            col for col in ['id', 'text', 'label_pred', 'confidence', 'explanation']
            if col in out_df.columns
        ]
        exp_df = out_df[exp_cols]
        exp_df.to_csv(f'{args.output}/explanations.csv', index=False)

        print(f"\nDone. {len(results)} tweets processed.")
        print(f"Results saved to {args.output}/predictions.csv")


if __name__ == '__main__':
    main()
