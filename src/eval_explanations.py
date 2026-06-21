"""Evaluate explanation quality metrics:
1. Citation density: % of explanation tokens that appear in the original tweet
2. Length check: average number of sentences
3. Label recoverability: can you guess the label from the explanation alone?
"""
import json, re, os
import pandas as pd
from collections import Counter


def tokenize_simple(text):
    """Simple whitespace+punctuation tokenizer."""
    return set(re.findall(r'[a-zA-Z]+', text.lower()))


def citation_density(explanation, tweet):
    """Fraction of explanation content words that appear in the tweet."""
    exp_tokens = tokenize_simple(explanation)
    tweet_tokens = tokenize_simple(tweet)
    if not exp_tokens:
        return 0.0
    overlap = exp_tokens & tweet_tokens
    return len(overlap) / len(exp_tokens)


def count_sentences(text):
    """Count sentences in explanation."""
    return max(1, len(re.findall(r'[.!?]+', text)))


def count_quoted_words(text):
    """Count quoted words/phrases in explanation."""
    return len(re.findall(r'"([^"]+)"', text))


def evaluate_all(predictions_path='outputs/predictions.csv',
                 phase1_path=None):
    """Compute all explanation quality metrics."""
    # Load predictions
    df = pd.read_csv(predictions_path)

    # Load key tokens for citation analysis if available
    phase1 = None
    if phase1_path and os.path.exists(phase1_path):
        with open(phase1_path) as f:
            phase1 = json.load(f)

    densities = []
    sentence_counts = []
    quote_counts = []

    for i, row in df.iterrows():
        exp = str(row['explanation'])
        text = str(row['text'])

        # Skip fallback explanations
        if 'template-based explanation' in exp.lower():
            continue

        densities.append(citation_density(exp, text))
        sentence_counts.append(count_sentences(exp))
        quote_counts.append(count_quoted_words(exp))

    fallbacks = sum(1 for _, row in df.iterrows()
                   if 'template-based explanation' in str(row['explanation']).lower())
    print(f"Fallback rate: {fallbacks}/{len(df)} ({fallbacks/len(df)*100:.1f}%)")

    if not densities:
        print("\nNo non-fallback explanations to evaluate.")
        return densities, sentence_counts, quote_counts

    print(f"\nEvaluated {len(densities)} explanations (non-fallback)")
    print(f"\n{'Metric':<30} {'Mean':>8} {'Median':>8} {'Min':>8} {'Max':>8}")
    print('-' * 65)

    import numpy as np
    for name, vals in [
        ('Citation Density', densities),
        ('Sentence Count', sentence_counts),
        ('Quoted Phrases', quote_counts),
    ]:
        arr = np.array(vals)
        print(f"{name:<30} {arr.mean():>8.3f} {np.median(arr):>8.3f} {arr.min():>8.3f} {arr.max():>8.3f}")

    return densities, sentence_counts, quote_counts


if __name__ == '__main__':
    evaluate_all()
