"""WordNet 同义词替换数据增强。"""

import random

import nltk
import numpy as np
import pandas as pd
import nlpaug.augmenter.word as naw

_NLTK_REQUIREMENTS = {
    "wordnet": ("corpora/wordnet", "corpora/wordnet.zip"),
    "omw-1.4": ("corpora/omw-1.4", "corpora/omw-1.4.zip"),
    "averaged_perceptron_tagger_eng": (
        "taggers/averaged_perceptron_tagger_eng",
        "taggers/averaged_perceptron_tagger_eng.zip",
    ),
}


def _has_nltk_resource(paths: tuple[str, ...]) -> bool:
    for path in paths:
        try:
            nltk.data.find(path)
            return True
        except LookupError:
            continue
    return False


def ensure_nltk_resources():
    """Fail loudly if WordNet augmentation dependencies are unavailable."""
    missing = [
        name for name, paths in _NLTK_REQUIREMENTS.items()
        if not _has_nltk_resource(paths)
    ]
    if missing:
        packages = " ".join(missing)
        raise RuntimeError(
            "Missing NLTK resources for WordNet augmentation: "
            f"{', '.join(missing)}. Install them with: "
            f"python -m nltk.downloader {packages}"
        )


def augment_dataset(df: pd.DataFrame, random_seed: int = 42) -> pd.DataFrame:
    """对少数类样本做 WordNet 同义词替换增强。

    规则：事件内某个类别的样本数 <=5 条 → 增强 10 倍；<=15 条 → 增强 5 倍。
    替换率 20%，每条生成唯一增强文本。
    """
    ensure_nltk_resources()
    random.seed(random_seed)
    np.random.seed(random_seed)

    syn_aug = naw.SynonymAug(aug_src='wordnet', aug_p=0.2)
    df = df.copy()
    df['source'] = 'original'

    augmented_rows = []
    seen_texts = set(df['text'].tolist())

    for event_id in df['event'].unique():
        event_mask = df['event'] == event_id
        for label_val in [0, 1]:
            subset = df[event_mask & (df['label'] == label_val)]
            count = len(subset)

            if count <= 5:
                n_aug = 10
            elif count <= 15:
                n_aug = 5
            else:
                continue

            for _, row in subset.iterrows():
                text = str(row['text'])
                for _ in range(n_aug):
                    aug_text = syn_aug.augment(text)
                    if isinstance(aug_text, list):
                        aug_text = aug_text[0]
                    if aug_text and aug_text != text and aug_text not in seen_texts:
                        seen_texts.add(aug_text)
                        new_row = row.to_dict()
                        new_row['text'] = aug_text
                        new_row['source'] = 'augmented'
                        augmented_rows.append(new_row)

    if augmented_rows:
        df = pd.concat([df, pd.DataFrame(augmented_rows)], ignore_index=True)

    return df
