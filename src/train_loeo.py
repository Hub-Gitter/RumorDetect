"""Leave-One-Event-Out 训练脚本。7 轮，每轮留一个 event 做验证。"""

import os
import sys
import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.model import DEFAULT_MODEL_NAME, build_model, get_tokenizer, FocalLoss
from src.dataset import RumorDataset

# 数据增强模块在 P2 实现，P0 阶段用 identity fallback
try:
    from src.data_augment import augment_dataset
except ImportError:
    def augment_dataset(df):
        return df


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_epoch(model, classifier, loader, optimizer, criterion, device):
    model.train()
    classifier.train()
    total_loss = 0.0
    for batch in loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        cls_hidden = outputs.last_hidden_state[:, 0, :]
        logits = classifier(cls_hidden)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * input_ids.size(0)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, classifier, loader, device):
    model.eval()
    classifier.eval()
    all_preds, all_labels = [], []
    for batch in loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        cls_hidden = outputs.last_hidden_state[:, 0, :]
        logits = classifier(cls_hidden)
        preds = logits.argmax(dim=-1)

        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    acc = accuracy_score(all_labels, all_preds)
    # handle edge case: only one class in val set (e.g. Event 2 all-rumor)
    unique_labels = set(all_labels)
    if len(unique_labels) < 2:
        # Macro F1 is ill-defined for single-class; report as NaN for reference
        macro_f1 = float('nan')
        f1_per_class = {}
        if 0 in unique_labels:
            f1_per_class[0] = f1_score(all_labels, all_preds, pos_label=0)
            f1_per_class[1] = float('nan')
        else:
            f1_per_class[0] = float('nan')
            f1_per_class[1] = f1_score(all_labels, all_preds, pos_label=1)
    else:
        macro_f1 = f1_score(all_labels, all_preds, average='macro')
        f1_per_class = {
            0: f1_score(all_labels, all_preds, pos_label=0),
            1: f1_score(all_labels, all_preds, pos_label=1),
        }
    return acc, macro_f1, f1_per_class


def main(model_name: str = DEFAULT_MODEL_NAME, max_epochs: int = 5,
         batch_size: int = 16, use_augment: bool = True, seed: int = 42,
         focal_loss: bool = True, focal_gamma: float = 2.0):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Base model: {model_name}")
    print(f"Seed: {seed}")
    print(f"Focal loss: {focal_loss}" + (f" (gamma={focal_gamma})" if focal_loss else ""))
    set_seed(seed)

    df = pd.read_csv('data/train.csv')
    events = sorted(df['event'].unique())
    tokenizer = get_tokenizer(model_name)

    results = []

    for val_event in events:
        print(f"\n{'='*50}")
        print(f"Round {len(results)+1}/{len(events)}: val_event={val_event}")
        print(f"{'='*50}")

        train_df = df[df['event'] != val_event].copy()
        val_df = df[df['event'] == val_event].copy()

        # 数据增强仅在训练 fold 内
        if use_augment:
            train_df = augment_dataset(train_df, random_seed=seed + len(results))

        train_ds = RumorDataset(
            train_df['text'].tolist(), train_df['label'].tolist(), tokenizer
        )
        val_ds = RumorDataset(
            val_df['text'].tolist(), val_df['label'].tolist(), tokenizer
        )

        round_seed = seed + len(results)
        set_seed(round_seed)
        generator = torch.Generator()
        generator.manual_seed(round_seed)
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True, generator=generator
        )
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

        # 每轮从头初始化
        model, classifier = build_model(model_name)
        model.to(device)
        classifier.to(device)

        optimizer = torch.optim.AdamW(
            list(model.parameters()) + list(classifier.parameters()),
            lr=5e-4,
            weight_decay=0.01,
        )

        # class weight: sklearn 风格的 balanced 权重
        class_counts = train_df['label'].value_counts().to_dict()
        n_samples = len(train_df)
        n_classes = 2
        weight_0 = n_samples / (n_classes * class_counts.get(0, 1))
        weight_1 = n_samples / (n_classes * class_counts.get(1, 1))
        class_weights = torch.tensor([weight_0, weight_1], device=device)

        if focal_loss:
            criterion = FocalLoss(alpha=class_weights, gamma=focal_gamma)
        else:
            criterion = nn.CrossEntropyLoss(weight=class_weights)

        best_val_loss = float('inf')
        best_state = None
        patience_counter = 0
        best_epoch = 0
        for epoch in range(1, max_epochs + 1):
            train_loss = train_one_epoch(
                model, classifier, train_loader, optimizer, criterion, device
            )

            # compute val loss for early stopping
            model.eval()
            classifier.eval()
            val_loss_total = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    labels = batch['label'].to(device)
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    cls_hidden = outputs.last_hidden_state[:, 0, :]
                    logits = classifier(cls_hidden)
                    loss = criterion(logits, labels)
                    val_loss_total += loss.item() * input_ids.size(0)
            val_loss = val_loss_total / len(val_ds)

            acc, macro_f1, f1_per_class = evaluate(
                model, classifier, val_loader, device
            )

            f1_0 = f1_per_class.get(0, float('nan'))
            f1_1 = f1_per_class.get(1, float('nan'))

            print(f"  Epoch {epoch}: train_loss={train_loss:.4f}, "
                  f"val_loss={val_loss:.4f}, acc={acc:.4f}, "
                  f"macro_f1={macro_f1:.4f}, f1_0={f1_0:.4f}, f1_1={f1_1:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {
                    'model': model.state_dict(),
                    'classifier': classifier.state_dict(),
                }
                best_epoch = epoch
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= 2:
                    print(f"  Early stopping at epoch {epoch}")
                    break

        # 加载最佳权重，计算最终指标
        model.load_state_dict(best_state['model'])
        classifier.load_state_dict(best_state['classifier'])
        final_acc, final_f1, final_f1_per = evaluate(
            model, classifier, val_loader, device
        )

        results.append({
            'round': len(results) + 1,
            'val_event': val_event,
            'train_size': len(train_df),
            'val_size': len(val_df),
            'accuracy': round(final_acc, 4),
            'macro_f1': round(final_f1, 4) if not np.isnan(final_f1) else 'nan',
            'f1_class0': round(final_f1_per.get(0, float('nan')), 4),
            'f1_class1': round(final_f1_per.get(1, float('nan')), 4),
            'best_epoch': best_epoch,
        })

        print(f"  Final (best epoch {best_epoch}): "
              f"acc={final_acc:.4f}, macro_f1={final_f1:.4f}")

    # 汇总输出
    results_df = pd.DataFrame(results)
    os.makedirs('outputs', exist_ok=True)
    results_df.to_csv('outputs/loeo_results.csv', index=False)

    print(f"\n{'='*50}")
    print("LOEO Results Summary")
    print(f"{'='*50}")
    print(results_df.to_string(index=False))

    avg_acc = results_df['accuracy'].mean()
    valid_f1 = results_df['macro_f1']
    valid_f1 = valid_f1[valid_f1.apply(lambda x: isinstance(x, (int, float)) and not np.isnan(x))]
    avg_f1 = valid_f1.mean() if len(valid_f1) > 0 else float('nan')
    print(f"\nAverage: acc={avg_acc:.4f}, macro_f1={avg_f1:.4f} "
          f"(over {len(valid_f1)}/{len(events)} valid rounds)")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Leave-One-Event-Out training")
    parser.add_argument('--model-name', default=DEFAULT_MODEL_NAME,
                        help='HuggingFace base model name')
    parser.add_argument('--epochs', type=int, default=5,
                        help='Maximum epochs per LOEO fold')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Training batch size')
    parser.add_argument('--no-augment', action='store_true',
                        help='Disable WordNet synonym augmentation')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for augmentation, initialization, and shuffle')
    parser.add_argument('--no-focal-loss', action='store_false', dest='focal_loss',
                        default=True,
                        help='Disable Focal Loss, use standard CrossEntropyLoss')
    parser.add_argument('--focal-gamma', type=float, default=2.0,
                        help='Gamma parameter for Focal Loss (default=2.0)')
    args = parser.parse_args()
    main(
        model_name=args.model_name,
        max_epochs=args.epochs,
        batch_size=args.batch_size,
        use_augment=not args.no_augment,
        seed=args.seed,
        focal_loss=args.focal_loss,
        focal_gamma=args.focal_gamma,
    )
