"""全量数据训练最终模型。LOEO 验证通过后使用全部 2840 条数据训练。"""

import os
import sys
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.model import DEFAULT_MODEL_NAME, build_model, get_tokenizer, FocalLoss
from src.dataset import RumorDataset
from src.data_augment import augment_dataset


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main(model_name: str = DEFAULT_MODEL_NAME, num_epochs: int = 5,
         batch_size: int = 16, limit: int = None,
         use_augment: bool = True, output_dir: str = "checkpoints",
         seed: int = 42, learning_rate: float = 5e-4,
         weight_decay: float = 0.01, classifier_dropout: float = 0.3,
         use_class_weights: bool = True, tuning_mode: str = "lora",
         focal_loss: bool = True, focal_gamma: float = 2.0,
         train_csv: str = "data/train.csv",
         rdrop: bool = False, rdrop_alpha: float = 1.0):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Base model: {model_name}")
    print(f"Seed: {seed}")
    print(f"Learning rate: {learning_rate}")
    print(f"Weight decay: {weight_decay}")
    print(f"Classifier dropout: {classifier_dropout}")
    print(f"Class weights: {use_class_weights}")
    print(f"Focal loss: {focal_loss}" + (f" (gamma={focal_gamma})" if focal_loss else ""))
    print(f"R-Drop: {rdrop}" + (f" (alpha={rdrop_alpha})" if rdrop else ""))
    print(f"Tuning mode: {tuning_mode}")
    set_seed(seed)

    df = pd.read_csv(train_csv)
    if limit:
        df = df.sample(n=min(limit, len(df)), random_state=42).reset_index(drop=True)
        print(f"Using a training subset: {len(df)} samples")
    tokenizer = get_tokenizer(model_name)

    has_preaug = "source" in df.columns and (df["source"] == "llm_augmented").any()
    if use_augment and not has_preaug:
        print("Augmenting training data...")
        df = augment_dataset(df, random_seed=seed)
        print(f"  After augmentation: {len(df)} samples "
              f"(original: {(df['source']=='original').sum()}, "
              f"augmented: {(df['source']=='augmented').sum()})")
    elif has_preaug:
        print(f"Using pre-augmented data: {len(df)} samples "
              f"(original: {(df['source']=='original').sum()}, "
              f"LLM augmented: {(df['source']=='llm_augmented').sum()})")

    dataset = RumorDataset(
        df['text'].tolist(), df['label'].tolist(), tokenizer
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, generator=generator
    )

    model, classifier = build_model(
        model_name, classifier_dropout=classifier_dropout,
        tuning_mode=tuning_mode,
    )
    model.to(device)
    classifier.to(device)

    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(classifier.parameters()),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    class_counts = df['label'].value_counts().to_dict()
    n_samples = len(df)
    n_classes = 2
    if use_class_weights:
        weight_0 = n_samples / (n_classes * class_counts.get(0, 1))
        weight_1 = n_samples / (n_classes * class_counts.get(1, 1))
        class_weights = torch.tensor([weight_0, weight_1], device=device)
    else:
        class_weights = None

    if focal_loss:
        criterion = FocalLoss(alpha=class_weights, gamma=focal_gamma)
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights)

    print(f"\nTraining for {num_epochs} epochs on {len(df)} samples...")

    for epoch in range(1, num_epochs + 1):
        model.train()
        classifier.train()
        total_loss = 0.0

        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            optimizer.zero_grad()

            outputs1 = model(input_ids=input_ids, attention_mask=attention_mask)
            logits1 = classifier(outputs1.last_hidden_state[:, 0, :])

            if rdrop:
                outputs2 = model(input_ids=input_ids, attention_mask=attention_mask)
                logits2 = classifier(outputs2.last_hidden_state[:, 0, :])

                ce_loss = (criterion(logits1, labels) + criterion(logits2, labels)) / 2

                p = logits1.softmax(dim=-1)
                q = logits2.softmax(dim=-1)
                kl = (F.kl_div(p.log(), q, reduction='batchmean') +
                      F.kl_div(q.log(), p, reduction='batchmean')) / 2

                loss = ce_loss + rdrop_alpha * kl
            else:
                loss = criterion(logits1, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * input_ids.size(0)

        avg_loss = total_loss / len(dataset)
        print(f"  Epoch {epoch}/{num_epochs}: loss={avg_loss:.4f}")

    # 保存
    classifier_path = os.path.join(output_dir, 'classifier.pt')
    if tuning_mode == "lora":
        model_path = os.path.join(output_dir, 'lora_adapter')
    elif tuning_mode == "full":
        model_path = os.path.join(output_dir, 'full_model')
    else:
        raise ValueError(f"Unsupported tuning_mode: {tuning_mode}")
    os.makedirs(model_path, exist_ok=True)
    model.save_pretrained(model_path)
    torch.save(classifier.state_dict(), classifier_path)
    print(f"\nModel saved to {model_path}/ and {classifier_path}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Train final rumor classifier")
    parser.add_argument('--model-name', default=DEFAULT_MODEL_NAME,
                        help='HuggingFace base model name')
    parser.add_argument('--epochs', type=int, default=5,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Training batch size')
    parser.add_argument('--limit', type=int, default=None,
                        help='Optional subset size for smoke tests')
    parser.add_argument('--no-augment', action='store_true',
                        help='Disable WordNet synonym augmentation')
    parser.add_argument('--output-dir', default='checkpoints',
                        help='Directory for lora_adapter/ and classifier.pt')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--learning-rate', type=float, default=5e-4,
                        help='AdamW learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.01,
                        help='AdamW weight decay')
    parser.add_argument('--classifier-dropout', type=float, default=0.3,
                        help='Dropout in the classifier head')
    parser.add_argument('--no-class-weights', action='store_true',
                        help='Disable balanced class weights')
    parser.add_argument('--tuning-mode', choices=['lora', 'full'], default='lora',
                        help='Fine-tuning mode')
    parser.add_argument('--no-focal-loss', action='store_false', dest='focal_loss',
                        default=True,
                        help='Disable Focal Loss')
    parser.add_argument('--focal-gamma', type=float, default=2.0,
                        help='Focal Loss gamma')
    parser.add_argument('--train-csv', default='data/train.csv',
                        help='Training CSV path')
    parser.add_argument('--rdrop', action='store_true', dest='rdrop',
                        default=False,
                        help='Enable R-Drop consistency regularization')
    parser.add_argument('--rdrop-alpha', type=float, default=1.0,
                        help='R-Drop KL divergence weight')
    args = parser.parse_args()
    main(
        model_name=args.model_name,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        limit=args.limit,
        use_augment=not args.no_augment,
        output_dir=args.output_dir,
        seed=args.seed,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        classifier_dropout=args.classifier_dropout,
        use_class_weights=not args.no_class_weights,
        tuning_mode=args.tuning_mode,
        focal_loss=args.focal_loss,
        focal_gamma=args.focal_gamma,
        train_csv=args.train_csv,
        rdrop=args.rdrop,
        rdrop_alpha=args.rdrop_alpha,
    )
