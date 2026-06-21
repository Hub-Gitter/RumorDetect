import json
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from peft import LoraConfig, get_peft_model


class FocalLoss(nn.Module):
    """Focal Loss for imbalanced classification.

    FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)

    When γ=0, this reduces to standard weighted cross-entropy.
    """

    def __init__(self, alpha=None, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.alpha, reduction="none")
        p_t = torch.exp(-ce_loss)
        focal_loss = (1 - p_t) ** self.gamma * ce_loss
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


DEFAULT_MODEL_NAME = "cardiffnlp/twitter-roberta-base"
DEFAULT_MAX_LENGTH = 128


def resolve_model_name(model_name: str = None) -> str:
    """Return the configured HuggingFace base model name."""
    return model_name or os.environ.get("RUMOR_BASE_MODEL", DEFAULT_MODEL_NAME)


def get_tokenizer(model_name: str = None):
    """Load the tokenizer for the configured base model."""
    return AutoTokenizer.from_pretrained(resolve_model_name(model_name), use_fast=True)


def build_classifier(hidden_size: int, dropout: float = 0.3) -> nn.Module:
    """Build the task classification head."""
    return nn.Sequential(
        nn.Linear(hidden_size, 256),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(256, 2),
    )


def get_hidden_size(model) -> int:
    """Read hidden size from a base or PEFT-wrapped encoder."""
    config = getattr(model, "config", None)
    if config is None and hasattr(model, "base_model"):
        config = getattr(model.base_model, "config", None)
    if config is None or not hasattr(config, "hidden_size"):
        raise ValueError("Cannot infer hidden_size from model config.")
    return config.hidden_size


def build_model(model_name: str = None, classifier_dropout: float = 0.3,
                tuning_mode: str = "lora"):
    """返回 (encoder, classifier)。tuning_mode 支持 lora 或 full。"""
    model_name = resolve_model_name(model_name)
    base_model = AutoModel.from_pretrained(model_name)

    if tuning_mode == "lora":
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["query", "value"],
            lora_dropout=0.1,
            bias="none",
        )
        model = get_peft_model(base_model, lora_config)
    elif tuning_mode == "full":
        model = base_model
    else:
        raise ValueError(f"Unsupported tuning_mode: {tuning_mode}")

    classifier = build_classifier(
        get_hidden_size(base_model), dropout=classifier_dropout
    )

    return model, classifier


def _adapter_base_model_name(adapter_path: str) -> str:
    config_path = os.path.join(adapter_path, "adapter_config.json")
    if not os.path.exists(config_path):
        return ""
    with open(config_path, encoding="utf-8") as f:
        config = json.load(f)
    return config.get("base_model_name_or_path", "")


def _assert_checkpoint_matches(adapter_path: str, model_name: str):
    adapter_base = _adapter_base_model_name(adapter_path)
    if adapter_base and adapter_base != model_name:
        raise ValueError(
            "Checkpoint/base model mismatch: "
            f"adapter was trained for '{adapter_base}', but current base model is "
            f"'{model_name}'. Retrain with src/train_final.py or src/train_loeo.py "
            "before running inference."
        )


def load_trained_model(device: str = "cpu", model_name: str = None,
                       adapter_path: str = "checkpoints/lora_adapter",
                       classifier_path: str = "checkpoints/classifier.pt",
                       tuning_mode: str = "lora",
                       full_model_path: str = None):
    """Load a trained encoder and classifier head for inference."""
    from peft import PeftModel

    model_name = resolve_model_name(model_name)
    if not os.path.exists(classifier_path):
        raise FileNotFoundError(
            f"Classifier checkpoint not found at '{classifier_path}'. "
            "Train the model first."
        )

    if tuning_mode == "lora":
        if not os.path.exists(adapter_path):
            raise FileNotFoundError(
                f"LoRA adapter not found at '{adapter_path}'. Train the model first."
            )
        _assert_checkpoint_matches(adapter_path, model_name)
        base_model = AutoModel.from_pretrained(model_name)
        model = PeftModel.from_pretrained(base_model, adapter_path)
        hidden_source = base_model
    elif tuning_mode == "full":
        if full_model_path is None:
            raise ValueError("full_model_path is required when tuning_mode='full'.")
        if not os.path.exists(full_model_path):
            raise FileNotFoundError(
                f"Full model checkpoint not found at '{full_model_path}'."
            )
        model = AutoModel.from_pretrained(full_model_path)
        hidden_source = model
    else:
        raise ValueError(f"Unsupported tuning_mode: {tuning_mode}")

    model.to(device)

    classifier = build_classifier(get_hidden_size(hidden_source))
    classifier.load_state_dict(torch.load(
        classifier_path, map_location=device, weights_only=True
    ))
    classifier.to(device)

    model.eval()
    classifier.eval()
    return model, classifier


def predict(model, classifier, tokenizer, text: str, device: str = "cpu",
            max_length: int = DEFAULT_MAX_LENGTH) -> dict:
    """输入原始 text，返回 {"label": int, "confidence": float}。"""
    model.eval()
    classifier.eval()

    encoded = tokenizer(
        text,
        padding='max_length',
        truncation=True,
        max_length=max_length,
        return_tensors='pt',
    )
    input_ids = encoded['input_ids'].to(device)
    attention_mask = encoded['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        cls_hidden = outputs.last_hidden_state[:, 0, :]
        logits = classifier(cls_hidden)
        probs = logits.softmax(dim=-1)
        label = probs.argmax(dim=-1).item()
        confidence = probs.max(dim=-1).values.item()
        prob_non_rumor = probs[0, 0].item()
        prob_rumor = probs[0, 1].item()

    return {
        "label": label,
        "confidence": confidence,
        "prob_non_rumor": prob_non_rumor,
        "prob_rumor": prob_rumor,
    }
