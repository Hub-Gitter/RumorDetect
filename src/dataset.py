import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase
from src.preprocess import preprocess


class RumorDataset(Dataset):
    """谣言检测 Dataset。输入 text，输出 tokenized 张量 + label。"""

    def __init__(self, texts: list, labels: list,
                 tokenizer: PreTrainedTokenizerBase,
                 max_length: int = 128):
        self.texts = [preprocess(t) for t in texts]
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoded = self.tokenizer(
            self.texts[idx],
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt',
        )
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }
