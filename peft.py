"""
PEFT (LoRA) training script aligned with the LoRA section in main.ipynb.

This script:
1) Loads cleaned CSV splits (train/val/test)
2) Builds mT5 tokenized datasets
3) Wraps MT5ForSequenceClassification with LoRA (PEFT)
4) Trains/evaluates with weighted cross-entropy
5) Saves best LoRA adapter + tokenizer + metrics
"""

import json
import os
import random
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.utils.class_weight import compute_class_weight
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoTokenizer,
    MT5ForSequenceClassification,
    get_linear_schedule_with_warmup,
)

from peft import LoraConfig, TaskType, get_peft_model


@dataclass
class Config:
    train_path: str = "tamilmix_train.csv"
    val_path: str = "tamilmix_val.csv"
    test_path: str = "tamilmix_test.csv"
    model_name: str = "google/mt5-small"
    num_labels: int = 5
    max_length: int = 128
    batch_size: int = 16
    num_epochs: int = 5
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    grad_clip: float = 1.0
    output_dir: str = "mt5_lora"
    results_dir: str = "results_lora"
    seed: int = 42


LABEL_NAMES = ["Positive", "Negative", "Mixed_feelings", "unknown_state", "not-Tamil"]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class TamilSentimentDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, tokenizer, max_length: int):
        self.texts = dataframe["text_clean"].tolist()
        self.labels = dataframe["label_int"].tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def load_splits(cfg: Config):
    train_df = pd.read_csv(cfg.train_path)
    val_df = pd.read_csv(cfg.val_path)
    test_df = pd.read_csv(cfg.test_path)

    for df in (train_df, val_df, test_df):
        df["text_clean"] = df["text_clean"].fillna("").astype(str)
        df["label_int"] = df["label"].astype(int)

    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


def build_class_weights(train_df: pd.DataFrame, num_labels: int, device: torch.device):
    train_labels = train_df["label_int"].values
    present_classes = np.unique(train_labels)
    computed_weights = compute_class_weight(
        class_weight="balanced",
        classes=present_classes,
        y=train_labels,
    )
    class_weights = np.ones(num_labels, dtype=np.float32)
    for cls, weight in zip(present_classes, computed_weights):
        class_weights[cls] = weight
    return torch.tensor(class_weights, dtype=torch.float).to(device)


def train_epoch(model, loader, optimizer, scheduler, loss_fn, device, grad_clip):
    model.train()
    total_loss = 0.0
    all_preds, all_labels = [], []

    for step, batch in enumerate(loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        loss = loss_fn(logits, labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        preds = torch.argmax(logits, dim=-1).detach().cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.detach().cpu().numpy())

        if (step + 1) % 100 == 0:
            print(f"    Step {step + 1}/{len(loader)} - loss: {loss.item():.4f}")

    avg_loss = total_loss / max(1, len(loader))
    acc = accuracy_score(all_labels, all_preds)
    return avg_loss, acc


def evaluate(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            loss = loss_fn(logits, labels)

            total_loss += loss.item()
            preds = torch.argmax(logits, dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / max(1, len(loader))
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
    return avg_loss, acc, f1, all_preds, all_labels


def main():
    cfg = Config()
    os.makedirs(cfg.output_dir, exist_ok=True)
    os.makedirs(cfg.results_dir, exist_ok=True)

    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading splits...")
    train_df, val_df, test_df = load_splits(cfg)
    print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    print(f"Loading tokenizer: {cfg.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=False)

    train_dataset = TamilSentimentDataset(train_df, tokenizer, cfg.max_length)
    val_dataset = TamilSentimentDataset(val_df, tokenizer, cfg.max_length)
    test_dataset = TamilSentimentDataset(test_df, tokenizer, cfg.max_length)

    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=0)

    class_weights_tensor = build_class_weights(train_df, cfg.num_labels, device)

    print(f"Loading base model: {cfg.model_name}")
    base_model = MT5ForSequenceClassification.from_pretrained(
        cfg.model_name,
        num_labels=cfg.num_labels,
    )

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q", "v"],
    )

    model = get_peft_model(base_model, lora_config).to(device)
    model.print_trainable_parameters()

    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )
    total_steps = len(train_loader) * cfg.num_epochs
    warmup_steps = int(total_steps * cfg.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    loss_fn = nn.CrossEntropyLoss(weight=class_weights_tensor)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "val_f1": []}
    best_val_f1 = 0.0
    best_epoch = 0

    print("=" * 60)
    print("TRAINING - mT5 LoRA (PEFT)")
    print("=" * 60)

    for epoch in range(1, cfg.num_epochs + 1):
        print(f"\nEpoch {epoch}/{cfg.num_epochs}")
        print("-" * 40)
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, scheduler, loss_fn, device, cfg.grad_clip
        )
        val_loss, val_acc, val_f1, _, _ = evaluate(model, val_loader, loss_fn, device)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)

        print(f"  Train - Loss: {train_loss:.4f}  Acc: {train_acc:.4f}")
        print(f"  Val   - Loss: {val_loss:.4f}  Acc: {val_acc:.4f}  F1: {val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            model.save_pretrained(cfg.output_dir)
            tokenizer.save_pretrained(cfg.output_dir)
            print(f"  Saved best LoRA adapter (val F1 = {best_val_f1:.4f})")

    print(f"\nBest epoch: {best_epoch} | Best val F1: {best_val_f1:.4f}")

    test_loss, test_acc, test_f1, test_preds, test_labels = evaluate(
        model, test_loader, loss_fn, device
    )
    print("\n" + "=" * 60)
    print("TEST RESULTS - mT5 LoRA")
    print("=" * 60)
    print(f"Loss: {test_loss:.4f}")
    print(f"Accuracy: {test_acc:.4f}")
    print(f"F1 (weighted): {test_f1:.4f}")
    print(classification_report(test_labels, test_preds, target_names=LABEL_NAMES, zero_division=0))

    metrics = {
        "best_epoch": best_epoch,
        "best_val_f1": best_val_f1,
        "test_loss": test_loss,
        "test_accuracy": test_acc,
        "test_f1_weighted": test_f1,
        "history": history,
    }
    metrics_path = os.path.join(cfg.results_dir, "metrics_lora.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics: {metrics_path}")


if __name__ == "__main__":
    main()
