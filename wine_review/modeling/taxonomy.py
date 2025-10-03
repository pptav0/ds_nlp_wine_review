# wine_review/modeling/taxonomy.py
"""
Auxiliary classifiers for wine taxonomy:
- Variety predictor
- Country predictor
"""

import os, json
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from typing import List, Dict


def taxo_metrics(eval_pred):
    """Compute accuracy, precision, recall, f1 for taxonomy classifiers."""
    logits, labels = eval_pred
    y_pred = logits.argmax(axis=-1)
    acc = accuracy_score(labels, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        labels, y_pred, average="macro", zero_division=0
    )
    return {
        "accuracy": acc,
        "macro_precision": prec,
        "macro_recall": rec,
        "macro_f1": f1,
    }


def train_one_classifier(
    train_ds,
    val_ds,
    label_list: List[str],
    out_dir: str,
    base_model: str = "distilbert-base-uncased",
    epochs: int = 3,
    batch_size: int = 16,
    lr: float = 5e-5,
):
    """
    Train a single-label classifier (variety or country).
    Saves tokenizer, model, and labels.json to out_dir.
    """
    os.makedirs(out_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model, num_labels=len(label_list)
    )

    args = TrainingArguments(
        output_dir=out_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        weight_decay=0.01,
        logging_dir=os.path.join(out_dir, "logs"),
        logging_steps=50,
        save_total_limit=1,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=taxo_metrics,
        tokenizer=tokenizer,
    )

    trainer.train()

    # save artifacts
    trainer.save_model(out_dir)
    tokenizer.save_pretrained(out_dir)
    with open(os.path.join(out_dir, "labels.json"), "w") as f:
        json.dump({"labels": label_list}, f, indent=2)

    return trainer


def load_classifier(model_dir: str, device: str | None = None):
    """Reload trained taxonomy classifier bundle."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(device).eval()

    with open(os.path.join(model_dir, "labels.json")) as f:
        labels = json.load(f)["labels"]

    return {"model": model, "tokenizer": tokenizer, "labels": labels, "device": device}


def predict_cls(texts: List[str], bundle: Dict, max_len: int = 256):
    """Run inference and return predicted label + softmax probs."""
    tok = bundle["tokenizer"]
    mdl = bundle["model"]
    labels = bundle["labels"]
    device = bundle["device"]

    enc = tok(
        texts,
        padding=True,
        truncation=True,
        max_length=max_len,
        return_tensors="pt",
    )
    enc = {k: v.to(device) for k, v in enc.items()}

    with torch.inference_mode():
        probs = torch.softmax(mdl(**enc).logits, dim=-1).cpu().numpy()

    preds = probs.argmax(axis=-1)
    out = []
    for i, p in enumerate(preds):
        out.append(
            {
                "label": labels[p],
                "probs": {labels[j]: float(probs[i, j]) for j in range(len(labels))},
            }
        )
    return out
