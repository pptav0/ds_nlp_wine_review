# modeling/sentiment.py
from typing import List, Dict
import os, json, numpy as np, torch, pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def points_to_sentiment(p: float, neu_th: float, neg_th: float) -> str:
    """Map rating to neg/neu/pos using thresholds decided upstream (e.g., quantiles)."""
    if p <= neg_th: return "neg"
    elif p <= neu_th: return "neu"
    else: return "pos"

class SingleLabelTextDS(torch.utils.data.Dataset):
    """Dataset for single-label (multi-class) classification."""
    def __init__(self, texts: List[str], label_ids: List[int], tokenizer, max_len: int = 256):
        self.texts = [str(t) for t in texts]
        self.labels = [int(i) for i in label_ids]
        self.tokenizer = tokenizer
        self.max_len = max_len
    def __len__(self): return len(self.texts)
    def __getitem__(self, idx):
        enc = self.tokenizer(self.texts[idx], truncation=True, padding="max_length", max_length=self.max_len, return_tensors="pt")
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

def sent_metrics(eval_pred) -> Dict[str, float]:
    logits, labels = eval_pred
    y_pred = logits.argmax(axis=-1)
    acc = accuracy_score(labels, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(labels, y_pred, average="macro", zero_division=0)
    return {"accuracy": acc, "macro_precision": prec, "macro_recall": rec, "macro_f1": f1}

def predict_sentiment(texts: List[str], model_path: str, device: str | None = None, max_len: int = 256):
    """Return label + softmax probabilities for each text from a saved classifier dir."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(model_path)
    mdl = AutoModelForSequenceClassification.from_pretrained(model_path).to(device).eval()

    enc = tok(texts, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
    with torch.inference_mode():
        probs = torch.softmax(mdl(**{k: v.to(device) for k, v in enc.items()}).logits, dim=-1).cpu().numpy()

    with open(os.path.join(model_path, "labels.json")) as f:
        labels = json.load(f)["labels"]

    preds = probs.argmax(axis=-1)
    out = []
    for i, p in enumerate(preds):
        out.append({"label": labels[p], "probs": {labels[j]: float(probs[i, j]) for j in range(len(labels))}})
    return out
