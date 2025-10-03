# modeling/flavor.py
import os, json, torch, numpy as np
from typing import Optional, List
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class MultiLabelTextDS(torch.utils.data.Dataset):
    """Dataset for multi-label text classification."""
    def __init__(self, texts: List[str], labels: np.ndarray, tokenizer, max_len: int = 256, sample_weight: Optional[np.ndarray] = None):
        self.texts = [str(t) for t in texts]
        self.labels = labels.astype("float32")
        self.sample_weight = sample_weight.astype("float32") if sample_weight is not None else None
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self): return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(self.texts[idx], truncation=True, padding="max_length", max_length=self.max_len, return_tensors="pt")
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.float)
        if self.sample_weight is not None:
            item["sample_weight"] = torch.tensor(self.sample_weight[idx], dtype=torch.float)
        return item

def infer_probs(texts: List[str], tokenizer, model, max_len: int = 256, batch_size: int = 64) -> np.ndarray:
    """Batched sigmoid probabilities for a multi-label head."""
    model.eval()
    all_probs = []
    device = next(model.parameters()).device
    with torch.inference_mode():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            enc = tokenizer(batch, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            logits = model(**enc).logits
            all_probs.append(torch.sigmoid(logits).cpu().numpy())
    return np.vstack(all_probs)


class FlavorTagger:
	"""
	A class for tagging wine reviews with flavor profiles using a pre-trained transformer model.

	Attributes:
	----------
	tokenizer : transformers.PreTrainedTokenizer
		Tokenizer for encoding input text.
	model : transformers.PreTrainedModel
		Pre-trained transformer model for sequence classification.
	device : str
		The device used for computation ('cuda', 'mps', or 'cpu').
	labels : list
		List of flavor labels used for classification.
	thresholds : float or dict
		Threshold(s) for determining classification probabilities.

	Methods:
	-------
	__init__(model_dir: str, device: str | None = None):
		Initializes the FlavorTagger with the specified model directory and device.
	predict(texts, max_length: int = 256):
		Predicts flavor tags for the given input texts.
	"""
	def __init__(self, model_dir: str, device: Optional[str] = None):
		self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
		self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
		self.model.eval()

		if device is None:
			if torch.cuda.is_available():
				device = "cuda"
			elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
				device = "mps"
			else:
				device = "cpu"

		self.device = device
		self.model.to(self.device)

		with open(os.path.join(model_dir, "labels.json")) as f:
			meta = json.load(f)

		self.labels = meta["labels"]

		# support either single or per-label thresholds
		self.thresholds = meta.get("thresholds") or meta.get("threshold") or 0.5

	def predict(self, texts, max_length: int = 256):
		enc = self.tokenizer(texts,
			padding=True, truncation=True,
			max_length=max_length, return_tensors='pt')
		enc = {k: v.to(self.device) for k,v in enc.items()}

		with torch.inference_mode():
			probs = torch.sigmoid(self.model(**enc).logits).cpu().numpy()

		results = []
		if isinstance(self.thresholds, dict):
			thr_vec = np.array([self.thresholds.get(l, 0.5) for l in self.labels])
		else:
			thr_vec = float(self.thresholds)

		for p in probs:
			idx = np.where(p >= thr_vec)[0] if np.ndim(thr_vec)>0 else np.where(p >= thr_vec)[0]
			tags = [self.labels[i] for i in idx]
			results.append({
				"tags": tags,
				"probs": {self.labels[i]: float(p[i]) for i in range(len(self.labels))}
			})

		return results
