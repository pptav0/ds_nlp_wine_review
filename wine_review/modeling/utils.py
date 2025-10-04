# modeling/utils.py
from transformers import Trainer
import torch
import torch.nn as nn
from typing import Optional

class WeightedBCETrainer(Trainer):
	"""Trainer with BCEWithLogits + pos_weight and optional per-sample weights."""
	def __init__(self, *args, pos_weight: Optional[torch.Tensor] = None, **kwargs):
		super().__init__(*args, **kwargs)
		self.pos_weight = pos_weight

	def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
		"""
        Compute the loss for the given model and inputs.

        Parameters:
        ----------
        model : torch.nn.Module
            The model being trained.
        inputs : dict
            The input data, including labels and optional sample weights.
        return_outputs : bool, optional
            Whether to return the model outputs along with the loss, by default False.

        Returns:
        -------
        torch.Tensor or tuple
            The computed loss, and optionally the model outputs.
        """
		labels = inputs.pop("labels")
		sample_w = inputs.pop("sample_weight", None)
		outputs = model(**inputs)
		logits = outputs.logits

		loss_fn = nn.BCEWithLogitsLoss(
			pos_weight=self.pos_weight.to(logits.device) if self.pos_weight is not None else None,
			reduction="none"
		)
		loss_mat = loss_fn(logits, labels)              # [batch, num_labels]
		loss_per_sample = loss_mat.mean(dim=1)          # [batch]
		loss = (loss_per_sample * sample_w.to(logits.device)).mean() if sample_w is not None else loss_per_sample.mean()

		return (loss, outputs) if return_outputs else loss
