# wine_review/evalviz/plots.py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def plot_cm(y_true, y_pred, labels, normalize=True, max_labels=25, figsize=(8,6)):
    """
    Plot a confusion matrix heatmap.

    Parameters
    ----------
    y_true : array-like
        Ground truth labels
    y_pred : array-like
        Predicted labels
    labels : list
        Label names (in order of indices)
    normalize : bool
        Whether to normalize row-wise
    max_labels : int
        Limit number of labels displayed (for readability)
    figsize : tuple
        Size of the plot
    """
    if len(labels) > max_labels:
        labels = labels[:max_labels]
        keep_ids = list(range(len(labels)))
        mask = np.isin(y_true, keep_ids) & np.isin(y_pred, keep_ids)
        y_true, y_pred = y_true[mask], y_pred[mask]

    cm = confusion_matrix(y_true, y_pred, labels=range(len(labels)))
    if normalize:
        cm = cm.astype("float") / (cm.sum(axis=1, keepdims=True) + 1e-12)

    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="YlGnBu",
                xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Normalized Confusion Matrix")
    plt.tight_layout()
    plt.show()
