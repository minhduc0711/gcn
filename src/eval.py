import torch
from sklearn import metrics
from pytorch_lightning.metrics.functional import \
    accuracy, average_precision, auroc, precision, recall, f1


def compute_metrics(proba_pos, y_true, threshold=0.5):
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(proba_pos, torch.Tensor):
        proba_pos = proba_pos.detach().cpu().numpy()
    if proba_pos.ndim > 1:
        proba_pos = proba_pos[:, 1]
    y_pred = proba_pos > threshold
    p, r, _ = metrics.precision_recall_curve(y_true, proba_pos)
    return {
        "precision": metrics.precision_score(y_true, y_pred),
        "recall": metrics.recall_score(y_true, y_pred, average='binary'),
        "f1": metrics.f1_score(y_true, y_pred),
        "auROC": metrics.roc_auc_score(y_true, proba_pos),
        "auPR": metrics.auc(r, p),
        "avg precision": metrics.average_precision_score(y_true, proba_pos),
    }


def compute_torch_metrics(y_pred, y_true,
                    threshold=0.5):
    if self.num_classes == 2:
        y_pred = y_pred[:, 1]
    return {
        "acc": accuracy(y_pred, y_true),
        "avg_precision": average_precision(y_pred, y_true),
        "auroc": auroc(y_pred, y_true)
    }
