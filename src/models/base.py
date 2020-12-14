from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from src.eval import compute_metrics


class BaseGNN(pl.LightningModule, ABC):
    def __init__(self, lr=1e-3, class_weights=None):
        super(BaseGNN, self).__init__()
        self.save_hyperparameters()

        self.lr = lr
        self.loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.lr)
        return opt

    @abstractmethod
    def forward(self, g, x):
        pass

    def forward_pass(self, g, feats, labels, node_mask):
        logits = self(g, feats)
        logits = logits[node_mask]
        labels = labels[node_mask]

        loss = self.loss_fn(logits, labels)
        proba_pred = F.softmax(logits, dim=1)
        metrics = compute_metrics(proba_pred, labels)

        return proba_pred, loss, metrics

    def training_step(self, batch, batch_idx):
        proba_pred, loss, metrics = self.forward_pass(
            batch["g"], batch["feats"], batch["labels"], batch["node_mask"]
        )
        self.log("train/loss", loss)
        for k in metrics.keys():
            self.log(f"train/{k}", metrics[k])

        return loss

    def validation_step(self, batch, batch_idx):
        proba_pred, loss, metrics = self.forward_pass(
            batch["g"], batch["feats"], batch["labels"], batch["node_mask"]
        )
        self.log("val/loss", loss)
        for k in metrics.keys():
            self.log(f"val/{k}", metrics[k])

    def test_step(self, batch, batch_idx):
        proba_pred, loss, metrics = self.forward_pass(
            batch["g"], batch["feats"], batch["labels"], batch["node_mask"]
        )
        self.log("test/loss", loss)
        for k in metrics.keys():
            self.log(f"test/{k}", metrics[k])
