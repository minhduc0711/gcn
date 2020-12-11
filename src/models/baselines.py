import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from src.eval import compute_metrics

class DNN(pl.LightningModule):
    def __init__(self, in_feats, hidden_feats, num_classes, num_hidden_layers,
                 learning_rate=1e-3,
                 class_weights=None):
        super(DNN, self).__init__()

        self.learning_rate = learning_rate
        self.num_classes = num_classes

        last_input_dim = in_feats
        layers = []
        for _ in range(num_hidden_layers):
            layers.extend([
                nn.Linear(last_input_dim, hidden_feats),
                nn.ReLU()
            ])
            last_input_dim = hidden_feats
        layers.append(nn.Linear(last_input_dim, num_classes))
        self.layers = nn.Sequential(*layers)

        self.loss_fn = nn.CrossEntropyLoss(weight=class_weights)
        self.save_hyperparameters()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def forward(self, x):
        return self.layers(x)

    def training_step(self, batch, batch_idx):
        mask = batch["node_mask"]
        x = batch["feats"][mask]
        y = batch["labels"][mask]

        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.log("train/loss", loss)

        proba_pred = F.softmax(logits, dim=1)
        metrics = compute_metrics(proba_pred, y)
        for k in metrics.keys():
            self.log(f"train/{k}", metrics[k])

        return loss

    def validation_step(self, batch, batch_idx):
        mask = batch["node_mask"]
        x = batch["feats"][mask]
        y = batch["labels"][mask]

        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.log("val/loss", loss)

        proba_pred = F.softmax(logits, dim=1)
        metrics = compute_metrics(proba_pred, y)
        for k in metrics.keys():
            self.log(f"val/{k}", metrics[k])

        return loss

    def test_step(self, batch, batch_idx):
        mask = batch["node_mask"]
        x = batch["feats"][mask]
        y = batch["labels"][mask]

        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.log("test/loss", loss)

        proba_pred = F.softmax(logits, dim=1)
        metrics = compute_metrics(proba_pred, y)
        for k in metrics.keys():
            self.log(f"test/{k}", metrics[k])

        return loss
