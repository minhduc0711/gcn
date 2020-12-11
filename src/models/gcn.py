import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from dgl.nn import GraphConv
import dgl.function as fn

from src.eval import compute_metrics


class GCN(pl.LightningModule):
    def __init__(
        self, in_feats, hidden_feats, num_classes, num_hidden_layers,
        learning_rate=1e-2,
        class_weights=None,
        dropout_proba=None,
        weight_decay=1e-3
    ):
        super(GCN, self).__init__()

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.num_classes = num_classes

        self.layers = nn.ModuleList()
        for i in range(num_hidden_layers + 1):
            if i == 0:
                self.layers.append(GraphConv(in_feats, hidden_feats))
            elif i == num_hidden_layers:
                self.layers.append(GraphConv(hidden_feats, num_classes))
            else:
                self.layers.append(GraphConv(hidden_feats, hidden_feats))
        if dropout_proba is not None:
            self.dropout = nn.Dropout(p=dropout_proba)
        else:
            self.dropout = None
        self.loss_fn = nn.CrossEntropyLoss(weight=class_weights)

        self.save_hyperparameters()

    def configure_optimizers(self):
        param_list = []
        # apply l2 regularization only to the first GCN layer
        # for i, params in enumerate(self.parameters()):
        #     param_list.append({
        #         "params": params,
        #         "weight_decay": self.weight_decay if i < 2 else 0
        #     })
        # opt = torch.optim.Adam(param_list, lr=self.learning_rate)
        opt = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return opt
        # sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        #     opt, "min", patience=5, factor=0.1
        # )
        # return {
        #     "optimizer": opt,
        #     "lr_scheduler": sched,
        #     "monitor": "train/loss"
        # }

    def forward(self, g, x):
        for i, layer in enumerate(self.layers):
            if self.dropout is not None and i != 0:
                x = self.dropout(x)
            x = layer(g, x)
            if i != len(self.layers) - 1:
                x = F.relu(x)
        return x

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
