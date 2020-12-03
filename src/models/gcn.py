import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

import dgl
from dgl.nn import GraphConv
import dgl.function as fn


class MyGraphConv(nn.Module):
    # message passing functions
    gcn_msg = fn.copy_src(src="h", out="m")
    gcn_reduce = fn.mean(msg="m", out="h")

    def __init__(self, in_feats, out_feats,
                 norm="right"):
        super(MyGraphConv, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)
        self.norm = norm

    def forward(self, g, x):
        # TODO: implement symmetric norm
        with g.local_scope():
            g.ndata["h"] = x
            g.update_all(self.gcn_msg, self.gcn_reduce)
            h = g.ndata["h"]
            return self.linear(h)


class GCN(pl.LightningModule):
    def __init__(
        self, in_feats, hidden_feats, num_classes, num_hidden_layers, dropout_proba=None,
        weight_decay=1e-3
    ):
        super(GCN, self).__init__()

        self.weight_decay = weight_decay

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
        self.loss_fn = nn.CrossEntropyLoss()

        self.save_hyperparameters()

    def configure_optimizers(self):
        param_list = []
        # apply l2 regularization only to the first GCN layer
        for i, params in enumerate(self.parameters()):
            param_list.append({
                "params": params,
                "weight_decay": self.weight_decay if i < 2 else 0
            })
        opt = torch.optim.Adam(param_list, lr=1e-2)
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
        class_pred = torch.argmax(logits, dim=1)
        acc = (class_pred == labels).sum() * 1.0 / labels.shape[0]
        return loss, acc

    def training_step(self, batch, batch_idx):
        loss, acc = self.forward_pass(
            batch["g"], batch["feats"], batch["labels"], batch["node_mask"]
        )
        self.log("train/loss", loss)
        self.log("train/accuracy", acc)

        return loss

    def validation_step(self, batch, batch_idx):
        loss, acc = self.forward_pass(
            batch["g"], batch["feats"], batch["labels"], batch["node_mask"]
        )
        self.log("val/loss", loss)
        self.log("val/accuracy", acc)

    def test_step(self, batch, batch_idx):
        loss, acc = self.forward_pass(
            batch["g"], batch["feats"], batch["labels"], batch["node_mask"]
        )
        self.log("test/loss", loss)
        self.log("test/accuracy", acc)
