import torch
import torch.nn as nn
import torch.nn.functional as F

from dgl.nn import GraphConv
from src.models.base import BaseGNN


class GCN(BaseGNN):
    def __init__(
        self, in_feats, hidden_feats, num_classes, num_hidden_layers,
        dropout_proba=None,
        weight_decay=1e-3,
        **kwargs
    ):
        super(GCN, self).__init__(**kwargs)

        self.weight_decay = weight_decay

        self.layers = nn.ModuleList()
        last_input_dim = in_feats
        for i in range(num_hidden_layers):
            self.layers.append(GraphConv(last_input_dim, hidden_feats, activation=F.relu))
            last_input_dim = hidden_feats
        self.layers.append(GraphConv(last_input_dim, num_classes))

        if dropout_proba is not None:
            self.dropout = nn.Dropout(p=dropout_proba)
        else:
            self.dropout = None

    def configure_optimizers(self):
        # param_list = []
        # apply l2 regularization only to the first GCN layer
        # for i, params in enumerate(self.parameters()):
        #     param_list.append({
        #         "params": params,
        #         "weight_decay": self.weight_decay if i < 2 else 0
        #     })
        # opt = torch.optim.Adam(param_list, lr=self.learning_rate)
        opt = torch.optim.Adam(self.parameters(), lr=self.lr)
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
        return x
