from dgl.nn import SAGEConv, GraphConv
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from src.eval import compute_metrics
from .base import BaseGNN


class GraphSAGE(BaseGNN):
    def __init__(self, in_feats, hidden_feats, num_classes, num_hidden_layers,
                 agg="mean",
                 **kwargs):
        super().__init__(**kwargs)
        self.save_hyperparameters()

        self.layers = nn.ModuleList()
        last_input_dim = in_feats
        for i in range(num_hidden_layers):
            self.layers.append(SAGEConv(
                in_feats=last_input_dim, out_feats=hidden_feats, aggregator_type=agg,
                activation=F.relu
            ))
            last_input_dim = hidden_feats
        self.layers.append(SAGEConv(
            in_feats=last_input_dim, out_feats=num_classes, aggregator_type=agg
        ))

    def forward(self, g, x):
        for layer in self.layers:
            x = layer(g, x)
        return x
