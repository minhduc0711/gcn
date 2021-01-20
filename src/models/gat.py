from dgl.nn import GATConv
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from src.eval import compute_metrics
from .base import BaseGNN


class GAT(BaseGNN):
    def __init__(self, in_feats, hidden_feats, num_classes, num_hidden_layers,
                 num_heads=3,
                 feat_drop=0.0,
                 attn_drop=0.0,
                 **kwargs):
        super().__init__(**kwargs)
        self.save_hyperparameters()

        self.layers = nn.ModuleList()
        last_input_dim = in_feats
        for i in range(num_hidden_layers):
            self.layers.append(GATConv(
                in_feats=last_input_dim,
                out_feats=hidden_feats,
                num_heads=num_heads,
                feat_drop=feat_drop,
                attn_drop=attn_drop,
                activation=F.relu
            ))
            last_input_dim = hidden_feats * num_heads
        self.layers.append(GATConv(
            in_feats=last_input_dim,
            out_feats=num_classes,
            num_heads=num_heads,
            feat_drop=feat_drop,
            attn_drop=attn_drop,
            activation=F.relu
        ))

    def forward(self, g, x):
        batch_size = x.shape[0]
        for i, layer in enumerate(self.layers):
            x = layer(g, x)
            if i != len(self.layers) - 1:
                # concatenate attention heads at hidden layers
                x = x.reshape(batch_size, -1)
            else:
                # average over all attention heads at output layer
                x = x.mean(axis=1)
        return x
