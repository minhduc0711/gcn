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
                 attn_drop=0.5,
                 **kwargs):
        super().__init__(**kwargs)
        self.save_hyperparameters()

        self.layers = nn.ModuleList()
        last_input_dim = in_feats
        for i in range(num_hidden_layers):
            self.layers.append(GATConv(
                in_feats=last_input_dim, out_feats=hidden_feats,
                num_heads=num_heads,
                attn_drop=attn_drop
            ))
            last_input_dim = hidden_feats
        self.layers.append(GATConv(
            in_feats=last_input_dim, out_feats=num_classes,
            num_heads=num_heads,
            attn_drop=attn_drop
        ))

    def forward(self, g, x):
        for layer in self.layers:
            x = layer(g, x)
            # average over all attention heads
            x = x.mean(axis=1)
        return x
