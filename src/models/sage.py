from dgl.nn import SAGEConv, GraphConv
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from src.eval import compute_metrics
from .base import BaseGNN


class GraphSAGE(BaseGNN):
    def __init__(self, in_feats, hidden_feats, num_classes, **kwargs):
        super().__init__(**kwargs)
        self.conv1 = SAGEConv(
            in_feats=in_feats, out_feats=hidden_feats, aggregator_type="mean"
        )
        self.conv2 = SAGEConv(
            in_feats=hidden_feats, out_feats=num_classes, aggregator_type="mean"
        )

    def forward(self, graph, inputs):
        # inputs are features of nodes
        h = self.conv1(graph, inputs)
        h = F.relu(h)
        h = self.conv2(graph, h)
        return h
