from .base import BaseGNN

import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import GraphConv, HeteroGraphConv


class RGCN(BaseGNN):
    def __init__(self, in_feats, hidden_feats, num_classes, rel_names, **kwargs):
        super(RGCN, self).__init__(**kwargs)

        self.conv1 = HeteroGraphConv({
            rel: GraphConv(in_feats, hidden_feats)
            for rel in rel_names}, aggregate='mean')
        self.conv2 = HeteroGraphConv({
            rel: GraphConv(hidden_feats, num_classes)
            for rel in rel_names}, aggregate='mean')

    def forward(self, g, x):
        x = {"review": x}
        # print(x)
        x = self.conv1(g, x)
        x = {k: F.relu(v) for k, v in x.items()}
        x = self.conv2(g, x)
        return x["review"]
