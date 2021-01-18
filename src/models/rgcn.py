from .base import BaseGNN

import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import GraphConv, HeteroGraphConv


class RGCN(BaseGNN):
    def __init__(self, in_feats, hidden_feats, num_classes, num_hidden_layers, rel_names, **kwargs):
        super(RGCN, self).__init__(**kwargs)
        self.save_hyperparameters()
        
        self.layers = nn.ModuleList()
        last_input_dim = in_feats
        for i in range(num_hidden_layers):
            self.layers.append(HeteroGraphConv(
                {rel: GraphConv(last_input_dim, hidden_feats, activation=F.relu) for rel in rel_names},
                aggregate='mean'
            ))
            last_input_dim = hidden_feats
        self.layers.append(HeteroGraphConv(
            {rel: GraphConv(last_input_dim, num_classes) for rel in rel_names},
            aggregate='mean'
        ))

    def forward(self, g, x):
        x = {"review": x}
        for layer in self.layers:
            x = layer(g, x)
        return x["review"]
