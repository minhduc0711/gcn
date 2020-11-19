import torch
from torch.utils.data import Dataset
import dgl
from dgl.data.citation_graph import CiteseerGraphDataset


class CiteSeerDataset(Dataset):
    def __init__(self, subset=None,
                 no_node_features=False):
        ds = CiteseerGraphDataset()
        self.graph = dgl.add_self_loop(ds[0])  # only 1 graph in dataset
        if no_node_features:
            self.feats = torch.eye(self.graph.num_nodes())
        else:
            self.feats = self.graph.ndata["feat"]

        if subset == "train":
            self.node_mask = self.graph.ndata["train_mask"]
        elif subset == "val":
            self.node_mask = self.graph.ndata["val_mask"]
        elif subset == "test":
            self.node_mask = self.graph.ndata["test_mask"]
        else:
            self.node_mask = None

    def __len__(self):
        return 1  # does not support minibatch atm

    def __getitem__(self, idx):
        return {
            "graph": self.graph,
            "feats": self.feats,
            "labels": self.graph.ndata["label"],
            "node_mask": self.node_mask
        }
