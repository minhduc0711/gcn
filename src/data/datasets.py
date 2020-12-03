import torch
from torch.utils.data import Dataset
import dgl
from dgl.data.citation_graph import CiteseerGraphDataset


class Citeseer(Dataset):
    g = dgl.add_self_loop(CiteseerGraphDataset()[0])

    def __init__(self, subset=None, no_node_features=False):
        if no_node_features:
            self.feats = torch.eye(self.g.num_nodes())
        else:
            self.feats = self.g.ndata["feat"]

        if subset == "train":
            self.node_mask = self.g.ndata["train_mask"]
        elif subset == "val":
            self.node_mask = self.g.ndata["val_mask"]
        elif subset == "test":
            self.node_mask = self.g.ndata["test_mask"]
        else:
            self.node_mask = None

    def __len__(self):
        return 1  # does not support minibatch atm

    def __getitem__(self, idx):
        return {
            "g": self.g,
            "feats": self.feats,
            "labels": self.g.ndata["label"],
            "node_mask": self.node_mask
        }
