from abc import ABC, abstractmethod

import numpy as np
import scipy
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset
import dgl
from dgl.data.citation_graph import CiteseerGraphDataset


def standardize(x):
    mus = x.mean(dim=1, keepdim=True)
    sigmas = x.std(dim=1, keepdim=True)
    return (x - mus) / sigmas


class GraphDataset(Dataset, ABC):
    g = None

    def __init__(self, subset=None,
                 no_node_features=False):
        # avoid loading the graph again when creating a new dataset instance
        if type(self).g is None:
            type(self).g = self.load_graph()
        if no_node_features:
            self.feats = torch.eye(self.g.num_nodes())
        else:
            self.feats = standardize(self.g.ndata["feat"])
        self.labels = self.g.ndata["label"]
        self.num_classes = len(self.labels.unique())
        self.node_mask = self.get_node_mask(subset)

    @abstractmethod
    def load_graph(self):
        pass

    @abstractmethod
    def get_node_mask(self, subset):
        pass

    def __len__(self):
        return 1  # does not support minibatch atm

    def __getitem__(self, idx):
        return {
            "g": self.g,
            "feats": self.feats,
            "labels": self.labels,
            "node_mask": self.node_mask
        }


class Citeseer(GraphDataset):
    def get_node_mask(self, subset):
        return self.g.ndata[f"{subset}_mask"]

    def load_graph(self):
        return dgl.add_self_loop(CiteseerGraphDataset()[0])


class YelpChi(GraphDataset):
    def __init__(self,
                 graph_path="data/YelpChi_handcrafted.mat",
                 homo_graph=True,
                 train_size=0.2,
                 val_size=0.4,
                 **kwargs):
        self.graph_path = graph_path
        self.homo_graph = homo_graph
        self.train_size = train_size
        self.val_size = val_size
        super(YelpChi, self).__init__(**kwargs)

    def get_node_mask(self, subset):
        if subset not in ["train", "val", "test"]:
            raise RuntimeError("subset must be one of: train, val, test")
        train_mask, eval_mask = train_test_split(
            np.arange(self.feats.shape[0]),
            train_size=self.train_size,
            stratify=self.labels,
            random_state=42,
        )
        if subset == "train":
            return train_mask

        test_size = 1 - self.train_size - self.val_size
        val_mask, test_mask = train_test_split(
            eval_mask,
            test_size=test_size / (test_size + self.val_size),
            stratify=self.labels[eval_mask],
            random_state=42
        )
        return val_mask if subset == "val" else test_mask

    def load_graph(self):
        d = scipy.io.loadmat(self.graph_path)

        if self.homo_graph:
            g = dgl.from_scipy(d["homo"])
            g = dgl.add_self_loop(g)
        else:
            g = dgl.heterograph({
                ("review", "rur", "review"): d["net_rur"].nonzero(),
                ("review", "rsr", "review"): d["net_rsr"].nonzero(),
                ("review", "rtr", "review"): d["net_rtr"].nonzero(),
            })
        # g = dgl.DGLGraph()
        # g.add_nodes(d["label"].shape[1])
        # g = dgl.add_self_loop(g)

        g.ndata["feat"] = torch.from_numpy(d["features"].todense()).to(dtype=torch.float)
        g.ndata["label"] = torch.from_numpy(d["label"].flatten())
        return g


class Amazon(GraphDataset):
    # copy pasted code from YelpChi, since the data format is the same
    def __init__(self,
                 graph_path="data/Amazon.mat",
                 homo_graph=True,
                 train_size=0.2,
                 val_size=0.4,
                 **kwargs):
        self.graph_path = graph_path
        self.homo_graph = homo_graph
        self.train_size = train_size
        self.val_size = val_size
        super(Amazon, self).__init__(**kwargs)

    def get_node_mask(self, subset):
        if subset not in ["train", "val", "test"]:
            raise RuntimeError("subset must be one of: train, val, test")
        train_mask, eval_mask = train_test_split(
            np.arange(self.feats.shape[0]),
            train_size=self.train_size,
            stratify=self.labels,
            random_state=42,
        )
        if subset == "train":
            return train_mask

        test_size = 1 - self.train_size - self.val_size
        val_mask, test_mask = train_test_split(
            eval_mask,
            test_size=test_size / (test_size + self.val_size),
            stratify=self.labels[eval_mask],
            random_state=42
        )
        return val_mask if subset == "val" else test_mask

    def load_graph(self):
        d = scipy.io.loadmat(self.graph_path)

        if self.homo_graph:
            g = dgl.from_scipy(d["homo"])
            g = dgl.add_self_loop(g)
        else:
            g = dgl.heterograph({
                ("review", "upu", "review"): d["net_upu"].nonzero(),
                ("review", "usu", "review"): d["net_usu"].nonzero(),
                ("review", "uvu", "review"): d["net_uvu"].nonzero()
            })

        g.ndata["feat"] = torch.from_numpy(d["features"].todense()).to(dtype=torch.float)
        g.ndata["label"] = torch.from_numpy(d["label"].flatten()).to(dtype=torch.long)
        return g
