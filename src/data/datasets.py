from abc import ABC, abstractmethod

import numpy as np
import scipy
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset
import dgl
from dgl.data.citation_graph import CiteseerGraphDataset


class GraphDataset(Dataset, ABC):
    g = None

    def __init__(self, subset=None,
                 no_node_features=False,
                 graph_kwargs=None,
                 node_mask_kwargs=None):
        graph_kwargs = {} if graph_kwargs is None else graph_kwargs
        node_mask_kwargs = {} if node_mask_kwargs is None else node_mask_kwargs

        if type(self).g is None:
            type(self).g = self.load_graph(**graph_kwargs)
        if no_node_features:
            self.feats = torch.eye(self.g.num_nodes())
        else:
            self.feats = self.g.ndata["feat"]
        self.labels = self.g.ndata["label"]
        self.num_classes = len(self.labels.unique())
        self.node_mask = self.get_node_mask(subset, **node_mask_kwargs)

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
    g = None

    def __init__(self,
                 graph_path="data/YelpChi_handcrafted.mat",
                 convert_graph_to_homo=True,
                 train_size=0.2,
                 val_size=0.4,
                 **kwargs):
        graph_kwargs = {"graph_path": graph_path,
                        "convert_graph_to_homo": convert_graph_to_homo}
        node_mask_kwargs = {"train_size": train_size, "val_size": val_size}
        super(YelpChi, self).__init__(graph_kwargs=graph_kwargs,
                                      node_mask_kwargs=node_mask_kwargs,
                                      **kwargs)

    def get_node_mask(self, subset, train_size, val_size):
        if subset not in ["train", "val", "test"]:
            raise RuntimeError("subset must be one of: train, val, test")
        train_mask, eval_mask = train_test_split(
            np.arange(self.feats.shape[0]),
            train_size=train_size,
            stratify=self.labels,
            random_state=42,
        )
        if subset == "train":
            return train_mask

        test_size = 1 - train_size - val_size
        val_mask, test_mask = train_test_split(
            eval_mask,
            test_size=test_size / (test_size + val_size),
            stratify=self.labels[eval_mask],
            random_state=42
        )
        return val_mask if subset == "val" else test_mask

    def load_graph(self, graph_path, convert_graph_to_homo):
        d = scipy.io.loadmat(graph_path)
        g = dgl.heterograph(
            {
                ("review", "same_user", "review"): d["net_rur"].nonzero(),
                ("review", "same_prod_rating", "review"): d["net_rsr"].nonzero(),
                ("review", "same_prod_month", "review"): d["net_rtr"].nonzero(),
            }
        )
        # TODO: fix heterograph self loop!
        if convert_graph_to_homo:
            g = dgl.to_homogeneous(g)
            g = dgl.add_self_loop(g)

        # g = dgl.DGLGraph()
        # g.add_nodes(d["label"].shape[1])
        # g = dgl.add_self_loop(g)

        g.ndata["feat"] = torch.from_numpy(d["features"].todense()).to(dtype=torch.float)
        g.ndata["label"] = torch.from_numpy(d["label"].flatten())
        return g
