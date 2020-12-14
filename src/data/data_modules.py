from torch.utils.data import DataLoader
import pytorch_lightning as pl

from .datasets import YelpChi, Citeseer


def collate_fn(batch):
    return batch[0]


class CiteseerDataModule(pl.LightningDataModule):
    num_classes = 6

    def __init__(self, no_node_features=False):
        super(CiteseerDataModule, self).__init__()
        self.ds_kwargs = {
            "no_node_features": no_node_features
        }

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_ds = Citeseer(subset="train", **self.ds_kwargs)
            self.val_ds = Citeseer(subset="val", **self.ds_kwargs)
            self.dims = self.train_ds.feats[0].shape
        if stage == "test" or stage is None:
            self.test_ds = Citeseer(subset="test", **self.ds_kwargs)
            self.dims = self.test_ds.feats[0].shape

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=1, collate_fn=collate_fn)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=1, collate_fn=collate_fn)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=1, collate_fn=collate_fn)


class YelpDataModule(pl.LightningDataModule):
    num_classes = 2

    def __init__(self,
                 graph_path="data/YelpChi_handcrafted.mat",
                 homo_graph=True,
                 no_node_features=False,
                 train_size=0.2,
                 val_size=0.4):
        super(YelpDataModule, self).__init__()
        self.ds_kwargs = {
            "no_node_features": no_node_features,
            "graph_path": graph_path,
            "homo_graph": homo_graph,
            "train_size": train_size,
            "val_size": val_size
        }

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_ds = YelpChi(subset="train", **self.ds_kwargs)
            self.val_ds = YelpChi(subset="val", **self.ds_kwargs)
            self.dims = self.train_ds.feats[0].shape
        if stage == "test" or stage is None:
            self.test_ds = YelpChi(subset="test", **self.ds_kwargs)
            self.dims = self.test_ds.feats[0].shape

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=1, collate_fn=collate_fn)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=1, collate_fn=collate_fn)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=1, collate_fn=collate_fn)
