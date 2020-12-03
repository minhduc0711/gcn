import argparse as ap


import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping

from src.data.datasets import Citeseer
from src.models.gcn import GCN


def collate_fn(batch):
    return batch[0]


parser = ap.ArgumentParser()
parser.add_argument("--no-node-features", action="store_true")
parser.add_argument("--weight-decay", type=float, default=1e-3)
args = parser.parse_args()

train_dataloader = DataLoader(
    Citeseer(subset="train", no_node_features=args.no_node_features),
    batch_size=1,
    collate_fn=collate_fn,
)
val_dataloader = DataLoader(
    Citeseer(subset="val", no_node_features=args.no_node_features),
    batch_size=1,
    collate_fn=collate_fn,
)
test_dataloader = DataLoader(
    Citeseer(subset="test", no_node_features=args.no_node_features),
    batch_size=1,
    collate_fn=collate_fn,
)

if args.no_node_features:
    in_feats = 3327
else:
    in_feats = 3703
model = GCN(in_feats=in_feats, hidden_feats=16, num_classes=6, num_hidden_layers=1,
            dropout_proba=0.5, weight_decay=args.weight_decay)

es_callback = EarlyStopping(monitor="val/loss", patience=10)
trainer = pl.Trainer(max_epochs=500, callbacks=[es_callback])
trainer.fit(model, train_dataloader=train_dataloader, val_dataloaders=val_dataloader)
trainer.test(model, test_dataloaders=test_dataloader)
