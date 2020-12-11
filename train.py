import argparse as ap

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping

from src.data.data_modules import YelpDataModule, CiteseerDataModule
from src.models import GCN, DNN


parser = ap.ArgumentParser()
parser.add_argument("--dataset", type=str, default="yelp")
parser.add_argument("--no-node-features", action="store_true")

parser.add_argument("--model", type=str, default="gcn")
parser.add_argument("--lr", type=float, default=1e-2)
parser.add_argument("--weight-decay", type=float, default=1e-3)
parser.add_argument("--dropout", type=float, default=None)
parser.add_argument("--hidden-size", type=int, default=16)
parser.add_argument("--num-hidden-layers", type=int, default=1)

args = parser.parse_args()

if args.dataset == "yelp":
    dm = YelpDataModule(no_node_features=args.no_node_features)
elif args.dataset == "citeseer":
    dm = CiteseerDataModule(no_node_features=args.no_node_features)
else:
    raise RuntimeError("dataset must be one of: yelp, citeseer")
dm.setup()

classes, class_freqs = torch.unique(dm.train_ds.labels, return_counts=True)
class_weights = len(dm.train_ds.labels) / (len(classes) * class_freqs)
# class_weights = None

if args.model == "gcn":
    model = GCN(
        in_feats=dm.dims[0],
        hidden_feats=args.hidden_size,
        num_classes=dm.num_classes,
        num_hidden_layers=args.num_hidden_layers,
        dropout_proba=args.dropout,
        class_weights=class_weights,
        learning_rate=args.lr,
        weight_decay=args.weight_decay
    )
elif args.model == "dnn":
    model = DNN(
        in_feats=dm.dims[0],
        hidden_feats=args.hidden_size,
        num_classes=dm.num_classes,
        num_hidden_layers=args.num_hidden_layers,
        class_weights=class_weights,
        learning_rate=args.lr
    )


es_callback = EarlyStopping(monitor="val/f1", patience=500, mode="max")
trainer = pl.Trainer(max_epochs=1000,
                     callbacks=[es_callback],
                     log_every_n_steps=1)
trainer.fit(model, datamodule=dm)
trainer.test(model, datamodule=dm)
