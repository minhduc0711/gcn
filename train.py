import argparse as ap
from collections import defaultdict

import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping

from src.data.data_modules import YelpDataModule, CiteseerDataModule, \
    AmazonDataModule
from src.models import GCN, DNN, GraphSAGE, RGCN, GAT


parser = ap.ArgumentParser()
parser.add_argument("--dataset", type=str, default="yelp")
parser.add_argument("--no-node-features", action="store_true")

parser.add_argument("--num-runs", type=int, default=10)
parser.add_argument("--model", type=str, default="gcn")
parser.add_argument("--lr", type=float, default=1e-2)
parser.add_argument("--weight-decay", type=float, default=1e-3)
parser.add_argument("--dropout", type=float, default=None)
parser.add_argument("--hidden-size", type=int, default=16)
parser.add_argument("--num-hidden-layers", type=int, default=1)

parser = pl.Trainer.add_argparse_args(parser)

args = parser.parse_args()

homo_graph = False if args.model == "rgcn" else True
if args.dataset == "yelp":
    dm = YelpDataModule(no_node_features=args.no_node_features,
                        homo_graph=homo_graph)
elif args.dataset == "amazon":
    dm = AmazonDataModule(no_node_features=args.no_node_features,
                          homo_graph=homo_graph)
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
        lr=args.lr,
        weight_decay=args.weight_decay
    )
elif args.model == "sage":
    model = GraphSAGE(
        in_feats=dm.dims[0],
        hidden_feats=args.hidden_size,
        num_classes=dm.num_classes,
        num_hidden_layers=args.num_hidden_layers,
        class_weights=class_weights,
        lr=args.lr,
        agg="pool"
    )
elif args.model == "rgcn":
    model = RGCN(
        in_feats=dm.dims[0],
        hidden_feats=args.hidden_size,
        num_classes=dm.num_classes,
        num_hidden_layers=args.num_hidden_layers,
        class_weights=class_weights,
        lr=args.lr,
        rel_names=dm.train_ds.g.etypes
    )
elif args.model == "gat":
    model = GAT(
        in_feats=dm.dims[0],
        hidden_feats=args.hidden_size,
        num_classes=dm.num_classes,
        num_hidden_layers=args.num_hidden_layers,
        class_weights=class_weights,
        num_heads=3,
        attn_drop=0.5
    )
elif args.model == "dnn":
    model = DNN(
        in_feats=dm.dims[0],
        hidden_feats=args.hidden_size,
        num_classes=dm.num_classes,
        num_hidden_layers=args.num_hidden_layers,
        class_weights=class_weights,
        lr=args.lr
    )
else:
    raise ValueError(f"unknown model: {args.model}")

result_dict = defaultdict(list)
for _ in range(args.num_runs):
    es_callback = EarlyStopping(monitor="val/f1", patience=300, mode="max")
    trainer = pl.Trainer.from_argparse_args(args,
                                            max_epochs=10000,
                                            callbacks=[es_callback],
                                            log_every_n_steps=1)
    trainer.fit(model, datamodule=dm)
    d = trainer.test(model, datamodule=dm, verbose=False)[0]
    for k in d.keys():
        result_dict[k].append(d[k])

for k, vals in result_dict.items():
    avg = np.mean(vals)
    std = np.std(vals)
    print(f"{k}: {avg:.4f} \u00B1 {std:.4f}")

for metric in ["precision", "recall", "f1", "auROC", "auPR"]:
    vals = result_dict[f"test/{metric}"]
    avg = np.mean(vals)
    std = np.std(vals)
    print(f"& {avg:.4f} \u00B1 {std:.4f} ", end="")
print("\\\\")
