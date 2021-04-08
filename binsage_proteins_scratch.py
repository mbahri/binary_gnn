import argparse

import os
from pathlib import Path

import torch
import torch.nn.functional as F

import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, SAGEConv

from ogb.nodeproppred import PygNodePropPredDataset, Evaluator

from sage.logger import Logger

import sage.binsage as bs
import binary

import numpy as np

import wandb


class SAGE(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        num_layers,
        dropout,
        binary_inputs=False,
        binary_weights=False,
        pseudo_quantize=False,
    ):
        super(SAGE, self).__init__()

        self.num_layers = num_layers

        self.convs = torch.nn.ModuleList()
        self.activs = torch.nn.ModuleList()

        self.binary_inputs = binary_inputs
        self.binary_weights = binary_weights
        self.pseudo_quantize = pseudo_quantize

        # First layer - real inputs
        self.convs.append(
            bs.BinSAGEConv(
                in_channels,
                hidden_channels,
                binary_inputs=False,
                binary_weights=self.binary_weights,
                pseudo_quantize=False,
                inner_activation=False,
                center=None,
                name="sageconv_00",
            )
        )
        self.activs.append(binary.PReLU())

        # Middle layers
        for i in range(num_layers - 2):
            self.convs.append(
                bs.BinSAGEConv(
                    hidden_channels,
                    hidden_channels,
                    binary_inputs=self.binary_inputs,
                    binary_weights=self.binary_weights,
                    pseudo_quantize=self.pseudo_quantize,
                    inner_activation=False,
                    name=f"sageconv_{i:02d}",
                )
            )
            self.activs.append(binary.PReLU())

        # Last layer - no activation
        self.convs.append(
            bs.BinSAGEConv(
                hidden_channels,
                out_channels,
                binary_inputs=self.binary_inputs,
                binary_weights=self.binary_weights,
                pseudo_quantize=self.pseudo_quantize,
                inner_activation=False,
                name=f"sageconv_{num_layers:02d}",
            )
        )

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adj_t):
        structures = []
        for conv, activ in zip(self.convs[:-1], self.activs):
            x = conv(x, adj_t)
            x1 = activ(x)
            structures.append(x1)
            x = F.dropout(x1, p=self.dropout, training=self.training)

        x = self.convs[-1](x, adj_t)
        return x, structures


def train(model, data, train_idx, optimizer):
    model.train()
    criterion = torch.nn.BCEWithLogitsLoss()

    optimizer.zero_grad()
    out, structures = model(data.x, data.adj_t)

    # Keep only the training set
    out = out[train_idx]
    structures = [ss[train_idx] for ss in structures]

    loss = criterion(out, data.y[train_idx].to(torch.float))
    loss.backward()
    optimizer.step()

    return loss.item()


@torch.no_grad()
def test(model, data, split_idx, evaluator):
    model.eval()

    y_pred, _ = model(data.x, data.adj_t)

    train_rocauc = evaluator.eval(
        {
            "y_true": data.y[split_idx["train"]],
            "y_pred": y_pred[split_idx["train"]],
        }
    )["rocauc"]
    valid_rocauc = evaluator.eval(
        {
            "y_true": data.y[split_idx["valid"]],
            "y_pred": y_pred[split_idx["valid"]],
        }
    )["rocauc"]
    test_rocauc = evaluator.eval(
        {
            "y_true": data.y[split_idx["test"]],
            "y_pred": y_pred[split_idx["test"]],
        }
    )["rocauc"]

    return train_rocauc, valid_rocauc, test_rocauc


def main():
    parser = argparse.ArgumentParser(
        description="OGBN-Proteins (GNN) with BinSAGE from scratch"
    )
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--log_steps", type=int, default=1)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--hidden_channels", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--eval_steps", type=int, default=5)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument(
        "--exp_name",
        type=str,
        default="binsage_scratch",
        metavar="N",
        help="Name of the experiment",
    )
    args = parser.parse_args()
    print(args)

    wandb.init("binary_gnn_demo")
    wandb.config.update(args)
    wandb.config.dataset = "ogbn_proteins"
    wandb.config.distill = False

    device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    dataset = PygNodePropPredDataset(
        name="ogbn-proteins", root="./sage/data", transform=T.ToSparseTensor()
    )
    data = dataset[0]

    # Move edge features to node features.
    data.x = data.adj_t.mean(dim=1)
    data.adj_t.set_value_(None)

    split_idx = dataset.get_idx_split()
    train_idx = split_idx["train"].to(device)

    print("Using BinSAGE")
    model = SAGE(
        data.num_features,
        args.hidden_channels,
        112,
        args.num_layers,
        args.dropout,
        binary_inputs=True,
        binary_weights=True,
        pseudo_quantize=False,
    ).to(device)

    data = data.to(device)

    # Watch model
    wandb.watch(model, log="all", log_freq=50)

    evaluator = Evaluator(name="ogbn-proteins")
    logger = Logger(args.runs, args)

    for run in range(args.runs):
        OUTPUT_DIR = f"./sage/out/proteins/{args.exp_name}/run_{run}"
        Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

        model.reset_parameters()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        for epoch in range(1, 1 + args.epochs):
            loss = train(model, data, train_idx, optimizer)
            wandb.log({"run": run + 1, "epoch": epoch, "training_loss": loss})

            if epoch % args.eval_steps == 0:
                result = test(model, data, split_idx, evaluator)
                logger.add_result(run, result)

                train_rocauc, valid_rocauc, test_rocauc = result

                wandb.log(
                    {
                        "run": run + 1,
                        "epoch": epoch,
                        "train": train_rocauc,
                        "valid": valid_rocauc,
                        "test": test_rocauc,
                    }
                )

                if epoch % args.log_steps == 0:
                    print(
                        f"Run: {run + 1:02d}, "
                        f"Epoch: {epoch:02d}, "
                        f"Loss: {loss:.4f}, "
                        f"Train: {100 * train_rocauc:.2f}%, "
                        f"Valid: {100 * valid_rocauc:.2f}% "
                        f"Test: {100 * test_rocauc:.2f}%"
                    )

        logger.print_statistics(run)
        out_name = f"binsage_proteins_scratch_run_{run:02d}.pt"
        torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, out_name))
        wandb.save(os.path.join(OUTPUT_DIR, out_name))

    logger.print_statistics()


if __name__ == "__main__":
    main()
