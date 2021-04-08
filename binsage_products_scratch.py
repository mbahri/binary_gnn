#!/usr/bin/env python
# coding: utf-8

import argparse
import os.path as osp
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
from torch_geometric.data import NeighborSampler
from torch_geometric.nn import SAGEConv

import binary
import sage.binsage as bs

import wandb


root = "./sage/data"
dataset = PygNodePropPredDataset("ogbn-products", root)
split_idx = dataset.get_idx_split()
evaluator = Evaluator(name="ogbn-products")
data = dataset[0]

train_idx = split_idx["train"]
train_loader = NeighborSampler(
    data.edge_index,
    node_idx=train_idx,
    sizes=[15, 10, 5],
    batch_size=1024,
    shuffle=True,
    num_workers=4,
)
subgraph_loader = NeighborSampler(
    data.edge_index,
    node_idx=None,
    sizes=[-1],
    batch_size=4096,
    shuffle=False,
    num_workers=4,
)


class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers):
        super(SAGE, self).__init__()

        self.num_layers = num_layers

        self.convs = torch.nn.ModuleList()
        self.activs = torch.nn.ModuleList()

        self.convs.append(
            bs.BinSAGEConv(
                in_channels,
                hidden_channels,
                binary_inputs=False,
                binary_weights=True,
                pseudo_quantize=False,
                inner_activation=False,
                center=None,
                name="sageconv_00",
            )
        )
        self.activs.append(nn.Identity())

        for i in range(num_layers - 2):
            self.convs.append(
                bs.BinSAGEConv(
                    hidden_channels,
                    hidden_channels,
                    binary_inputs=True,
                    binary_weights=True,
                    pseudo_quantize=False,
                    inner_activation=False,
                    name="sageconv_{:02d}".format(i + 1),
                )
            )
            self.activs.append(nn.Identity())

        self.convs.append(
            bs.BinSAGEConv(
                hidden_channels,
                out_channels,
                binary_inputs=True,
                binary_weights=True,
                pseudo_quantize=False,
                inner_activation=False,
                name="sageconv_{:02d}".format(num_layers - 1),
            )
        )

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adjs):
        # `train_loader` computes the k-hop neighborhood of a batch of nodes,
        # and returns, for each layer, a bipartite graph object, holding the
        # bipartite edges `edge_index`, the index `e_id` of the original edges,
        # and the size/shape `size` of the bipartite graph.
        # Target nodes are also included in the source nodes so that one can
        # easily apply skip-connections or add self-loops.
        for i, (edge_index, _, size) in enumerate(adjs):
            x_target = x[: size[1]]  # Target nodes are always placed first.
            x = self.convs[i]((x, x_target), edge_index)
            if i != self.num_layers - 1:
                x = self.activs[i](x)
                x = F.dropout(x, p=0.5, training=self.training)
        return x.log_softmax(dim=-1)

    def inference(self, x_all):
        pbar = tqdm(total=x_all.size(0) * self.num_layers)
        pbar.set_description("Evaluating")

        # Compute representations of nodes layer by layer, using *all*
        # available edges. This leads to faster computation in contrast to
        # immediately computing the final representations of each batch.
        total_edges = 0
        for i in range(self.num_layers):
            xs = []
            for batch_size, n_id, adj in subgraph_loader:
                edge_index, _, size = adj.to(device)
                total_edges += edge_index.size(1)
                x = x_all[n_id].to(device)
                x_target = x[: size[1]]
                x = self.convs[i]((x, x_target), edge_index)
                if i != self.num_layers - 1:
                    x = self.activs[i](x)
                xs.append(x.cpu())

                pbar.update(batch_size)

            x_all = torch.cat(xs, dim=0)

        pbar.close()

        return x_all


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SAGE(dataset.num_features, 256, dataset.num_classes, num_layers=3)
model = model.to(device)

x = data.x.to(device)
y = data.y.squeeze().to(device)


def build_optimizer_params(model):
    "Ensures we apply weight decay on the rescaling parameters only"
    for k, v in model.named_parameters():
        tmp = k.split(".")
        if tmp[0] == "convs":
            if tmp[3] == "rescale":
                yield {"params": v, "weight_decay": 1e-4}
            else:
                yield {"params": v, "weight_decay": 0.0}
        else:
            yield {"params": v, "weight_decay": 0.0}


def train(epoch):
    model.train()

    pbar = tqdm(total=train_idx.size(0))
    pbar.set_description(f"Epoch {epoch:02d}")

    total_loss = total_correct = 0
    for batch_size, n_id, adjs in train_loader:
        # `adjs` holds a list of `(edge_index, e_id, size)` tuples.
        adjs = [adj.to(device) for adj in adjs]

        optimizer.zero_grad()
        out = model(x[n_id], adjs)
        #         import pdb; pdb.set_trace()
        loss = F.nll_loss(out, y[n_id[:batch_size]])
        loss.backward()
        optimizer.step()

        total_loss += float(loss)
        total_correct += int(out.argmax(dim=-1).eq(y[n_id[:batch_size]]).sum())
        pbar.update(batch_size)

    pbar.close()

    loss = total_loss / len(train_loader)
    approx_acc = total_correct / train_idx.size(0)

    return loss, approx_acc


@torch.no_grad()
def test():
    model.eval()

    out = model.inference(x)

    y_true = y.cpu().unsqueeze(-1)
    y_pred = out.argmax(dim=-1, keepdim=True)

    train_acc = evaluator.eval(
        {
            "y_true": y_true[split_idx["train"]],
            "y_pred": y_pred[split_idx["train"]],
        }
    )["acc"]
    val_acc = evaluator.eval(
        {
            "y_true": y_true[split_idx["valid"]],
            "y_pred": y_pred[split_idx["valid"]],
        }
    )["acc"]
    test_acc = evaluator.eval(
        {
            "y_true": y_true[split_idx["test"]],
            "y_pred": y_pred[split_idx["test"]],
        }
    )["acc"]

    return train_acc, val_acc, test_acc


runs_models = {}


parser = argparse.ArgumentParser(
    description="OGBn Products Benchmark with Binary GraphSAGE from scratch"
)
parser.add_argument(
    "--exp_name",
    type=str,
    default="binsage_scratch",
    metavar="N",
    help="Name of the experiment",
)

args = parser.parse_args()

wandb.init("binary_gnn_demo")
wandb.config.update(args)
wandb.config.prelu = False
wandb.config.dataset = "ogbn_products"
wandb.config.distill = False

test_accs = []
for run in tqdm(range(1, 11)):
    print("")
    print(f"Run {run:02d}:")
    print("")

    OUTPUT_DIR = f"./sage/out/products/{args.exp_name}/run_{run}"
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    model.reset_parameters()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=0.5,
        patience=5,
        cooldown=2,
        min_lr=1e-4,
        verbose=True,
        mode="max",
    )

    run_accs = []
    best_val_acc = final_test_acc = 0

    for epoch in range(1, 21):
        loss, acc = train(epoch)
        print(
            f'Epoch {epoch:02d}, Loss: {loss:.4f}, Approx. Train: {acc:.4f}, lr: {optimizer.param_groups[0]["lr"]:.3e}'
        )
        wandb.log({"epoch": epoch, "loss": loss, "approx_train_acc": acc})

        if epoch > 5:
            train_acc, val_acc, test_acc = test()
            print(
                f"Train: {train_acc:.4f}, Val: {val_acc:.4f}, " f"Test: {test_acc:.4f}"
            )

            run_accs.append(
                {
                    "run": run,
                    "epoch": epoch,
                    "train": train_acc,
                    "val": val_acc,
                    "test": test_acc,
                }
            )

            wandb.log(
                {
                    "run": run,
                    "epoch": epoch,
                    "train": train_acc,
                    "val": val_acc,
                    "test": test_acc,
                    "lr": optimizer.param_groups[0]["lr"],
                }
            )

            scheduler.step(val_acc)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                final_test_acc = test_acc
                sd = model.state_dict()
                runs_models[run] = sd

    wandb.log({"run": run, "final_test_acc": final_test_acc})
    test_accs.append(final_test_acc)
    torch.save(run_accs, f"{OUTPUT_DIR}/{args.exp_name}_accs_run_{run}.pt")
    torch.save(best_val_acc, f"{OUTPUT_DIR}/{args.exp_name}_best_val_acc_run_{run}.pt")
    torch.save(sd, f"{OUTPUT_DIR}/{args.exp_name}_run_{run}.pt")


test_acc = torch.tensor(test_accs)
print("============================")
print(f"Final Test: {test_acc.mean():.4f} ± {test_acc.std():.4f}")


torch.save(test_acc, f"./sage/out/{args.exp_name}/{args.exp_name}_all_accs.pt")
torch.save(runs_models, f"./sage/out/{args.exp_name}/{args.exp_name}_all_runs.pt")
