"""
@Author: Mehdi Bahri
@Contact: m.bahri@imperial.ac.uk
@File: binsage_products_distill.py
"""

import os
import os.path as osp
from pathlib import Path

import wandb
import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F

from torch_geometric.data import NeighborSampler
from torch_geometric.nn import SAGEConv

from ogb.nodeproppred import PygNodePropPredDataset, Evaluator

torch.set_num_threads(4)

import binary
import distillation
import sage.binsage as bs

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
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

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
        structures = []
        for i, (edge_index, _, size) in enumerate(adjs):
            x_target = x[: size[1]]  # Target nodes are always placed first.
            x = self.convs[i]((x, x_target), edge_index)
            x1 = x.clone()
            if i != self.num_layers - 1:
                x1 = F.relu(x)
                x = F.dropout(x, p=0.5, training=self.training)
            structures.append((x1, edge_index))

        return x.log_softmax(dim=-1), structures

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
                    x = F.relu(x)
                xs.append(x.cpu())

                pbar.update(batch_size)

            x_all = torch.cat(xs, dim=0)

        pbar.close()

        return x_all


class BinSAGE(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        num_layers=3,
        binary_inputs=False,
        binary_weights=False,
        pseudo_quantize=False,
    ):
        super().__init__()

        self.num_layers = num_layers

        self.convs = torch.nn.ModuleList()
        self.activs = torch.nn.ModuleList()

        self.binary_inputs = binary_inputs
        self.binary_weights = binary_weights
        self.pseudo_quantize = pseudo_quantize

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
        self.activs.append(torch.nn.Identity())

        self.convs.append(
            bs.BinSAGEConv(
                hidden_channels,
                hidden_channels,
                binary_inputs=self.binary_inputs,
                binary_weights=self.binary_weights,
                pseudo_quantize=self.pseudo_quantize,
                inner_activation=False,
                name="sageconv_01",
            )
        )
        self.activs.append(torch.nn.Identity())

        self.convs.append(
            bs.BinSAGEConv(
                hidden_channels,
                out_channels,
                binary_inputs=self.binary_inputs,
                binary_weights=self.binary_weights,
                pseudo_quantize=self.pseudo_quantize,
                inner_activation=False,
                name="sageconv_02",
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
        structures = []
        for i, (edge_index, _, size) in enumerate(adjs):
            x_target = x[: size[1]]  # Target nodes are always placed first.
            x = self.convs[i]((x, x_target), edge_index)
            x1 = x.clone()
            if i != self.num_layers - 1:
                x1 = self.activs[i](x)
                x = F.dropout(x1, p=0.5, training=self.training)

            structures.append((x1, edge_index))

        return x.log_softmax(dim=-1), structures

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


def train_distill(
    device,
    train_loader,
    student,
    teacher,
    student_optimizer,
    lsp_evaluators=[],
    lsp_points=[-1],
    lambda_lsp=1e2,
    TemperatureCE=None,
    alpha=0.1,
    run=0,
    epoch=0,
):
    student.train()

    losses = []
    kls = [(0.0, 0.0, 0.0), (0.0, 0.0, 0.0)]
    logmats = []

    total_loss = 0

    pbar = tqdm(total=train_idx.size(0))
    pbar.set_description(f"Epoch {epoch:02d}")

    total_loss = total_correct = 0
    for batch_size, n_id, adjs in train_loader:
        # `adjs` holds a list of `(edge_index, e_id, size)` tuples.
        adjs = [adj.to(device) for adj in adjs]

        student_optimizer.zero_grad()
        out, structures_s = student(x[n_id], adjs)

        with torch.no_grad():
            pred_t, structures_t = teacher(x[n_id], adjs)  # , ls_t

        # ll = []
        # for lsp_evaluator, i in zip(lsp_evaluators, lsp_points):
        #     kl = lsp_evaluator(structures_s[i][0], structures_t[i][0], structures_s[i][1], structures_t[i][1])
        #     ll.append(kl)
        # kls.append(tuple(k.item() for k in ll))

        loss = F.nll_loss(out, y[n_id[:batch_size]])
        if TemperatureCE is not None:
            logmat_loss = TemperatureCE(out, pred_t)
            loss = (1 - alpha) * loss + alpha * logmat_loss
            logmats.append(logmat_loss.item())

        # for k in ll:
        #     loss += lambda_lsp * k / len(lsp_points)

        loss.backward()
        student_optimizer.step()

        total_loss += float(loss)
        total_correct += int(out.argmax(dim=-1).eq(y[n_id[:batch_size]]).sum())
        pbar.update(batch_size)

    pbar.close()

    loss = total_loss / len(train_loader)
    approx_acc = total_correct / train_idx.size(0)

    return loss, approx_acc, kls, logmats


@torch.no_grad()
def test(model, y, split_idx):
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


def training_loop_distill(
    device,
    student,
    teacher,
    student_opt,
    lsp_evaluators,
    ls_points,
    temperature_crossentropy=None,
    alpha=1e-1,
    lambda_lsp=1e2,
    N_epochs=21,
    best_on_val=True,
    loaders=(),
    scheduler=None,
    stage=0,
    run=0,
):

    run_accs = []
    best_val_acc = final_test_acc = 0

    best_state_dict = {}

    student = student.to(device)
    teacher = teacher.to(device)

    teacher.eval()

    for epoch in range(1, N_epochs):
        loss, acc, kls, logmats = train_distill(
            device,
            train_loader,
            student,
            teacher,
            student_opt,
            lsp_evaluators,
            ls_points,
            lambda_lsp,
            temperature_crossentropy,
            alpha,
            run=run,
            epoch=epoch,
        )

        print(
            f'Epoch {epoch:02d}, Loss: {loss:.4f}, Approx. Train: {acc:.4f}, lr: {student_opt.param_groups[0]["lr"]:.3e}'
        )

        kls = np.concatenate(kls).reshape(-1, len(ls_points)).mean(0)
        logmats = np.mean(logmats)

        wandb.log(
            {
                "run": run,
                "epoch": epoch,
                "loss": loss,
                "approx_train_acc": acc,
                "kls": kls,
                "logmats": logmats,
            }
        )

        if epoch > 5:
            train_acc, val_acc, test_acc = test(student, y, split_idx)
            print(
                f"Train: {train_acc:.4f}, Val: {val_acc:.4f}, " f"Test: {test_acc:.4f}"
            )

            wandb.log(
                {
                    "run": run,
                    "epoch": epoch,
                    "train": train_acc,
                    "val": val_acc,
                    "test": test_acc,
                    "lr": student_opt.param_groups[0]["lr"],
                }
            )

            scheduler.step(val_acc)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                final_test_acc = test_acc
                sd = student.state_dict()
                best_state_dict = sd

    wandb.log(
        {"run": run, "final_test_acc": final_test_acc, "best_val_acc": best_val_acc}
    )

    return best_val_acc, best_state_dict, final_test_acc


###################################

parser = argparse.ArgumentParser(
    description="OGBn Products Benchmark with Binary GraphSAGE and distillation"
)
parser.add_argument("--run_id", type=int, default=1, metavar="N", help="ID of the run")
parser.add_argument(
    "--exp_name",
    type=str,
    default="binsage_distill",
    metavar="N",
    help="Name of the experiment",
)
parser.add_argument(
    "--LSP_lambda",
    type=float,
    default=1e2,
    metavar="M",
    help="Local structure preserving loss weight.",
)
parser.add_argument(
    "--KD_T",
    type=float,
    default=2,
    metavar="M",
    help="Temperature in softmax for logit matching.",
)
parser.add_argument(
    "--KD_alpha", type=float, default=0.1, metavar="M", help="Alpha for logit matching."
)
parser.add_argument(
    "--sched_gamma",
    type=float,
    default=0.1,
    help="Rate of decrease for the learning rate scheduler",
)
parser.add_argument(
    "--sched_patience",
    type=int,
    default=5,
    help="Patience for the learning rate scheduler",
)
parser.add_argument(
    "--sched_cooldown",
    type=int,
    default=2,
    help="Cooldown for the learning rate scheduler",
)

args = parser.parse_args()

OUTPUT_DIR = f"./sage/out/products/{args.exp_name}/run_{args.run_id}"

Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)


wandb.init("binary_gnn_demo")
wandb.config.update(args)
wandb.config.dataset = "ogbn_products"
wandb.config.distill = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
x = data.x.to(device)
y = data.y.squeeze().to(device)

TempCE = distillation.CrossEntropyWithTemperature(T=args.KD_T)
LSP_l2_l2 = distillation.StructuralSimilarity(
    kernel_s=distillation.rbf_l2, kernel_t=distillation.rbf_l2
)

print("Stage 1")

teacher = SAGE(dataset.num_features, 256, dataset.num_classes, num_layers=3)
teacher_state = torch.load("./sage/pretrained/baseline_run_10.pt", map_location=device)
teacher.load_state_dict(teacher_state)
teacher = teacher.to(device)

bin0_ = BinSAGE(
    dataset.num_features,
    256,
    dataset.num_classes,
    num_layers=3,
    binary_inputs=False,
    binary_weights=False,
    pseudo_quantize=True,
)
bin0_ = bin0_.to(device)
wandb.watch(bin0_, log="all", log_freq=10)

bin_0_optimizer_ = torch.optim.AdamW(bin0_.parameters(), lr=5e-3, weight_decay=1e-5)
scheduler0 = torch.optim.lr_scheduler.ReduceLROnPlateau(
    bin_0_optimizer_,
    factor=args.sched_gamma,
    patience=args.sched_patience,
    cooldown=args.sched_cooldown,
    min_lr=1e-4,
    verbose=True,
    mode="max",
)

baseline_bin0_ = training_loop_distill(
    device,
    bin0_,
    teacher,
    bin_0_optimizer_,
    [LSP_l2_l2, LSP_l2_l2, LSP_l2_l2],
    [0, 1, 2],
    temperature_crossentropy=TempCE,
    alpha=args.KD_alpha,
    lambda_lsp=args.LSP_lambda,
    N_epochs=21,
    best_on_val=True,
    loaders=(train_loader,),
    scheduler=scheduler0,
    stage=1,
    run=args.run_id,
)
sd = baseline_bin0_[1]
torch.save(baseline_bin0_, os.path.join(OUTPUT_DIR, "pseudo_bin_distill.pt"))
wandb.log(
    {
        "stage": 1,
        "best_val_acc": baseline_bin0_[0],
        "final_test_acc": baseline_bin0_[2],
        "run": args.run_id,
    }
)

print("Stage 1 done.")
print(f"Best val acc: {baseline_bin0_[0]}")
print(f"Final test acc: {baseline_bin0_[2]}")

print("Training stage 2: binary activations")

bin0_.load_state_dict(sd)
bin0_ba_ = BinSAGE(
    dataset.num_features,
    256,
    dataset.num_classes,
    num_layers=3,
    binary_inputs=True,
    binary_weights=False,
    pseudo_quantize=False,
)
bin0_ba_ = bin0_ba_.to(device)
wandb.watch(bin0_ba_, log="all", log_freq=10)

bin_0_ba_optimizer_ = torch.optim.AdamW(
    bin0_ba_.parameters(), lr=5e-3, weight_decay=1e-5
)
scheduler0_ba = torch.optim.lr_scheduler.ReduceLROnPlateau(
    bin_0_ba_optimizer_,
    factor=args.sched_gamma,
    patience=args.sched_patience,
    cooldown=args.sched_cooldown,
    min_lr=1e-4,
    verbose=True,
    mode="max",
)

baseline_bin0_ba_ = training_loop_distill(
    device,
    bin0_ba_,
    bin0_,
    bin_0_ba_optimizer_,
    [LSP_l2_l2, LSP_l2_l2, LSP_l2_l2],
    [0, 1, 2],
    temperature_crossentropy=TempCE,
    alpha=args.KD_alpha,
    lambda_lsp=args.LSP_lambda,
    N_epochs=21,
    best_on_val=True,
    loaders=(train_loader,),
    scheduler=scheduler0_ba,
    stage=2,
    run=args.run_id,
)
sd_ba = baseline_bin0_ba_[1]
torch.save(baseline_bin0_ba_, os.path.join(OUTPUT_DIR, "binary_activ_distill.pt"))
wandb.log(
    {
        "stage": 2,
        "best_val_acc": baseline_bin0_ba_[0],
        "final_test_acc": baseline_bin0_ba_[2],
        "run": args.run_id,
    }
)

print("Stage 2 done.")
print(f"Best val acc: {baseline_bin0_ba_[0]}")
print(f"Final test acc: {baseline_bin0_ba_[2]}")

bin0_ba_.load_state_dict(sd_ba)
bin0_bb_ = BinSAGE(
    dataset.num_features,
    256,
    dataset.num_classes,
    num_layers=3,
    binary_inputs=True,
    binary_weights=True,
    pseudo_quantize=False,
)
bin0_bb_ = bin0_bb_.to(device)
wandb.watch(bin0_bb_, log="all", log_freq=10)

bin_0_bb_optimizer_ = torch.optim.AdamW(bin0_bb_.parameters(), lr=5e-3, weight_decay=0)
scheduler0_bb = torch.optim.lr_scheduler.ReduceLROnPlateau(
    bin_0_bb_optimizer_,
    factor=args.sched_gamma,
    patience=args.sched_patience,
    cooldown=args.sched_cooldown,
    min_lr=1e-4,
    verbose=True,
    mode="max",
)
baseline_bin0_bb_ = training_loop_distill(
    device,
    bin0_bb_,
    bin0_ba_,
    bin_0_bb_optimizer_,
    [LSP_l2_l2, LSP_l2_l2, LSP_l2_l2],
    [0, 1, 2],
    temperature_crossentropy=TempCE,
    alpha=args.KD_alpha,
    lambda_lsp=args.LSP_lambda,
    N_epochs=21,
    best_on_val=True,
    loaders=(train_loader,),
    scheduler=scheduler0_bb,
    stage=2,
    run=args.run_id,
)
sd_bb = baseline_bin0_bb_[1]
torch.save(baseline_bin0_bb_, os.path.join(OUTPUT_DIR, "binary_activ_distill.pt"))
wandb.log(
    {
        "stage": 3,
        "best_val_acc": baseline_bin0_bb_[0],
        "final_test_acc": baseline_bin0_bb_[2],
        "run": args.run_id,
    }
)

print("Stage 3 done.")
print(f"Best val acc: {baseline_bin0_bb_[0]}")
print(f"Final test acc: {baseline_bin0_bb_[2]}")
