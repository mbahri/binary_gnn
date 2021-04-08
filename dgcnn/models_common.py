"""
@Author: Mehdi Bahri
@Contact: m.bahri@imperial.ac.uk
@File: models_common.py
"""

import torch
import torch.nn as nn

import sklearn.metrics as metrics
import numpy as np

import binary

import argparse

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def test_universal(model, test_loader, device, io, prefix=''):
    #Try to load models
    model = model.eval()
    test_acc = 0.0
    count = 0.0
    test_true = []
    test_pred = []
    for data, label in test_loader:
        data, label = data.to(device), label.to(device).squeeze()
        data = data.permute(0, 2, 1)
        batch_size = data.size()[0]
        logits, _ = model(data)
        preds = logits.max(dim=1)[1]
        test_true.append(label.cpu().numpy())
        test_pred.append(preds.detach().cpu().numpy())
    test_true = np.concatenate(test_true)
    test_pred = np.concatenate(test_pred)
    test_acc = metrics.accuracy_score(test_true, test_pred)
    avg_per_class_acc = metrics.balanced_accuracy_score(test_true, test_pred)
    outstr = 'Test :: %stest acc: %.6f, %stest avg acc: %.6f'%(prefix, test_acc, prefix, avg_per_class_acc)
    io.cprint(outstr)


def knn_lp(x, k, p):
    x_ = x.transpose(1, 2)
    pairwise_distance = -torch.cdist(x_, x_, p=p)
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    return idx


def knn_l2(x, k):
    return knn_lp(x, k, p=2)


def knn_lp_randproj(x, k, D, p):
    R = torch.randn(x.shape[1], D, device=x.device) * 1 / np.sqrt(D)
    x_ = x.transpose(1, 2) @ R

    pairwise_distance = -torch.cdist(x_, x_, p=p)
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    return idx


def knn_l2_randproj(x, k, D):
    return knn_lp_randproj(x, k, D, p=2)


@torch.jit.script
def fast_pairwise_hamming(X, Y):
    """Calculates all pair-wise Hamming distances for vectors valued in {-1, 1}.
    Tested for correctness.
    """
    B, N, C = X.shape
    return torch.nn.functional.relu(
        -(torch.matmul(X, Y.transpose(1,2)) - C)
    )


def knn_hamming(x, k: int):
    # Formula is for N x L x C but x is N x C x L
    # Note: Could optimize the code by rewriting all knns to expect N x L x C
    x_ = x.transpose(1, 2)

    # Technically this is 2 * Hamming but it doesn't matter for KNN
    # pairwise_distance = - (-(x_[:,:,None,:] * x_[:,None,:,:])+1).sum(-1)
    pairwise_distance = -fast_pairwise_hamming(x_, x_)

    idx = pairwise_distance.topk(k=k, dim=-1)[1]
    return idx


def knn_l1(x, k):
    return knn_lp(x, k, p=1)


KNN_OPS = {
    'l1': knn_l1,
    'l2': knn_l2,
    'hamming': knn_hamming
}


def sub(x_j, x_i):
    return x_j - x_i


def xor(x_j, x_i):
    "Simulates XOR operation for vectors in F{-1,1}"
    return -x_j*x_i


def empty(device):
    return torch.as_tensor([], device=device)


def get_graph_feature(x, k=20, idx=None, knn=knn_l2, edge_op=sub):
    batch_size = x.size(0)
    num_points = x.size(2)

    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)   # (batch_size, num_points, k)

    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points

    idx = idx + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    feature = torch.cat((edge_op(feature, x), x), dim=3).permute(0, 3, 1, 2).contiguous()

    return feature


def get_graph_feature_lsp(x, k=20, idx=None, knn=knn_l2, edge_op=sub):
    batch_size = x.size(0)
    num_points = x.size(2)

    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)   # (batch_size, num_points, k)

    device = x.device

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points

    idx = idx + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    feature = torch.cat((edge_op(feature, x), x), dim=3).permute(0, 3, 1, 2).contiguous()

    # Compute an edge_index representation of the graph
    row_0 = (torch.arange(num_points, device=device).unsqueeze(0).unsqueeze(2).repeat((batch_size, 1, k)) + idx_base).view(-1)
    edge_index = torch.stack((row_0, idx), dim=0)

    return feature, edge_index


class BinWrapperBlock(nn.Module):
    def __init__(self,
                    core_operator,
                    core_operator_kwargs,
                    center_operator,
                    center_operator_kwargs,
                    rescale_operator,
                    rescale_operator_kwargs,
                    binary_weights=True,
                    binary_inputs=True,
                    activation=True,
                    tanh_inputs=False,
                    name=''):
        super().__init__()

        self.quantize_weights = binary_weights
        self.quantize_inputs = binary_inputs
        self.tanh_inputs = tanh_inputs
        self.do_activation = activation

        self.activ = binary.PReLU() if activation else None

        self.center = center_operator(**center_operator_kwargs)

        core_operator_kwargs['binary_weights'] = self.quantize_weights
        self.core_op = core_operator(**core_operator_kwargs)

        self.rescale = rescale_operator(**rescale_operator_kwargs)

        self.name = name

        if self.quantize_inputs:
            self.quantizer = binary.STEQuantizer()
        elif self.tanh_inputs:
            self.quantizer = nn.Tanh()
        else:
            self.quantizer = nn.Identity()

    def reset_parameters(self):
        self.core_op.reset_parameters()
        if self.do_activation:
            self.activ.reset_parameters()

        self.rescale.reset_parameters()

    def forward(self, x):
        x = self.center(x)
        x = self.quantizer(x)

        x = self.core_op(x)
        x = self.rescale(x)
        if self.do_activation:
            x = self.activ(x)

        return x


class BinLinearBlockBN(BinWrapperBlock):
    def __init__(self, in_channels, out_channels, bn_momentum=0.999, **kwargs):
        super().__init__(
            core_operator=binary.BinLinear,
            core_operator_kwargs={'in_channels': in_channels, 'out_channels': out_channels},
            center_operator=nn.BatchNorm1d,
            center_operator_kwargs={'num_features': in_channels, 'momentum': bn_momentum},
            rescale_operator=binary.LearnedRescaleLayer0d,
            rescale_operator_kwargs={'input_shapes': (1, out_channels)},
            **kwargs
        )


class BinConv1dBlockBN(BinWrapperBlock):
    def __init__(self, in_channels, out_channels, bn_momentum=0.999, rescale_L=0, **kwargs):
        super().__init__(
            core_operator=binary.BinConv1d,
            core_operator_kwargs={'in_channels': in_channels, 'out_channels': out_channels, 'kernel_size': 1},
            center_operator=nn.BatchNorm1d,
            center_operator_kwargs={'num_features': in_channels, 'momentum': bn_momentum},
            rescale_operator=binary.LearnedRescaleLayer1d,
            rescale_operator_kwargs={'input_shapes': (1, out_channels, max(1, rescale_L))},
            **kwargs
        )


class BinConv2dBlockBN(BinWrapperBlock):
    def __init__(self, in_channels, out_channels, bn_momentum=0.999, rescale_H=0, rescale_W=0, **kwargs):
        super().__init__(
            core_operator=binary.BinConv2d,
            core_operator_kwargs={'in_channels': in_channels, 'out_channels': out_channels, 'kernel_size': 1},
            center_operator=nn.BatchNorm2d,
            center_operator_kwargs={'num_features': in_channels, 'momentum': bn_momentum},
            rescale_operator=binary.LearnedRescaleLayer2d,
            rescale_operator_kwargs={'input_shapes': (1, out_channels, max(1, rescale_H), max(1, rescale_W))},
            **kwargs
        )

class BinLinearBlockNoBN(BinWrapperBlock):
    def __init__(self, in_channels, out_channels, bn_momentum=0.999, **kwargs):
        super().__init__(
            core_operator=binary.BinLinear,
            core_operator_kwargs={'in_channels': in_channels, 'out_channels': out_channels},
            center_operator=nn.BatchNorm1d,
            center_operator_kwargs={'num_features': in_channels, 'momentum': bn_momentum},
            rescale_operator=binary.LearnedRescaleLayer0d,
            rescale_operator_kwargs={'input_shapes': (1, out_channels)},
            **kwargs
        )

    def forward(self, x):
        x = self.quantizer(x)

        x = self.core_op(x)
        x = self.rescale(x)
        if self.do_activation:
            x = self.activ(x)

        return x