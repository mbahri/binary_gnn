"""
@Author: Mehdi Bahri
@Contact: m.bahri@imperial.ac.uk
@File: models_extended.py

Based on "model.py" by Yue Wang
https://github.com/WangYueFt/dgcnn/blob/master/pytorch/model.py
"""

import os
import sys
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import dgcnn.models_common as mc

import binary


class BaseDGCNN(nn.Module):
    def __init__(self, args, output_channels=40, max_pool=True, avg_pool=True):
        super().__init__()
        self.args = args
        self.k = args.k

        self.max_pool = max_pool
        self.avg_pool = avg_pool

        if self.max_pool and self.avg_pool:
            embd_in_scale = 2
        else:
            embd_in_scale = 1

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(args.emb_dims)

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 128, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(128*2, 256, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(512, args.emb_dims, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.linear1 = nn.Linear(args.emb_dims*embd_in_scale, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=args.dropout)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=args.dropout)
        self.linear3 = nn.Linear(256, output_channels)

    def forward(self, x0):
        batch_size = x0.size(0)
        num_points = x0.size(2)

        x, ei0 = mc.get_graph_feature_lsp(x0, k=self.k)
        x = self.conv1(x)
        x1 = x.max(dim=-1, keepdim=False)[0]

        x, ei1 = mc.get_graph_feature_lsp(x1, k=self.k)
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0]

        x, ei2 = mc.get_graph_feature_lsp(x2, k=self.k)
        x = self.conv3(x)
        x3 = x.max(dim=-1, keepdim=False)[0]

        x, ei3 = mc.get_graph_feature_lsp(x3, k=self.k)
        x = self.conv4(x)
        x4 = x.max(dim=-1, keepdim=False)[0]

        x = torch.cat((x1, x2, x3, x4), dim=1)

        x = self.conv5(x)

        if self.max_pool:
            x_max = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        else:
            x_max = mc.empty(x.device)

        if self.avg_pool:
            x_avg = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        else:
            x_avg = mc.empty(x.device)

        x = torch.cat((x_max, x_avg), 1)

        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        x = self.linear3(x)
        return x, (
            (x0.transpose(2,1).contiguous().view(batch_size*num_points, -1), ei0),
            (x1.transpose(2,1).contiguous().view(batch_size*num_points, -1), ei1),
            (x2.transpose(2,1).contiguous().view(batch_size*num_points, -1), ei2),
            (x3.transpose(2,1).contiguous().view(batch_size*num_points, -1), ei3)
        )


class BinDGCNN_RF(nn.Module):
    """Binary DGCNN with real-valued graph features
    XNOR-Net++
    """
    def __init__(self, args, output_channels=40, max_pool=True, avg_pool=True):
        super().__init__()

        self.args = args
        self.k = args.k

        self.max_pool = max_pool
        self.avg_pool = avg_pool

        if self.max_pool and self.avg_pool:
            embd_in_scale = 2
        else:
            embd_in_scale = 1

        # Parameters for XNOR-Net++
        self.bn_momentum = args.bin_bn_momentum
        self.binary_weights = args.bin_quantize_weights
        self.binary_inputs = args.bin_quantize_inputs
        self.activ_in_blocks = args.bin_prelu_in_blocks
        self.pseudo_quantize = args.bin_pseudo_quantize
        self.rescale_H = args.num_points if args.bin_rescale_H else 1
        self.rescale_W = args.k if args.bin_rescale_W else 1
        self.rescale_L = args.num_points if args.bin_rescale_L else 1

        self.last_layer_binary_weights = args.bin_ll_quantize_weights
        self.last_layer_binary_inputs = args.bin_ll_quantize_inputs
        self.last_layer_pseudo_quantize = args.bin_ll_pseudo_quantize

        self.conv1 = mc.BinConv2dBlockBN(6, 64, bn_momentum=self.bn_momentum,
                            binary_weights=self.binary_weights,
                            binary_inputs=False,
                            activation=self.activ_in_blocks,
                            tanh_inputs=self.pseudo_quantize,
                            name='edgeconv1_conv2d',
                            rescale_H=self.rescale_H,
                            rescale_W=self.rescale_W
        )

        self.conv2 = mc.BinConv2dBlockBN(64*2, 64, bn_momentum=self.bn_momentum,
                                    binary_weights=self.binary_weights,
                                    binary_inputs=self.binary_inputs,
                                    activation=self.activ_in_blocks,
                                    tanh_inputs=self.pseudo_quantize,
                                    name='edgeconv2_conv2d',
                                    rescale_H=self.rescale_H,
                                    rescale_W=self.rescale_W
        )

        self.conv3 = mc.BinConv2dBlockBN(64*2, 128, bn_momentum=self.bn_momentum,
                                    binary_weights=self.binary_weights,
                                    binary_inputs=self.binary_inputs,
                                    activation=self.activ_in_blocks,
                                    tanh_inputs=self.pseudo_quantize,
                                    name='edgeconv3_conv2d',
                                    rescale_H=self.rescale_H,
                                    rescale_W=self.rescale_W
        )

        self.conv4 = mc.BinConv2dBlockBN(128*2, 256, bn_momentum=self.bn_momentum,
                                    binary_weights=self.binary_weights,
                                    binary_inputs=self.binary_inputs,
                                    activation=self.activ_in_blocks,
                                    tanh_inputs=self.pseudo_quantize,
                                    name='edgeconv4_conv2d',
                                    rescale_H=self.rescale_H,
                                    rescale_W=self.rescale_W
        )

        self.conv5 = mc.BinConv1dBlockBN(512, args.emb_dims, bn_momentum=self.bn_momentum,
                                    binary_weights=self.binary_weights,
                                    binary_inputs=self.binary_inputs,
                                    activation=self.activ_in_blocks,
                                    tanh_inputs=self.pseudo_quantize,
                                    name='map_to_latent_conv1d',
                                    rescale_L=self.rescale_L
        )

        ##################################################################################################################

        self.linear1 = mc.BinLinearBlockBN(args.emb_dims*embd_in_scale, 512, bn_momentum=self.bn_momentum,
                                    binary_weights=self.binary_weights,
                                    binary_inputs=self.binary_inputs,
                                    activation=self.activ_in_blocks,
                                    tanh_inputs=self.pseudo_quantize,
                                    name='mlp_lin1'
        )

        self.linear2 = mc.BinLinearBlockBN(512, 256, bn_momentum=self.bn_momentum,
                                    binary_weights=self.binary_weights,
                                    binary_inputs=self.binary_inputs,
                                    activation=self.activ_in_blocks,
                                    tanh_inputs=self.pseudo_quantize,
                                    name='mlp_lin2'
        )

        self.linear3 = mc.BinLinearBlockBN(256, output_channels, bn_momentum=self.bn_momentum,
                                    binary_weights=self.last_layer_binary_weights,
                                    binary_inputs=self.last_layer_binary_inputs,
                                    activation=False,
                                    tanh_inputs=self.last_layer_pseudo_quantize,
                                    name='mlp_lin3'
        )

        self.dp1 = nn.Dropout(p=args.dropout) if args.dropout > 0 else nn.Identity()
        self.dp2 = nn.Dropout(p=args.dropout) if args.dropout > 0 else nn.Identity()

    def forward(self, x0):
        batch_size = x0.size(0)
        num_points = x0.size(2)

        x, ei0 = mc.get_graph_feature_lsp(x0, k=self.k)
        x = self.conv1(x)
        x1 = x.max(dim=-1, keepdim=False)[0]

        x, ei1 = mc.get_graph_feature_lsp(x1, k=self.k)
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0]

        x, ei2 = mc.get_graph_feature_lsp(x2, k=self.k)
        x = self.conv3(x)
        x3 = x.max(dim=-1, keepdim=False)[0]

        x, ei3 = mc.get_graph_feature_lsp(x3, k=self.k)
        x = self.conv4(x)
        x4 = x.max(dim=-1, keepdim=False)[0]

        x = torch.cat((x1, x2, x3, x4), dim=1)

        x = self.conv5(x)

        if self.max_pool:
            x_max = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        else:
            x_max = mc.empty(x.device)

        if self.avg_pool:
            x_avg = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        else:
            x_avg = mc.empty(x.device)

        x = torch.cat((x_max, x_avg), 1)

        x = self.linear1(x)
        x = self.dp1(x)
        x = self.linear2(x)
        x = self.dp2(x)
        x = self.linear3(x)

        return x, (
            (x0.transpose(2,1).contiguous().view(batch_size*num_points, -1), ei0),
            (x1.transpose(2,1).contiguous().view(batch_size*num_points, -1), ei1),
            (x2.transpose(2,1).contiguous().view(batch_size*num_points, -1), ei2),
            (x3.transpose(2,1).contiguous().view(batch_size*num_points, -1), ei3)
        )


class BinDGCNN_BF1(nn.Module):
    def __init__(self, args, output_channels=40, max_pool=True, avg_pool=True):
        super().__init__()
        self.args = args
        self.k = args.k

        self.max_pool = max_pool
        self.avg_pool = avg_pool

        if self.max_pool and self.avg_pool:
            embd_in_scale = 2
        else:
            embd_in_scale = 1

        # Parameters for XNOR-Net++
        self.bn_momentum = args.bin_bn_momentum
        self.binary_weights = args.bin_quantize_weights
        self.binary_inputs = args.bin_quantize_inputs
        self.activ_in_blocks = args.bin_prelu_in_blocks
        self.pseudo_quantize = args.bin_pseudo_quantize
        self.rescale_H = args.num_points if args.bin_rescale_H else 1
        self.rescale_W = args.k if args.bin_rescale_W else 1
        self.rescale_L = args.num_points if args.bin_rescale_L else 1

        self.last_layer_binary_weights = args.bin_ll_quantize_weights
        self.last_layer_binary_inputs = args.bin_ll_quantize_inputs
        self.last_layer_pseudo_quantize = args.bin_ll_pseudo_quantize

        self.conv_prelu = args.bin_conv_prelu

        self.bn1 = nn.BatchNorm1d(64, momentum=self.bn_momentum)
        self.bn2 = nn.BatchNorm1d(64, momentum=self.bn_momentum)
        self.bn3 = nn.BatchNorm1d(128, momentum=self.bn_momentum)
        self.bn4 = nn.BatchNorm1d(256, momentum=self.bn_momentum)
        self.bn5 = nn.BatchNorm1d(args.emb_dims, momentum=self.bn_momentum)

        self.conv1 = nn.Sequential(binary.BinConv2d(6, 64, kernel_size=1, bias=False, binary_weights=self.binary_weights),
                                   binary.LearnedRescaleLayer2d((1, 64, self.rescale_H, self.rescale_W)),
                                   nn.PReLU() if self.conv_prelu else nn.Identity() )

        self.conv2 = nn.Sequential(binary.BinConv2d(64*2, 64, kernel_size=1, bias=False, binary_weights=self.binary_weights),
                                   binary.LearnedRescaleLayer2d((1, 64, self.rescale_H, self.rescale_W)),
                                   nn.PReLU() if self.conv_prelu else nn.Identity() )

        self.conv3 = nn.Sequential(binary.BinConv2d(64*2, 128, kernel_size=1, bias=False, binary_weights=self.binary_weights),
                                   binary.LearnedRescaleLayer2d((1, 128, self.rescale_H, self.rescale_W)),
                                   nn.PReLU() if self.conv_prelu else nn.Identity() )

        self.conv4 = nn.Sequential(binary.BinConv2d(128*2, 256, kernel_size=1, bias=False, binary_weights=self.binary_weights),
                                   binary.LearnedRescaleLayer2d((1, 256, self.rescale_H, self.rescale_W)),
                                   nn.PReLU() if self.conv_prelu else nn.Identity() )

        self.conv5 = nn.Sequential(binary.BinConv1d(512, args.emb_dims, kernel_size=1, bias=False, binary_weights=self.binary_weights),
                                   binary.LearnedRescaleLayer1d((1, args.emb_dims, self.rescale_L)),
                                   nn.PReLU() if self.conv_prelu else nn.Identity() )

        ##################################################################################################################

        self.linear1 = mc.BinLinearBlockBN(args.emb_dims*embd_in_scale, 512, bn_momentum=self.bn_momentum,
                                    binary_weights=self.binary_weights,
                                    binary_inputs=self.binary_inputs,
                                    activation=self.activ_in_blocks,
                                    tanh_inputs=self.pseudo_quantize,
                                    name='mlp_lin1'
        )

        self.linear2 = mc.BinLinearBlockBN(512, 256, bn_momentum=self.bn_momentum,
                                    binary_weights=self.binary_weights,
                                    binary_inputs=self.binary_inputs,
                                    activation=self.activ_in_blocks,
                                    tanh_inputs=self.pseudo_quantize,
                                    name='mlp_lin2'
        )

        self.linear3 = mc.BinLinearBlockBN(256, output_channels, bn_momentum=self.bn_momentum,
                                    binary_weights=self.last_layer_binary_weights,
                                    binary_inputs=self.last_layer_binary_inputs,
                                    activation=False,
                                    tanh_inputs=self.last_layer_pseudo_quantize,
                                    name='mlp_lin3'
        )

        self.knn_op = mc.KNN_OPS[args.bin_knn_op]

        if self.binary_inputs:
            self.quantizer = binary.STEQuantizer()
            self.edge_op = mc.xor
        elif self.pseudo_quantize:
            self.quantizer = nn.Tanh()
            self.edge_op = mc.sub
        else:
            self.quantizer = nn.Identity()
            self.edge_op = mc.sub

        self.dp1 = nn.Dropout(p=args.dropout) if args.dropout > 0 else nn.Identity()
        self.dp2 = nn.Dropout(p=args.dropout) if args.dropout > 0 else nn.Identity()

    def forward(self, x0):
        batch_size = x0.size(0)
        num_points = x0.size(2)

        x, ei0 = mc.get_graph_feature_lsp(x0, k=self.k)
        x = self.conv1(x)
        x1 = x.max(dim=-1, keepdim=False)[0]
        x1 = self.quantizer(self.bn1(x1))

        x, ei1 = mc.get_graph_feature_lsp(x1, k=self.k, knn=self.knn_op, edge_op=self.edge_op)
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0]
        x2 = self.quantizer(self.bn2(x2))

        x, ei2 = mc.get_graph_feature_lsp(x2, k=self.k, knn=self.knn_op, edge_op=self.edge_op)
        x = self.conv3(x)
        x3 = x.max(dim=-1, keepdim=False)[0]
        x3 = self.quantizer(self.bn3(x3))

        x, ei3 = mc.get_graph_feature_lsp(x3, k=self.k, knn=self.knn_op, edge_op=self.edge_op)
        x = self.conv4(x)
        x4 = x.max(dim=-1, keepdim=False)[0]
        x4 = self.quantizer(self.bn4(x4))

        x = torch.cat((x1, x2, x3, x4), dim=1)

        x = self.bn5(self.conv5(x))

        if self.max_pool:
            x_max = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        else:
            x_max = mc.empty(x.device)

        if self.avg_pool:
            x_avg = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        else:
            x_avg = mc.empty(x.device)

        x = torch.cat((x_max, x_avg), 1)

        x = self.linear1(x)
        x = self.dp1(x)
        x = self.linear2(x)
        x = self.dp2(x)
        x = self.linear3(x)
        return x, (
            (x0.transpose(2,1).contiguous().view(batch_size*num_points, -1), ei0),
            (x1.transpose(2,1).contiguous().view(batch_size*num_points, -1), ei1),
            (x2.transpose(2,1).contiguous().view(batch_size*num_points, -1), ei2),
            (x3.transpose(2,1).contiguous().view(batch_size*num_points, -1), ei3)
        )


class BinDGCNN_BF2(BinDGCNN_BF1):
    def __init__(self, args, output_channels=40, max_pool=True, avg_pool=True):
        super().__init__(args, output_channels, max_pool, avg_pool)

        self.bn1 = nn.BatchNorm2d(64, momentum=self.bn_momentum)
        self.bn2 = nn.BatchNorm2d(64, momentum=self.bn_momentum)
        self.bn3 = nn.BatchNorm2d(128, momentum=self.bn_momentum)
        self.bn4 = nn.BatchNorm2d(256, momentum=self.bn_momentum)
        self.bn5 = nn.BatchNorm1d(args.emb_dims, momentum=self.bn_momentum)


    def forward(self, x0):
        batch_size = x0.size(0)
        num_points = x0.size(2)

        x, ei0 = mc.get_graph_feature_lsp(x0, k=self.k)
        x = self.bn1(self.conv1(x))
        x1 = x.max(dim=-1, keepdim=False)[0]
        x1 = self.quantizer(x1)

        x, ei1 = mc.get_graph_feature_lsp(x1, k=self.k, knn=self.knn_op, edge_op=self.edge_op)
        x = self.bn2(self.conv2(x))
        x2 = x.max(dim=-1, keepdim=False)[0]
        x2 = self.quantizer(x2)

        x, ei2 = mc.get_graph_feature_lsp(x2, k=self.k, knn=self.knn_op, edge_op=self.edge_op)
        x = self.bn3(self.conv3(x))
        x3 = x.max(dim=-1, keepdim=False)[0]
        x3 = self.quantizer(x3)

        x, ei3 = mc.get_graph_feature_lsp(x3, k=self.k, knn=self.knn_op, edge_op=self.edge_op)
        x = self.bn4(self.conv4(x))
        x4 = x.max(dim=-1, keepdim=False)[0]
        x4 = self.quantizer(x4)

        x = torch.cat((x1, x2, x3, x4), dim=1)

        x = self.bn5(self.conv5(x))

        if self.max_pool:
            x_max = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        else:
            x_max = mc.empty(x.device)

        if self.avg_pool:
            x_avg = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        else:
            x_avg = mc.empty(x.device)

        x = torch.cat((x_max, x_avg), 1)

        x = self.linear1(x)
        x = self.dp1(x)
        x = self.linear2(x)
        x = self.dp2(x)
        x = self.linear3(x)
        return x, (
            (x0.transpose(2,1).contiguous().view(batch_size*num_points, -1), ei0),
            (x1.transpose(2,1).contiguous().view(batch_size*num_points, -1), ei1),
            (x2.transpose(2,1).contiguous().view(batch_size*num_points, -1), ei2),
            (x3.transpose(2,1).contiguous().view(batch_size*num_points, -1), ei3)
        )


class BinDGCNN_BF1_BAL(BinDGCNN_BF1):
    def __init__(self, args, output_channels=40, max_pool=True, avg_pool=True):
        super().__init__(args, output_channels, max_pool, avg_pool)

        self.max_pool = max_pool
        self.avg_pool = avg_pool

        if self.max_pool and self.avg_pool:
            embd_in_scale = 2
        else:
            embd_in_scale = 1

        self.bn1 = nn.BatchNorm2d(64, momentum=self.bn_momentum)
        self.bn2 = nn.BatchNorm2d(64, momentum=self.bn_momentum)
        self.bn3 = nn.BatchNorm2d(128, momentum=self.bn_momentum)
        self.bn4 = nn.BatchNorm2d(256, momentum=self.bn_momentum)
        self.bn5 = nn.BatchNorm1d(args.emb_dims, momentum=self.bn_momentum)

        self.linear1 = mc.BinLinearBlockNoBN(args.emb_dims*embd_in_scale, 512, bn_momentum=self.bn_momentum,
                                    binary_weights=self.binary_weights,
                                    binary_inputs=self.binary_inputs,
                                    activation=self.activ_in_blocks,
                                    tanh_inputs=self.pseudo_quantize,
                                    name='mlp_lin1'
        )

        self.global_balance_fcts = {
            'none': nn.Identity(),
            'mean': binary.MeanCenter(axis=args.bin_global_balance_axis),
            'median': binary.MedianCenter(axis=args.bin_global_balance_axis),
        }
        self.global_balance_fct = self.global_balance_fcts[args.bin_global_balance_op]

        self.edge_balance_fcts = {
            'none': nn.Identity(),
            'mean': binary.MeanCenter(axis=args.bin_edge_balance_axis),
            'median': binary.MedianCenter(axis=args.bin_edge_balance_axis),
        }
        self.edge_balance_fct = self.edge_balance_fcts[args.bin_edge_balance_op]


    def forward(self, x0):
        batch_size = x0.size(0)
        num_points = x0.size(2)

        x, ei0 = mc.get_graph_feature_lsp(x0, k=self.k)
        x = self.bn1(self.conv1(x))
        x1 = x.max(dim=-1, keepdim=False)[0]
        x1 = self.quantizer( self.edge_balance_fct(x1) )

        x, ei1 = mc.get_graph_feature_lsp(x1, k=self.k, knn=self.knn_op, edge_op=self.edge_op)
        x = self.bn2(self.conv2(x))
        x2 = x.max(dim=-1, keepdim=False)[0]
        x2 = self.quantizer( self.edge_balance_fct(x2) )

        x, ei2 = mc.get_graph_feature_lsp(x2, k=self.k, knn=self.knn_op, edge_op=self.edge_op)
        x = self.bn3(self.conv3(x))
        x3 = x.max(dim=-1, keepdim=False)[0]
        x3 = self.quantizer( self.edge_balance_fct(x3) )

        x, ei3 = mc.get_graph_feature_lsp(x3, k=self.k, knn=self.knn_op, edge_op=self.edge_op)
        x = self.bn4(self.conv4(x))
        x4 = x.max(dim=-1, keepdim=False)[0]
        x4 = self.quantizer( self.edge_balance_fct(x4) )

        x = torch.cat((x1, x2, x3, x4), dim=1)

        x = self.bn5(self.conv5(x))

        if self.max_pool:
            x_max = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        else:
            x_max = mc.empty(x.device)

        if self.avg_pool:
            x_avg = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        else:
            x_avg = mc.empty(x.device)

        x = torch.cat((x_max, x_avg), 1)
        x = self.global_balance_fct(x)

        x = self.linear1(x)
        x = self.dp1(x)
        x = self.linear2(x)
        x = self.dp2(x)
        x = self.linear3(x)
        return x, (
            (x0.transpose(2,1).contiguous().view(batch_size*num_points, -1), ei0),
            (x1.transpose(2,1).contiguous().view(batch_size*num_points, -1), ei1),
            (x2.transpose(2,1).contiguous().view(batch_size*num_points, -1), ei2),
            (x3.transpose(2,1).contiguous().view(batch_size*num_points, -1), ei3)
        )


MODELS_DICT = {
    'dgcnn': BaseDGCNN,
    'bin_rf': BinDGCNN_RF,
    'bin_bf1': BinDGCNN_BF1,
    'bin_bf2': BinDGCNN_BF2,
    'bin_bf1_bal': BinDGCNN_BF1_BAL,
}
