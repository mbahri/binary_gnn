#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Mehdi Bahri
@Contact: m.bahri@imperial.ac.uk
@File: distill_dgcnn.py

Based on "main.py" by Yue Wang
https://github.com/WangYueFt/dgcnn/blob/master/pytorch/main.py
"""


from __future__ import print_function
import os
import sys
import copy
import wandb
import argparse

import numpy as np
import sklearn.metrics as metrics
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR, StepLR

from dgcnn.data import ModelNet40
from dgcnn.util import cal_loss, IOStream
from dgcnn import models_extended
from dgcnn import models_common as mc

import distillation



def _init_():
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    if not os.path.exists('checkpoints/'+args.exp_name):
        os.makedirs('checkpoints/'+args.exp_name)
    if not os.path.exists('checkpoints/'+args.exp_name+'/'+'models'):
        os.makedirs('checkpoints/'+args.exp_name+'/'+'models')
    os.system('cp dgcnn_distill.py checkpoints'+'/'+args.exp_name+'/'+'dgcnn_distill.py.backup')
    os.system('cp dgcnn/models_extended.py checkpoints' + '/' + args.exp_name + '/' + 'models_extended.py.backup')
    os.system('cp dgcnn/util.py checkpoints' + '/' + args.exp_name + '/' + 'util.py.backup')
    os.system('cp dgcnn/data.py checkpoints' + '/' + args.exp_name + '/' + 'data.py.backup')

def train(args, io):
    train_loader = DataLoader(ModelNet40(partition='train', num_points=args.num_points), num_workers=8,
                              batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(ModelNet40(partition='test', num_points=args.num_points), num_workers=8,
                             batch_size=args.test_batch_size, shuffle=True, drop_last=False)

    device = torch.device("cuda" if args.cuda else "cpu")

    # Try to load models
    model = models_extended.MODELS_DICT[args.student_model](args).to(device)

    print(str(model))
    model = nn.DataParallel(model)

    # Copy the args and override the parameters of the binarization with those given
    # for the teacher (would be better with yaml config files...)
    teacher_args = copy.deepcopy(args)
    teacher_args.bin_bn_momentum = args.teacher_bn_momentum
    teacher_args.bin_quantize_weights = args.teacher_quantize_weights
    teacher_args.bin_quantize_inputs = args.teacher_quantize_inputs
    teacher_args.bin_prelu_in_blocks = args.teacher_prelu_in_blocks
    teacher_args.bin_pseudo_quantize = args.teacher_pseudo_quantize
    teacher_args.bin_rescale_H = args.teacher_rescale_H
    teacher_args.bin_rescale_W = args.teacher_rescale_W
    teacher_args.bin_rescale_L = args.teacher_rescale_L
    teacher_args.bin_knn_op = args.teacher_knn_op
    teacher_args.bin_conv_prelu = args.bin_conv_prelu

    teacher_args.bin_global_balance_axis = args.teacher_global_balance_axis
    teacher_args.bin_global_balance_op = args.teacher_global_balance_op
    teacher_args.bin_edge_balance_axis = args.teacher_edge_balance_axis
    teacher_args.bin_edge_balance_op = args.teacher_edge_balance_op

    teacher_args.bin_ll_quantize_weights = args.teacher_ll_quantize_weights
    teacher_args.bin_ll_quantize_inputs = args.teacher_ll_quantize_inputs
    teacher_args.bin_ll_pseudo_quantize = args.teacher_ll_pseudo_quantize

    teacher = models_extended.MODELS_DICT[args.teacher_model](teacher_args).to(device)

    teacher = nn.DataParallel(teacher)
    teacher_state = torch.load(args.teacher_path, map_location=device)
    teacher.load_state_dict(teacher_state)
    teacher.eval()

    print('Checking teacher accuracy on the test set:')
    mc.test_universal(teacher, test_loader, device, io, 'teacher_')

    if args.init_student_with_teacher_weights:
        print('Loading teacher weights in the student')
        model.load_state_dict(teacher_state)
    else:
        print('Student starting from scratch')

    print('Checking student accuracy on the test set')
    mc.test_universal(model, test_loader, device, io, 'init_')

    print("Let's use", torch.cuda.device_count(), "GPUs!")

    # Initialize the distillation modules
    if args.LSP_kernels == 'l2_l2':
        lsp_kernel_student, lsp_kernel_teacher = distillation.rbf_l2, distillation.rbf_l2
    elif args.LSP_kernels == 'hamming_l2':
        lsp_kernel_student, lsp_kernel_teacher = distillation.rbf_hamming, distillation.rbf_l2
    elif args.LSP_kernels == 'hamming_sq_l2':
        lsp_kernel_student, lsp_kernel_teacher = distillation.rbf_hamming_sq, distillation.rbf_l2
    elif args.LSP_kernels == 'hamming_hamming':
        lsp_kernel_student, lsp_kernel_teacher = distillation.rbf_hamming, distillation.rbf_hamming
    elif args.LSP_kernels == 'h_sq_h_sq':
        lsp_kernel_student, lsp_kernel_teacher = distillation.rbf_hamming_sq, distillation.rbf_hamming_sq
    else:
        raise NotImplementedError('Invalid kernel functions.')
    LSP_KERNELS = distillation.StructuralSimilarity(kernel_s=lsp_kernel_student, kernel_t=lsp_kernel_teacher)
    CET = distillation.CrossEntropyWithTemperature(T=args.KD_T)

    if args.use_sgd:
        print("Use SGD")
        opt = optim.SGD(model.parameters(), lr=args.lr*100, momentum=args.momentum, weight_decay=args.wd)
    else:
        print("Use Adam")
        opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

    # Different schedulers with the options used for the experiments
    if args.scheduler == 'step':
        scheduler = StepLR(opt, step_size=50, gamma=0.5)
    elif args.scheduler == 'multistep':
        scheduler = MultiStepLR(opt, milestones=[args.epochs // 2, 3 * args.epochs // 4], gamma=0.5)
    elif args.scheduler == 'cosine':
        scheduler = CosineAnnealingLR(opt, args.epochs, eta_min=args.lr)
    else:
        raise NotImplemented('Invalid LR scheduler, must be one of step, multistep, or cosine.')

    criterion = cal_loss

    wandb.watch(model, log='all', log_freq=50)
    wandb.watch(teacher, log='all', log_freq=50)
    wandb.watch(LSP_KERNELS, log='all', log_freq=50)
    wandb.watch(CET, log='all', log_freq=50)

    best_test_acc = 0
    best_avg_per_class_acc = 0
    for epoch in tqdm(range(args.epochs)):
        ####################
        # Train
        ####################
        train_loss = 0.0
        count = 0.0
        model.train()
        train_pred = []
        train_true = []
        kl1s, kl2s, kl3s, klls = [], [], [], []
        for data, label in tqdm(train_loader):
            data, label = data.to(device), label.to(device).squeeze()
            data = data.permute(0, 2, 1)
            batch_size = data.size()[0]
            opt.zero_grad()

            logits, structures_s = model(data)
            with torch.no_grad():
                logits_t, structures_t = teacher(data)


            # Similarity between (binary?) student features and real teacher features
            KL1 = LSP_KERNELS(structures_s[1][0], structures_t[1][0], structures_s[1][1], structures_t[1][1])
            KL2 = LSP_KERNELS(structures_s[2][0], structures_t[2][0], structures_s[2][1], structures_t[2][1])
            KL3 = LSP_KERNELS(structures_s[3][0], structures_t[3][0], structures_s[3][1], structures_t[3][1])

            # Logit matching loss
            KLL = CET(logits, logits_t)

            kl1s.append(KL1.item())
            kl2s.append(KL2.item())
            kl3s.append(KL3.item())
            klls.append(KLL.item())

            loss = (1 - args.KD_alpha) * criterion(logits, label) + args.KD_alpha * KLL + args.LSP_lambda * (KL1 + KL2 + KL3) / 3.0

            loss.backward()
            opt.step()
            preds = logits.max(dim=1)[1]
            count += batch_size
            train_loss += loss.item() * batch_size
            train_true.append(label.cpu().numpy())
            train_pred.append(preds.detach().cpu().numpy())
        scheduler.step()
        train_true = np.concatenate(train_true)
        train_pred = np.concatenate(train_pred)

        avg_kl1, avg_kl2, avg_kl3, avg_kll = np.mean(kl1s), np.mean(kl2s), np.mean(kl3s), np.mean(klls)

        outstr = 'Train %d, loss: %.6f, train acc: %.6f, train avg acc: %.6f | KL1: %.3e - KL2: %.3e - KL3: %.3e - KLL: %.3e' % (epoch,
                                                                                 train_loss*1.0/count,
                                                                                 metrics.accuracy_score(
                                                                                     train_true, train_pred),
                                                                                 metrics.balanced_accuracy_score(
                                                                                     train_true, train_pred),
                                                                                 avg_kl1, avg_kl2, avg_kl3, avg_kll
                                                                                )
        wandb.log({
            'train_loss': train_loss*1.0/count,
            'accuracy': metrics.accuracy_score(train_true, train_pred),
            'balanced_accuracy': metrics.balanced_accuracy_score(train_true, train_pred),
            'avg_kl1': avg_kl1,
            'avg_kl2': avg_kl2,
            'avg_kl3': avg_kl3,
            'avg_logmal': avg_kll,
            'lr': opt.param_groups[0]["lr"]
        })
        io.cprint(outstr)

        ####################
        # Test
        ####################
        test_loss = 0.0
        count = 0.0
        model.eval()
        test_pred = []
        test_true = []
        for data, label in test_loader:
            data, label = data.to(device), label.to(device).squeeze()
            data = data.permute(0, 2, 1)
            batch_size = data.size()[0]
            logits, _ = model(data)
            loss = criterion(logits, label)
            preds = logits.max(dim=1)[1]
            count += batch_size
            test_loss += loss.item() * batch_size
            test_true.append(label.cpu().numpy())
            test_pred.append(preds.detach().cpu().numpy())
        test_true = np.concatenate(test_true)
        test_pred = np.concatenate(test_pred)
        test_acc = metrics.accuracy_score(test_true, test_pred)
        avg_per_class_acc = metrics.balanced_accuracy_score(test_true, test_pred)
        outstr = 'Test %d, loss: %.6f, test acc: %.6f, test avg acc: %.6f' % (epoch,
                                                                              test_loss*1.0/count,
                                                                              test_acc,
                                                                              avg_per_class_acc)
        io.cprint(outstr)
        if test_acc >= best_test_acc:
            best_test_acc = test_acc
            torch.save(model.state_dict(), 'checkpoints/%s/models/model.t7' % args.exp_name)
        if avg_per_class_acc >= best_avg_per_class_acc:
            best_avg_per_class_acc = avg_per_class_acc
            torch.save(model.state_dict(), 'checkpoints/%s/models/model_best_avg.t7' % args.exp_name)

        wandb.log(
            {
                'test_acc': test_acc,
                'test_avg_acc': avg_per_class_acc,
                'best_test_acc': best_test_acc,
                'best_test_avg_acc': best_avg_per_class_acc,
            }
        )


def test(args, io):
    test_loader = DataLoader(ModelNet40(partition='test', num_points=args.num_points),
                             batch_size=args.test_batch_size, shuffle=True, drop_last=False)

    device = torch.device("cuda" if args.cuda else "cpu")

    #Try to load models
    model = DGCNN(args).to(device)
    model = nn.DataParallel(model)
    model.load_state_dict(torch.load(args.model_path))

    mc.test_universal(model, test_loader, device, io)


if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='Point Cloud Recognition')
    parser.add_argument('--exp_name', type=str, default='exp', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--stage', type=int, default=1, metavar='N')
    parser.add_argument('--suffix', type=str, default='', metavar='N')
    parser.add_argument('--additional_suffix', type=str, default='', metavar='N')

    parser.add_argument('--student_model', type=str, default='dgcnn', metavar='N')

    parser.add_argument('--dataset', type=str, default='modelnet40', metavar='N',
                        choices=['modelnet40'])
    parser.add_argument('--batch_size', type=int, default=32, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--test_batch_size', type=int, default=16, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=250, metavar='N',
                        help='number of episode to train ')
    parser.add_argument('--use_sgd', type=mc.str2bool, default=False,
                        help='Use SGD')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001, 0.1 if using sgd)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--no_cuda', type=mc.str2bool, default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--eval', type=mc.str2bool,  default=False,
                        help='evaluate the model')
    parser.add_argument('--num_points', type=int, default=1024,
                        help='num of points to use')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout rate')
    parser.add_argument('--emb_dims', type=int, default=1024, metavar='N',
                        help='Dimension of embeddings')
    parser.add_argument('--k', type=int, default=20, metavar='N',
                        help='Num of nearest neighbors to use')
    parser.add_argument('--model_path', type=str, default='', metavar='N',
                        help='Pretrained model path')

    parser.add_argument('--scheduler', type=str, default='multistep', metavar='N', help='Learning rate scheduler to use')

    parser.add_argument('--wd', type=float, default=1e-5, metavar='M', help='Weight decay.')

    # Binary networks and multi-stage experiments - STUDENT NETWORK
    parser.add_argument('--bin_bn_momentum', type=float, default=0.999, help='Momentum for BN used in the binary blocks')
    parser.add_argument('--bin_quantize_weights', type=mc.str2bool, default=False, help='Binarize the weights in the conv/linear')
    parser.add_argument('--bin_quantize_inputs', type=mc.str2bool, default=False, help='Binarize the input to the conv / the graph features (depending on the model)')
    parser.add_argument('--bin_prelu_in_blocks', type=mc.str2bool, default=True, help='Apply PReLU activation in the binary blocks')
    parser.add_argument('--bin_pseudo_quantize', type=mc.str2bool, default=True, help='Pseudo-quantize with tanh instead of sign')
    parser.add_argument('--bin_rescale_H', type=mc.str2bool, default=False, help='Learnable rescaling in the H dimension (# points)')
    parser.add_argument('--bin_rescale_W', type=mc.str2bool, default=False, help='Learnable rescaling in the W dimension (k neighbours)')
    parser.add_argument('--bin_rescale_L', type=mc.str2bool, default=False, help='Learnable rescaling in the L dimension for the last conv before the MLP (# points)')
    parser.add_argument('--bin_knn_op', type=str, default='l2', choices=['l1', 'l2', 'hamming'], help='Distance used to find the KNNs in feature space')
    parser.add_argument('--bin_conv_prelu', type=mc.str2bool, default=True, help='Use PReLU after bin conv')

    parser.add_argument('--bin_global_balance_axis', type=int, default=0, help='Axis on which to perform mean/median centering')
    parser.add_argument('--bin_global_balance_op', type=str, default='none', choices=['mean', 'median', 'none'], help='Global balance function')
    parser.add_argument('--bin_edge_balance_axis', type=int, default=0, help='Axis on which to perform mean/median centering')
    parser.add_argument('--bin_edge_balance_op', type=str, default='none', choices=['mean', 'median', 'none'], help='Edge balance function')

    parser.add_argument('--bin_ll_quantize_weights', type=mc.str2bool, default=False, help='Quantize the weights in the last linear layer')
    parser.add_argument('--bin_ll_quantize_inputs', type=mc.str2bool, default=False, help='Quantize the inputs of the last linear layer')
    parser.add_argument('--bin_ll_pseudo_quantize', type=mc.str2bool, default=False, help='Pseudo-quantize the input of the last linear layer')

    # Binary networks and multi-stage experiments - TEACHER NETWORK
    parser.add_argument('--teacher_model', type=str, default='dgcnn', metavar='N')
    parser.add_argument('--teacher_path', type=str, default='pretrained/dgcnn_1024_teacher/models/model.t7', metavar='N',
                        help='Pretrained model path')
    parser.add_argument('--init_student_with_teacher_weights', type=mc.str2bool, default=False, help='Initialize the weights of the students with those of the trained teacher')

    parser.add_argument('--teacher_bn_momentum', type=float, default=0.999, help='Momentum for BN used in the binary blocks')
    parser.add_argument('--teacher_quantize_weights', type=mc.str2bool, default=False, help='Binarize the weights in the conv/linear')
    parser.add_argument('--teacher_quantize_inputs', type=mc.str2bool, default=False, help='Binarize the input to the conv / the graph features (depending on the model)')
    parser.add_argument('--teacher_prelu_in_blocks', type=mc.str2bool, default=True, help='Apply PReLU activation in the binary blocks')
    parser.add_argument('--teacher_pseudo_quantize', type=mc.str2bool, default=True, help='Pseudo-quantize with tanh instead of sign')
    parser.add_argument('--teacher_rescale_H', type=mc.str2bool, default=False, help='Learnable rescaling in the H dimension (# points)')
    parser.add_argument('--teacher_rescale_W', type=mc.str2bool, default=False, help='Learnable rescaling in the W dimension (k neighbours)')
    parser.add_argument('--teacher_rescale_L', type=mc.str2bool, default=False, help='Learnable rescaling in the L dimension for the last conv before the MLP (# points)')
    parser.add_argument('--teacher_knn_op', type=str, default='l2', choices=['l1', 'l2', 'hamming'], help='Distance used to find the KNNs in feature space')
    parser.add_argument('--teacher_conv_prelu', type=mc.str2bool, default=True, help='Use PReLU after bin conv')

    parser.add_argument('--teacher_global_balance_axis', type=int, default=0, help='Axis on which to perform mean/median centering')
    parser.add_argument('--teacher_global_balance_op', type=str, default='none', choices=['mean', 'median', 'none'], help='Global balance function')
    parser.add_argument('--teacher_edge_balance_axis', type=int, default=0, help='Axis on which to perform mean/median centering')
    parser.add_argument('--teacher_edge_balance_op', type=str, default='none', choices=['mean', 'median', 'none'], help='Edge balance function')

    parser.add_argument('--teacher_ll_quantize_weights', type=mc.str2bool, default=False, help='Quantize the weights in the last linear layer')
    parser.add_argument('--teacher_ll_quantize_inputs', type=mc.str2bool, default=False, help='Quantize the inputs of the last linear layer')
    parser.add_argument('--teacher_ll_pseudo_quantize', type=mc.str2bool, default=False, help='Pseudo-quantize the input of the last linear layer')

    #############################
    parser.add_argument('--LSP_lambda', type=float, default=1e2, metavar='M', help='Local structure preserving loss weight.')
    parser.add_argument('--LSP_kernels', type=str, default='l2_l2', metavar='M', choices=['l2_l2', 'hamming_l2', 'hamming_sq_l2', 'hamming_hamming', 'h_sq_h_sq'], help='Local structure preserving similarity kernels.')
    parser.add_argument('--KD_T', type=float, default=3, metavar='M', help='Temperature in softmax for logit matching.')
    parser.add_argument('--KD_alpha', type=float, default=0.1, metavar='M', help='Alpha for logit matching.')




    args = parser.parse_args()

    _init_()
    wandb.init(project='binary_gnn_demo')
    wandb.config.update(args)

    io = IOStream('checkpoints/' + args.exp_name + '/run.log')
    io.cprint(str(args))

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    if args.cuda:
        io.cprint(
            'Using GPU : ' + str(torch.cuda.current_device()) + ' from ' + str(torch.cuda.device_count()) + ' devices')
        torch.cuda.manual_seed(args.seed)
    else:
        io.cprint('Using CPU')

    if not args.eval:
        train(args, io)
    else:
        test(args, io)
