"""
@Author: Mehdi Bahri
@Contact: m.bahri@imperial.ac.uk
@File: binary.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


class _STEQuantizer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, in_):
        ctx.save_for_backward(in_)

        x = torch.sign(in_)

        return x

    @staticmethod
    def backward(ctx, grad_out_):
        in_, = ctx.saved_tensors

        cond = in_.abs() <= 1
        zeros = torch.zeros_like(grad_out_)
        x = torch.where(cond, grad_out_, zeros)

        return x


class STEQuantizer(torch.nn.Module):
    def forward(self, x):
        return _STEQuantizer.apply(x)


quantize = _STEQuantizer.apply


class PReLU(nn.PReLU):
    def __init__(self, num_parameters: int = 1, init: float = 0.25):
        super().__init__(num_parameters=num_parameters, init=init)
        self.init = init

    def reset_parameters(self):
        nn.init.constant_(self.weight, self.init)


class ReLU(nn.ReLU):
    def __init__():
        super().__init__()

    def reset_parameters(self):
        pass


class LearnedRescaleLayer2d(nn.Module):
    def __init__(self, input_shapes):
        super(LearnedRescaleLayer2d, self).__init__()
        """Implements the learned activation rescaling XNOR-Net++ style.
            This is used to scale the outputs of the binary convolutions in the Strong
            Baseline networks. [(Bulat & Tzimiropoulos,
            2019)](https://arxiv.org/abs/1909.13863)
        """

        self.shapes = input_shapes
        self.scale_a = nn.Parameter(torch.Tensor(self.shapes[1], 1, 1).fill_(1))
        self.scale_b = nn.Parameter(torch.Tensor(1, self.shapes[2], 1).fill_(1))
        self.scale_c = nn.Parameter(torch.Tensor(1, 1, self.shapes[3]).fill_(1))

    def reset_parameters(self):
        nn.init.ones_(self.scale_a)
        nn.init.ones_(self.scale_b)
        nn.init.ones_(self.scale_c)

    def forward(self, x):
        out = x * self.scale_a * self.scale_b * self.scale_c

        return out


class LearnedRescaleLayer1d(nn.Module):
    def __init__(self, input_shapes):
        super(LearnedRescaleLayer1d, self).__init__()
        """Implements the learned activation rescaling XNOR-Net++ style.
            This is used to scale the outputs of the binary convolutions in the Strong
            Baseline networks. [(Bulat & Tzimiropoulos,
            2019)](https://arxiv.org/abs/1909.13863)
        """

        self.shapes = input_shapes
        self.scale_a = nn.Parameter(torch.Tensor(self.shapes[1], 1).fill_(1))
        self.scale_b = nn.Parameter(torch.Tensor(1, self.shapes[2]).fill_(1))


    def reset_parameters(self):
        nn.init.ones_(self.scale_a)
        nn.init.ones_(self.scale_b)

    def forward(self, x):
        out = x * self.scale_a * self.scale_b

        return out


class LearnedRescaleLayer0d(nn.Module):
    def __init__(self, input_shapes):
        super(LearnedRescaleLayer0d, self).__init__()
        """Implements the learned activation rescaling XNOR-Net++ style.
            This is used to scale the outputs of the binary convolutions in the Strong
            Baseline networks. [(Bulat & Tzimiropoulos,
            2019)](https://arxiv.org/abs/1909.13863)
        """

        self.shapes = input_shapes
        self.scale_a = nn.Parameter(torch.Tensor(self.shapes[1],).fill_(1))

    def reset_parameters(self):
        nn.init.ones_(self.scale_a)

    def forward(self, x):
        out = x * self.scale_a

        return out


class LearnedRescaleLayer1db(nn.Module):
    def __init__(self, input_shapes):
        super().__init__()
        """Implements the learned activation rescaling XNOR-Net++ style.
            This is used to scale the outputs of the binary convolutions in the Strong
            Baseline networks. [(Bulat & Tzimiropoulos,
            2019)](https://arxiv.org/abs/1909.13863)
        """

        self.shapes = input_shapes
        self.scale_a = nn.Parameter(torch.Tensor(self.shapes[0],self.shapes[1]).fill_(1))

    def forward(self, x):
        out = x * self.scale_a

        return out


class Transpose(nn.Module):
    def __init__(self, a, b):
        super().__init__()
        self.a = a
        self.b = b

    def forward(self, X):
        return X.transpose(self.a, self.b)


class BinLinear(nn.Module):
    def __init__(self, in_channels, out_channels, binary_weights=True, bias=False):
        super(BinLinear, self).__init__()
        """
        An implementation of a Linear layer.

        Parameters:
        - weight: the learnable weights of the module of shape (in_channels, out_channels).
        - bias: the learnable bias of the module of shape (out_channels).
        """
        self.in_channel = in_channels
        self.out_channels = out_channels
        self.binary_weights = binary_weights

        self.weights_real = nn.Parameter(torch.Tensor(out_channels, in_channels))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weights_real)

    def forward(self, x):
        """
        Input:
        - x: Input data of shape (N, *, H) where * means any number of additional
        dimensions and H = in_channels
        Output:
        - out: Output data of shape (N, *, H') where * means any number of additional
        dimensions and H' = out_channels
        """
        if self.binary_weights:
            # self.weights_real.data = torch.clamp(self.weights_real.data - self.weights_real.data.mean(1, keepdim=True), -1, 1)
            self.weights_real.data = torch.clamp(self.weights_real.data, -1, 1)
            weights = quantize(self.weights_real)
        else:
            weights = self.weights_real

        out = F.linear(x, weights)

        return out


class RescaledDotProduct(nn.Module):
    def __init__(self, dot_product, dot_product_args, rescaler, rescaler_args):
        super().__init__()

        self.inner = dot_product(**dot_product_args)
        self.rescaler = rescaler(**rescaler_args)

    def reset_parameters():
        self.inner.reset_parameters()
        self.rescaler.reset_parameters()

    def forward(self, x):
        x = self.inner(x)
        return self.rescaler(x)


class NoOp(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, x):
        return x


class Identity(nn.Identity):
    def __init__(self):
        super().__init__()

    def reset_parameters(self):
        pass


class BinConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False, binary_weights=True):
        super(BinConv1d, self).__init__()
        """
        An implementation of a Linear layer.

        Parameters:
        - weight: the learnable weights of the module of shape (in_channels, out_channels).
        - bias: the learnable bias of the module of shape (out_channels).
        """
        self.in_channel = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        self.binary_weights = binary_weights

        self.weights_real = nn.Parameter(torch.Tensor(out_channels, in_channels, kernel_size))
        nn.init.kaiming_normal_(self.weights_real)

    def forward(self, x):
        """
        Input:
        - x: Input data of shape (N, *, H) where * means any number of additional
        dimensions and H = in_channels
        Output:
        - out: Output data of shape (N, *, H') where * means any number of additional
        dimensions and H' = out_channels
        """
        if self.binary_weights:
            self.weights_real.data = torch.clamp(self.weights_real.data, -1, 1)
            weights = quantize(self.weights_real)
        else:
            weights = self.weights_real

        out = F.conv1d(x, weights, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups)

        return out


class BinConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False, binary_weights=True):
        super(BinConv2d, self).__init__()
        """
        An implementation of a Linear layer.

        Parameters:
        - weight: the learnable weights of the module of shape (in_channels, out_channels).
        - bias: the learnable bias of the module of shape (out_channels).
        """
        self.in_channel = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        self.binary_weights = binary_weights

        if isinstance(kernel_size, int):
            k1, k2 = kernel_size, kernel_size
        else:
            k1, k2 = kernel_size

        self.weights_real = nn.Parameter(torch.Tensor(out_channels, in_channels, k1, k2))
        nn.init.kaiming_normal_(self.weights_real)

    def forward(self, x):
        """
        Input:
        - x: Input data of shape (N, *, H) where * means any number of additional
        dimensions and H = in_channels
        Output:
        - out: Output data of shape (N, *, H') where * means any number of additional
        dimensions and H' = out_channels
        """
        if self.binary_weights:
            self.weights_real.data = torch.clamp(self.weights_real.data, -1, 1)
            weights = quantize(self.weights_real)
        else:
            weights = self.weights_real

        out = F.conv2d(x, weights, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups)

        return out


class MedianCenter(nn.Module):
    def __init__(self, axis):
        super().__init__()
        self.axis = axis

    def forward(self, x):
        return x - x.median(self.axis, keepdim=True)[0]

    def reset_parameters(self):
        pass


class MeanCenter(nn.Module):
    def __init__(self, axis):
        super().__init__()
        self.axis = axis

    def forward(self, x):
        return x - x.mean(self.axis, keepdim=True)[0]

    def reset_parameters(self):
        pass

def count_trainable_parameters(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params