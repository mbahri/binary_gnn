from typing import Union, Tuple
from torch_geometric.typing import OptPairTensor, Adj, Size

from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch_sparse import SparseTensor, matmul
from torch_geometric.nn.conv import MessagePassing

import binary


class BinLinearBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        binary_weights=True,
        binary_inputs=True,
        activation=True,
        pseudo_quantize=False,
        center="bn",
        name="",
    ):
        super().__init__()

        self.quantize_weights = binary_weights
        self.quantize_inputs = binary_inputs
        self.pseudo_quantize = pseudo_quantize

        self.do_activation = activation
        self.activ = binary.PReLU() if activation else binary.Identity()

        if self.quantize_inputs:
            self.input_quantizer = binary.STEQuantizer()
        elif self.pseudo_quantize:
            self.input_quantizer = nn.Tanh()
        else:
            self.input_quantizer = binary.Identity()

        self.center_method = center
        if self.center_method == "bn":
            self.center = nn.BatchNorm1d(in_channels, momentum=0.999)
        elif self.center_method == "median":
            self.center = binary.MedianCenter(0)
        elif self.center_method is None:
            self.center = binary.Identity()
        else:
            raise NotImplementedError("Unsupported input centerer")

        self.lin = binary.BinLinear(
            in_channels, out_channels, binary_weights=self.quantize_weights
        )

        self.rescale = binary.LearnedRescaleLayer1d((1, 1, out_channels))

        self.name = name

    def reset_parameters(self):
        self.lin.reset_parameters()
        if self.do_activation:
            self.activ.reset_parameters()

        self.rescale.reset_parameters()
        self.center.reset_parameters()

    def forward(self, x):
        x = self.center(x)
        x = self.input_quantizer(x)

        x = self.lin(x)
        x = self.rescale(x)
        x = self.activ(x)

        return x


class BinSAGEConv(MessagePassing):
    r"""The GraphSAGE operator from the `"Inductive Representation Learning on
    Large Graphs" <https://arxiv.org/abs/1706.02216>`_ paper
    .. math::
        \mathbf{x}^{\prime}_i = \mathbf{W}_1 \mathbf{x}_i + \mathbf{W_2} \cdot
        \mathrm{mean}_{j \in \mathcal{N(i)}} \mathbf{x}_j
    Args:
        in_channels (int or tuple): Size of each input sample. A tuple
            corresponds to the sizes of source and target dimensionalities.
        out_channels (int): Size of each output sample.
        normalize (bool, optional): If set to :obj:`True`, output features
            will be :math:`\ell_2`-normalized, *i.e.*,
            :math:`\frac{\mathbf{x}^{\prime}_i}
            {\| \mathbf{x}^{\prime}_i \|_2}`.
            (default: :obj:`False`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """

    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        normalize: bool = False,
        bias: bool = True,
        name="",
        binary_inputs=False,
        binary_weights=False,
        pseudo_quantize=False,
        inner_activation=False,
        center="bn",
        **kwargs
    ):  # yapf: disable

        kwargs.setdefault("aggr", "mean")
        super().__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize

        self.name = name

        self.binary_inputs = binary_inputs
        self.binary_weights = binary_weights
        self.pseudo_quantize = pseudo_quantize
        self.inner_activation = inner_activation
        self.center = center

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        self.lin_l = BinLinearBlock(
            in_channels[0],
            out_channels,
            binary_weights=self.binary_weights,
            binary_inputs=self.binary_inputs,
            activation=self.inner_activation,
            center=self.center,
            name=self.name + "_l",
        )
        self.lin_r = BinLinearBlock(
            in_channels[1],
            out_channels,
            binary_weights=self.binary_weights,
            binary_inputs=self.binary_inputs,
            activation=self.inner_activation,
            center=self.center,
            name=self.name + "_r",
        )

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_l.reset_parameters()
        self.lin_r.reset_parameters()

    def forward(
        self, x: Union[Tensor, OptPairTensor], edge_index: Adj, size: Size = None
    ) -> Tensor:
        """"""
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        # propagate_type: (x: OptPairTensor)
        out = self.propagate(edge_index, x=x, size=size)
        out = self.lin_l(out)

        x_r = x[1]
        if x_r is not None:
            out += self.lin_r(x_r)

        if self.normalize:
            out = F.normalize(out, p=2.0, dim=-1)

        return out

    def message(self, x_j: Tensor) -> Tensor:
        return x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: OptPairTensor) -> Tensor:
        adj_t = adj_t.set_value(None, layout=None)
        return matmul(adj_t, x[0], reduce=self.aggr)

    def __repr__(self):
        return "{}({}, {})".format(
            self.__class__.__name__, self.in_channels, self.out_channels
        )
