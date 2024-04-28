import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

class Quantize(nn.Module):
    def __init__(self, bits: int):
        super().__init__()
        self.bits = bits

    def forward(self, w: Tensor):
        # range [w_min, w_max]
        w_min = w.min()
        w_max = w.max()
        # range [0, w_max - w_min]
        w = w - w_min
        # range [0, 2 ** bits - 1]
        s = (w_max - w_min) / (2 ** self.bits - 1)
        w = w / (s + 1e-10)
        w = torch.round(w)
        # range [0, w_max - w_min]
        w = w * s
        # range [w_min, w_max]
        w = w + w_min
        return w

    def backward(self, grad_output):
        grad_input = grad_output.clone()
        return grad_input

class QLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool, bits: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None
        self.quantize = Quantize(bits)

    def forward(self, x: Tensor):
        weight_q = self.quantize(self.weight)
        return F.linear(x, weight_q, bias=self.bias)