import math

import torch

from torch import Tensor

import model.core
import model.interface

from fastfeedforward import FFF

class FastFeedForwardSubLayer(model.core.TransformerLayerPart, model.interface.IFeedForwardSubLayer):
    def __init__(self):
        super().__init__()
        hp = self.hparams
        d_ff = int(hp.d_model * hp.feedforward_d_model_ratio)
        d_leaf = d_ff // hp.n_leaf
        depth = int(math.log2(hp.n_leaf))
        activation = lambda x: torch.square(torch.relu(x))
        self.fff = FFF(
            hp.d_model,
            d_leaf,
            hp.d_model,
            depth,
            activation,
            hp.dropout,
            hp.train_hardened
        )
    def forward(self, x : Tensor):
        return self.fff.training_forward(
            x,
            return_entropies=False,
            use_hard_decisions=self.hparams.train_hardened
        )