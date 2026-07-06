# models/recurrent_block.py
from models.conv import DepthwiseCausalConv1d
import torch.nn as nn
import torch.nn.functional as F
from .rg_lru import RGLRU
from .conv import DepthwiseCausalConv1d

class RecurrentBlock(nn.Module):
    def __init__(self, d_model: int, d_rnn: int, conv_kernel: int, dropout: float = 0.0):
        super().__init__()
        self.lin_u = nn.Linear(d_model, d_rnn, bias=False)
        self.lin_y = nn.Linear(d_model, d_rnn, bias=False)
        self.conv = DepthwiseCausalConv1d(d_rnn, kernel_size=conv_kernel)
        self.rglru = RGLRU(d_rnn)
        self.out = nn.Linear(d_rnn, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        u = self.lin_u(x)
        y = F.gelu(self.lin_y(x))
        u = self.conv(u)
        h, _ = self.rglru(u)
        return self.out(self.dropout(h * y))
