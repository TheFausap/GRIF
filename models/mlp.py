import torch.nn as nn
import torch.nn.functional as F

class GatedMLP(nn.Module):
    def __init__(self, d_model: int, mult: int = 3, dropout: float = 0.0):
        super().__init__()
        d_hidden = d_model * mult
        self.up1 = nn.Linear(d_model, d_hidden)
        self.up2 = nn.Linear(d_model, d_hidden)
        self.down = nn.Linear(d_hidden, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.down(self.dropout(F.gelu(self.up1(x)) * self.up2(x)))
    