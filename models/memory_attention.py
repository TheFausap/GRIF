# models/memory_attention.py
import torch.nn as nn

class MemoryAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gate = nn.Linear(dim, 1)

    def forward(self, local_out, memory_out):
        g = torch.sigmoid(self.gate(local_out))
        return g * memory_out + (1 - g) * local_out
