import torch.nn as nn
import torch.nn.functional as F

class DepthwiseCausalConv1d(nn.Module):
    def __init__(self, channels: int, kernel_size: int):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(
            channels, channels, kernel_size=kernel_size, groups=channels, bias=False
        )

    def forward(self, x):
        x = x.transpose(1, 2)  # [B, C, T]
        x = F.pad(x, (self.kernel_size - 1, 0))
        x = self.conv(x)
        return x.transpose(1, 2)