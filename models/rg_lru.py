# models/rg_lru.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class RGLRU(nn.Module):
    """
    Practical PyTorch RG-LRU approximation aligned with the Griffin/RecurrentGemma family:
      g_x = sigmoid(Wx x_t + bx)
      g_a = sigmoid(Wa x_t + ba)
      log a_t = -8 * g_a * softplus(lambda)
      h_t = a_t * h_{t-1} + sqrt(1 - a_t^2) * (g_x * x_t)
    Gates depend only on current input; the recurrence is diagonal.
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.in_gate = nn.Linear(dim, dim)
        self.rec_gate = nn.Linear(dim, dim)
        self.recurrent_param = nn.Parameter(torch.empty(dim))
        self.eps = eps
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.in_gate.weight, std=0.02)
        nn.init.zeros_(self.in_gate.bias)
        nn.init.normal_(self.rec_gate.weight, std=0.02)
        nn.init.zeros_(self.rec_gate.bias)
        nn.init.uniform_(self.recurrent_param, a=-5.0, b=-3.0)

    def forward(self, x, h0=None):
        B, T, R = x.shape
        gx = torch.sigmoid(self.in_gate(x))
        ga = torch.sigmoid(self.rec_gate(x))
        log_a = -8.0 * ga * F.softplus(self.recurrent_param).view(1, 1, R)
        a = torch.exp(log_a)
        gamma_sq = torch.clamp(1.0 - torch.exp(2.0 * log_a), min=self.eps)
        gamma = torch.sqrt(gamma_sq)

        u = gamma * (gx * x)

        h = torch.zeros(B, R, device=x.device, dtype=torch.float32) if h0 is None else h0.float()
        ys = []
        for t in range(T):
            h = a[:, t].float() * h + u[:, t].float()
            ys.append(h.to(x.dtype))
        y = torch.stack(ys, dim=1)
        return y, h
