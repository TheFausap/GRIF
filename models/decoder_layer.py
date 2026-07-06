# models/decoder_layer.py
from configs.griffin import GriffinConfig
from models.attention import SlidingWindowMQA
from models.recurrent_block import RecurrentBlock
import torch.nn as nn
from .norm import RMSNorm
from .mlp import GatedMLP

class DecoderLayer(nn.Module):
    def __init__(self, cfg: GriffinConfig, layer_type: str):
        super().__init__()
        self.layer_type = layer_type
        self.norm1 = RMSNorm(cfg.d_model)
        self.norm2 = RMSNorm(cfg.d_model)
        if layer_type == "recurrent":
            self.temporal = RecurrentBlock(cfg.d_model, cfg.d_rnn, cfg.conv_kernel, cfg.dropout)
        elif layer_type == "attention":
            self.temporal = SlidingWindowMQA(cfg.d_model, cfg.n_heads, cfg.n_kv_heads, cfg.attention_window, cfg.dropout)
        else:
            raise ValueError(f"Unknown layer type: {layer_type}")
        self.mlp = GatedMLP(cfg.d_model, cfg.mlp_mult, cfg.dropout)

    def forward(self, x):
        x = x + self.temporal(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x
