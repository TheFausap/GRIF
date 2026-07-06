# models/griffin_lm.py
from configs.griffin import GriffinConfig
import torch.nn as nn
import torch.nn.functional as F
from .decoder_layer import DecoderLayer
from .norm import RMSNorm

class GriffinLM(nn.Module):
    def __init__(self, cfg: GriffinConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        pattern = [x.strip() for x in cfg.layer_pattern.split(",")]
        self.layers = nn.ModuleList(
            [DecoderLayer(cfg, pattern[i % len(pattern)]) for i in range(cfg.n_layers)]
        )
        self.final_norm = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight  # tied embeddings
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)
        elif isinstance(module, nn.Conv1d):
            nn.init.normal_(module.weight, std=0.02)

    def forward(self, input_ids, labels=None):
        x = self.embed(input_ids)
        for layer in self.layers:
            x = layer(x)
        x = self.final_norm(x)
        logits = self.lm_head(x)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
        return logits, loss
