"""Decoders (owner: Anikait - decoder).

Both take frozen ResNet-50 grid features (B, 49, 2048) and teacher-forced
input tokens (B, T) and return logits (B, T, V).
  - LSTMDecoder: baseline. Mean-pooled image vector initialises h0/c0.
  - TransformerDecoder: main model. Cross-attends over all 49 grid cells.
"""
import torch
from torch import nn


class LSTMDecoder(nn.Module):
    def __init__(self, vocab_size, d=512, feat_dim=2048, dropout=0.3):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d, padding_idx=0)
        self.init_h = nn.Linear(feat_dim, d)
        self.init_c = nn.Linear(feat_dim, d)
        self.lstm = nn.LSTM(d, d, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(d, vocab_size)

    def forward(self, feats, tokens):
        g = feats.mean(1)                                   # (B, 2048)
        h0 = torch.tanh(self.init_h(g)).unsqueeze(0)
        c0 = torch.tanh(self.init_c(g)).unsqueeze(0)
        x = self.drop(self.emb(tokens))
        y, _ = self.lstm(x, (h0, c0))
        return self.out(self.drop(y))


class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, d=512, feat_dim=2048, heads=8, layers=3,
                 ff=1024, dropout=0.3, max_len=64, grid=49):
        super().__init__()
        self.d = d
        self.emb = nn.Embedding(vocab_size, d, padding_idx=0)
        self.pos = nn.Embedding(max_len, d)                 # learned token positions
        self.mem_proj = nn.Sequential(nn.Linear(feat_dim, d), nn.LayerNorm(d))
        self.mem_pos = nn.Parameter(torch.zeros(1, grid, d))  # learned grid positions
        nn.init.normal_(self.mem_pos, std=0.02)
        layer = nn.TransformerDecoderLayer(d, heads, ff, dropout, batch_first=True,
                                           norm_first=True)
        self.dec = nn.TransformerDecoder(layer, layers)
        self.norm = nn.LayerNorm(d)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(d, vocab_size)

    def forward(self, feats, tokens):
        B, T = tokens.shape
        mem = self.mem_proj(feats) + self.mem_pos
        pos = torch.arange(T, device=tokens.device)
        x = self.drop(self.emb(tokens) + self.pos(pos))
        mask = torch.triu(torch.ones(T, T, device=tokens.device, dtype=torch.bool), 1)
        y = self.dec(x, mem, tgt_mask=mask)  # pads sit at the end, so causal mask hides them
        return self.out(self.norm(y))


def build(name, vocab_size):
    return {"lstm": LSTMDecoder, "transformer": TransformerDecoder}[name](vocab_size)
