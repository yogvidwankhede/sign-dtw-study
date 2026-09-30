"""Small sequence classifiers on (batch, frames, dims) inputs. All are intentionally modest in size."""
import torch
from torch import nn


class CNN1D(nn.Module):
    def __init__(self, d_in, n_cls, width=128, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(d_in, width, 5, padding=2), nn.BatchNorm1d(width), nn.ReLU(),
            nn.Conv1d(width, width, 5, padding=2), nn.BatchNorm1d(width), nn.ReLU(),
            nn.Conv1d(width, 2 * width, 3, padding=1), nn.BatchNorm1d(2 * width), nn.ReLU())
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(4 * width, n_cls))

    def forward(self, x):
        h = self.net(x.transpose(1, 2))
        return self.head(torch.cat([h.mean(-1), h.amax(-1)], dim=-1))


class BiGRU(nn.Module):
    def __init__(self, d_in, n_cls, hidden=128, dropout=0.3):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(d_in, hidden), nn.ReLU(), nn.Dropout(dropout))
        self.rnn = nn.GRU(hidden, hidden, num_layers=2, batch_first=True, bidirectional=True, dropout=dropout)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(4 * hidden, n_cls))

    def forward(self, x):
        h, _ = self.rnn(self.proj(x))
        return self.head(torch.cat([h.mean(1), h.amax(1)], dim=-1))


class SmallTransformer(nn.Module):
    def __init__(self, d_in, n_cls, dim=128, layers=3, heads=4, frames=32, dropout=0.1):
        super().__init__()
        self.proj = nn.Linear(d_in, dim)
        self.pos = nn.Parameter(torch.zeros(1, frames, dim))
        layer = nn.TransformerEncoderLayer(dim, heads, 2 * dim, dropout, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, layers)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, n_cls)

    def forward(self, x):
        h = self.enc(self.proj(x) + self.pos)
        return self.head(self.norm(h).mean(1))


MODELS = {"cnn1d": CNN1D, "bigru": BiGRU, "transformer": SmallTransformer}


def build(name, d_in, n_cls, **kw):
    return MODELS[name](d_in, n_cls, **kw)


def n_params(model):
    return sum(p.numel() for p in model.parameters())
