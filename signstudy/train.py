"""Train one classifier for a fixed number of gradient steps (no early stopping, no validation labels used)."""
import time

import numpy as np
import torch
from torch import nn

from . import models as M


def device():
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def standardise(train_x, *others):
    flat = train_x.reshape(-1, train_x.shape[-1])
    mu, sd = flat.mean(0), flat.std(0)
    sd = np.where(sd < 1e-3, 1.0, sd)
    return [(a - mu) / sd for a in (train_x, *others)]


def augment(x, gen):
    """Light augmentation: per-sample time stretch (resample), additive noise, and global feature scale jitter."""
    b, t, d = x.shape
    stretch = 0.85 + 0.3 * torch.rand(b, device=x.device, generator=gen)
    grid = (torch.linspace(0, 1, t, device=x.device)[None, :] * stretch[:, None]).clamp(max=1.0) * (t - 1)
    lo = grid.floor().long().clamp(max=t - 2)
    w = (grid - lo)[..., None]
    idx = torch.arange(b, device=x.device)[:, None]
    x = x[idx, lo] * (1 - w) + x[idx, lo + 1] * w
    x = x * (0.9 + 0.2 * torch.rand(b, 1, 1, device=x.device, generator=gen))
    return x + 0.05 * torch.randn(x.shape, device=x.device, generator=gen)


def fit_predict(name, train_x, train_y, eval_xs, n_cls, seed, steps, lr=1e-3, batch=64, wd=1e-2, aug=False,
                model_kw=None):
    """Returns (list of logits arrays for each eval set, training seconds)."""
    dev = device()
    torch.manual_seed(seed)
    gen = torch.Generator(device=dev if dev.type != "mps" else "cpu").manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    xs = standardise(train_x, *eval_xs)
    tx = torch.tensor(xs[0], dtype=torch.float32, device=dev)
    ty = torch.tensor(train_y, dtype=torch.long, device=dev)
    model = M.build(name, tx.shape[-1], n_cls, **(model_kw or {})).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.05)
    n = len(tx)
    perm_rng = np.random.default_rng(seed)
    t0 = time.time()
    model.train()
    order = np.concatenate([perm_rng.permutation(n) for _ in range(int(np.ceil(steps * batch / n)) + 1)])
    for step in range(steps):
        idx = torch.as_tensor(order[step * batch:(step + 1) * batch], device=dev)
        xb, yb = tx[idx], ty[idx]
        if aug:
            xb = augment(xb, None) if dev.type != "cpu" else augment(xb, gen)
        if len(xb) < 2:
            continue
        opt.zero_grad(set_to_none=True)
        loss_fn(model(xb), yb).backward()
        opt.step()
        sched.step()
    train_s = time.time() - t0
    model.eval()
    outs = []
    with torch.no_grad():
        for x in xs[1:]:
            ex = torch.tensor(x, dtype=torch.float32, device=dev)
            outs.append(torch.cat([model(ex[i:i + 512]) for i in range(0, len(ex), 512)]).cpu().numpy())
    return outs, train_s, model


def steps_for(n_train, batch=64, epochs=30, lo=600, hi=4000):
    return int(min(max(np.ceil(epochs * n_train / batch), lo), hi))
