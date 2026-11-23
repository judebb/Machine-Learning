# MSML 612 - Step 3: SR3-style diffusion model for super-resolution
# Training: add random noise to the sharp image, teach a UNet to predict that noise,
# given the 256x256 bicubic image (condition), the noise step t, and the land-use class.
# Sampling: DDIM, starting from pure noise and denoising in a few tens of steps.
import math

import torch
from torch import nn
import torch.nn.functional as F


def timestep_embedding(t, dim):
    # turns the step number t (0..steps-1) into a vector of sines and cosines
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    args = t[:, None].float() * freqs[None]
    return torch.cat([args.sin(), args.cos()], 1)


class Block(nn.Module):
    # basic building block: two 3x3 convolutions plus the step/class embedding
    def __init__(self, cin, cout, tdim):
        super().__init__()
        self.n1 = nn.GroupNorm(8, cin)
        self.c1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.t = nn.Linear(tdim, cout)  # maps the embedding to one value per channel
        self.n2 = nn.GroupNorm(8, cout)
        self.c2 = nn.Conv2d(cout, cout, 3, padding=1)
        # 1x1 conv so the input can be added to the output when channel counts differ
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x, t):
        h = self.c1(F.silu(self.n1(x)))
        h = h + self.t(t)[:, :, None, None]  # inject the noise-step (+ class) information
        h = self.c2(F.silu(self.n2(h)))
        return h + self.skip(x)


class UNet(nn.Module):
    # input: 6 channels = bicubic condition (3) + noisy target (3)
    # output: 3 channels = the noise the network thinks was added
    def __init__(self, base=32, num_classes=21, use_class=True):
        super().__init__()
        tdim = base * 4
        self.base = base
        self.use_class = use_class
        self.temb = nn.Sequential(nn.Linear(base, tdim), nn.SiLU(), nn.Linear(tdim, tdim))
        # land-use label -> vector, added to the timestep embedding (skipped when use_class is False)
        self.cemb = nn.Embedding(num_classes, tdim)
        self.inc = nn.Conv2d(6, base, 3, padding=1)
        # going down: fewer pixels, more channels
        self.d1 = Block(base, base, tdim)
        self.d2 = Block(base, base * 2, tdim)
        self.d3 = Block(base * 2, base * 4, tdim)
        self.mid = Block(base * 4, base * 4, tdim)
        # going up: input channels are doubled because we concatenate the matching down-layer
        self.u3 = Block(base * 8, base * 2, tdim)
        self.u2 = Block(base * 4, base, tdim)
        self.u1 = Block(base * 2, base, tdim)
        self.out = nn.Conv2d(base, 3, 3, padding=1)
        self.down = nn.AvgPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode="nearest")

    def forward(self, x, t, label):
        e = self.temb(timestep_embedding(t, self.base))
        if self.use_class:
            e = e + self.cemb(label)
        h1 = self.d1(self.inc(x), e)    # full size
        h2 = self.d2(self.down(h1), e)  # 1/2 size
        h3 = self.d3(self.down(h2), e)  # 1/4 size
        u = self.u3(torch.cat([self.mid(h3, e), h3], 1), e)
        u = self.u2(torch.cat([self.up(u), h2], 1), e)
        u = self.u1(torch.cat([self.up(u), h1], 1), e)
        return self.out(u)


class SR3(nn.Module):
    def __init__(self, steps=200, base=32, num_classes=21, use_class=True):
        super().__init__()
        self.steps = steps
        self.net = UNet(base, num_classes, use_class)
        # noise schedule: betas = how much noise is added at each step (small -> larger)
        # rescaled so fewer steps still end up fully noisy
        betas = torch.linspace(1e-4 * 1000 / steps, 0.02 * 1000 / steps, steps)
        self.register_buffer("betas", betas)
        self.register_buffer("abar", torch.cumprod(1 - betas, 0))  # total signal kept after t steps

    def loss(self, cond, target, label):
        # 1) pick a random noise step for each image in the batch
        t = torch.randint(0, self.steps, (target.size(0),), device=target.device)
        # 2) forward noising: corrupt the sharp image to step t
        noise = torch.randn_like(target)
        a = self.abar[t][:, None, None, None]
        noisy = a.sqrt() * target + (1 - a).sqrt() * noise
        # 3) the network sees bicubic condition + noisy image + t + class and must guess the noise
        return F.mse_loss(self.net(torch.cat([cond, noisy], 1), t, label), noise)

    @torch.no_grad()
    def sample(self, cond, label, ddim_steps=50):
        # DDIM (eta = 0): deterministic, uses only ddim_steps of the training steps
        x = torch.randn_like(cond)
        ts = torch.linspace(self.steps - 1, 0, ddim_steps).round().long().tolist()
        for k, i in enumerate(ts):
            t = torch.full((cond.size(0),), i, device=cond.device, dtype=torch.long)
            eps = self.net(torch.cat([cond, x], 1), t, label)
            a = self.abar[i]
            a_prev = self.abar[ts[k + 1]] if k + 1 < len(ts) else torch.ones_like(a)
            x0 = ((x - (1 - a).sqrt() * eps) / a.sqrt()).clamp(-1, 1)  # current guess of the sharp image
            x = a_prev.sqrt() * x0 + (1 - a_prev).sqrt() * eps
        return x.clamp(-1, 1)
