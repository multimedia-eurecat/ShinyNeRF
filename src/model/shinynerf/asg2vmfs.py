# ------------------------------------------------------------------------------------
# ShinyNeRF
# Copyright (c) 2026 Barreiro, Albert. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import math

# class ASG2vMFs(nn.Module):
#     def __init__(self, d_model=20, hidden_dim=128, num_laterals=3, log=True):
#         super(ASG2vMFs, self).__init__()
        
#         # Positional encoding
#         self.d_model = d_model
#         div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
#         self.register_buffer('div_term', div_term)

#         # MLP
#         self.n_inputs = 3
#         self.log = log
#         self.fc1 = nn.Linear(d_model * self.n_inputs + self.n_inputs, hidden_dim)
#         self.fc2 = nn.Linear(hidden_dim, hidden_dim)
#         self.fc3 = nn.Linear(hidden_dim, hidden_dim // 2)
        
#         # von Mises-Fisher Params
#         self.concentration = nn.Linear(hidden_dim // 2, num_laterals+1)
#         self.elevation = nn.Linear(hidden_dim // 2, num_laterals)
        
#         # von Mises-Fisher Weights
#         self.weighting = nn.Linear(hidden_dim // 2, num_laterals+1)

#         # Inits
#         nn.init.xavier_uniform_(self.fc1.weight)
#         nn.init.xavier_uniform_(self.fc2.weight)
#         nn.init.xavier_uniform_(self.fc3.weight)
#         nn.init.xavier_uniform_(self.elevation.weight)
#         nn.init.xavier_uniform_(self.concentration.weight)
#         nn.init.xavier_uniform_(self.weighting.weight)

#     @torch.no_grad()
#     def positional_encoding(self, x):
#         pe_k = [torch.sin(x[...,0:1] * self.div_term), torch.cos(x[...,0:1] * self.div_term)]
#         pe_b = [torch.sin(x[...,1:2] * self.div_term), torch.cos(x[...,1:2] * self.div_term)]
#         pe_e = [torch.sin(x[...,2:3] * self.div_term), torch.cos(x[...,2:3] * self.div_term)]
#         return torch.cat(pe_k + pe_b + pe_e, dim=-1)

#     def forward(self, bandwidths):
#         if self.log:
#             bandwidths = torch.log(bandwidths)
#             if self.n_inputs == 3:
#                 ellipticity_ratio = bandwidths[...,1:2] - (bandwidths[...,0:1] + 1e-4) # Bandwidths' ratio: minor axis / major axis in log space
#                 bandwidths = torch.cat([bandwidths, ellipticity_ratio], dim=-1)
#         # Ellipticity ratio
#         elif self.n_inputs == 3:
#             ellipticity_ratio = bandwidths[...,1:2] / (bandwidths[...,0:1] + 1e-4) # Bandwidths' ratio: minor axis / major axis
#             bandwidths = torch.cat([bandwidths, ellipticity_ratio], dim=-1)

#         # Positional Encoding
#         if self.d_model > 1:
#             pe = self.positional_encoding(bandwidths)
#             bandwidths = torch.cat([bandwidths, pe], -1)

#         # Shared MLP backbone
#         bandwidths = F.leaky_relu(self.fc1(bandwidths))
#         bandwidths = F.leaky_relu(self.fc2(bandwidths))
#         bandwidths = F.leaky_relu(self.fc3(bandwidths))

#         # vMFs params prediction
#         k = F.softplus(self.concentration(bandwidths), beta=0.1)
#         thetas = torch.sigmoid(self.elevation(bandwidths)) * torch.pi / 2
#         weights = F.softplus(self.weighting(bandwidths))
#        # weights_norm = weights / (weights.sum(dim=-1, keepdim=True)  + 1e-6 ) we normalize in the loss

#         return k, thetas, weights

# # Custom Dataset to handle bandwidth pairs (major and minor axis)
# class ASGDataset(Dataset):
#     def __init__(self, bandwidth_lambdas, bandwidth_mus):
#         self.data = []
#         for l, mus_per_lambda in zip(bandwidth_lambdas, bandwidth_mus):
#             for m in mus_per_lambda:
#                 self.data.append((l.item(), m.item()))
#     def __len__(self):
#         return len(self.data)
    
#     def __getitem__(self, idx):
#         bandwidth_lambda, bandwidth_mu = self.data[idx]
#         return torch.tensor([bandwidth_lambda, bandwidth_mu], dtype=torch.float32)  # Each item is a tensor of shape (2,)


# class ASG2vMFs(nn.Module):
#     def __init__(self, d_model=20, hidden_dim=128, num_laterals=5, log=True, k_max=1e6):
#         super().__init__()
#         self.d_model = d_model
#         self.n_inputs = 3  # (log λ, log μ, log μ - log λ)
#         self.log = log
#         self.k_max = k_max

#         # positional encoding (sin/cos with shared div_term)
#         div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
#         self.register_buffer('div_term', div_term)

#         # MLP
#         in_dim = d_model * self.n_inputs + self.n_inputs
#         self.fc1 = nn.Linear(in_dim, hidden_dim)
#         self.fc2 = nn.Linear(hidden_dim, hidden_dim)
#         self.fc3 = nn.Linear(hidden_dim, hidden_dim // 2)

#         # heads
#         self.head_k   = nn.Linear(hidden_dim // 2, num_laterals + 1)   # central + laterals
#         self.head_el  = nn.Linear(hidden_dim // 2, num_laterals)       # elevations (sorted)
#         self.head_w   = nn.Linear(hidden_dim // 2, num_laterals + 1)   # mixture weights

#         for m in [self.fc1, self.fc2, self.fc3, self.head_k, self.head_el, self.head_w]:
#             nn.init.xavier_uniform_(m.weight)
#             nn.init.zeros_(m.bias)

#     @torch.no_grad()
#     def positional_encoding(self, x):
#         # x: (..., 3)
#         pe_k = [torch.sin(x[...,0:1] * self.div_term), torch.cos(x[...,0:1] * self.div_term)]
#         pe_b = [torch.sin(x[...,1:2] * self.div_term), torch.cos(x[...,1:2] * self.div_term)]
#         pe_e = [torch.sin(x[...,2:3] * self.div_term), torch.cos(x[...,2:3] * self.div_term)]
#         return torch.cat(pe_k + pe_b + pe_e, dim=-1)

#     def forward(self, bw):
#         # bw: [B,2] = (λ, μ)
#         if self.log:
#             bw = torch.log(bw.clamp_min(1e-12))
#             # log‑ratio exactly, no epsilon
#             ell = bw[...,1:2] - bw[...,0:1]
#             x = torch.cat([bw, ell], dim=-1)
#         else:
#             # linear ratio uses eps to avoid /0
#             ell = bw[...,1:2] / (bw[...,0:1] + 1e-8)
#             x = torch.cat([bw, ell], dim=-1)

#         if self.d_model > 1:
#             pe = self.positional_encoding(x)
#             x = torch.cat([x, pe], -1)

#         h = F.leaky_relu(self.fc1(x))
#         h = F.leaky_relu(self.fc2(h))
#         h = F.leaky_relu(self.fc3(h))

#         # concentrations with huge dynamic range
#         log_k = self.head_k(h)
#         k = torch.exp(log_k).clamp_max(self.k_max)

#         # monotone, sorted elevations in (0, π/2)
#         raw = self.head_el(h)
#         gaps = F.softplus(raw) + 1e-6
#         thetas = torch.cumsum(gaps, dim=-1)
#         thetas = (thetas / (thetas[..., -1:] + 1e-6)) * (math.pi/2 - 1e-3)

#         # positive weights (normalised in loss)
#         weights = F.softplus(self.head_w(h))
#         return k, thetas, weights



# class ASG2vMFs(nn.Module):
#     def __init__(self, d_model=20, hidden_dim=128, num_laterals=5, log=True,
#                  k_min=1.0, k_max=5000.0, tanh_scale=6.0):
#         super().__init__()
#         self.d_model = d_model
#         self.n_inputs = 3  # (log λ, log μ, log μ - log λ)
#         self.log = log
#         self.k_min = float(k_min)
#         self.k_max = float(k_max)
#         self._log_span = math.log(max(self.k_max - self.k_min, 1.0))
#         self._tanh_scale = float(tanh_scale)

#         # positional encoding (sin/cos with shared div_term)
#         div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
#         self.register_buffer('div_term', div_term)

#         # MLP
#         in_dim = d_model * self.n_inputs + self.n_inputs
#         self.fc1 = nn.Linear(in_dim, hidden_dim)
#         self.fc2 = nn.Linear(hidden_dim, hidden_dim)
#         self.fc3 = nn.Linear(hidden_dim, hidden_dim // 2)

#         # heads
#         self.head_k   = nn.Linear(hidden_dim // 2, num_laterals + 1)   # central + laterals
#         self.head_el  = nn.Linear(hidden_dim // 2, num_laterals)       # elevations (sorted)
#         self.head_w   = nn.Linear(hidden_dim // 2, num_laterals + 1)   # mixture weights

#         for m in [self.fc1, self.fc2, self.fc3, self.head_k, self.head_el, self.head_w]:
#             nn.init.xavier_uniform_(m.weight)
#             nn.init.zeros_(m.bias)

#     @torch.no_grad()
#     def positional_encoding(self, x):
#         # x: (..., 3)
#         pe_k = [torch.sin(x[...,0:1] * self.div_term), torch.cos(x[...,0:1] * self.div_term)]
#         pe_b = [torch.sin(x[...,1:2] * self.div_term), torch.cos(x[...,1:2] * self.div_term)]
#         pe_e = [torch.sin(x[...,2:3] * self.div_term), torch.cos(x[...,2:3] * self.div_term)]
#         return torch.cat(pe_k + pe_b + pe_e, dim=-1)

#     def forward(self, bw):
#         # bw: [B,2] = (λ, μ)
#         if self.log:
#             bw = torch.log(bw.clamp_min(1e-12))
#             # log‑ratio exactly, no epsilon
#             ell = bw[...,1:2] - bw[...,0:1]
#             x = torch.cat([bw, ell], dim=-1)
#         else:
#             # linear ratio uses eps to avoid /0
#             ell = bw[...,1:2] / (bw[...,0:1] + 1e-8)
#             x = torch.cat([bw, ell], dim=-1)

#         if self.d_model > 1:
#             pe = self.positional_encoding(x)
#             x = torch.cat([x, pe], -1)

#         h = F.leaky_relu(self.fc1(x))
#         h = F.leaky_relu(self.fc2(h))
#         h = F.leaky_relu(self.fc3(h))

#         # concentrations with huge dynamic range
#         s = self.head_k(h) / self._tanh_scale
#         log_extra = self._log_span * torch.tanh(s)
#         k = self.k_min + torch.exp(log_extra)

#         # monotone, sorted elevations in (0, π/2)
#         raw = self.head_el(h)
#         gaps = F.softplus(raw) + 1e-6
#         thetas = torch.cumsum(gaps, dim=-1)
#         thetas = (thetas / (thetas[..., -1:] + 1e-6)) * (math.pi/2 - 1e-3)

#         # positive weights (normalised in loss)
#         weights = F.softplus(self.head_w(h))
#         return k, thetas, weights



class ASG2vMFs(nn.Module):
    def __init__(self, d_model=20, hidden_dim=128, num_laterals=5, log=True,
                 k_min=1.0, k_max=5000.0, tanh_scale=6.0):
        super().__init__()
        self.d_model = d_model
        self.n_inputs = 3  # (log λ, log μ, log μ - log λ)
        self.log = log
        self.k_min = float(k_min)
        self.k_max = float(k_max)
        self._log_span = math.log(max(self.k_max - self.k_min, 1.0))
        self._tanh_scale = float(tanh_scale)

        # positional encoding (sin/cos with shared div_term)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        self.register_buffer('div_term', div_term)

        # MLP
        in_dim = d_model * self.n_inputs + self.n_inputs
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim // 2)

        # heads
        self.head_k   = nn.Linear(hidden_dim // 2, num_laterals + 1)   # central + laterals
        self.head_el  = nn.Linear(hidden_dim // 2, num_laterals)       # elevations (sorted)
        self.head_w   = nn.Linear(hidden_dim // 2, num_laterals + 1)   # mixture weights

        for m in [self.fc1, self.fc2, self.fc3, self.head_k, self.head_el, self.head_w]:
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)

    @torch.no_grad()
    def positional_encoding(self, x):
        # x: (..., 3)
        pe_k = [torch.sin(x[...,0:1] * self.div_term), torch.cos(x[...,0:1] * self.div_term)]
        pe_b = [torch.sin(x[...,1:2] * self.div_term), torch.cos(x[...,1:2] * self.div_term)]
        pe_e = [torch.sin(x[...,2:3] * self.div_term), torch.cos(x[...,2:3] * self.div_term)]
        return torch.cat(pe_k + pe_b + pe_e, dim=-1)

    def forward(self, bw):
        # bw: [B,2] = (λ, μ)
        if self.log:
            bw = torch.log(bw.clamp_min(1e-12))
            # log-ratio exactly, no epsilon
            ell = bw[...,1:2] - bw[...,0:1]
            x = torch.cat([bw, ell], dim=-1)
        else:
            # linear ratio uses eps to avoid /0
            ell = bw[...,1:2] / (bw[...,0:1] + 1e-8)
            x = torch.cat([bw, ell], dim=-1)

        if self.d_model > 1:
            pe = self.positional_encoding(x)
            x = torch.cat([x, pe], -1)

        h = F.leaky_relu(self.fc1(x))
        h = F.leaky_relu(self.fc2(h))
        h = F.leaky_relu(self.fc3(h))

        # concentrations with huge dynamic range
        s = self.head_k(h) / self._tanh_scale
        log_extra = self._log_span * torch.tanh(s)
        k = self.k_min + torch.exp(log_extra)

        # monotone, sorted elevations in (0, π/2)
        raw = self.head_el(h)
        gaps = F.softplus(raw) + 1e-6
        thetas = torch.cumsum(gaps, dim=-1)
        thetas = (thetas / (thetas[..., -1:] + 1e-6)) * (math.pi/2 - 1e-3)

        # positive weights (normalised in loss)
        weights = F.softplus(self.head_w(h))
        return k, thetas, weights

import torch
import math
from torch.utils.data import Dataset

class ASGDatasetAug(Dataset):
    def __init__(self, lambdas, mus, jitter=True,
                 scale_jitter_sigma=0.10,        # ±10 % in log‑space
                 ratio_jitter_sigma=0.15):       # ±15 % in log‑ratio
        self.pairs   = [(l.item(), m.item()) for l, mu_list in zip(lambdas, mus) for m in mu_list]
        self.jitter  = jitter
        self.sigma_s = scale_jitter_sigma
        self.sigma_r = ratio_jitter_sigma

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        λ, μ = self.pairs[idx]

        if self.jitter:
            # 1. multiplicative scale on both λ and μ  (keeps ratio unchanged)
            log_s = torch.randn(1).item() * self.sigma_s
            s     = math.exp(log_s)
            λ    *= s
            μ    *= s

            # 2. multiplicative jitter **only** on the ratio r = μ/λ
            log_rj = torch.randn(1).item() * self.sigma_r
            r      = μ / λ
            r     *= math.exp(log_rj)
            r      = max(r, 1e-4)          # clamp to avoid 0
            r      = min(r, 1.0)           # enforce μ ≤ λ
            μ      = λ * r                 # recompute μ

        return torch.tensor([λ, μ], dtype=torch.float32)
