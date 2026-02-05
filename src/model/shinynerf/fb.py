"""
FB‑5 → spherical‑harmonic encoder (PyTorch 1.11‑compatible)
Revision 7.1 (2025‑07‑01) — **bug‑fix** release
────────────────────────────────────────────────────────────
* Fixes the broadcasting error that triggered
  `RuntimeError: einsum(): operands do not broadcast…` when the input
  batch carried extra latent dimensions (e.g. `[batch, K]`).  The
  offending two‑stage `einsum` has been replaced with a shape‑safe dot
  product.
* Rolled the rotation kernel back to the proven per‑ℓ loop; its share of
  total runtime is < 3 % and the old vectorised attempt was getting
  fiddly with rank‑4 shapes.
* Kept the **vectorised `d_nu_sq`** construction and the JIT hints –
  those are safe and give most of the speed‑up.

If you still need more speed, the next win is to move the whole
`fb_standard_coeffs` into `torch.compile()` (PyTorch 2.x) — that removes
Python overhead without touching tensor algebra.
"""
from __future__ import annotations
import math, functools
import torch
from torch import Tensor
from e3nn.o3 import wigner_D
#from src.model.trefnerf.wigner_rotation import wigner_D


PI = math.pi
SQRT_PI = math.sqrt(PI)
DTYPE_R = torch.float64
DTYPE_C = torch.complex128
idx_shift = lambda l_max, m: m + l_max
Max_N = 100
Max_T = 84

# ---------------------------------------------------------------------
# Helpers: finite checks &  sin(πq/2)/q
# ---------------------------------------------------------------------

def _check_tensor(name: str, x: Tensor, k=None, b=None):
    if torch.isnan(x).any():
        raise RuntimeError(f"NaNs in {name}, {k.max()}, {k.min()}, {b.max()},{b.min()}")
    if torch.isinf(x).any():
        raise RuntimeError(f"Infs in {name}, {k.max()}, {k.min()}, {b.max()}, {b.min()}")


def sinc_pi_half(q: Tensor) -> Tensor:
    x = (PI/2)*q
    small = q.abs() < 1e-4
    return torch.where(small,
                       1. - x**2/6 + x**4/120,
                       torch.sin(x)/q)

# ---------------------------------------------------------------------
# Modified‑Bessel  I_{n+½}(κ)  (Eq. 20)
# ---------------------------------------------------------------------


# def iv_half_integer(x: torch.Tensor,
#                     n_max: int,
#                     extra: int = 32) -> torch.Tensor:
#     """
#     Stable (Miller) computation of [I_{0.5}(x), …, I_{n_max+0.5}(x)]
#     for every element of x (broadcasted).  Uses only float64.
#     extra  : how many “dummy” orders to start above n_max
#              (≥20 is usually enough; Miller converges fast).
#     """
#     # Ensure x is in float64 for numerical stability
#     x = x.to(torch.float64)

#     # Small epsilon value to avoid division by zero
#     epsilon = 1e-4

#     # Clamp x to avoid division by zero
#     x = torch.clamp(x, min=epsilon)

#     N = n_max + extra  # highest order we march down from

#     # --- Miller downward recurrence for the spherical i_n -------------
#     # relationship: i_n(x) = sqrt(pi/(2x)) * I_{n+0.5}(x)
#     i_np1 = torch.zeros_like(x)  # i_{N+1} = 0
#     i_n = torch.ones_like(x)     # i_{N}  = 1  (arbitrary scale)
#     bag = []  # store i_n down to n=0

#     for n in range(N, -1, -1):  # n = N … 0
#         # i_{n-1} = i_{n+1} + (2n+1)/x * i_n   (stable direction)
#         coef = (2 * n + 1) / x
#         i_nm1 = i_np1 + coef * i_n
#         if n <= n_max:
#             bag.append(i_n)  # keep what we need
#         i_np1, i_n = i_n, i_nm1  # shift window

#     bag.reverse()  # now [i_0, …, i_{n_max}]
#     i = torch.stack(bag, dim=-1)  # shape …, n_max+1

#     # --- rescale by matching the exact i_0 -----------------------------
#     i0_exact = torch.sinh(x) / x
#     scale = i0_exact / i[..., 0]  # rescale to match i_0
#     i = i * scale.unsqueeze(-1)  # broadcast
#     _check_tensor("i", i, x, n_max)
#     _check_tensor("scale", scale, x, n_max)

#     # --- convert back to I_{n+0.5} -------------------------------------
#     root = torch.sqrt(torch.pi / (2 * x))
#     _check_tensor("root", root, x, n_max)
#     return (i / root.unsqueeze(-1)).squeeze()


def iv_half_integer_masked(x: Tensor,
                           n_max_each: Tensor,
                           *,
                           dtype: torch.dtype = torch.float64) -> Tensor:
    """
    Miller-style I_{n+½}(x) for variable n_max, no in-place ops.
    Returns (..., max(n_max_each)+1) with zeros beyond each sample’s n_max.
    """
    extra = max(8, torch.ceil(1.5*n_max_each.max()**(1/3)).clamp_max(Max_N))
    # broadcast & prep (all functional ops)
    x, n_max_each = torch.broadcast_tensors(
        x.to(dtype),
        n_max_each.to(torch.int64)
    )

    x = torch.where(x<=1e-12, 1e-12 + 0.5*x.square(), x)

    topN_each  = n_max_each + extra
    N_global   = int(topN_each.max())
    B          = x.numel()

    x_flat     = x.reshape(B)
    topN_flat  = topN_each.reshape(B)
    n_max_flat = n_max_each.reshape(B)
    M_collect  = int(n_max_flat.max())

    # Miller downward recurrence (fully functional)
    i_np1 = torch.zeros_like(x_flat)
    i_n   = torch.ones_like(x_flat)
    rows  = []

    for n in range(N_global, -1, -1):
        active = (n <= topN_flat)
        coef   = (2*n + 1) / x_flat
        i_nm1  = i_np1 + coef * i_n

        # freeze rows not yet active
        i_nm1 = torch.where(active, i_nm1, i_n)
        i_np1 = torch.where(active, i_n,  i_np1)

        # collect only up to M_collect
        if n <= M_collect:
            rows.append(i_n)

        i_n = i_nm1

    rows.reverse()
    i_all = torch.stack(rows, dim=-1)            # (B, M_collect+1)

    # rescale so that i_0 is exact
    i0_exact = torch.sinh(x_flat) / x_flat
    scale    = i0_exact / i_all[..., 0]
    i_all    = i_all * scale.unsqueeze(-1)

    # convert to I_{n+½}(x)
    root  = torch.sqrt(torch.pi / (2 * x_flat))
    I_all = i_all / root.unsqueeze(-1)           # (B, M_collect+1)

    # zero-out beyond each sample’s n_max (out-of-place)
    orders = torch.arange(M_collect + 1,
                          device=x.device,
                          dtype=torch.int64)
    mask   = (orders <= n_max_flat.unsqueeze(-1))
    mask_f = mask.to(dtype)                      # promote to float
    I_all  = I_all * mask_f

    # restore original shape
    return I_all.reshape(x.shape[:-1] + (M_collect + 1,))

# ---------------------------------------------------------------------
# Wigner‑d(π/2) table via Trapani–Navaza recursion
# ---------------------------------------------------------------------


def d_from_halfpi(theta: torch.Tensor, d_half: torch.Tensor) -> torch.Tensor:
    """
    Eq. (21): build d^ℓ_{m,n}(θ) from the π/2 table.
        theta  (*B₁, …, *B_k)         – radians
        d_half (L+1, 2L+1, 2L+1)       – real, at β = π/2
    returns
        (*B₁, …, *B_k, L+1, 2L+1, 2L+1)  – complex64/128
    """
    L = d_half.shape[0] - 1
    S = 2 * L + 1
    u = torch.arange(-L, L + 1, device=theta.device)          # (S,)

    # e^{i u θ}  – shape (*batch, S)
    phase_u = torch.exp(1j * theta[..., None] * u)

    # promote to complex once
    if d_half.dtype == torch.float32:
        d_half_c = d_half.to(torch.complex64)
    else:
        d_half_c = d_half.to(torch.complex128)

    # ----------  Σ_u d(u,m) d(u,n) e^{iuθ}  -----------------------------
    #   '...u, eum, eun -> ...emn'
    dθ = torch.einsum('...u,eum,eun->...emn', phase_u, d_half_c, d_half_c)

    # ----------  multiply by  i^{n-m}  ---------------------------------
    m = u                                                        # (S,)
    i_phase = (1j) ** (m[None] - m[:, None])                     # (S, S)
    dθ = dθ * i_phase.to(dθ.dtype)                               # broadcasts

    return dθ



@functools.lru_cache(maxsize=None)
def get_d_table(l_max: int,
                device: torch.device = torch.device("cpu"),
                dtype: torch.dtype    = torch.float64) -> torch.Tensor:
    """
    Cached π/2 Wigner-d stack.

    Returns
    -------
    d_half : (l_max+1, 2*l_max+1, 2*l_max+1)  real tensor on `device`
    """
    # ─── build angle tensors ON THE SAME DEVICE ────────────────────────
    zeros    = torch.tensor(0.0,        dtype=dtype)
    half_pi  = torch.tensor(math.pi/2,  dtype=dtype)

    rows = []
    for ℓ in range(l_max + 1):
        dℓ = wigner_D(ℓ, alpha=zeros, beta=half_pi, gamma=zeros)  # (1, 2ℓ+1, 2ℓ+1)
        rows.append(dℓ.squeeze(0).to(dtype=dtype, device=device))                         # strip batch dim

    # ─── pad to common square (M×M) and stack ─────────────────────────
    M = 2 * l_max + 1
    padded = torch.zeros((l_max + 1, M, M), dtype=dtype, device=device)
    for ℓ, dℓ in enumerate(rows):
        s = l_max - ℓ                      # offset of the (2ℓ+1) block
        padded[ℓ, s:s+2*ℓ+1, s:s+2*ℓ+1] = dℓ

    return padded     # will be memo-cached keyed by (l_max, device, dtype)


def _C_kappa_beta(kappa: torch.Tensor,
                  beta:   torch.Tensor,
                  *,
                  dtype: torch.dtype) -> torch.Tensor:
    """Truncated series for *C(κ,β)* with the same cut‑off R = 1.5 κ + 24."""
    kappa, beta = torch.broadcast_tensors(kappa, beta)
    device = kappa.device

    k64 = kappa.to(torch.float64)
    b64 = beta.to(torch.float64)

    R = (1.5*k64 + 24).ceil().to(torch.int64).clamp_max(Max_N)       # TODO CHANGE per‑batch cut‑off
    Rmax = int(R.max().item())

    r = torch.arange(Rmax + 1, dtype=torch.float64, device=device)  # (Rmax+1,)
    log_coeff = torch.lgamma(r + 0.5) - torch.lgamma(r + 1)          # log Γ‑ratio
    _check_tensor("log_coeff", log_coeff, kappa, beta)

    log_term = (log_coeff
                + 2.0*r*torch.log(b64)
                - (2.0*r+0.5)*(torch.log(k64) - math.log(2.0)))
    _check_tensor("log_term", log_term, kappa, beta)

    # I_{2r+½}(κ)
    I_all = iv_half_integer_masked(k64, 2*R)             # (B, 2Rmax+1)
    _check_tensor("I_all", I_all, kappa, beta)

    idx   = (2*r).to(torch.long)                     # (R,)
    idx   = idx.unsqueeze(0).expand(k64.shape[0], -1)  # (B,R)
    I_sel = torch.gather(I_all, -1, idx)             # (B,R)
    _check_tensor("I_sel", I_sel, kappa, beta)

    term = torch.exp(log_term) * I_sel
    C = 2*math.pi * term.sum(dim=-1)                 # (B,)
    _check_tensor("C", C, kappa, beta)

    return C.to(dtype)

# core utility ---------------------------------------------------------------
def _pad_up(l_max, device, dtype):
    """Matrix U[ell, a] that stores u' = -ell…ell centred in a
       vector of length 2*l_max+1; invalid entries are 0."""
    up = torch.zeros(l_max+1, 2*l_max+1, dtype=dtype, device=device)
    for ell in range(l_max+1):
        rng = torch.arange(-ell, ell+1, device=device, dtype=dtype)
        up[ell, l_max-ell:l_max+ell+1] = rng
    return up                                                    # (ℓ,Â)
# ---------------------------------------------------------------------
#   Standard‑frame coefficients  (vectorised d_nu_sq; per‑m loop is back)
# ---------------------------------------------------------------------

# def fb_standard_coeffs(kappa: torch.Tensor,
#                            beta : torch.Tensor,
#                            l_max: int,
#                            dtab : torch.Tensor) -> torch.Tensor:
#     """
#     Vectorised, memory-friendly version (PyTorch 1.11).
#     Shapes:  kappa, beta  →  (*B, 1)
#              return       →  (*B, l_max+1, 2*l_max+1)  (complex)
#     """
#     dev = kappa.device
#     kappa = kappa.squeeze(-1).to(DTYPE_R).clamp_min(1e-6)
#     beta  = beta.squeeze(-1).to(DTYPE_R).clamp_min(1e-6)
#     Bsh   = kappa.shape                                    # batch dims

#     # ---------- cut-offs ---------------------------------------------------
#     N   = (1.5*kappa + 24).ceil().int(); maxN = int(N.max())
#     T   = (1.44*beta + 12).ceil().int(); maxT = int(T.max())
#     Inu = iv_half_integer(kappa, maxN)                     # (*B,N+1)
#     two_n1 = torch.arange(maxN+1, device=dev, dtype=DTYPE_R)*2 + 1

#     if dtab.size(0)-1 < maxN:                              # enlarge Wigner-d
#         dtab = get_d_table(maxN, dev)                      # (N+1,2N+1,N+1)

#     # ---------- A_u  (∑_n (2n+1) I_n d²) ----------------
#     idx0  = (dtab.size(0)-1 - torch.arange(maxN+1, device=dev)).long()
#     rows  = idx0[:,None] + torch.arange(-maxN,maxN+1,device=dev).long()
#     d_nu_sq = dtab[torch.arange(maxN+1, device=dev)[:,None],
#                    rows, idx0[:,None]].pow(2)              # (N+1, 2N+1)
#     Au = torch.einsum('...n,nu->...u', Inu*two_n1, d_nu_sq)  # (*B, 2N+1)

#     # ---------- d_{ℓu′0} and d_{ℓu′m} tables -------------
#     Â = 2*l_max + 1
#     up_pad = torch.zeros(l_max+1, Â, dtype=DTYPE_R, device=dev)
#     for ℓ in range(l_max+1):
#         up_pad[ℓ, l_max-ℓ:l_max+ℓ+1] = torch.arange(-ℓ, ℓ+1, device=dev)

#     idx_col = idx0[:l_max+1]                               # (ℓ,)

#     row0 = idx_col[:,None] + up_pad.long()                 # (ℓ,Â)
#     ℓ_idx = torch.arange(l_max+1, device=dev)[:,None].expand_as(row0)
#     col0  = idx_col[:,None].expand_as(row0)
#     d_l_up0 = dtab[ℓ_idx, row0, col0]                      # (ℓ,Â)

#     m_even = torch.arange(0, l_max+1, 2, device=dev)       # (M,)
#     M      = m_even.numel()

#     row_m  = row0[:,None,:].expand(-1, M, -1)              # (ℓ,M,Â)
#     col_m  = (idx_col[:,None] + m_even[None,:])[:,:,None]  # (ℓ,M,1)
#     col_m  = col_m.expand_as(row_m)
#     d_l_upm = dtab[ℓ_idx[:,None,:], row_m, col_m]          # (ℓ,M,Â)
#     valid   = (m_even[None,:] <= torch.arange(l_max+1, device=dev)[:,None])
#     d_l_upm = d_l_upm * valid[:, :, None]

#     dl_prod = d_l_up0.unsqueeze(0) * d_l_upm.permute(1,0,2)   # (M,ℓ,Â)

#     # ---------- sinc_q(u,ℓ,a) --------------------------------------------
#     u = torch.arange(-maxN, maxN+1, dtype=DTYPE_R, device=dev)
#     q = (u[:,None,None] + up_pad[None,:,:])                   # (2N+1,ℓ,Â)
#     sinc_q = sinc_pi_half(q)                                  # (2N+1,ℓ,Â)

#     # ---------- Σ_u  A_u · sinc_q  →  B_{ℓa} -----------------------------
#     B_la = torch.einsum('...u,ula->...la', Au, sinc_q)        # (*B,ℓ,Â)

#     # ---------- Σ_t  S_t · γ   →  S̃_m  -------------------------------
#     t = torch.arange(maxT+1, device=dev, dtype=DTYPE_R)
#     m_half = (m_even//2).to(DTYPE_R)
#     log_beta = torch.log(beta/2).unsqueeze(-1)                # (*B,1)
#     t_term   = 2*t[:,None] + m_half[None,:]                   # (T,M)
#     log_S  = (log_beta.unsqueeze(-1)*t_term) \
#              - torch.lgamma(t+1)[:,None] \
#              - torch.lgamma(t[:,None]+m_half[None,:]+1)
#     S_t   = torch.exp(log_S)                                  # (*B,T,M)

#     p = 4*t[:,None] + m_even[None,:] + 1
#     log_g = 0.5*math.log(PI) + torch.lgamma((p+1)/2) - torch.lgamma(p/2+1)
#     γ     = torch.exp(log_g)                                  # (T,M)

#     S_bar = torch.einsum('...tm,tm->...m', S_t, γ)            # (*B,M)

#     # ---------- term(b,m,ℓ)  = Σ_a B·S̄·d-product ------------------------
#     term_tmp = S_bar.unsqueeze(-1).unsqueeze(-1) * B_la.unsqueeze(-3)  # (*B,M,ℓ,Â)
#     term = (term_tmp * dl_prod.unsqueeze(0).unsqueeze(0)).sum(-1)      # (*B,M,ℓ)

#     # ---------- prefactor & assemble coefficient tensor -----------------
#     logI0 = torch.log(torch.i0(beta) + 1e-300)
#     invC  = torch.exp(-kappa - logI0 - math.log(2*PI))        # (*B)
#     sqrt  = torch.sqrt((2*torch.arange(l_max+1, device=dev)+1) /
#                        (2*kappa.unsqueeze(-1)))               # (*B,ℓ)
#     pref  = PI * invC.unsqueeze(-1) * sqrt                    # (*B,ℓ)

#     coeff = torch.zeros(*Bsh, l_max+1, 2*l_max+1,
#                         dtype=DTYPE_C, device=dev)

#     phase   = ((1j)**(-m_even)).to(DTYPE_C)                   # (M,)
#     idx_pos = (m_even + l_max).long()                         # (M,)

#     rhs = term.transpose(-2, -1).to(DTYPE_C)   # (*B, ℓ, M)
#     rhs = rhs * phase[None, None, :]           #   broadcast on last axis (M)
#     rhs = rhs * pref.to(DTYPE_C).unsqueeze(-1) # (*B, ℓ, 1) → broadcasts vs M
#     coeff[..., :, idx_pos] = rhs               # works, shapes match

#     m_all = torch.arange(1, l_max+1, device=dev)
#     coeff[..., :, l_max-m_all] = torch.conj(coeff[..., :, l_max+m_all]) \
#                                  * ((-1)**m_all)
#     return coeff

import math, functools, torch
from typing import Optional

# ------------------------------------------------------------
# utility: choose matching complex dtype once
# ------------------------------------------------------------
def _to_cplx(x: torch.Tensor, real_dtype: torch.dtype) -> torch.Tensor:
    return x.to(torch.complex128 if real_dtype == torch.float64
                             else torch.complex64)

# ------------------------------------------------------------
# main routine
# ------------------------------------------------------------
# def fb_standard_coeffs(kappa:  torch.Tensor,
#                        beta:   torch.Tensor,
#                        l_max:  int,
#                        d_tbl:  torch.Tensor,
#                        *,
#                        device: Optional[torch.device] = None,
#                        dtype:  torch.dtype            = torch.float64
#                       ) -> torch.Tensor:
#     r"""
#     Vectorised evaluation of

#         (f)_ℓ^m   for  |m| ≤ ℓ ≤ ℓ_max

#     on *one* or *many* (κ,β) pairs.  Return shape
#         (*B,  ℓ_max+1,  2ℓ_max+1)     ‑‑ complex
#     where axis‑2 is ordered with m = −ℓ_max … +ℓ_max.
#     """

#     # ‑‑‑ housekeeping -------------------------------------------------
#     kappa, beta = torch.broadcast_tensors(
#         kappa.to(dtype), beta.to(dtype))
#     if device is None:
#         device = kappa.device
#     kappa, beta = kappa.to(device), beta.to(device)

#     # Check the shapes and types of input tensors
#     _check_tensor("kappa", kappa)
#     _check_tensor("beta", beta)

#     orig_shape  = kappa.shape[:2]

#     kappa  = kappa.reshape(-1, 1)
#     beta  = beta.reshape(-1, 1)

#     B          = kappa.shape[0]
#     L          = l_max + 1                        # number of ℓ rows
#     M          = 2 * l_max + 1                   # full m‑axis length
#     Mpos       = l_max + 1                       #  m = 0…ℓ_max
#     m_pos      = torch.arange(0, l_max + 1,      device=device, dtype=dtype)
#     l_range    = torch.arange(0, l_max + 1,      device=device, dtype=dtype)

#     # Check shapes of computed tensors
#     _check_tensor("m_pos", m_pos)
#     _check_tensor("l_range", l_range)

#     # ‑‑‑ cut‑offs -----------------------------------------------------
#     r_cut = (1.5 * kappa + 24).ceil().to(torch.int64)       # (B,)
#     t_cut = ((36/25) * beta + 12).ceil().to(torch.int64)    # (B,)
#     Rmax  = int(r_cut.max().item())
#     Tmax  = int(t_cut.max().item())

#     # Check cut-off tensors
#     _check_tensor("r_cut", r_cut)
#     _check_tensor("t_cut", t_cut)

#     # ‑‑‑ index tensors ------------------------------------------------
#     n_idx  = torch.arange(0, Rmax + 1,          device=device, dtype=dtype)   # (N,)
#     u_idx  = torch.arange(-Rmax, Rmax + 1,      device=device, dtype=dtype)   # (U,)
#     t_idx  = torch.arange(0, Tmax + 1,          device=device, dtype=dtype)   # (T,)
#     up_idx = torch.arange(-l_max, l_max + 1,    device=device, dtype=dtype)   # (Up,)

#     # Check index tensors
#     _check_tensor("n_idx", n_idx)
#     _check_tensor("u_idx", u_idx)
#     _check_tensor("t_idx", t_idx)
#     _check_tensor("up_idx", up_idx)

#     N, U, Up, T = n_idx.numel(), u_idx.numel(), up_idx.numel(), t_idx.numel()

#     # ================================================================
#     # 1.  (2n+1) I_{n+½}(κ)   …  bn     (B,N)
#     # ================================================================
#     I_n_all   = iv_half_integer_masked(kappa, r_cut).to(dtype)      # (B,N)
#     pref_bn   = (2*n_idx + 1)[None, :] * I_n_all            # (B,N)

#     mask_n    = (n_idx[None, :] <= r_cut[:, None]).squeeze()
#     bn        = torch.where(mask_n, pref_bn, torch.zeros_like(pref_bn))

#     # Check tensor bn
#     _check_tensor("bn", bn)

#     # ================================================================
#     # 2.  dⁿ_{u0}(π/2)²         …  nu     (N,U)
#     # ================================================================
#     pad_R     = get_d_table(Rmax, device=device, dtype=dtype)    # (Rmax+1, 2R+1, 2R+1)
#     d_n_u0    = pad_R[:, :, Rmax]        # pick m'=0 column  (N,U)
#     nu        = d_n_u0.pow(2)            # square once, reuse everywhere

#     # Check tensor nu
#     _check_tensor("nu", nu)

#     # ================================================================
#     # 3.  β‑dependent factor     …  btm    (B,T,Mpos)
#     # ================================================================
#     tF = t_idx[None, :, None]            # (1,T,1)
#     mF = m_pos[None, None, :]            # (1,1,Mpos)
#     log_fac  = ( (2*tF + mF/2) * torch.log(beta[:,None,None]/2)
#                - torch.lgamma(tF + 1)
#                - torch.lgamma(tF + mF/2 + 1) )
#     btm_full = torch.exp(log_fac)                        # (B,T,Mpos)
#     mask_t   = (t_idx[None, :, None] <= t_cut[:,None,None])
#     btm      = torch.where(mask_t, btm_full, torch.zeros_like(btm_full))

#     # Check tensor btm
#     _check_tensor("btm", btm, kappa, beta)

#     # ================================================================
#     # 4.  d^ℓ_{u'0} d^ℓ_{u'm}    …  lpm    (L,Up,Mpos)
#     # ================================================================
#     off0      = l_max
#     dℓ_up0    = d_tbl[:, :, off0]                        # (L,Up)
#     cols_m    = (off0 + m_pos).to(torch.long)            # (Mpos,)
#     dℓ_upm    = d_tbl[:, :, cols_m]                     # (L,Up,Mpos)
#     lpm       = dℓ_up0.unsqueeze(-1) * dℓ_upm           # (L,Up,Mpos)

#     # Check tensor lpm
#     _check_tensor("lpm", lpm)

#     # ================================================================
#     # 5.  G(4t+m+1, u+u')        …  tupm   (T,U,Up,Mpos)    complex
#     # ================================================================
#     p_tm  = (4 * t_idx[:, None] + m_pos[None, :] + 1).to(dtype)    # (T, Mpos)
#     q_up  = (u_idx[:, None] + up_idx[None, :]).to(dtype)           # (U, Up)

#     # Check tensors p_tm and q_up
#     _check_tensor("p_tm", p_tm)
#     _check_tensor("q_up", q_up)

#     ln_Γp2   = torch.lgamma(p_tm + 2)
#     ln_p1    = torch.log(p_tm + 1)

#     # now broadcast to (T,M,U,Up)
#     p4d  = p_tm    [:, :, None, None]
#     lnΓ  = ln_Γp2  [:, :, None, None]
#     #lp   = ln_p    [:, :, None, None]
#     lp1  = ln_p1   [:, :, None, None]

#     q4d  = q_up[None, None, :, :]

#     arg1 = (p4d + q4d + 2)/2
#     arg2 = (p4d - q4d + 2)/2

#     # Including the missing 2^p term
#     log_amp = ( math.log(math.pi)
#                 + lnΓ
#                 - math.log(2.0) * p4d        # factor of 2^p in the denominator
#                 - lp1
#                 - torch.lgamma(arg1) - torch.lgamma(arg2) )

#     amp = torch.exp(log_amp)                     # (T,M,U,Up)

#     # Check tensor amp
#     _check_tensor("amp", amp)

#     phase = q4d * (math.pi/2)
#     phase_c = torch.cos(phase).to(dtype) #+ 1j*torch.sin(phase).to(dtype) #same as torch.exp(1j * q4d * (math.pi / 2))

#     G_tmp   = _to_cplx(amp, dtype) * phase_c       # complex  (T,M,U,Up)
#     tupm    = G_tmp.permute(0, 2, 3, 1)            # (T,U,Up,Mpos)

#     # Check tensor tupm
#     _check_tensor("tupm", tupm)


#     # ================================================================
#     # 6.  big inner contraction      →   S⁺ᵇ_{ℓm}  (B,L,Mpos)
#     # ================================================================
#     btm_c  = _to_cplx(btm,  dtype)
#     lpm_c  = _to_cplx(lpm,  dtype)
#     bn_c = _to_cplx(bn, dtype)           # (B, N)  → complex
#     nu_c = _to_cplx(nu, dtype)           # (N, U)

#     # Check tensors before contraction
#     _check_tensor("btm_c", btm_c)
#     _check_tensor("lpm_c", lpm_c)
#     _check_tensor("bn_c", bn_c)
#     _check_tensor("nu_c", nu_c)

#     # 6-A  R[b,u] = Σ_n  bn · nu
#     R = torch.einsum('bn,nu->bu', bn_c, nu_c)                 # (B, U)  # TODO VERIFY

#     # 6-B  P[b,t,up,m] = Σ_u  R · G
#     P = torch.einsum('bu,tupm->btpm', R, tupm)                # (B, T, Up, Mpos) # TODO VERIFY

#     # 6-C  inner[b,l,m] = Σ_{t,up}  P · β · L
#     inner = torch.einsum('btpm,bltm,lpm->blm',
#                         P, btm_c, lpm_c)                     # (B, L, Mpos) # TODO VERIFY

#     # ------------------------------------------------------------
#     # 7.  front prefactor  (π i^{-m} / C) √((2ℓ+1)/(2κ))
#     # ------------------------------------------------------------
#     C      = _C_kappa_beta(kappa, beta, dtype=dtype)          # (B,)
#     sqrtL  = torch.sqrt((2*l_range + 1) / (2*kappa))          # (B, L)
#     phaseM = torch.exp(-0.5j * math.pi * m_pos) #(1j) ** (-m_pos)                                 # (Mpos,)

#     S_pos = (math.pi * phaseM)[None, None, :]        \
#             / C[:, None, None]                       \
#             * sqrtL[:, :, None] * inner              # (B, L, Mpos)
#     _check_tensor("inner", inner)
#     _check_tensor("sqrtL", sqrtL)
#     _check_tensor("C", C)
#     _check_tensor("S_pos", S_pos)

#     # ------------------------------------------------------------
#     # 8.  assemble ±m into rectangular (B, L, 2ℓ_max+1)
#     # ------------------------------------------------------------
#     F = torch.zeros((B, L, M), dtype=S_pos.dtype, device=device)

#     # +m side
#     col_pos = (m_pos + l_max).to(torch.long)                  # (Mpos,)
#     F.index_copy_(2, col_pos, S_pos)

#     # −m side  (−m = 1 … ℓ_max)
#     if l_max > 0:
#         m_neg   = m_pos[1:]                                   # 1 … ℓ_max
#         col_neg = (l_max - m_neg).to(torch.long)
#         factor  = (-1.0) ** m_neg
#         F_neg   = factor[None, None, :] * torch.conj(S_pos[:, :, 1:])
#         F.index_copy_(2, col_neg, F_neg)
#     _check_tensor("F", F)

#     return F.view(*orig_shape, l_max+1, 2*l_max+1)          # (B, ℓ_max+1, 2ℓ_max+1)   complex

# def fb_standard_coeffs(
#         kappa: torch.Tensor,
#         beta:  torch.Tensor,
#         l_max: int,
#         d_tbl: torch.Tensor,
#         *,
#         device: Optional[torch.device] = None,
#         eps64: torch.dtype = torch.float64,   # high-prec dtype
#         eps32: torch.dtype = torch.float32    # low-prec dtype
#     ) -> torch.Tensor:
#     """
#     Same signature as fb_standard_coeffs(), but:
#       • exp/Γ/Bessel work in float64
#       • large linear pieces use float32/complex64
#     Output is still complex128.
#     """

#     # ---------- housekeeping -------------------------------------------
#     kappa, beta = torch.broadcast_tensors(kappa.to(eps64), beta.to(eps64))
#     if device is None:
#         device = kappa.device
#     kappa, beta = kappa.to(device), beta.to(device)

#     orig_shape = kappa.shape[:2]
#     kappa = kappa.reshape(-1, 1)
#     beta  = beta.reshape(-1, 1)
#     B     = kappa.shape[0]

#     L      = l_max + 1
#     M      = 2 * l_max + 1
#     Mpos   = l_max + 1
#     m_pos  = torch.arange(0, l_max + 1, device=device, dtype=eps64)
#     l_rng  = torch.arange(0, l_max + 1, device=device, dtype=eps64)

#     # ---------- cut-offs & indices -------------------------------------
#     r_cut = (1.5 * kappa + 24).ceil().to(torch.int64)
#     t_cut = ((36/25) * beta + 12).ceil().to(torch.int64)
#     Rmax  = int(r_cut.max())
#     Tmax  = int(t_cut.max())

#     n_idx  = torch.arange(0, Rmax + 1,  device=device, dtype=eps64)
#     u_idx  = torch.arange(-Rmax, Rmax + 1, device=device, dtype=eps64)
#     t_idx  = torch.arange(0, Tmax + 1, device=device, dtype=eps64)
#     up_idx = torch.arange(-l_max, l_max + 1, device=device, dtype=eps64)

#     # ---------- 1.  bn  (64-bit) ---------------------------------------
#     I_n   = iv_half_integer_masked(kappa, r_cut)         # float64
#     bn64  = (2*n_idx + 1)[None, :] * I_n
#     bn64  = torch.where(n_idx[None, :] <= r_cut, bn64, torch.zeros_like(bn64))

#     # ---------- 2.  nu  (64-bit) --------------------------------------
#     pad_R = get_d_table(Rmax, device=device, dtype=eps64)
#     nu64  = pad_R[:, :, Rmax].pow(2)                      # (N,U)

#     # ---------- 3.  btm  (64-bit) -------------------------------------
#     tF   = t_idx[None, :, None]
#     mF   = m_pos[None, None, :]
#     logf = ((2*tF + mF/2) * torch.log(beta[:,None,None]/2)
#             - torch.lgamma(tF+1)
#             - torch.lgamma(tF + mF/2 + 1))
#     btm64 = torch.exp(logf)
#     btm64 = torch.where(t_idx[None,:,None] <= t_cut[:,None,None],
#                         btm64, torch.zeros_like(btm64))

#     # ---------- 4.  lpm  (64-bit – but small tensor) -------------------
#     off0  = l_max
#     d_up0 = d_tbl[:, :, off0]
#     cols  = (off0 + m_pos).long()
#     lpm64 = d_up0.unsqueeze(-1) * d_tbl[:, :, cols]

#     # ---------- 5.  G tensor (large)  → float32/complex64 --------------
#     p_tm = (4*t_idx[:,None] + m_pos[None,:] + 1).to(eps64)
#     q_up = (u_idx[:,None] + up_idx[None,:]).to(eps64)

#     lnΓ = torch.lgamma(p_tm + 2)[:, :, None, None]   # (T, Mpos, 1, 1)
#     ln1 = torch.log(p_tm + 1)   [:, :, None, None]   # (T, Mpos, 1, 1)
#     p4d = p_tm[:, :, None, None]; q4d = q_up[None,None,:,:]
#     arg1 = (p4d + q4d + 2)/2
#     arg2 = (p4d - q4d + 2)/2

#     log_amp = ( math.log(math.pi) + lnΓ
#                 - math.log(2.0)*p4d - ln1
#                 - torch.lgamma(arg1) - torch.lgamma(arg2) )
#     amp32   = torch.exp(log_amp).to(eps32)               # → float32
#     phase32 = torch.cos(q4d * (math.pi/2)).to(eps32)

#     tupm = (_to_cplx(amp32, eps32) * phase32).permute(0,2,3,1)  # (T,U,Up,Mpos)

#     # ---------- 6.  big contractions in float32 -----------------------
#     bn32  = _to_cplx(bn64.to(eps32), eps32)
#     nu32  = _to_cplx(nu64.to(eps32), eps32)
#     btm32 = _to_cplx(btm64.to(eps32), eps32)
#     lpm32 = _to_cplx(lpm64.to(eps32), eps32)

#     R32 = torch.einsum('bn,nu->bu', bn32, nu32)          # (B,U)
#     P32 = torch.einsum('bu,tupm->btpm', R32, tupm)       # (B,T,Up,Mpos)
#     inner64 = torch.einsum('btpm,bltm,lpm->blm',
#                            P32, btm32, lpm32).double()   # back to 64-bit

#     # ---------- 7.  front prefactor (64-bit) --------------------------
#     C    = _C_kappa_beta(kappa, beta, dtype=eps64)       # (B,)
#     sqrtL= torch.sqrt((2*l_rng + 1) / (2*kappa))         # (B,L)
#     phaseM = torch.exp(-0.5j*math.pi*m_pos).to(torch.complex128)

#     S_pos = (math.pi * phaseM)[None,None,:] / C[:,None,None] \
#             * sqrtL[:,:,None] * inner64                     # (B,L,Mpos)

#     # ---------- 8.  assemble ±m  (64-bit) -----------------------------
#     F = torch.zeros((B, L, M),
#                     dtype=torch.complex128, device=device)
#     col_pos = (m_pos + l_max).long()
#     F.index_copy_(2, col_pos, S_pos)

#     if l_max > 0:
#         m_neg  = m_pos[1:]; col_neg = (l_max - m_neg).long()
#         factor = (-1.0)**m_neg
#         F_neg  = factor[None,None,:] * torch.conj(S_pos[:,:,1:])
#         F.index_copy_(2, col_neg, F_neg)

#     return F.view(*orig_shape, l_max+1, 2*l_max+1)        # complex128



def fb_standard_coeffs_real(
        kappa: torch.Tensor,
        beta:  torch.Tensor,
        l_max: int,
        d_tbl: torch.Tensor,
        *,
        device: Optional[torch.device] = None,
        eps64: torch.dtype = torch.float64,
        eps32: torch.dtype = torch.float32
    ) -> torch.Tensor:
    """
    Return *real* float64 tensor  F[..., ℓ, m]   (m = –ℓ..ℓ).
    The numerically heavy exp/Γ/Bessel parts stay in 64-bit,
    the big linear slabs switch to 32-bit to save RAM.
    """

    # ─── bookkeeping ----------------------------------------------------
    kappa, beta = torch.broadcast_tensors(kappa.to(eps64), beta.to(eps64))
    if device is None:
        device = kappa.device
    kappa, beta = kappa.to(device), beta.to(device)

    orig_shape = kappa.shape[:2]
    kappa = kappa.reshape(-1, 1)
    beta  = beta.reshape(-1, 1)
    Btot  = kappa.shape[0]

    L      = l_max + 1
    M      = 2 * l_max + 1
    Mpos   = l_max + 1
    m_pos  = torch.arange(0, l_max + 1, device=device, dtype=eps64)
    l_rng  = torch.arange(0, l_max + 1, device=device, dtype=eps64)

    # ─── cut-offs & indices --------------------------------------------
    r_cut = (1.5 * kappa + 24).ceil().to(torch.int64)
    t_cut = ((36/25) * beta + 12).ceil().to(torch.int64)
    Rmax  = int(r_cut.max());  Tmax = int(t_cut.max())

    n_idx  = torch.arange(0, Rmax + 1,  device=device, dtype=eps64)
    u_idx  = torch.arange(-Rmax, Rmax + 1, device=device, dtype=eps64)
    t_idx  = torch.arange(0, Tmax + 1, device=device, dtype=eps64)
    up_idx = torch.arange(-l_max, l_max + 1, device=device, dtype=eps64)

    # ─── 1. bn (64-bit) -------------------------------------------------
    I_n = iv_half_integer_masked(kappa, r_cut)               # (B,N)
    bn64 = (2*n_idx + 1)[None,:] * I_n
    bn64 = torch.where(n_idx[None,:] <= r_cut, bn64, 0.)

    # ─── 2. nu (64-bit) -------------------------------------------------
    nu64 = get_d_table(Rmax, device, eps64)[:, :, Rmax].pow(2)   # (N,U)

    # ─── 3. btm (64-bit) ----------------------------------------------
    tF, mF = t_idx[None,:,None], m_pos[None,None,:]
    logf = ((2*tF + mF/2) * torch.log(beta[:,None,None]/2)
            - torch.lgamma(tF+1) - torch.lgamma(tF + mF/2 + 1))
    btm64 = torch.exp(logf)
    btm64 = torch.where(t_idx[None,:,None] <= t_cut[:,None,None], btm64, 0.)
    _check_tensor("logf", logf)
    _check_tensor("btm64", btm64)

    # ─── 4. lpm (64-bit, small) ---------------------------------------
    off0 = l_max
    lpm64 = d_tbl[:, :, off0].unsqueeze(-1) * d_tbl[:, :, off0+m_pos.long()]
    _check_tensor("lpm64", lpm64)

    # ─── 5. G-tensor  →  float32 ( **still real** ) --------------------
    p_tm = (4*t_idx[:,None] + m_pos[None,:] + 1).to(eps64)
    q_up = (u_idx[:,None]    + up_idx[None,:]).to(eps64)
    _check_tensor("p_tm", p_tm)
    _check_tensor("q_up", q_up)

    lnΓ = torch.lgamma(p_tm + 2)[:, :, None, None]
    ln1 = torch.log(p_tm + 1)[:, :, None, None]
    p4d = p_tm[:, :, None, None]; q4d = q_up[None,None,:,:]
    arg1, arg2 = (p4d + q4d + 2)/2, (p4d - q4d + 2)/2
    _check_tensor("lnΓ", lnΓ)
    _check_tensor("ln1", ln1)
    _check_tensor("p4d", p4d)
    _check_tensor("arg1", arg1)
    _check_tensor("arg2", arg2)

    log_amp = ( math.log(math.pi) + lnΓ - math.log(2.)*p4d - ln1 # not defined for negative arg1, arg2
                - torch.lgamma(arg1) - torch.lgamma(arg2) )
    amp32   = torch.exp(log_amp).to(eps32)                   # (T,Mpos,U,Up)
    phase32 = torch.cos(q4d * (math.pi/2)).to(eps32)         #   same shape
    tupm32  = (amp32 * phase32).permute(0,2,3,1)             # (T,U,Up,Mpos)
    _check_tensor("amp32", amp32)
    _check_tensor("phase32", phase32)
    _check_tensor("tupm32", tupm32)

    # ─── 6. big contractions in float32 -------------------------------
    bn32, nu32 = bn64.to(eps32), nu64.to(eps32)
    btm32, lpm32 = btm64.to(eps32), lpm64.to(eps32)
    _check_tensor("bn32", bn32)
    _check_tensor("nu32", nu32)
    _check_tensor("btm32", btm32)
    _check_tensor("lpm32", lpm32)


    R32 = torch.einsum('bn,nu->bu', bn32, nu32)              # (B,U)
    P32 = torch.einsum('bu,tupm->btpm', R32, tupm32)         # (B,T,Up,Mpos)
    inner64 = torch.einsum('btpm,bltm,lpm->blm',
                           P32, btm32, lpm32).double()       # back to 64-bit
    _check_tensor("R32", R32)
    _check_tensor("P32", P32)
    _check_tensor("inner64", inner64)

    # ─── 7. front prefactor (real!) ------------------------------
    # i^{-m} = cos(πm/2) – the sine term cancels in the real part.
    phaseM = torch.cos(0.5 * math.pi * m_pos)                # (Mpos,)
    C      = _C_kappa_beta(kappa, beta, dtype=eps64)         # (B,)
    sqrtL  = torch.sqrt((2*l_rng + 1) / (2*kappa))           # (B,L)
    _check_tensor("phaseM", phaseM)
    _check_tensor("C", C)
    _check_tensor("sqrtL", sqrtL)

    S_pos = (math.pi * phaseM)[None,None,:] / C[:,None,None] \
            * sqrtL[:,:,None] * inner64                      # real (B,L,Mpos)
    _check_tensor("S_pos", S_pos)
    return S_pos.view(*orig_shape, l_max + 1, l_max + 1)     # real64, packed
    
    # # ─── 8. assemble ±m into full real tensor -------------------------
    # F = torch.zeros((Btot, L, M), dtype=eps64, device=device)
    # col_pos = (m_pos + l_max).long()
    # F.index_copy_(2, col_pos, S_pos)

    # if l_max > 0:
    #     m_neg  = m_pos[1:]; col_neg = (l_max - m_neg).long()
    #     # F(−m)= (−1)^m F(+m)  for *real* SH basis
    #     F.index_copy_(2, col_neg, ((-1.)**m_neg)[None,None,:] * S_pos[:,:,1:])

    # return F.view(*orig_shape, l_max+1, 2*l_max+1)           # float64, real


# For regular code
# def fb_standard_coeffs_bucketed(kappa: torch.Tensor,
#                                 beta : torch.Tensor,
#                                 l_max: int,
#                                 d_tbl: torch.Tensor,
#                                 *,
#                                 device=None,
#                                 dtype=torch.float64):

#     # ---------- 0.  remember original shape, then flatten ---------------
#     orig_shape   = kappa.shape[:-1]            # e.g. (B1, B2, …)
#     kappa_flat   = kappa.reshape(-1)
#     beta_flat    = beta.reshape(-1)
#     Btot         = kappa_flat.numel()

#     # ---------- 1.  bucket key = (Rmax, Tmax) per sample ----------------
#     Rcut = (1.5*kappa_flat + 24).ceil().to(torch.int64).clamp_max(Max_N)
#     Tcut = ((36/25)*beta_flat + 12).ceil().to(torch.int64).clamp_max(Max_T)

#         # choose how much you are willing to over-allocate
#     STEP_R = 16          # group every n in Rcut
#     STEP_T = 16          # group every n in Tcut

#     # quantise *down* so all samples in a bucket fit its cut-off
#     Rkey = ((Rcut + STEP_R - 1) // STEP_R) * STEP_R      # e.g. 37→40
#     Tkey = ((Tcut + STEP_T - 1) // STEP_T) * STEP_T

#     pair_code = torch.stack((Rkey, Tkey), 1)             # (Btot, 2)

#     #pair_code      = torch.stack((Rcut, Tcut), 1)              # (Btot, 2)
#     uniq_pairs, inv = torch.unique(pair_code, dim=0, return_inverse=True)

#     # ---------- 2.  pre-allocate flat output ---------------------------
#     out_flat = torch.empty(
#         (Btot, l_max+1, 2*l_max+1),
#         dtype=torch.complex128 if dtype==torch.float64 else torch.complex64,
#         device=device if device else kappa.device)

#     # ---------- 3.  loop over the small set of buckets -----------------
#     for bid in range(uniq_pairs.shape[0]):
#         idx = (inv == bid).nonzero(as_tuple=True)[0]           # 1-D indices
#         coeff = fb_standard_coeffs_real(kappa_flat[idx],
#                                    beta_flat[idx],
#                                    l_max, d_tbl,
#                                    device=device)
#         out_flat[idx] = coeff                                  # scatter back

#     # ---------- 4.  restore original batch shape -----------------------
#     return out_flat.view(*orig_shape, l_max+1, 2*l_max+1)

# ---------------------------------------------------------------------
# helper: bucketed call to fb_standard_coeffs_real
# ---------------------------------------------------------------------
def fb_standard_coeffs_bucketed(
    kappa: torch.Tensor,
    beta:  torch.Tensor,
    l_max: int,
    d_tbl: torch.Tensor,
    *,
    device:      Optional[torch.device] = None,
    dtype:       torch.dtype           = torch.float64,
    step_r:      int                   = 16,   # bucket granularity in Rcut
    step_t:      int                   = 16,   # bucket granularity in Tcut
    max_r:       int                   = 256,  # hard safety cap
    max_t:       int                   = 256,
) -> torch.Tensor:
    """
    Computes the *real* standard–frame FB-5 coefficients in small buckets
    so that the internal Rmax×Tmax tensor never blows up.  The result is
    scattered back to the ORIGINAL batch order.

    Returns  tensor of shape  (*B, ℓ_max+1, 2ℓ_max+1)  and dtype float64.
    """

    # 0 ─── flatten batch so we only have ONE index dimension ------------
    orig_batch   = kappa.shape[:-1]          # e.g. (B₁, B₂, …)
    kappa_flat   = kappa.reshape(-1)
    beta_flat    = beta.reshape(-1)
    Btot         = kappa_flat.numel()

    if device is None:
        device = kappa.device

    # 1 ─── per-sample cut-offs, quantised into “fuzzy” buckets ----------
    Rcut = (1.5 * kappa_flat + 24).ceil().clamp_max(max_r).to(torch.int64)
    Tcut = ((36/25) * beta_flat + 12).ceil().clamp_max(max_t).to(torch.int64)

    Rkey = ((Rcut + step_r - 1) // step_r) * step_r   # round *up*
    Tkey = ((Tcut + step_t - 1) // step_t) * step_t

    bucket_code            = torch.stack((Rkey, Tkey), 1)    # (Btot, 2)
    uniq_pairs, inv_bucket = torch.unique(bucket_code, dim=0, return_inverse=True)

    # 2 ─── output tensor, still REAL -----------------------------------
    out_flat = torch.empty(
        (Btot, l_max + 1, l_max + 1),
        dtype=dtype,          # float64 by default
        device=device
    )

    # 3 ─── loop over each bucket (usually only a handful) ---------------
    for bid in range(uniq_pairs.shape[0]):
        sel = (inv_bucket == bid).nonzero(as_tuple=True)[0]   # 1-D indices

        coeff_real = fb_standard_coeffs_real(
            kappa_flat[sel],
            beta_flat [sel],
            l_max, d_tbl,
            device=device,                 # forwarded
            eps64=dtype                    # keep double precision
        )

        out_flat[sel] = coeff_real        # scatter back in original order

    # 4 ─── restore original batch shape --------------------------------
    out = out_flat.view(*orig_batch, l_max + 1, l_max + 1)   # float64
    return out


# ---------------------------------------------------------------------
# Euler & rotation  (classic per‑ℓ loop, safe & fast enough)
# ---------------------------------------------------------------------

# def euler_from_basis(mu: Tensor, eta1: Tensor, eta2: Tensor):
#     θ = torch.acos(mu[...,2].clamp(-1+1e-7,1-1e-7))
#     φ = torch.atan2(mu[...,1], mu[...,0])
#     xpx,xpy,xpz = torch.cos(φ)*torch.cos(θ), torch.sin(φ)*torch.cos(θ), -torch.sin(θ)
#     ypx,ypy = -torch.sin(φ), torch.cos(φ)
#     ω = torch.atan2(eta1[...,1]*ypy + eta1[...,0]*ypx,
#                     eta1[...,1]*xpy + eta1[...,0]*xpx + eta1[...,2]*xpz)
#     return φ, θ, ω

# def euler_from_basis(mu: torch.Tensor, eta1: torch.Tensor, eta2: torch.Tensor):
#     # Compute theta (polar angle)
#     θ = torch.acos(mu[..., 2].clamp(-1 + 1e-7, 1 - 1e-7))
    
#     # Compute phi (azimuthal angle)
#     φ = torch.atan2(mu[..., 1], mu[..., 0])
    
#     # Compute omega (roll angle)
#     # The formula can be simplified for the identity case since there is no rotation around the z-axis.
#     ω = torch.atan2(eta1[..., 2], eta1[..., 0] * torch.cos(θ) + eta1[..., 1] * torch.sin(θ))

#     return φ, θ, ω


# def _angle_from_tan(
#     axis: str, other_axis: str, data, horizontal: bool, tait_bryan: bool
# ) -> torch.Tensor:
#     i1, i2 = {"X": (2, 1), "Y": (0, 2), "Z": (1, 0)}[axis]
#     if horizontal:
#         i2, i1 = i1, i2
#     even = (axis + other_axis) in ["XY", "YZ", "ZX"]
#     if horizontal == even:
#         return torch.atan2(data[..., i1], data[..., i2])
#     if tait_bryan:
#         return torch.atan2(-data[..., i2], data[..., i1])
#     return torch.atan2(data[..., i2], -data[..., i1])

# def _index_from_letter(letter: str) -> int:
#     if letter == "X":
#         return 0
#     if letter == "Y":
#         return 1
#     if letter == "Z":
#         return 2
#     raise ValueError("letter must be either X, Y or Z.")


# def matrix_to_euler_angles(R: torch.Tensor,
#                            convention: str = "ZYZ",
#                            eps: float = 1e-8) -> torch.Tensor:
#     # … validation omitted for brevity …

#     i0, i2 = _index_from_letter(convention[0]), _index_from_letter(convention[2])
#     tait_bryan = i0 != i2

#     theta = ( torch.asin(R[..., i0, i2] * (-1.0 if i0 - i2 in [-1, 2] else 1.0))
#               if tait_bryan
#               else torch.acos(torch.clamp(R[..., i0, i0], -1.0, 1.0)) )

#     omega = _angle_from_tan(convention[0], convention[1], R[..., i2], False, tait_bryan)
#     phi   = _angle_from_tan(convention[2], convention[1], R[..., i0, :], True , tait_bryan)

#     # gimbal corrections (identical to the snippet you pasted) …
#     near_zero = theta.abs() < eps
#     near_pi   = (theta - math.pi).abs() < eps

#     if near_zero.any():
#         omega = torch.where(near_zero, torch.atan2(R[...,1,0], R[...,0,0]), omega)
#         phi   = torch.where(near_zero, torch.zeros_like(phi),                   phi)
#     if near_pi.any():
#         omega = torch.where(near_pi, torch.atan2(-R[...,1,0], R[...,0,0]), omega)
#         phi   = torch.where(near_pi,  torch.zeros_like(phi),                  phi)

#     return torch.stack((omega, theta, phi), -1)        # [ω, θ, ϕ]

# ---------------------------------------------------------------------
#  final wrapper
# ---------------------------------------------------------------------
# def basis_to_euler(mu: torch.Tensor,
#                    eta1: torch.Tensor,
#                    eta2: torch.Tensor,
#                    convention: str = "ZYZ"):
#     """Return (ϕ, ϑ, ω) for a single set or a batch of bases."""
#     # make sure inputs are float and unit length
#     mu   = torch.nn.functional.normalize(mu  , dim=-1)
#     eta1 = torch.nn.functional.normalize(eta1, dim=-1)
#     eta2 = torch.nn.functional.normalize(eta2, dim=-1)

#     R = torch.stack((mu, eta1, eta2), dim=-1)               # (...,3,3)
#     ωθϕ = matrix_to_euler_angles(R, convention)             # (...,3) = [ω, θ, ϕ]
#     return ωθϕ[..., 2], ωθϕ[..., 1], ωθϕ[..., 0]            # (ϕ, θ, ω)


import torch, math

def euler_from_basis(mu, eta1, eta2, eps=1e-6):
    """
    Z–Y–Z Euler angles with gradient-safe numerics.

    Arguments
    ---------
    mu, eta1, eta2 : [..., 3]  (z, x, y basis vectors)
    eps            : float     numerical safety margin for gradients

    Returns
    -------
    φ, θ, ω : tensors with shape [...]
    """
    # --- θ ---------------------------------------------------------------
    z3   = mu[..., 2].clamp(-1 + eps, 1 - eps)     # pull away from ±1
    theta = torch.atan2(torch.sqrt(1 - z3*z3), z3) # smoother than acos

    # --- φ ---------------------------------------------------------------
    # safe_atan2 keeps the denominator at least eps
    def safe_atan2(y, x):
        return torch.atan2(y, torch.where(x.abs() < eps, x.sign()*eps, x))

    phi = safe_atan2(mu[..., 1], mu[..., 0])       # atan2(R23, R13)

    # --- ω ---------------------------------------------------------------
    sin_theta = torch.sin(theta)                   # already > eps
    omega_raw = safe_atan2(eta2[..., 2], -eta1[..., 2])

    # singular handling
    safe  = sin_theta > eps
    omega = torch.where(safe, omega_raw, torch.zeros_like(omega_raw))
    phi   = torch.where(
        safe,
        phi,
        safe_atan2(eta1[..., 1], eta1[..., 0])
    )

    # --- wrap to (-π, π] --------------------------------------------------
    wrap = lambda a: (a + math.pi) % (2 * math.pi) - math.pi

    real_dtype = torch.float64
    return wrap(phi).to(real_dtype), wrap(theta).to(real_dtype), wrap(omega).to(real_dtype)



# def rotate_coeffs(c_std: torch.Tensor,
#                   φ: torch.Tensor, θ: torch.Tensor, ω: torch.Tensor,
#                   d_half: torch.Tensor) -> torch.Tensor:
#     """
#     Rotate coeffs with pre-built π/2 table  d_half.
#     Shapes:
#         c_std : (*B, L, M)   complex
#         φ,ω,θ : (*B,)        real
#         d_half: (L, M, M)    real
#     """
#     l_max = c_std.shape[-2] - 1
#     M     = c_std.shape[-1]
#     dev   = c_std.device
#     dtype_r = c_std.real.dtype

#     # build d(θ) once for the whole batch
#     dθ = d_from_halfpi(θ, d_half).to(c_std.dtype)               # (*B,L,M,M)

#     m = torch.arange(-l_max, l_max+1, device=dev, dtype=dtype_r)

#     # right phase e^{-i m' ω}
#     g = c_std * torch.exp(-1j * ω[..., None] * m)[..., None, :]

#     # middle multiply
#     tmp = torch.matmul(dθ, g.unsqueeze(-1)).squeeze(-1) #tmp = torch.matmul(g.unsqueeze(-2), dθ).squeeze(-2)

#     # left phase e^{-i m φ}
#     out = tmp * torch.exp(-1j * φ[..., None] * m)[..., None, :]
#     return out


def rotate_coeffs_real(c_real: torch.Tensor,
                       φ: torch.Tensor, θ: torch.Tensor, ω: torch.Tensor,
                       d_half: torch.Tensor) -> torch.Tensor:
    """
    Convert real coeffs → complex just-in-time, apply Z–Y–Z rotation.
    Returns complex64/128 (matches `c_real` promotion).
    """
    l_max = c_real.shape[-2] - 1
    dev   = c_real.device
    cdtype= torch.complex128 if c_real.dtype==torch.float64 else torch.complex64

    # promote once
    c = c_real.to(cdtype)

    # Wigner-d(θ)
    dθ = d_from_halfpi(θ.double(), d_half).to(cdtype)

    m64 = torch.arange(-l_max, l_max+1, device=dev, dtype=torch.float64)
    phase_ω = torch.exp(-1j * ω.double()[...,None] * m64).to(cdtype)
    phase_φ = torch.exp(-1j * φ.double()[...,None] * m64).to(cdtype)

    g   = c * phase_ω[..., None, :]
    tmp = torch.matmul(dθ, g.unsqueeze(-1)).squeeze(-1)
    return tmp * phase_φ[..., None, :]

def rotate_coeffs_real_packed(
    c_pos: torch.Tensor,               # (..., L, L+1)  real64
    φ: torch.Tensor, θ: torch.Tensor, ω: torch.Tensor,
    d_half: torch.Tensor
) -> torch.Tensor:
    """
    Take packed real coeffs (m ≥ 0 only), reconstruct the negative-m half,
    convert to complex, and apply the Z–Y–Z rotation.

    Returns the FULL complex tensor (..., L, 2L+1).
    """
    l_max = c_pos.shape[-2] - 1
    dev   = c_pos.device
    cdtype= torch.complex128 if c_pos.dtype == torch.float64 else torch.complex64

    # -- unpack:   F(-m) = (-1)^m F(+m)   (still real) ------------------
    m_pos = torch.arange(0, l_max + 1, device=dev, dtype=c_pos.dtype)
    if l_max > 0:
        neg_slice = ((-1.)**m_pos[1:])[None, None, :] * c_pos[..., 1:]
        c_full = torch.cat([neg_slice.flip(-1), c_pos], dim=-1)   # (...,L,2L+1)
    else:
        c_full = c_pos                                            # L=0 case

    # promote once to complex
    c = c_full.to(cdtype)

    # ---------- rotation (unchanged) ----------------------------------
    dθ = d_from_halfpi(θ.double(), d_half).to(cdtype)

    m64 = torch.arange(-l_max, l_max + 1, device=dev, dtype=torch.float64)
    phase_ω = torch.exp(-1j * ω.double()[..., None] * m64).to(cdtype)
    phase_φ = torch.exp(-1j * φ.double()[..., None] * m64).to(cdtype)

    g   = c * phase_ω[..., None, :]
    tmp = torch.matmul(dθ, g.unsqueeze(-1)).squeeze(-1)
    return tmp * phase_φ[..., None, :]


# ---------------------------------------------------------------------
# Public wrapper
# ---------------------------------------------------------------------

# def fb5_harmonics(refdirs: Tensor, Tnw: Tensor, Bnw: Tensor,
#                   concentration: Tensor, eccentricity: Tensor, *, l_max: int):
#     dev = refdirs.device
#     dtab = get_d_table(l_max, dev)
#     coeff_std = fb_standard_coeffs_bucketed(concentration, eccentricity, l_max, dtab)
#     _check_tensor("coeff_std", coeff_std)
#     φ,θ,ω = euler_from_basis(refdirs, Tnw, Bnw)

#     coeff = rotate_coeffs_real(coeff_std, φ, θ, ω, dtab)
#     return compact_coeff_view(coeff, l_max)


def fb5_harmonics(refdirs: Tensor, Tnw: Tensor, Bnw: Tensor,
                  concentration: Tensor, eccentricity: Tensor, *,
                  l_max: int):
    dev  = refdirs.device
    dtab = get_d_table(l_max, dev)

    coeff_std = fb_standard_coeffs_bucketed(   # REAL float64
        concentration, eccentricity, l_max, dtab)
    print("coeff_std", coeff_std.max())
    _check_tensor("coeff_std", coeff_std)

    φ, θ, ω = euler_from_basis(refdirs, Tnw, Bnw)
    coeff   = rotate_coeffs_real_packed(coeff_std, φ, θ, ω, dtab)   # complex
    print("coeff", coeff.max())

    return compact_coeff_view(coeff, l_max)


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

def compact_coeff_view(coeff_full: Tensor, l_max: int):
    mid = l_max
    return torch.cat([coeff_full[...,ℓ,mid-ℓ:mid+ℓ+1] for ℓ in range(l_max+1)], dim=-1)

def as_real_view(harm: Tensor):
    return torch.cat((harm.real, harm.imag), dim=-1)

def fb5_encoder_size(l_max: int, *, real_scalars: bool=True):
    n = (l_max+1)**2
    return 2*n if real_scalars else n
