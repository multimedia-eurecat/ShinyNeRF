# ------------------------------------------------------------------------------------
# ShinyNeRF
# Copyright (c) 2026 Albert Barreiro.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------
# Modified from nerf-factory (https://github.com/kakaobrain/NeRF-Factory)
# ------------------------------------------------------------------------------------

import numpy as np
import torch


def img2mse(x, y, weight=None, eps=1e-6):
    """
    x, y     : (B, H, W, 3) tensors (RGB images)
    weight   : (B, H, W, 1) tensor or None
    returns  : scalar loss
    """
    mse_per_pixel = (x - y) ** 2

    if weight is not None:
        mse_per_pixel = weight * mse_per_pixel

    return mse_per_pixel.mean()


def mse2psnr(x):
    return -10.0 * torch.log(x) / np.log(10)


def linear_to_srgb(linear, eps=1e-10):
    eps = torch.finfo(torch.float32).eps
    srgb0 = 323 / 25 * linear
    srgb1 = (
        211 * torch.fmax(torch.full_like(linear, eps), linear) ** (5 / 12) - 11
    ) / 200
    return torch.where(linear <= 0.0031308, srgb0, srgb1)


def sample_along_rays(
    rays_o,
    rays_d,
    radii,
    num_samples,
    near,
    far,
    randomized,
    lindisp,
    ray_shape,
):
    bsz = rays_o.shape[0]
    t_vals = torch.linspace(0.0, 1.0, num_samples + 1, device=rays_o.device)

    if not isinstance(near, float):
        t_vals = torch.broadcast_to(t_vals, (bsz, num_samples + 1))
        near = near.unsqueeze(-1)
        far = far.unsqueeze(-1)

    if lindisp:
        t_vals = 1.0 / (1.0 / near * (1.0 - t_vals) + 1.0 / far * t_vals)
    else:
        t_vals = near * (1.0 - t_vals) + far * t_vals

    if randomized:
        mids = 0.5 * (t_vals[..., 1:] + t_vals[..., :-1])
        upper = torch.cat([mids, t_vals[..., -1:]], -1)
        lower = torch.cat([t_vals[..., :1], mids], -1)
        t_rand = torch.rand((bsz, num_samples + 1), device=rays_o.device)
        t_vals = lower + (upper - lower) * t_rand
    else:
        t_vals = torch.broadcast_to(t_vals, (bsz, num_samples + 1))

    means, covs = cast_rays(t_vals, rays_o, rays_d, radii, ray_shape)
    return t_vals, (means, covs)


def resample_along_rays(
    rays_o,
    rays_d,
    radii,
    t_vals,
    weights,
    randomized,
    ray_shape,
    stop_level_grad,
    resample_padding,
    num_samples=None,
):

    weights_pad = torch.cat([weights[..., :1], weights, weights[..., -1:]], dim=-1)
    weights_max = torch.fmax(weights_pad[..., :-1], weights_pad[..., 1:])
    weights_blur = 0.5 * (weights_max[..., :-1] + weights_max[..., 1:])

    weights = weights_blur + resample_padding

    # weights has shape [B,S], bins has shape [B,S+1]
    if num_samples is None:
        num_samples = t_vals.shape[-1] - 1  # number of intervals

    new_t_vals = sorted_piecewise_constant_pdf(t_vals, weights, num_samples, randomized)

    # new_t_vals must be [B, num_samples+1]
    assert new_t_vals.shape[-1] == int(num_samples) + 1, (new_t_vals.shape, num_samples)

    if stop_level_grad:
        new_t_vals = new_t_vals.detach()

    means, covs = cast_rays(new_t_vals, rays_o, rays_d, radii, ray_shape)
    return new_t_vals, (means, covs)


def sorted_piecewise_constant_pdf(bins, weights, num_samples, randomized, float_min_eps=2**-32):
    """
    bins:    [B, N+1] (ray depth boundaries)
    weights: [B, N]   (interval weights between bins)
    num_samples: K    (desired number of intervals after resampling)
    returns new_bins: [B, K+1] (new boundaries)
    """
    B = bins.shape[0]
    device = bins.device

    K = int(num_samples)
    if K < 1:
        raise ValueError("num_samples must be >= 1")

    # If K == 1 -> just keep endpoints (single interval)
    if K == 1:
        return torch.cat([bins[..., :1], bins[..., -1:]], dim=-1)

    # We need K+1 boundaries -> 2 endpoints + (K-1) interior
    K_inner = K - 1

    eps = 1e-5
    weight_sum = weights.sum(dim=-1, keepdim=True)
    padding = torch.clamp(eps - weight_sum, min=0.0)
    weights = weights + padding / weights.shape[-1]
    weight_sum = weight_sum + padding

    pdf = weights / weight_sum  # [B,N]

    # CDF with length N+1
    cdf = torch.cumsum(pdf[..., :-1], dim=-1)  # [B,N-1]
    cdf = torch.minimum(torch.ones_like(cdf), cdf)
    cdf = torch.cat(
        [
            torch.zeros((B, 1), device=device),
            cdf,
            torch.ones((B, 1), device=device),
        ],
        dim=-1,
    )  # [B, N+1]

    # Stratified samples in [0,1)
    if randomized:
        u = (torch.arange(K_inner, device=device, dtype=bins.dtype) +
             torch.rand(K_inner, device=device, dtype=bins.dtype)) / K_inner
    else:
        u = (torch.arange(K_inner, device=device, dtype=bins.dtype) + 0.5) / K_inner

    u = torch.minimum(u, torch.ones_like(u) * (1.0 - float_min_eps))  # [K_inner]

    # Invert CDF (vectorized)
    mask = u[None, None, :] >= cdf[:, :, None]  # [B, N+1, K_inner]

    bin0 = (mask * bins[:, :, None] + (~mask) * bins[:, :1, None]).max(dim=-2)[0]
    bin1 = ((~mask) * bins[:, :, None] + mask * bins[:, -1:, None]).min(dim=-2)[0]

    cdf0 = (mask * cdf[:, :, None] + (~mask) * cdf[:, :1, None]).max(dim=-2)[0]
    cdf1 = ((~mask) * cdf[:, :, None] + mask * cdf[:, -1:, None]).min(dim=-2)[0]

    t = torch.clip(torch.nan_to_num((u[None, :] - cdf0) / (cdf1 - cdf0), nan=0.0), 0.0, 1.0)
    samples = bin0 + t * (bin1 - bin0)  # [B, K_inner]

    # Build new boundaries: endpoints + sorted interior
    samples, _ = torch.sort(samples, dim=-1)
    new_bins = torch.cat([bins[..., :1], samples, bins[..., -1:]], dim=-1)  # [B, K+1]
    return new_bins


def _angular_similarity_2(n, n_pred):
    """|cos θ| between two normal vectors."""
    return(torch.sum(n * n_pred, dim=-1))         # [B,H,W,S]


 # ------------------------------------------------------------
# helpers: interval centres and interval weights
# ------------------------------------------------------------


def _centres(s):                 # s: [..., N]
    return 0.5 * (s[..., :-1] + s[..., 1:])         # [..., N-1]


# ------------------------------------------------------------
# helpers
# ------------------------------------------------------------


def ensure_tuple(x, n):
    return (x,) * n if not isinstance(x, (tuple, list)) else x


# ---------------------------------------------------------------------
#  tiny numeric helpers
# ---------------------------------------------------------------------
_F32_EPS  = torch.finfo(torch.float32).eps
_F32_EPS2 = _F32_EPS ** 2


def plus_eps(x, eps=_F32_EPS):
    """ x -> x + ε · max(1,|x|)  (matches math.plus_eps in Zip-NeRF) """
    return x + eps * torch.maximum(torch.ones_like(x), x.abs())


def minus_eps(x, eps=_F32_EPS):
    return x - eps * torch.maximum(torch.ones_like(x), x.abs())


# ---------------------------------------------------------------------
#  utilities that Zip-NeRF assumes
# ---------------------------------------------------------------------
# ───────────────────────────────────────────────────────────────
# strict check for a step-function histogram  (edges, values)
# ───────────────────────────────────────────────────────────────


def assert_valid_stepfun(t: torch.Tensor, y: torch.Tensor):
    """
    Valid step function means
       • t.shape[-1] == y.shape[-1] + 1
    """
    if t.shape[-1] != y.shape[-1] + 1:
        raise ValueError(f"Invalid shapes {t.shape} vs {y.shape} for a step-function.")


def assert_valid_linspline(t, y):
  """Assert that piecewise linear spline (t, y) has a valid shape."""
  if t.shape[-1] != y.shape[-1]:
    raise ValueError(
        f'Invalid shapes ({t.shape}, {y.shape}) for a linear spline.'
    )


def weight_to_pdf(t, w):
    """
    Convert sample-wise weights (len N) or interval weights (N-1)
    into a piece-wise constant PDF.
    """
    if w.shape[-1] == t.shape[-1]:              # N  (per sample)
        w_int = 0.5 * (w[..., :-1] + w[..., 1:])   # N-1
    else:
        w_int = w                                # already N-1
    assert_valid_stepfun(t, w_int)
    dt = torch.diff(t)
    return w_int / torch.clamp(dt, min=_F32_EPS)


# ---------------------------------------------------------------------
#  (2) blur step-function  – verbatim PyTorch port of loss_utils.blur_stepfun
# ---------------------------------------------------------------------


def blur_stepfun(ts, ys, halfwidth):
    """
    Convolve a step function (ts,ys) with a box filter of half-width `halfwidth`.
    Returns knots (tp) and values (yp) of a **piece-wise linear** spline.
    """
    assert_valid_stepfun(ts, ys)

    ts_lo = ts - halfwidth
    ts_hi = plus_eps(ts) + halfwidth    # ensure strictly greater

    # slopes (positive ramp then negative ramp)
    ys0 = torch.cat([torch.zeros_like(ys[..., :1]), ys,
                     torch.zeros_like(ys[..., :1])], dim=-1)
    dy  = torch.diff(ys0) / (ts_hi - ts_lo)

    tp  = torch.cat([ts_lo, ts_hi], dim=-1)     # 2×N knots
    dyp = torch.cat([dy, -dy],  dim=-1)         # matching slopes

    # sort both knot vector and slope vector
    idx = torch.argsort(tp, dim=-1)
    tp  = torch.take_along_dim(tp,  idx,  dim=-1)
    dyp = torch.take_along_dim(dyp, idx[..., :-2], dim=-1)  # lose 2 tails

    # integrate twice (ramp → quadratic) to get spline values at knots
    yp = torch.cumsum(
            torch.diff(tp)[..., :-1] * torch.cumsum(dyp, dim=-1),
            dim=-1
         )
    yp = torch.cat([torch.zeros_like(yp[..., :1]),
                    yp,
                    torch.zeros_like(yp[..., -1:])], dim=-1)

    return tp, yp


def resample_stepfun(src_edges, trg_edges, src_vals):
    """
    Piece-wise constant resampling of `src_vals` (defined on src bins)
    into the bins defined by `trg_edges`, using **overlap-weighted average**.

    src_edges : [..., N]
    trg_edges : [..., M]
    src_vals  : [..., N-1]
    returns   : [..., M-1]
    """
    N = src_edges.shape[-1] - 1
    M = trg_edges.shape[-1] - 1

    a_i, b_i = src_edges[..., :-1], src_edges[..., 1:]
    a_j, b_j = trg_edges[..., :-1], trg_edges[..., 1:]

    # broadcast to [..., N-1, M-1]
    a_i, b_i, v_i = [x.unsqueeze(-1) for x in (a_i, b_i, src_vals)]
    a_j, b_j      = [x.unsqueeze(-2) for x in (a_j, b_j)]

    overlap = torch.clamp(torch.min(b_i, b_j) - torch.max(a_i, a_j), min=0.0)
    weight  = overlap / torch.clip(b_j - a_j, min=_F32_EPS)        # normalise

    return (weight * v_i).sum(dim=-2)                              # [..., M-1]


# ------------------------------------------------------------------
# automatic blur: weighted geometric mean of overlapping proposal widths
# ------------------------------------------------------------------


def auto_blur_halfwidth(c, cp):
    """
    Implements Zip-NeRF paper’s rule:
    r_i  = exp(  ⟨ log Δ( cp ) ⟩  over proposal intervals overlapping NeRF bin i )
    """
    widths_log = torch.log(torch.clip(torch.diff(cp), min=_F32_EPS))
    # weighted average in log-space
    log_r = resample_stepfun(cp, c, widths_log)   # [..., S-1]
    return torch.exp(log_r)                       # half-width per NeRF interval


# ---------------------------------------------------------------------
#  (3) integrate linear spline → quadratic coeffs
# ---------------------------------------------------------------------


def compute_integral(t, y):
    assert_valid_linspline(t, y)
    dt = torch.diff(t)
    a  = torch.diff(y) / torch.clamp(2.0 * dt, min=_F32_EPS2)
    b  = y[..., :-1]
    c1 = 0.5 * torch.cumsum(dt[..., :-1] *
                            (y[..., :-2] + y[..., 1:-1]), dim=-1)
    c  = torch.cat([torch.zeros_like(y[..., :1]), c1], dim=-1)
    return a, b, c   # each [..., N-1]


# ---------------------------------------------------------------------
#  (3b) evaluate quadratic CDF at arbitrary points
# ---------------------------------------------------------------------


def interpolate_integral(tq, t, a, b, c):
    assert_valid_stepfun(t, a)
    assert_valid_stepfun(t, b)
    assert_valid_stepfun(t, c)

    tq = torch.clamp(tq, t[..., :1], minus_eps(t[..., -1:]))
    idx = torch.searchsorted(t, tq, right=False) - 1
    idx = idx.clamp(0, t.shape[-1] - 2)

    gather = lambda z: torch.gather(z, -1, idx)
    t0, a0, b0, c0 = map(gather, (t, a, b, c))

    td = tq - t0
    return a0 * td**2 + b0 * td + c0   # [..., K]


# ---------------------------------------------------------------------
#  (1-4) blur + resample NeRF weights into proposal bins  (Zip-NeRF §4.2)
# ---------------------------------------------------------------------


def blur_and_resample_weights(
        tq,              # [..., K]   proposal edges ( ˆs )
        t, w,            # [..., N] & [..., N-1]  NeRF edges & weights
        blur_halfwidth,  # scalar or tensor broadcastable to [..., N-1]
        eps=_F32_EPS):
    """
    Complete anti-aliased resampling (Fig. 5): histogram → PDF → blur →
    integrate → query → diff.
    """
    # 1) weights → PDF
    p = weight_to_pdf(t, w)

    # 2) box-blur
    t_lin, p_lin = blur_stepfun(t, p, blur_halfwidth)

    # 3) integrate spline → quadratic CDF  & query at proposal edges
    a, b, c = compute_integral(t_lin, p_lin)
    acc_wq  = interpolate_integral(tq, t_lin, a, b, c)

    # 4) finite diff gives weights on proposal intervals
    wq = torch.diff(acc_wq, dim=-1)
    wq = torch.clamp(wq, min=0.0)
    return wq

# ---------------------------------------------------------------------
# One-sided (directional) Wasserstein 1-distance: u -> v
# ---------------------------------------------------------------------


def to_interval_weights(t_vals, w):
    """
    Ensure `w` has length N-1 (interval weights).
    Accepts torch.Tensor or any sequence convertible to tensor.
    """
    # convert to tensor if it isn't one already
    if not torch.is_tensor(w):
        w = torch.as_tensor(w, dtype=t_vals.dtype, device=t_vals.device)

    if w.shape[-1] == t_vals.shape[-1] - 1:      # already interval
        return w
    if w.shape[-1] == t_vals.shape[-1]:          # per-sample
        return 0.5 * (w[..., :-1] + w[..., 1:])

    raise ValueError("Weights length must be N or N-1")


def integrated_pos_enc(means, covs, min_deg, max_deg):
    scales = torch.tensor([2**i for i in range(min_deg, max_deg)]).type_as(means)
    shape = list(means.shape[:-1]) + [-1]
    scaled_means = torch.reshape(means[..., None, :] * scales[:, None], shape)
    scaled_covs = torch.reshape(covs[..., None, :] * scales[:, None] ** 2, shape)

    return expected_sin(
        torch.cat([scaled_means, scaled_means + 0.5 * np.pi], dim=-1),
        torch.cat([scaled_covs] * 2, dim=-1),
    )[0]


def volumetric_rendering(rgb, density, t_vals, dirs, white_bkgd):
    t_mids = 0.5 * (t_vals[..., :-1] + t_vals[..., 1:])
    t_dists = t_vals[..., 1:] - t_vals[..., :-1]
    delta = t_dists * torch.norm(dirs[..., None, :], dim=-1)
    # Note that we're quietly turning density from [..., 0] to [...].
    density_delta = density[..., 0] * delta

    alpha = 1 - torch.exp(-density_delta)
    trans = torch.exp(
        -torch.cat(
            [
                torch.zeros_like(density_delta[..., :1]),
                torch.cumsum(density_delta[..., :-1], dim=-1),
            ],
            dim=-1,
        )
    )
    weights = alpha * trans

    comp_rgb = (weights[..., None] * rgb).sum(dim=-2)
    acc = weights.sum(dim=-1)

    distance = (weights * t_mids).sum(dim=-1)# / acc

    distance = torch.clip(distance, t_vals[:, 0], t_vals[:, -1])
    if white_bkgd:
        comp_rgb = comp_rgb + (1.0 - acc[..., None])
    distance_norm = (distance - t_mids.min()) / (t_mids.max() - t_mids.min())

    return comp_rgb, distance, distance_norm, acc, weights


def expected_sin(x, x_var):
    y = torch.exp(-0.5 * x_var) * torch.sin(x)
    y_var = 0.5 * (1 - torch.exp(-2 * x_var) * torch.cos(2 * x)) - y**2
    y_var = torch.fmax(torch.zeros_like(y_var), y_var)
    return y, y_var


def lift_gaussian(d, t_mean, t_var, r_var):

    mean = d[..., None, :] * t_mean[..., None]

    d_mag_sq = torch.sum(d**2, dim=-1, keepdim=True)
    thresholds = torch.ones_like(d_mag_sq) * 1e-10
    d_mag_sq = torch.fmax(d_mag_sq, thresholds)

    d_outer_diag = d**2
    null_outer_diag = 1 - d_outer_diag / d_mag_sq
    t_cov_diag = t_var[..., None] * d_outer_diag[..., None, :]
    xy_cov_diag = r_var[..., None] * null_outer_diag[..., None, :]
    cov_diag = t_cov_diag + xy_cov_diag

    return mean, cov_diag


# According to the link below, numerically stable implementations are required.
# https://github.com/google/mipnerf/blob/84c969e0a623edd183b75693aed72a7e7c22902d/internal/mip.py#L88


def conical_frustum_to_gaussian(d, t0, t1, radius):

    mu = (t0 + t1) / 2
    hw = (t1 - t0) / 2
    t_mean = mu + (2 * mu * hw**2) / (3 * mu**2 + hw**2)
    t_var = (hw**2) / 3 - (4 / 15) * (
        (hw**4 * (12 * mu**2 - hw**2)) / (3 * mu**2 + hw**2) ** 2
    )
    r_var = radius**2 * (
        (mu**2) / 4
        + (5 / 12) * hw**2
        - 4 / 15 * (hw**4) / (3 * mu**2 + hw**2)
    )

    return lift_gaussian(d, t_mean, t_var, r_var)


def cylinder_to_gaussian(d, t0, t1, radius):

    t_mean = (t0 + t1) / 2
    r_var = radius**2 / 4
    t_var = (t1 - t0) ** 2 / 12

    return lift_gaussian(d, t_mean, t_var, r_var)


def cast_rays(t_vals, origins, directions, radii, ray_shape):
    t0 = t_vals[..., :-1]
    t1 = t_vals[..., 1:]
    if ray_shape == "cone":
        gaussian_fn = conical_frustum_to_gaussian
    elif ray_shape == "cylinder":
        gaussian_fn = cylinder_to_gaussian
    else:
        assert False
    means, covs = gaussian_fn(directions, t0, t1, radii)
    means = means + origins[..., None, :]
    return means, covs


