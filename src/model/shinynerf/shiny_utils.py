# ------------------------------------------------------------------------------------
# ShinyNeRF
# Copyright (c) 2026 Barreiro, Albert. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------
# NeRF-Factory
# Copyright (c) 2022 POSTECH, KAIST, Kakao Brain Corp. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------
# Modified from Ref-NeRF (https://github.com/google-research/multinerf)
# Copyright (c) 2022 Google LLC. All Rights Reserved.
# ------------------------------------------------------------------------------------

import numpy as np
import torch


def reflect(viewdirs, normals):
    """Reflect view directions about normals.

    The reflection of a vector v about a unit vector n is a vector u such that
    dot(v, n) = dot(u, n), and dot(u, u) = dot(v, v). The solution to these two
    equations is u = 2 dot(n, v) n - v.

    Args:
        viewdirs: [..., 3] array of view directions.
        normals: [..., 3] array of normal directions (assumed to be unit vectors).

    Returns:
        [..., 3] array of reflection directions.
    """
    return (
        2.0 * torch.sum(normals * viewdirs, dim=-1, keepdims=True) * normals - viewdirs
    )


def l2_normalize(x, eps=torch.finfo(torch.float32).eps):
    """Normalize x to unit length along last axis."""

    return x / torch.sqrt(
        torch.fmax(torch.sum(x**2, dim=-1, keepdims=True), torch.full_like(x, eps))
    )


def compute_consistent_tangent_bitangent(normals, angle):
    # normal add it without normalize
    # Drop the z-component
    n_xy = normals[..., :2]  # shape: [4096, 128, 2]

    # 90° rotation in 2D: (x, y) → (-y, x)
    t_xy = torch.stack([-n_xy[..., 1], n_xy[..., 0]], dim=-1)  # shape: [4096, 128, 2]

    # Pad with 0 in z-direction to lift back to 3D
    z_pad = torch.zeros_like(t_xy[..., :1])  # shape: [4096, 128, 1]
    tangents = torch.cat([t_xy, z_pad], dim=-1)  # shape: [4096, 128, 3]

    # Normalize (avoid division by zero)
    tangents_norm = l2_normalize(tangents)

    bitangent = torch.cross(normals, tangents_norm, dim=-1)
    bitangent_norm = l2_normalize(bitangent)

    cos_a = torch.cos(angle)
    sin_a = torch.sin(angle)

    # Rotate tangent and bitangent in the tangent plane
    t_rot = cos_a * tangents_norm + sin_a * bitangent_norm
    b_rot = -sin_a * tangents_norm + cos_a * bitangent_norm

    return t_rot, b_rot


def compute_weighted_mae(weights, normals, normals_gt, mask=None):
    """Weighted mean angular error (degrees). Normals must be unit vectors (…,3)."""
    eps      = torch.finfo(weights.dtype).eps
    one_eps  = 1.0 - eps

    if mask is None:
        mask = torch.ones_like(weights, dtype=torch.bool)
    else:
        mask = mask.squeeze(-1)            # (3000, 31)
        mask = mask.bool()                  # ensure correct dtype
        if mask.shape != weights.shape:
            mask = mask.expand_as(weights)  # broadcast if needed

    # angle in radians, same shape as weights
    ang = torch.arccos(
        (normals * normals_gt).sum(-1).clamp(-one_eps, one_eps)
    )

    # use mask as multipliers instead of boolean indexing
    m = mask.float()
    num   = torch.sum(weights * ang * m)
    denom = torch.sum(weights * m) + eps

    return num / denom * 180.0 / np.pi


def compute_weighted_mae_picture_2(normals, normals_gt):
    """Compute weighted mean angular error, assuming normals are unit length."""
    one_eps = 1 - torch.finfo(torch.float32).eps
    return (
        (
            torch.arccos(
                torch.clamp(torch.sum(normals * normals_gt, -1), -one_eps, one_eps)
            )
        )
        * 180.0
        / np.pi
    )


def generalized_binomial_coeff(a, k):
    """Compute generalized binomial coefficients."""
    # return np.prod(a - np.arange(k)) / np.math.factorial(k)
    return np.prod(a - np.arange(k)) / np.math.factorial(k)


def assoc_legendre_coeff(l, m, k):
    """Compute associated Legendre polynomial coefficients.

    Returns the coefficient of the cos^k(theta)*sin^m(theta) term in the
    (l, m)th associated Legendre polynomial, P_l^m(cos(theta)).

    Args:
        l: associated Legendre polynomial degree.
        m: associated Legendre polynomial order.
        k: power of cos(theta).

    Returns:
        A float, the coefficient of the term corresponding to the inputs.
    """
    return (
        (-1) ** m
        * 2**l
        * np.math.factorial(l)
        / np.math.factorial(k)
        / np.math.factorial(l - k - m)
        * generalized_binomial_coeff(0.5 * (l + k + m - 1.0), l)
    )


def sph_harm_coeff(l, m, k):
    """Compute spherical harmonic coefficients."""
    # return (np.sqrt(
    #     (2.0 * l + 1.0) * np.math.factorial(l - m) /
    #     (4.0 * np.pi * np.math.factorial(l + m))) * assoc_legendre_coeff(l, m, k))
    return np.sqrt(
        (2.0 * l + 1.0)
        * np.math.factorial(l - m)
        / (4.0 * np.pi * np.math.factorial(l + m))
    ) * assoc_legendre_coeff(l, m, k)


def get_ml_array(deg_view):
    """Create a list with all pairs of (l, m) values to use in the encoding."""
    ml_list = []
    for i in range(deg_view):
        l = 2**i
        # Only use nonnegative m values, later splitting real and imaginary parts.
        for m in range(l + 1):
            ml_list.append((m, l))

    ml_array = np.array(ml_list).T
    return ml_array

def generate_ide_fn(deg_view):
    """Generate integrated directional encoding (IDE) function (stable for 2nd order).

    Matches Equations 6–8 of arxiv.org/abs/2112.03907 but avoids float-exponent pow
    (and its unstable gradients) by using integer Vandermonde bases via cumprod.

    Args:
        deg_view: number of spherical harmonics degrees to use. Must be <= 5.

    Returns:
        integrated_dir_enc_fn(xyz, kappa_inv) -> [..., 2*M] real tensor
    """
    if deg_view > 5:
        raise ValueError("Only deg_view of at most 5 is numerically stable.")

    ml_array = get_ml_array(deg_view)  # shape [2, M], rows are [m, l]
    l_max = 2 ** (deg_view - 1)

    # Build coefficient matrix mat[k, i] for z^k terms (see original code).
    mat_np = np.zeros((l_max + 1, ml_array.shape[1]))
    for i, (m, l) in enumerate(ml_array.T):
        for k in range(l - m + 1):
            mat_np[k, i] = sph_harm_coeff(l, m, k)

    # Store on CPU; cast/move inside the closure to match input dtype/device.
    mat_base = torch.tensor(mat_np, dtype=torch.float32)
    ml_base = torch.tensor(ml_array, dtype=torch.float32)  # rows: m, l

    def _vandermonde_lastdim(base, N):
        """Return [..., N] with powers base**[0..N-1] using cumprod (integer powers)."""
        # base: [..., 1]
        out = torch.ones(*base.shape[:-1], N, dtype=base.dtype, device=base.device)
        if N > 1:
            rep = base.expand(*base.shape[:-1], N - 1)
            out[..., 1:] = torch.cumprod(rep, dim=-1)
        return out

    def integrated_dir_enc_fn(xyz, kappa_inv):
        """
        Args:
            xyz: [..., 3] unit (or any) direction vectors (float32/float64).
            kappa_inv: [..., 1] reciprocal concentration of vMF (>=0 recommended).
        Returns:
            IDE features: real tensor [..., 2*M] (real || imag).
        """
        device = xyz.device
        ftype = xyz.dtype
        ctype = torch.complex64 if ftype == torch.float32 else torch.complex128

        # Move/convert constants to the right device/dtype
        mat = mat_base.to(device=device, dtype=ftype)          # [N, M]
        ml  = ml_base.to(device=device)                        # [2, M] (float32)
        m_idx = ml[0, :].to(torch.long)                        # [M] integer m (assumed >= 0)
        l_vals = ml[1, :].to(dtype=ftype)                      # [M] float (for sigma)

        # Split coords
        x = xyz[..., 0:1]
        y = xyz[..., 1:2]
        z = xyz[..., 2:3]

        # Vandermonde in z: [..., N] for integer powers 0..N-1
        N = mat.shape[0]
        vmz = _vandermonde_lastdim(z, N)                       # real [..., N]

        # Vandermonde in (x + i y): build up to max(m) and gather needed columns
        complex_xy = (x.to(ctype) + 1j * y.to(ctype))          # [..., 1] complex
        max_m = int(m_idx.max().item()) if m_idx.numel() > 0 else 0
        vmxy_all = _vandermonde_lastdim(complex_xy, max_m + 1) # [..., max_m+1] complex
        # NOTE: if your ml_array ever contains negative m, adapt here using SH symmetry.
        vmxy = vmxy_all.index_select(-1, m_idx)                # [..., M] complex

        # Spherical harmonics-like part: [..., M] complex
        sph_harms = vmxy * torch.matmul(vmz, mat).to(ctype)

        # vMF blur factor (sigma = 0.5*l*(l+1))
        sigma = 0.5 * l_vals * (l_vals + 1.0)                  # [M] real
        # Broadcast kappa_inv [...,1] with sigma [M] → [..., M]
        blur = torch.exp(-sigma * kappa_inv)                   # [..., M] real

        ide = sph_harms * blur.to(ctype)                       # [..., M] complex
        return torch.cat([ide.real.to(ftype), ide.imag.to(ftype)], dim=-1)

    return integrated_dir_enc_fn


def find_rotation_matrix(vec, target_vec, eps=1e-4, rot_axis_param=None):
    """Find the rotation matrix to align vector n with the z-axis."""

    #vec = vec.squeeze()
    batch_size, n_size, _ = vec.shape
    vec = vec.view(-1, 3)

    #vec = l2_normalize(vec) #/ add_eps_nan(torch.norm(vec), eps)
    if len(target_vec.shape) == 1:
        target_vec = target_vec.repeat(vec.size(0), 1)
    else:
        target_vec = target_vec.view(-1, 3)

    # Step 1: Find the axis of rotation
    if rot_axis_param == None:
        rot_axis = torch.linalg.cross(vec, target_vec)
    else:
        rot_axis = rot_axis_param.repeat(vec.size(0), 1)

    #rot_axis = rot_axis / torch.linalg.norm(rot_axis, dim=-1)
    rot_axis = l2_normalize(rot_axis)

    # Step 2: Find the angle of rotation
    dot_product = torch.sum(vec * target_vec , dim=-1)
    moduls = (torch.linalg.norm(vec, dim=-1)) * (torch.linalg.norm(target_vec, dim=-1)) + eps
    assert (moduls > 0).all(), "Moduls should be greater than 0"
    cos_theta = dot_product / moduls
    assert (cos_theta >= -1).all() and  (cos_theta <= 1).all(), "Cos theta not between -1 and 1"
    theta = torch.arccos(cos_theta)

    # Step 3: Construct the rotation matrix using axis-angle representation    
    c = torch.cos(theta)
    s = torch.sin(theta)
    t = 1 - c

    # TODO: optimize calculation, avoid repetitions
    rotation_matrix = torch.stack([
        torch.stack([t * rot_axis[:,0]**2 + c,                              t * rot_axis[:,0] * rot_axis[:,1] - s * rot_axis[:,2], t * rot_axis[:,0] * rot_axis[:,2] + s * rot_axis[:,1]]),
        torch.stack([t * rot_axis[:,0] * rot_axis[:,1] + s * rot_axis[:,2], t * rot_axis[:,1]**2 + c,                              t * rot_axis[:,1] * rot_axis[:,2] - s * rot_axis[:,0]]),
        torch.stack([t * rot_axis[:,0] * rot_axis[:,2] - s * rot_axis[:,1], t * rot_axis[:,1] * rot_axis[:,2] + s * rot_axis[:,0], t * rot_axis[:,2]**2 + c])
    ]).permute(2,0,1)

    vec_rotated = torch.matmul(rotation_matrix, (vec).unsqueeze(-1)).squeeze()

    rotation_matrix = rotation_matrix.view(batch_size, n_size, 3, 3)

    if rot_axis_param == None: # TODO DEBUGIN
        #print("torch.sum(l2_normalize(vec_rotated) * target_vec, dim=-1", torch.sum(l2_normalize(vec_rotated) * target_vec, dim=-1))
        assert (torch.abs(torch.sum(l2_normalize(vec_rotated) * target_vec, dim=-1)) > 0.95).all() ,  "Rotation not correct"
    return rotation_matrix


def spherical_to_cartesian(theta, phi, r=1):
    # theta is the elevation with respect to the z-axis
    # phi is the azimuth with respect to the x-axis
    # both theta and phi are expected in radians
    if r == 1:
        x = torch.sin(theta) * torch.cos(phi)
        y = torch.sin(theta) * torch.sin(phi)
        z = torch.cos(theta)
    else:
        x = r * torch.sin(theta) * torch.cos(phi)
        y = r * torch.sin(theta) * torch.sin(phi)
        z = r * torch.cos(theta)
    vec = torch.stack([x, y, z], dim=-1).squeeze(-2)  # shape: [batch, ..., 3]

    # unit_vec = l2_normalize(vec) # unnecessary
    return vec


def brdf_2_w_r_2(wr, normal, binormal, tangent, thetas, num_laterals):
    """
    wr:       [N, S, 3]
    normal:   [N, S, 3]
    binormal: [N, S, 3]
    tangent:  [N, S, 3]
    thetas:   [N, S, L] or [N, S, L, 1]
    return:   [N, S, L, 3, 2]  # (..., L, xyz, {L,R})
    """
    device = wr.device
    # Local BRDF basis (tangent, binormal, normal)
    R_n_basis = torch.stack((tangent, binormal, normal), dim=-1)  # [N,S,3,3]
    Rn_wr = find_rotation_matrix(normal, wr)                      # [N,S,3,3]
    R_brdf = torch.matmul(Rn_wr, R_n_basis)                       # [N,S,3,3]

    # Ensure thetas has shape [N,S,L]
    if thetas.dim() == 4 and thetas.size(-1) == 1:
        thetas_ = thetas.squeeze(-1)
    else:
        thetas_ = thetas  # [N,S,L]

    # Build phi tensors (broadcasted) for left/right
    pi2 = torch.tensor(torch.pi * 0.5, device=device, dtype=thetas_.dtype)
    phi_L = pi2
    phi_R = -pi2

    # Vectorized spherical->cartesian in BRDF frame
    v_L = spherical_to_cartesian(thetas_, phi_L)   # [N,S,L,3]
    v_R = spherical_to_cartesian(thetas_, phi_R)   # [N,S,L,3]

    # Rotate to world, broadcasting R over L
    v_L_world = torch.matmul(R_brdf.unsqueeze(-3), v_L.unsqueeze(-1)).squeeze(-1)  # [N,S,L,3]
    v_R_world = torch.matmul(R_brdf.unsqueeze(-3), v_R.unsqueeze(-1)).squeeze(-1)  # [N,S,L,3]

    # Normalize exactly like your original
    v_L_world = l2_normalize(v_L_world)
    v_R_world = l2_normalize(v_R_world)

    # Stack L/R as the last axis
    v_world = torch.stack((v_L_world, v_R_world), dim=-1)  # [N,S,L,3,2]
    return v_world


def compute_ide_from_vmf(refdirs, refdirs_laterals, roughness, weights, num_laterals, dir_enc_fn):
    """
    refdirs:          [B, S, 3]
    refdirs_laterals: [B, S, L, 3, 2]   (L=num_laterals, last dim=2 for left/right)
    roughness:        [B, S, L+1] or [B, S, L+1, 1]
    weights:          [B, S, L+1] or [B, S, L+1, 1]

    returns:          [B, S, C]  (rank-stable; safe for B=1)
    """

    # Normalize weights/roughness to [B,S,L+1]
    if roughness.dim() == 4:
        roughness = roughness.squeeze(-1)
    if weights.dim() == 4:
        weights = weights.squeeze(-1)

    B, S, _ = refdirs.shape
    L = num_laterals

    # Base (component 0)
    w0 = weights[..., 0:1]     # [B,S,1]
    r0 = roughness[..., 0:1]   # [B,S,1]
    comp0 = w0 * dir_enc_fn(refdirs, r0)   # [B,S,C]

    # Laterals (components 1..L)
    wl = weights[..., 1:]      # [B,S,L]
    rl = roughness[..., 1:]    # [B,S,L]

    # Split left/right from last dim
    left_dirs  = refdirs_laterals[..., 0]  # [B,S,L,3]
    right_dirs = refdirs_laterals[..., 1]  # [B,S,L,3]

    # Stack both sides along the lateral axis -> [B,S,2L,3]
    both_dirs = torch.cat([left_dirs, right_dirs], dim=-2)

    # Duplicate roughness/weights to match 2L laterals -> [B,S,2L,1]
    rl2 = torch.cat([rl, rl], dim=-1).unsqueeze(-1)
    wl2 = torch.cat([wl, wl], dim=-1).unsqueeze(-1)

    # Encode all laterals at once -> [B,S,2L,C]
    both_enc = dir_enc_fn(both_dirs, rl2)

    # Each lateral has left+right contributions averaged (0.5 each)
    lateral_sum = (0.5 * wl2 * both_enc).sum(dim=-2)  # [B,S,C]

    return comp0 + lateral_sum  # [B,S,C]