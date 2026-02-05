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

import os
from typing import Any, Callable
from pathlib import Path

import gin
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import math

import src.model.shinynerf.helper as helper
import src.model.shinynerf.shiny_utils as shiny_utils
import utils.store_image as store_image
from src.model.interface import LitModel
import tifffile as tiff

from src.model.shinynerf.asg2vmfs import ASG2vMFs

from skimage import io

GLOBAL_STEP_KNOW = 0

class ProposalMLP(nn.Module):
    """4-layer σ-only network, weights shared by “coarse” and “fine” calls."""
    def __init__(self, in_dim, hidden=256):
        super().__init__()
        self.fcs = nn.ModuleList(
            [nn.Linear(in_dim, hidden)] +
            [nn.Linear(hidden, hidden) for _ in range(2)] +
            [nn.Linear(hidden, hidden)])
        self.out = nn.Linear(hidden, 1)
        for m in self.fcs + [self.out]:
            nn.init.xavier_uniform_(m.weight)

    def forward(self, enc):                      # enc shape [B,S,C]
        h = enc
        for fc in self.fcs:
            h = F.relu(fc(h))
        return F.softplus(self.out(h))           # σ ≥ 0



@gin.configurable()
class ShinyNeRFMLP(nn.Module):
    def __init__(
        self,
        deg_view,
        min_deg_point: int = 0,
        max_deg_point: int = 16,
        netdepth: int = 8,
        netwidth: int = 256,
        bottleneck_width: int = 32,
        netdepth_viewdirs: int = 8,
        netwidth_viewdirs: int = 256,

        skip_layer: int = 4,
        skip_layer_dir: int = 4,
        perturb: float = 1.0,
        input_ch: int = 3,
        input_ch_view: int = 3,
        num_rgb_channels: int = 3,
        num_density_channels: int = 1,
        num_k_channels: int = 1,
        num_angle_channels: int = 1,

        bias_asg: float = 1.0,
        k_scale: float = 600,
        bottleneck_noise: float = 0.0,

        density_bias: float = -1.0,
        density_noise: float = 0.0,
        rgb_premultiplier: float = 1.0,

        rgb_bias: float = 0.0,
        rgb_padding: float = 0.001,
        num_normal_channels: int = 3,
        num_binormal_channels: int = 1,
        num_tint_channels: int = 3,
        num_anisotropy_angle_channels: int = 1,
        eccentricity_channels: int = 1,
        concentration_channels: int = 1,
        init_ecc: float = 4,
        beta_ecc: float = 3,

    ):
        for name, value in vars().items():
            if name not in ["self", "__class__"]:
                setattr(self, name, value)

        super(ShinyNeRFMLP, self).__init__()

        self.dir_enc_fn = shiny_utils.generate_ide_fn(self.deg_view)

        self.net_activation = nn.ReLU()
        self.concentration_activation = nn.Softplus()
        self.density_activation = nn.Softplus()
        self.rgb_activation = nn.Sigmoid()
        self.tint_activation = nn.Sigmoid()
        self.ecc_activation = nn.Softplus()
        self.anisotropy_angle_activation = nn.Sigmoid()

        pos_size = ((max_deg_point - min_deg_point) * 2) * input_ch
        view_pos_size = (2**deg_view - 1 + deg_view) * 2
        init_layer = nn.Linear(pos_size, self.netwidth)
        init.xavier_uniform_(init_layer.weight)
        pts_linear = [init_layer]

        for idx in range(self.netdepth - 1):
            if idx % self.skip_layer == 0 and idx > 0:
                module = nn.Linear(self.netwidth + pos_size, self.netwidth)
            else:
                module = nn.Linear(self.netwidth, self.netwidth)
            init.xavier_uniform_(module.weight)
            pts_linear.append(module)

        self.pts_linears = nn.ModuleList(pts_linear)

        views_linear = [
            nn.Linear(self.bottleneck_width + view_pos_size + 1, self.netwidth_viewdirs)
        ]
        for idx in range(self.netdepth_viewdirs - 1):
            if idx % self.skip_layer_dir == 0 and idx > 0:
                module = nn.Linear(
                    self.netwidth_viewdirs + self.bottleneck_width + view_pos_size + 1,
                    self.netwidth_viewdirs,
                )
            else:
                module = nn.Linear(self.netwidth_viewdirs, self.netwidth_viewdirs)
            init.xavier_uniform_(module.weight)
            views_linear.append(module)
        self.num_laterals = 14
        self.d_model = 20

        self.asg2vmf = ASG2vMFs(num_laterals=self.num_laterals, d_model=self.d_model)
        parent_path = Path(__file__).resolve().parent          # directory containing this .py file
        asg2vmf_path = parent_path / "ASG2vMFs_l_14.pth"

        self.asg2vmf.load_state_dict(torch.load(asg2vmf_path))

        for param in self.asg2vmf.parameters():
            param.requires_grad = False

        self.views_linear = nn.ModuleList(views_linear)

        self.bottleneck_layer = nn.Linear(self.netwidth, self.bottleneck_width)
        self.density_layer = nn.Linear(self.netwidth, num_density_channels)
        self.rgb_layer = nn.Linear(self.netwidth_viewdirs, num_rgb_channels)

        self.normal_layer = nn.Linear(self.netwidth, self.num_normal_channels)
        self.concentration_layer = nn.Linear(self.netwidth, self.concentration_channels)
        self.ecc_layer = nn.Linear(self.netwidth, self.eccentricity_channels)
        self.rgb_diffuse_layer = nn.Linear(self.netwidth, self.num_rgb_channels)
        self.tint_layer = nn.Linear(self.netwidth, self.num_tint_channels)
        self.anisotropy_angle_layer = nn.Linear(self.netwidth, self.num_anisotropy_angle_channels)

        init.xavier_uniform_(self.bottleneck_layer.weight)
        init.xavier_uniform_(self.density_layer.weight)
        init.xavier_uniform_(self.normal_layer.weight)
        init.xavier_uniform_(self.concentration_layer.weight)
        init.xavier_uniform_(self.ecc_layer.weight)
        init.xavier_uniform_(self.rgb_diffuse_layer.weight)
        init.xavier_uniform_(self.tint_layer.weight)
        nn.init.constant_(self.ecc_layer.bias, torch.log(torch.tensor(self.init_ecc - 1.0)) / self.beta_ecc)
        init.xavier_uniform_(self.anisotropy_angle_layer.weight)

        # ---- additions to reduce "memorization" of lighting via bottleneck ----
        self.g_logits = nn.Parameter(torch.tensor([0.0, 1.0]))
        self.bottleneck_dropout = nn.Dropout(p=0.2)               # only active in train

    """
    x: torch.Tensor, [batch, num_samples, feature]
    viewdirs: torch.Tensor, [batch, viewdirs]
    """
    def forward(self, samples, viewdirs):

        means, covs = samples
        epsilon = 1e-5

        with torch.set_grad_enabled(True):
            means.requires_grad_(True)
            x = helper.integrated_pos_enc(
                means=means,
                covs=covs,
                min_deg=self.min_deg_point,
                max_deg=self.max_deg_point,
            )

            num_samples, feat_dim = x.shape[1:]
            x = x.reshape(-1, feat_dim)

            inputs = x
            for idx in range(self.netdepth):
                x = self.pts_linears[idx](x)
                x = self.net_activation(x)
                if idx % self.skip_layer == 0 and idx > 0:
                    x = torch.cat([x, inputs], dim=-1)
            if torch.isnan(x).any():
                import pdb; pdb.set_trace()

            raw_density = self.density_layer(x)

            raw_density_grad = torch.autograd.grad(
                outputs=raw_density.sum(), inputs=means, retain_graph=True
            )[0]

            raw_density_grad = raw_density_grad.reshape(
                -1, num_samples, self.num_normal_channels
            )

            normals = -shiny_utils.l2_normalize(raw_density_grad)
            means.detach()

        density = self.density_activation(raw_density + self.density_bias)
        density = density.reshape(-1, num_samples, self.num_density_channels)

        angle_mlp = self.anisotropy_angle_layer(x)
        anisotropy_angle = torch.pi * self.anisotropy_angle_activation(angle_mlp).reshape(
            -1, num_samples, self.num_anisotropy_angle_channels
        )

        grad_pred = self.normal_layer(x).reshape(
            -1, num_samples, self.num_normal_channels
        )

        normals_pred = -shiny_utils.l2_normalize(grad_pred)
        normals_to_use = normals_pred

        tangents_pred, binormals_pred = shiny_utils.compute_consistent_tangent_bitangent(
            normals_to_use, anisotropy_angle
        )


        pred_concentration = self.concentration_layer(x).reshape(
            -1, num_samples, self.concentration_channels
        )

        concentration_mlp = self.concentration_activation(pred_concentration) + self.bias_asg

        pred_ecc = self.ecc_layer(x).reshape(
            -1, num_samples, self.eccentricity_channels
        )

        # ---- sum–split reparameterization (guarantees λ ≥ μ) ----
        ecc_mlp = torch.sigmoid(pred_ecc)                    # s in (0,1)
        mu_mlp = 0.5 * concentration_mlp * (1.0 - ecc_mlp) + epsilon
        lambda_mlp = 0.5 * concentration_mlp * (1.0 + ecc_mlp) + epsilon

        binormals_to_use = binormals_pred
        tangents_to_use = tangents_pred

        raw_rgb_diffuse = self.rgb_diffuse_layer(x)

        tint = self.tint_layer(x)
        tint = self.tint_activation(tint)

        bottleneck = self.bottleneck_layer(x)
        bottleneck = bottleneck.reshape(-1, num_samples, self.bottleneck_width)
        bottleneck = self.bottleneck_dropout(bottleneck)     # train-time only via module flag

        refdirs = shiny_utils.reflect(-viewdirs[..., None, :], normals_to_use)
        refdirs = shiny_utils.l2_normalize(refdirs)

        asg2vmf_input = torch.cat([lambda_mlp, mu_mlp], dim=-1)

        kent_output = self.asg2vmf(asg2vmf_input)
        ks, thetas, weights = kent_output
        ks += self.bias_asg
        weights = (weights / (weights.sum(dim=-1, keepdim=True) + 1e-8)).clamp_min(1e-8)

        roughness = 1 / (ks + epsilon)

        refdirs_laterals = shiny_utils.brdf_2_w_r_2(
            refdirs, normals_to_use, binormals_to_use, tangents_to_use, thetas, self.num_laterals
        )

        dir_enc = shiny_utils.compute_ide_from_vmf(
            refdirs, refdirs_laterals, roughness, weights, self.num_laterals, self.dir_enc_fn
        )
        dotprod = torch.sum(
            normals_to_use * viewdirs[..., None, :], dim=-1, keepdim=True
        )

        g_b, g_d = torch.softmax(self.g_logits, dim=0)
        # ---- gated view input (discourages lighting memorization through bottleneck) ----
        x = torch.cat([g_b * bottleneck, g_d * dir_enc, dotprod], dim=-1)


        debug_bottleneck = bottleneck
        debug_dir_enc = dir_enc

        x = x.reshape(-1, x.shape[-1])
        inputs = x
        for idx in range(self.netdepth_viewdirs):
            x = self.views_linear[idx](x)
            x = self.net_activation(x)
            if idx % self.skip_layer_dir == 0 and idx > 0:
                x = torch.cat([x, inputs], dim=-1)

        raw_rgb = self.rgb_layer(x)
        rgb_spec = self.rgb_activation(self.rgb_premultiplier * raw_rgb + self.rgb_bias)

        diffuse_linear = self.rgb_activation(raw_rgb_diffuse - np.log(3.0))
        specular_linear = tint * rgb_spec
        rgb = torch.clamp(
            helper.linear_to_srgb(specular_linear + diffuse_linear), 0.0, 1.0
        )

        rgb = rgb * (1 + 2 * self.rgb_padding) - self.rgb_padding
        rgb = rgb.reshape(-1, num_samples, self.num_rgb_channels)


        if torch.isnan(rgb).any() or torch.isnan(normals).any() or torch.isnan(density).any() or torch.isnan(binormals_pred).any() or torch.isnan(roughness).any():
            import pdb; pdb.set_trace()
        if torch.isinf(rgb).any() or torch.isinf(normals).any() or torch.isinf(density).any() or torch.isinf(binormals_pred).any() or torch.isinf(roughness).any():
            import pdb; pdb.set_trace()

        return dict(
            means=means,
            viewdirs=viewdirs,
            rgb=rgb,
            density=density,
            normals=normals,
            normals_pred=normals_pred,
            bitangents_pred=binormals_pred,
            tangents_pred=tangents_pred,
            eccentricity=ecc_mlp,
            roughness=1 / (concentration_mlp + epsilon),
            concentration=concentration_mlp,
            roughness_vMFs=1 / (ks + epsilon),
            weights_vMFs=weights,
            thetas_vMFs=thetas,
            diffuse=diffuse_linear.reshape(-1, num_samples, self.num_rgb_channels),
            specular=specular_linear.reshape(-1, num_samples, self.num_rgb_channels),
            tint=tint.reshape(-1, num_samples, self.num_rgb_channels),
            specular_rgb=rgb_spec.reshape(-1, num_samples, self.num_rgb_channels),
            lambda_asg=lambda_mlp,
            mu_asg=mu_mlp,
            g_b=g_b,
            # ---- debug tensors for sensitivity loss ----
            debug_dir_enc=debug_dir_enc,
            debug_bottleneck=debug_bottleneck,
        )


@gin.configurable()
class ShinyNeRF(nn.Module):
    def __init__(
        self,
        num_levels: int = 1,
        resample_padding: float = 0.01,
        stop_level_grad: bool = True,
        use_viewdirs: bool = True,
        lindisp: bool = False,
        ray_shape: str = "cone",
        deg_view: int = 5,
        rgb_padding: float = 0.001,
        num_prop_levels: int = 2,
        prop_samples: int = 64,
        num_samples: int = 32,
    ):
        # Layers
        for name, value in vars().items():
            if name not in ["self", "__class__"]:
                setattr(self, name, value)

        super(ShinyNeRF, self).__init__()
        self.mlp = ShinyNeRFMLP(self.deg_view)

        # ───── proposal nets ─────
        in_dim_enc = (self.mlp.max_deg_point - self.mlp.min_deg_point) * 2 * 3
        self.proposal_net = ProposalMLP(in_dim_enc)      # shared weights

        # ───── main NeRF ─────

    def forward(self, rays, randomized, white_bkgd, near, far, radius):
        """
        Levels 0 … (num_prop_levels-1)  →   proposal network (σ only)
        Level  num_prop_levels         →   full TRefNeRF (RGB + extras)
        """
        rays_o, rays_d, radii = rays["rays_o"], rays["rays_d"], rays["radii"]
        viewdirs = rays["viewdirs"]
        B = rays_o.shape[0]

        # ---------- near / far auto-compute ----------
        if near is None:
            d = rays_o.norm(dim=-1)
            near = d - radius
            far = d + radius

        results   = []                                   # list of dicts (one per level)
        weights   = None                                 # placeholder for resampling
        total_lvls = self.num_prop_levels + 1            # proposals + final NeRF

        for i_level in range(total_lvls):

            # -------- choose how many samples this level ----------
            if i_level < self.num_prop_levels:           # proposal stages
                cur_nsamp = self.prop_samples
            else:                                        # last stage = main NeRF
                cur_nsamp = self.num_samples

            # ----------- (re)sample t-values -----------
            if i_level == 0:
                t_vals, samples = helper.sample_along_rays(
                    rays_o, rays_d, radii,
                    num_samples=cur_nsamp,
                    near=near, far=far, randomized=randomized,
                    lindisp=self.lindisp, ray_shape=self.ray_shape)
            else:
                t_vals, samples = helper.resample_along_rays(
                    rays_o, rays_d, radii,
                    t_vals=t_vals, weights=weights,
                    randomized=randomized, ray_shape=self.ray_shape,
                    stop_level_grad=(i_level < total_lvls-1),        # stop grad for proposals
                    resample_padding=self.resample_padding,
                    num_samples=cur_nsamp)

            # ============ PROPOSAL LEVELS =============
            if i_level < self.num_prop_levels:
                means, covs = samples
                enc = helper.integrated_pos_enc(
                    means, covs,
                    min_deg=self.mlp.min_deg_point,
                    max_deg=self.mlp.max_deg_point,
                )
                sigma = self.proposal_net(enc.reshape(-1, enc.shape[-1])).reshape_as(means[..., :1])
                _, _, _, _, weights = helper.volumetric_rendering(
                    torch.zeros_like(sigma), sigma, t_vals, rays_d, white_bkgd=False)

                results.append(dict(
                    level      = f"proposal_{i_level}",
                    t_vals     = t_vals.detach(),
                    sigma      = sigma,
                    weights    = weights,
                ))
                continue       # go to next level

            # ============ FINAL NeRF LEVEL =============
            ray_results = self.mlp(samples, rays["viewdirs"])          # heavy MLP
            comp_rgb,  distance, distance_norm, acc, weights = helper.volumetric_rendering(
                ray_results["rgb"],      ray_results["density"],
                t_vals, rays_d, white_bkgd=white_bkgd)

            comp_diffuse, _, _, _, _ = helper.volumetric_rendering(
                ray_results["diffuse"],  ray_results["density"],
                t_vals, rays_d, white_bkgd=white_bkgd)

            comp_specular, _, _, _, _ = helper.volumetric_rendering(
                ray_results["specular"], ray_results["density"],
                t_vals, rays_d, white_bkgd=white_bkgd)

            comp_tint, _, _, _, _ = helper.volumetric_rendering(
                ray_results["tint"],     ray_results["density"],
                t_vals, rays_d, white_bkgd=white_bkgd)

            comp_specular_rgb, _, _, _, _ = helper.volumetric_rendering(
                ray_results["specular_rgb"], ray_results["density"],
                t_vals, rays_d, white_bkgd=white_bkgd)

            # ---- pack everything  ----
            ray_results.update({
                "comp_rgb":       comp_rgb,
                "distance":       distance,
                "distance_norm":  distance_norm,
                "acc":            acc,
                "weights":        weights,
                "t_vals":         t_vals,

                "diffuse":        comp_diffuse,
                "specular":       comp_specular,
                "tint":           comp_tint,
                "specular_rgb":   comp_specular_rgb,

                "rays_o":         rays_o,
                "rays_d":         rays_d,
            })
            results.append(ray_results)   # final dict

        return results   # len == num_prop_levels + 1



@gin.configurable()
class LitShinyNeRF(LitModel):
    def __init__(
        self,
        lr_init: float = 5.0e-4,
        lr_final: float = 5.0e-6,
        lr_delay_steps: int = 2500,
        lr_delay_mult: float = 0.01,
        spline_interlevel_weight: float = 3e-4,
        randomized: bool = True,
        orientation_loss_mult_s: float = 0.1,
        orientation_loss_mult_f: float = 0.01,
        predicted_normal_pred_loss_mult_s: float = 8e-3,
        predicted_normal_pred_loss_mult_f: float = 0.001,
        predicted_normal_grad_loss_mult: float = 3e-5,
        dist_init_mult: float = 3e-3,
        dist_final_mult: float = 6e-2,
        max_steps_pred: int = 2000,
        compute_normal_metrics: bool = False,
        compute_bitangent_metrics: bool = False,
        grad_max_norm: float = 0.001,
    ):
        for name, value in vars().items():
            if name not in ["self", "__class__"]:
                setattr(self, name, value)

        super(LitShinyNeRF, self).__init__()
        self.model = ShinyNeRF()
        self.first_number_test = -1

    def setup(self, stage):
        if hasattr(self.trainer.datamodule, 'near'):
            self.near = self.trainer.datamodule.near
            self.far = self.trainer.datamodule.far
            self.radius = None
        else:
            self.radius = self.trainer.datamodule.radius
            self.near = None
            self.far = None

        self.white_bkgd = self.trainer.datamodule.white_bkgd

    def training_step(self, batch, batch_idx):
        global GLOBAL_STEP_KNOW
        GLOBAL_STEP_KNOW = self.global_step

        rendered_results = self.model(
            batch, self.randomized, self.white_bkgd, self.near, self.far, self.radius
        )

        rgb_fine = rendered_results[-1]["comp_rgb"]
        target = batch["target"]

        w_fine = rendered_results[-1]["weights"]
        k_fine = (w_fine[..., None] * rendered_results[-1]["concentration"]).sum(dim=-2)

        loss1 = helper.img2mse(rgb_fine, target, None)

        loss = loss1
        if torch.isnan(rgb_fine).any():
            import pdb; pdb.set_trace()

        w_f   = rendered_results[-1]["weights"]
        ecc_f = (w_f[..., None] * rendered_results[-1]["eccentricity"]).sum(-2)
        mask = ecc_f > 1e-2
        if mask.any():
            ecc_f_val = ecc_f[mask].mean()
        else:
            ecc_f_val = torch.tensor(0.0, device=ecc_f.device)

        self.log("train/ecc_f", ecc_f_val, on_step=True)

        dist_loss = self.distortion_loss(rendered_results[-1]["t_vals"], weights = rendered_results[-1]["weights"], weight = self._lambda_dist())
        self.log("train/dist_loss", dist_loss, on_step=True)
        loss += dist_loss

        BLURS = (0.03, 0.003)
        MULTS = (1.0, 1.0)
        LAMBDA_PROP = self.spline_interlevel_weight
        prop_loss = LAMBDA_PROP * self.spline_interlevel_loss_torch(rendered_results, mults=MULTS, blurs=BLURS)
        self.log("train/prop_loss", prop_loss, on_step=True)
        loss += prop_loss

        lambda_comp = 1
        loss += lambda_comp * rendered_results[-1]["g_b"]
        self.log("train/g_b_importance", lambda_comp * rendered_results[-1]["g_b"], on_step=True)

        weights = rendered_results[-1]["weights"]
        weights_masks = weights != 0

        tangnet_pred_mod = torch.linalg.norm(rendered_results[-1]["tangents_pred"], dim=-1)
        self.log(f"train/tangent_pred_mu", (weights[weights_masks] * tangnet_pred_mod[weights_masks]).mean(), on_step=True)

        bitangents_pred_mod = torch.linalg.norm(rendered_results[-1]["bitangents_pred"], dim=-1)
        self.log(f"train/bitangents_pred_lambda", (weights[weights_masks] * bitangents_pred_mod[weights_masks]).mean(), on_step=True)

        if self.compute_normal_metrics:
            normal_mae = self.compute_normal_mae(rendered_results[-1], batch["normals"])
            self.log("train/normal_mae_grad", normal_mae, on_step=True)
            normal_mae = self.compute_normal_mae(rendered_results[-1],  batch["normals"], True)
            self.log("train/normal_mae_pred", normal_mae, on_step=True)
            self.log("train/mde_no_back", self.compute_distance_mde(rendered_results[-1], batch["distances"].cpu()))
            self.log("train/mde_back", self.compute_distance_mde(rendered_results[-1], batch["distances"].cpu(), error_plus_background=True))

        if self.compute_bitangent_metrics:
            means = rendered_results[-1]["means"].detach()
            alpha_gt = rendered_results[-1]["weights"].detach()
            temp_val = torch.where(means[..., 0] >= 0, torch.full_like(means[..., 0], True), torch.full_like(means[..., 0], False))
            ecc_GT = (alpha_gt * temp_val).sum(-1).unsqueeze(-1)
            bitangent_mae, bitangent_mae_masked = self.compute_bitangent_mae(rendered_results[-1], batch["bitangents"], mask=ecc_GT)
            self.log("train/bitangent_mae", bitangent_mae, on_step=True)
            self.log("train/bitangent_mae_ecc_filter", bitangent_mae_masked, on_step=True)

        if self._lambda_orientation() > 0:
            orientation_loss = self.orientation_loss(
                rendered_results, batch["viewdirs"]
            )
            self.log("train/orientation_loss", orientation_loss, on_step=True)
            loss += orientation_loss

        if (self._lambda_normals_pred() > 0) or (self.predicted_normal_grad_loss_mult > 0):
            pred_normal_pred_loss = self.predicted_normal_pred_loss(rendered_results)
            pred_normal_grad_loss = self.predicted_normal_gradient_loss(rendered_results)
            self.log("train/pred_normal_pred_loss", pred_normal_pred_loss, on_step=True)
            self.log("train/pred_normal_grad_loss", pred_normal_grad_loss, on_step=True)
            loss += pred_normal_pred_loss
            loss += pred_normal_grad_loss

        psnr1 = helper.mse2psnr(loss1)

        self.log("train/mse1", loss1, on_step=True, prog_bar=True, logger=True)
        self.log("train/psnr1", psnr1, on_step=True, prog_bar=True, logger=True)
        self.log("train/loss", loss, on_step=True)
        return loss

    def render_rays_val(self, batch, batch_idx):

        ret = {}
        rendered_results = self.model(
            batch, False, self.white_bkgd, self.near, self.far, self.radius
        )
        rgb_fine = rendered_results[-1]["comp_rgb"]
        target = batch["target"]
        ret["target"] = target
        ret["rgb"] = rgb_fine

        normals_deriv = (rendered_results[-1]["weights"][..., None] * rendered_results[-1]["normals"]).sum(1)
        ret["comp_normals"] = shiny_utils.l2_normalize(normals_deriv)
        normals_pred_deriv = (rendered_results[-1]["weights"][..., None] * rendered_results[-1]["normals_pred"]).sum(1)
        ret["comp_normals_pred"] = shiny_utils.l2_normalize(normals_pred_deriv)
        
        binormal_deriv = (rendered_results[-1]["weights"][..., None] * rendered_results[-1]["bitangents_pred"]).sum(1)
        ret["comp_binormals"] = shiny_utils.l2_normalize(binormal_deriv)

        ret["eccentricity"] = (rendered_results[-1]["weights"][..., None] * rendered_results[-1]["eccentricity"]).sum(1)
    
        rend_dist_cpu = rendered_results[-1]["distance"].cpu()
        max_num = rend_dist_cpu.max()
        min_num = rend_dist_cpu.min()
        ret["distance"] = ((rend_dist_cpu - min_num) / (max_num - min_num))
        ret["distance"] = ret["distance"][:, np.newaxis]



        ret["roughness"] = (rendered_results[-1]["weights"][..., None] * rendered_results[-1]["roughness"]).sum(1)

        ret["gt_normals"] =  batch["normals"]

        ret["gt_bitangents"] =  batch["bitangents"]

        if self.compute_normal_metrics:
            ret["sum_weights"] =  rendered_results[-1]["weights"].sum()
            ret["mae_grad"] = torch.Tensor([self.compute_normal_mae(rendered_results[-1], batch["normals"])])
            ret["mae_pred"] = torch.Tensor([self.compute_normal_mae(rendered_results[-1], batch["normals"], True)])
            ret["mde_no_back"] =  self.compute_distance_mde(rendered_results[-1], batch["distances"].cpu())
            ret["mde_back"] =  self.compute_distance_mde(rendered_results[-1], batch["distances"].cpu(), error_plus_background=True)
        
        if self.compute_bitangent_metrics:
            means = rendered_results[-1]["means"].detach()
            alpha_gt = rendered_results[-1]["weights"].detach()
            temp_val = torch.where(means[..., 0] >= 0,torch.full_like(means[..., 0], True),torch.full_like(means[..., 0], False))
            ecc_GT = (alpha_gt * temp_val).sum(-1).unsqueeze(-1)
            ret["mae_grad_bitangent"], ret["mae_grad_bitangent_ecc_filter"] = self.compute_bitangent_mae(rendered_results[-1], batch["bitangents"], mask=ecc_GT)
            ret["mae_grad_bitangent"], ret["mae_grad_bitangent_ecc_filter"] = torch.Tensor([ret["mae_grad_bitangent"]]), torch.Tensor([ret["mae_grad_bitangent_ecc_filter"]])

        return ret

    def render_rays_test(self, batch, batch_idx):
        ret = {}
        rendered_results = self.model(
            batch, False, self.white_bkgd, self.near, self.far, self.radius
        )
        rgb_fine = rendered_results[-1]["comp_rgb"]
        target = batch["target"]
        ret["target"] = target
        ret["rgb"] = rgb_fine

        if self.compute_normal_metrics:
            ret["sum_weights"] =  rendered_results[-1]["weights"].sum()
            ret["mae_grad"] = torch.Tensor([self.compute_normal_mae(rendered_results[-1], batch["normals"])])
            ret["mae_pred"] = torch.Tensor([self.compute_normal_mae(rendered_results[-1], batch["normals"], True)])
            ret["mde_no_back"] =  self.compute_distance_mde(rendered_results[-1], batch["distances"].cpu())
            ret["mde_back"] =  self.compute_distance_mde(rendered_results[-1], batch["distances"].cpu(), error_plus_background=True)

        normals_pred_deriv = (rendered_results[-1]["weights"][..., None] * rendered_results[-1]["normals_pred"]).sum(1)
        ret["normals_pred"] = shiny_utils.l2_normalize(normals_pred_deriv)

        return ret


    def save_val_images(self, target_img_idx, val_dataloader, path):

        with torch.no_grad():

            # (1) Run model for each batch of rays corresponding to the target image
            output, image_indx = [], []

            for _, val_batch in enumerate(val_dataloader):
                if target_img_idx in torch.unique(val_batch['image_indx']):
                    for k in val_batch:
                        val_batch[k] = val_batch[k].to(self.device)
                    rendered_results = self.model(
                        val_batch, False, self.white_bkgd, self.near, self.far, self.radius
                    )

                    ret = self.calculate_var(rendered_results, val_batch, 'test')
                    output.append(ret)
                    image_indx.append(val_batch['image_indx'])

                elif (target_img_idx < val_batch['image_indx']).all():
                    break
        
        # (2) Concatenate all batch outputs
        joined = dict(zip(output[0].keys(), [None]*len(output[0].keys())))
        valid_rays = torch.cat(image_indx) == target_img_idx
        for kk in output[0].keys():
            x = torch.cat([o[kk] for o in output])[valid_rays]
            size_img = int(math.sqrt(x.shape[0]))
            if x.ndim == 3:
                joined[kk] = x.view(size_img,size_img,x.shape[-2], x.shape[-1]) # size_img
            else:
                joined[kk] = x.view(size_img,size_img,x.shape[-1])
        
        # (3) Save output images
        self.save_img(joined, path)


    def save_single_file(self, image_dir, folder,  name_file, img, colormap=False):
            full_path = os.path.join(image_dir, folder)
            os.makedirs(full_path, exist_ok=True)
            if not colormap:
                store_image.store_single_image(full_path, name_file, img)
            else:
                store_image.store_single_image_colormap(full_path, name_file, img)

    def save_single_file_tiff(self, image_dir, folder,  name_file, img):

        full_path = os.path.join(image_dir, folder)
        os.makedirs(full_path, exist_ok=True)
        img_array = img.cpu().numpy().astype(np.float32)
        tiff.imwrite(f'{full_path}/{name_file:03d}.tif', img_array)

    def save_single_file_np(self, image_dir, folder,  i, img):
        full_path = os.path.join(image_dir, folder)
        os.makedirs(full_path, exist_ok=True)
        img_array = img.cpu().numpy().astype(np.float32)
        np.save(f'{full_path}/{i:03d}.npy', img_array)

    def save_img(self, output, image_dir):

        os.makedirs(image_dir, exist_ok=True)
        rgb = output['rgb']

        # Get the number of iterations already made
        rgb_dir = os.path.join(image_dir, 'rgb')
        os.makedirs(rgb_dir, exist_ok=True)
        i = self.count_files(rgb_dir)

        self.save_single_file_tiff(image_dir, 'rgb_tif',  i, rgb)
        self.save_single_file(image_dir, 'rgb',  i, rgb)
        del rgb


        diffuse = output['diffuse']
        self.save_single_file_tiff(image_dir, 'diffuse_tif',  i, diffuse)
        self.save_single_file(image_dir, 'diffuse',  i, diffuse)
        del diffuse


        specular = output['specular']
        self.save_single_file_tiff(image_dir, 'specular_tif',  i, specular)
        self.save_single_file(image_dir, 'specular',  i, specular)
        del specular


        tint = output['tint']
        self.save_single_file_tiff(image_dir, 'tint_tif',  i, tint)
        self.save_single_file(image_dir, 'tint',  i, tint)
        del tint


        specular_rgb = output['specular_rgb']
        self.save_single_file_tiff(image_dir, 'specular_rgb_tif', i, specular_rgb)
        self.save_single_file(image_dir, 'specular_rgb',  i, specular_rgb)
        del specular_rgb


        normals = output['comp_normals']
        self.save_single_file_tiff(image_dir, 'normals_tif',  i, normals)
        normals = (normals+1) / 2
        self.save_single_file(image_dir, 'comp_normals',  i, normals)
        del normals


        normals_pred = output['comp_normals_pred']
        self.save_single_file_tiff(image_dir, 'normals_pred_tif',  i, normals_pred)
        normals_pred = (normals_pred+1) / 2
        self.save_single_file(image_dir, 'comp_normals_pred',  i, normals_pred)
        del normals_pred


        tangnets_pred = output['comp_binormals']
        self.save_single_file_tiff(image_dir, 'comp_binormals_tif',  i, tangnets_pred)
        tangnets_pred = (tangnets_pred+1) / 2
        self.save_single_file(image_dir, 'comp_binormals',  i, tangnets_pred)
        del tangnets_pred


        tangnets_pred = output['comp_tangents']
        self.save_single_file_tiff(image_dir, 'comp_tangents_tif',  i, tangnets_pred)
        tangnets_pred = (tangnets_pred+1) / 2
        self.save_single_file(image_dir, 'comp_tangents',  i, tangnets_pred)
        del tangnets_pred


        ecc = output['eccentricity'] * (output['concentration'] / 2)
        self.save_single_file_tiff(image_dir, 'abs_ecc_tif',  i, ecc)
        ecc = (ecc - ecc.min()) / (ecc.max() - ecc.min())
        ecc_3d = ecc.repeat(1,1,3)
        self.save_single_file(image_dir, 'abs_ecc',  i, ecc_3d, True)
        del ecc

        tangnets_pred = output['comp_binormals']
        ecc = output['eccentricity']
        tangnets_pred = tangnets_pred * ecc
        self.save_single_file_tiff(image_dir, 'comp_binormals_masked_ecc_tif',  i, tangnets_pred)
        tangnets_pred = (tangnets_pred+1) / 2
        self.save_single_file(image_dir, 'comp_binormals_masked_ecc',  i, tangnets_pred)
        del tangnets_pred
        del ecc

        ecc_norm = output['eccentricity']

        self.save_single_file_tiff(image_dir, 'ecc_tif',  i, ecc_norm)
        ecc_norm = (ecc_norm - ecc_norm.min()) / (ecc_norm.max() - ecc_norm.min())
        ecc_norm_3d = ecc_norm.repeat(1,1,3)
        self.save_single_file(image_dir, 'ecc',  i, ecc_norm_3d, True)
        del ecc_norm


        roughness = output['roughness']

        self.save_single_file_tiff(image_dir, 'roughness_tif',  i, roughness)
        roughness = (roughness - roughness.min()) / (roughness.max() - roughness.min())
        roughness_3d = roughness.repeat(1,1,3)
        self.save_single_file(image_dir, 'roughness',  i, roughness_3d, True)
        del roughness


        concentration = output['concentration']

        self.save_single_file_tiff(image_dir, 'concentration_tif',  i, concentration)
        concentration = (concentration - concentration.min()) / (concentration.max() - concentration.min())
        concentration_3d = concentration.repeat(1,1,3)
        self.save_single_file(image_dir, 'concentration',  i, concentration_3d, True)
        del concentration


        lambda_asg = output['lambda_asg']

        self.save_single_file_tiff(image_dir, 'lambda_asg_tif',  i, lambda_asg)
        lambda_asg = (lambda_asg - lambda_asg.min()) / (lambda_asg.max() - lambda_asg.min())
        lambda_asg_3d = lambda_asg.repeat(1,1,3)
        self.save_single_file(image_dir, 'lambda_asg',  i, lambda_asg_3d, True)
        del lambda_asg

        mu_asg = output['mu_asg']

        self.save_single_file_tiff(image_dir, 'mu_asg_tif',  i, mu_asg)
        mu_asg = (mu_asg - mu_asg.min()) / (mu_asg.max() - mu_asg.min())
        mu_asg_3d = mu_asg.repeat(1,1,3)
        self.save_single_file(image_dir, 'mu_asg',  i, mu_asg_3d, True)
        del mu_asg


        distances = output['distance']
        self.save_single_file_tiff(image_dir, 'distances_tif',  i, distances)
        dist_3d = output['distance_norm'].repeat(1,1,3)
        self.save_single_file(image_dir, 'distance',  i, dist_3d, True)
        del distances
        del dist_3d


        if self.compute_normal_metrics:
            results = {}
            results['weights'] = output['weights']
            results['normals_pred'] = output['comp_normals_pred']
            gt_normals = output['gt_normals']

            mae_pred = self.compute_normal_mae(results, gt_normals, normals_pred=True, picture=True)

            self.save_single_file_tiff(image_dir, 'mae_normals_pred_tif',  i, mae_pred)

            mae_pred_3d = mae_pred.unsqueeze(-1).repeat(1,1,3)
            mae_pred_3d /= 359
            self.save_single_file(image_dir, 'mae_normals_pred',  i, mae_pred_3d, colormap=True)
            del results
            del gt_normals


            normals = output['comp_normals_pred']
            gt_normals = output['gt_normals']
            diff_normals = ((gt_normals[...,:-1] - normals)**2).sum(-1) * gt_normals[...,-1]
            self.save_single_file_tiff(image_dir, 'diff_normals_pred_tif',  i, diff_normals)
            diff_normals = (diff_normals - diff_normals.min()) / (diff_normals.max() - diff_normals.min())
            diff_normals3d = diff_normals.unsqueeze(-1).repeat(1,1,3)
            self.save_single_file(image_dir, 'diff_normals_pred',  i, diff_normals3d, colormap=True)
            del normals
            del gt_normals


            results = {}
            results['weights'] = output['weights']
            results['normals'] = output['comp_normals']
            gt_normals = output['gt_normals']

            mae_grad = self.compute_normal_mae(results, gt_normals, picture=True)

            self.save_single_file_tiff(image_dir, 'mae_normals_grad_tif',  i, mae_grad)

            mae_grad_3d = mae_grad.unsqueeze(-1).repeat(1,1,3)
            mae_grad_3d /= 359
            self.save_single_file(image_dir, 'mae_normals_grad',  i, mae_grad_3d, colormap=True)
            del results
            del gt_normals


            normals = output['comp_normals']
            gt_normals = output['gt_normals']
            diff_normals = ((gt_normals[...,:-1] - normals)**2).sum(-1) * gt_normals[...,-1]
            self.save_single_file_tiff(image_dir, 'diff_normals_grad_tif',  i, diff_normals)
            diff_normals = (diff_normals - diff_normals.min()) / (diff_normals.max() - diff_normals.min())
            diff_normals3d = diff_normals.unsqueeze(-1).repeat(1,1,3)
            self.save_single_file(image_dir, 'diff_normals_grad',  i, diff_normals3d, colormap=True)
            del normals
            del gt_normals

            results = {}
            results['distance'] = output['distance'].cpu()
            results['weights'] = output['weights'].cpu()
            gt_distance = output['gt_distance'].cpu()

            mde =  self.compute_distance_mde(results, gt_distance, error_plus_background=True, picture=True)
            self.save_single_file_tiff(image_dir, 'mde_back_tif',  i, mde)
            mde = (mde - mde.min()) / (mde.max() - mde.min())
            mde_3d = mde.repeat(1,1,3)
            self.save_single_file(image_dir, 'mde_back',  i, mde_3d, True)
            del results
            del gt_distance


            results = {}
            results['distance'] = output['distance'].cpu()
            results['weights'] = output['weights'].cpu()
            gt_distance = output['gt_distance'].cpu()

            mde =  self.compute_distance_mde(results, gt_distance, picture=True)
            self.save_single_file_tiff(image_dir, 'mde_no_back_tif',  i, mde)
            mde = (mde - mde.min()) / (mde.max() - mde.min())
            mde_3d = mde.repeat(1,1,3)
            self.save_single_file(image_dir, 'mde_no_back',  i, mde_3d, True)
            del results
            del gt_distance

    def calculate_var(self,rendered_results, batch_input, mode):
        ret = {}

        ret["means"] = rendered_results[-1]["means"]
        ret["viewdirs"] = rendered_results[-1]["viewdirs"]
        ret["rays_o"] = rendered_results[-1]["rays_o"]
        ret["rays_d"] = rendered_results[-1]["rays_d"]

        ret["rgb"] = rendered_results[-1]["comp_rgb"]

        ret["diffuse"] = rendered_results[-1]["diffuse"]
        ret["specular"] = rendered_results[-1]["specular"]
        ret["tint"] = rendered_results[-1]["tint"]
        ret["specular_rgb"] = rendered_results[-1]["specular_rgb"]

        normals_deriv = (rendered_results[-1]["weights"][..., None] * rendered_results[-1]["normals"]).sum(1)
        ret["comp_normals"] = shiny_utils.l2_normalize(normals_deriv)
        normals_pred_deriv = (rendered_results[-1]["weights"][..., None] * rendered_results[-1]["normals_pred"]).sum(1)
        ret["comp_normals_pred"] = shiny_utils.l2_normalize(normals_pred_deriv)

        if mode == 'test':
            ret["distance"] = rendered_results[-1]["distance"].cpu()[:, np.newaxis]
            ret["distance_norm"] = rendered_results[-1]["distance_norm"].cpu()[:, np.newaxis]


        binormal_deriv = (rendered_results[-1]["weights"][..., None] * rendered_results[-1]["bitangents_pred"]).sum(1)
        ret["comp_binormals"] = shiny_utils.l2_normalize(binormal_deriv)

        tangent_deriv = (rendered_results[-1]["weights"][..., None] * rendered_results[-1]["tangents_pred"]).sum(1)
        ret["comp_tangents"] = shiny_utils.l2_normalize(tangent_deriv)

        ret["eccentricity"] = (rendered_results[-1]["weights"][..., None] * rendered_results[-1]["eccentricity"]).sum(1)

        ret["concentration"] = (rendered_results[-1]["weights"][..., None] * rendered_results[-1]["concentration"]).sum(1)

        ret["roughness"] = (rendered_results[-1]["weights"][..., None] * rendered_results[-1]["roughness"]).sum(1)

        ret["lambda_asg"] = (rendered_results[-1]["weights"][..., None] * rendered_results[-1]["lambda_asg"]).sum(1)

        ret["mu_asg"] = (rendered_results[-1]["weights"][..., None] * rendered_results[-1]["mu_asg"]).sum(1)

        if self.compute_normal_metrics:
            ret["normals"] = rendered_results[-1]["normals"]
            ret["normals_pred"] = rendered_results[-1]["normals_pred"]
            ret["weights"] = rendered_results[-1]["weights"]
            ret["gt_normals"] = batch_input['normals']
            ret["gt_distance"] = batch_input['distances']

        if self.compute_bitangent_metrics:
            ret["bitangents_pred"] = rendered_results[-1]["bitangents_pred"]
            ret["gt_bitangents"] = batch_input['bitangents']

        ret["weights"] = rendered_results[-1]["weights"]

        ret["t_vals"] = rendered_results[-1]["t_vals"]

        return ret


    def validation_step(self, batch, batch_idx):

        return self.render_rays_val(batch, batch_idx)


    def get_folder_names(self, directory):
        # List all entries in the directory
        entries = os.listdir(directory)
        # Filter out only the directories
        folders = [entry for entry in entries if os.path.isdir(os.path.join(directory, entry))]
        return folders

    def count_files(self, directory):
        # List all entries in the directory
        entries = os.listdir(directory)
        # Filter out only the files
        files = [entry for entry in entries if os.path.isfile(os.path.join(directory, entry))]
        return len(files)

    def test_step(self, batch, batch_idx):

        image_dir = os.path.join(self.logdir, "render_model")

        id_batch = batch['image_indx'][-1].item() # Suppose an image is bigger than a batch
        if self.first_number_test < 0:
            os.makedirs(image_dir, exist_ok=True)
            self.first_number_test = id_batch # Get the first id of all
        
        folder_name = self.get_folder_names(image_dir)


        if len(folder_name) == 0 or id_batch - self.first_number_test > self.count_files(os.path.join(image_dir, folder_name[-1])) - 1:  
            print("saving image", id_batch - self.first_number_test)
            val_batch = self.trainer.test_dataloaders[0]
            self.save_val_images(id_batch, val_batch, image_dir)

        return self.render_rays_test(batch, batch_idx)

    def configure_optimizers(self):
        return torch.optim.Adam(
            params=self.parameters(), lr=self.lr_init, betas=(0.9, 0.999)
        )

    def optimizer_step(
        self,
        epoch,
        batch_idx,
        optimizer,
        optimizer_idx,
        optimizer_closure,
        on_tpu,
        using_native_amp,
        using_lbfgs,
    ):
        step = self.trainer.global_step
        max_steps = gin.query_parameter("run.max_steps")

        if self.lr_delay_steps > 0:
            delay_rate = self.lr_delay_mult + (1 - self.lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / self.lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0

        t = np.clip(step / max_steps, 0, 1)
        scaled_lr = np.exp(np.log(self.lr_init) * (1 - t) + np.log(self.lr_final) * t)
        new_lr = delay_rate * scaled_lr

        for pg in optimizer.param_groups:
            pg["lr"] = new_lr

        if self.grad_max_norm > 0:
            nn.utils.clip_grad_norm_(self.parameters(), self.grad_max_norm)

        optimizer.step(closure=optimizer_closure)

    def validation_epoch_end(self, outputs):
        path_rgb = os.path.join(self.logdir, "val_rgb_tiff")
        path_norm_tan_ecc = os.path.join(self.logdir, "val_norm_tan_ecc_tiff")
        os.makedirs(path_rgb, exist_ok=True)
        os.makedirs(path_norm_tan_ecc, exist_ok=True)

        val_image_sizes = self.trainer.datamodule.val_image_sizes
        rgbs = self.alter_gather_cat(outputs, "rgb", val_image_sizes)
        targets = self.alter_gather_cat(outputs, "target", val_image_sizes)


        psnr_mean = self.psnr_each(rgbs, targets).mean()
        ssim_mean = self.ssim_each(rgbs, targets).mean()
        msssim_mean = self.msssim_each(rgbs, targets).mean()
        gmsd_mean = self.gmsd_each(rgbs, targets).mean()
        msgmsd_mean = self.msgmsd_each(rgbs, targets).mean()
        mdsi_mean = self.mdsi_each(rgbs, targets).mean()
        haarpsi_mean = self.haarpsi_each(rgbs, targets).mean()
        lpips_mean = self.lpips_each(rgbs, targets).mean()
        self.log("val/psnr", psnr_mean.item(), on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val/ssim", ssim_mean.item(), on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val/msssim", msssim_mean.item(), on_epoch=True,  prog_bar=True, sync_dist=True)
        self.log("val/gmsd", gmsd_mean.item(), on_epoch=True,  prog_bar=True, sync_dist=True)
        self.log("val/msgmsd", msgmsd_mean.item(), on_epoch=True,  prog_bar=True, sync_dist=True)
        self.log("val/mdsi", mdsi_mean.item(), on_epoch=True,  prog_bar=True, sync_dist=True)
        self.log("val/haarpsi", haarpsi_mean.item(), on_epoch=True,  prog_bar=True, sync_dist=True)
        self.log("val/lpips", lpips_mean.item(), on_epoch=True, prog_bar=True, sync_dist=True)

        rr = norm8b255(rgbs[0].permute(2,0,1))
        tt = norm8b255(targets[0].permute(2,0,1))

        stack_rgb =  torch.stack([rr, tt])
        self.logger.experiment.add_images("val/images (Pred - GT)", stack_rgb, global_step=self.trainer.global_step)

        index = 6 if len(rgbs) > 7 else len(rgbs) // 2
        rr = norm8b255(rgbs[index].permute(2,0,1))
        tt = norm8b255(targets[index].permute(2,0,1))

        stack_rgb =  torch.stack([rr, tt])
        self.logger.experiment.add_images("val/images (Pred - GT) 2", stack_rgb, global_step=self.trainer.global_step)

        path = os.path.join(path_rgb , f"stack_rgb_{self.current_epoch}.tif")
        io.imsave(path, stack_rgb.cpu().numpy())

        del rgbs
        del targets
        del stack_rgb

        gt_normals_aplha = self.alter_gather_cat(outputs, "gt_normals", val_image_sizes)
        gt_normals = [m[:,:,:3] for m in gt_normals_aplha]


        comp_normals = self.alter_gather_cat(outputs, "comp_normals", val_image_sizes)
        comp_normals_pred = self.alter_gather_cat(outputs, "comp_normals_pred", val_image_sizes)
        comp_binormals = self.alter_gather_cat(outputs, "comp_binormals", val_image_sizes)
        eccentricity = self.alter_gather_cat(outputs, "eccentricity", val_image_sizes)

        distance = self.alter_gather_cat(outputs, "distance", val_image_sizes)
        roughness = self.alter_gather_cat(outputs, "roughness", val_image_sizes)


        mask = (eccentricity[0] > 0.05).all(dim=-1) & (distance[0] > 0.05).all(dim=-1)
        mask = mask.unsqueeze(-1).repeat(1,1,3)
        norm = norm8b255(comp_normals[0]).permute(2,0,1)
        norm_pred = norm8b255(comp_normals_pred[0]).permute(2,0,1)
        gt_normal = norm8b255(gt_normals[0]).permute(2,0,1)
        binormal = norm8b255(comp_binormals[0]).permute(2,0,1)
        ecc = norm8b255(eccentricity[0].repeat(1,1,3), apply_colormap=True).permute(2,0,1)
        ro = norm8b255(roughness[0].repeat(1,1,3),apply_colormap=True).permute(2,0,1)
        dist = norm8b255(distance[0].repeat(1,1,3),apply_colormap=True).permute(2,0,1)

        stack_norm_tan_ecc =  torch.stack([norm_pred, norm, gt_normal, binormal, ecc, dist, ro])
        self.logger.experiment.add_images("val/norms (Pred - Density - GT - Bnorm - Ecc - Dist - Ro)", norm8b(stack_norm_tan_ecc), global_step=self.trainer.global_step)

        path = os.path.join(path_norm_tan_ecc, f"stack_norm_tan_ecc_{self.current_epoch}_1.tif")
        io.imsave(path, stack_norm_tan_ecc.cpu().numpy())

        mask = (eccentricity[index] > 0.05).all(dim=-1) & (distance[index] > 0.05).all(dim=-1)

        norm = norm8b255(comp_normals[index]).permute(2,0,1)
        norm_pred = norm8b255(comp_normals_pred[index]).permute(2,0,1)
        gt_normal = norm8b255(gt_normals[index]).permute(2,0,1)
        binormal = norm8b255(comp_binormals[index]).permute(2,0,1)
        ecc = norm8b255(eccentricity[index].repeat(1,1,3), apply_colormap=True).permute(2,0,1)
        ro = norm8b255(roughness[index].repeat(1,1,3), apply_colormap=True).permute(2,0,1)
        dist = norm8b255(distance[index].repeat(1,1,3), apply_colormap=True).permute(2,0,1)


        stack_norm_tan_ecc =  torch.stack([norm_pred, norm, gt_normal, binormal, ecc, dist, ro])
        self.logger.experiment.add_images("val/norms (Pred - Density - GT - Bnorm - Ecc - Dist - RO) 2", norm8b(stack_norm_tan_ecc), global_step=self.trainer.global_step)

        path = os.path.join(path_norm_tan_ecc, f"stack_norm_tan_ecc_{self.current_epoch}_2.tif")
        io.imsave(path, stack_norm_tan_ecc.cpu().numpy())

        del stack_norm_tan_ecc
        del gt_normals_aplha
        del comp_normals
        del comp_normals_pred
        del comp_binormals
        del distance

        if self.compute_normal_metrics:
            mae_grad = torch.Tensor([m['mae_grad'] for m in outputs])
            mae_pred = torch.Tensor([m['mae_pred'] for m in outputs])
            mde_no_back = torch.Tensor([m['mde_no_back'] for m in outputs])
            mde_back = torch.Tensor([m['mde_back'] for m in outputs])

            mae_grad = torch.Tensor([m['mae_grad'] for m in outputs])
            mae_pred = torch.Tensor([m['mae_pred'] for m in outputs])
            sum_weights = torch.Tensor([m['sum_weights'] for m in outputs])

            self.log("val/normal_mae_grad", ((mae_grad * sum_weights).sum() / sum_weights.sum()), on_epoch=True)
            self.log("val/normal_mae_pred", ((mae_pred * sum_weights).sum() / sum_weights.sum()), on_epoch=True)

            self.log("val/mde_no_back", ((mde_no_back * sum_weights).sum() / sum_weights.sum()), on_epoch=True)
            self.log("val/mde_back", ((mde_back * sum_weights).sum() / sum_weights.sum()), on_epoch=True)

        if self.compute_bitangent_metrics:
            mae_grad_bitangent = torch.Tensor([m['mae_grad_bitangent'] for m in outputs])
            self.log("val/normal_mae_grad_bitangent", ((mae_grad_bitangent * sum_weights).sum() / sum_weights.sum()), on_epoch=True)

            mae_grad_bitangent_ecc_filter = torch.Tensor([m['mae_grad_bitangent_ecc_filter'] for m in outputs])
            self.log("val/normal_mae_grad_bitangent_ecc_filter", ((mae_grad_bitangent_ecc_filter * sum_weights).sum() / sum_weights.sum()), on_epoch=True)


    def test_epoch_end(self, outputs):
        dmodule = self.trainer.datamodule
        all_image_sizes = (
            dmodule.all_image_sizes
            if not dmodule.eval_test_only
            else dmodule.test_image_sizes
        )

        rgbs = self.alter_gather_cat(outputs, "rgb", all_image_sizes)
        targets = self.alter_gather_cat(outputs, "target", all_image_sizes)
        psnr = self.psnr(rgbs, targets, dmodule.i_train, dmodule.i_val, dmodule.i_test)
        ssim = self.ssim(rgbs, targets, dmodule.i_train, dmodule.i_val, dmodule.i_test)
        msssim = self.msssim(rgbs, targets, dmodule.i_train, dmodule.i_val, dmodule.i_test)
        gmsd = self.gmsd(rgbs, targets, dmodule.i_train, dmodule.i_val, dmodule.i_test)
        msgmsd = self.msgmsd(rgbs, targets, dmodule.i_train, dmodule.i_val, dmodule.i_test)
        mdsi = self.mdsi(rgbs, targets, dmodule.i_train, dmodule.i_val, dmodule.i_test)
        haarpsi = self.haarpsi(rgbs, targets, dmodule.i_train, dmodule.i_val, dmodule.i_test)
        vsi = self.vsi(rgbs, targets, dmodule.i_train, dmodule.i_val, dmodule.i_test)
        fsim = self.fsim(rgbs, targets, dmodule.i_train, dmodule.i_val, dmodule.i_test)
        lpips = self.lpips(
            rgbs, targets, dmodule.i_train, dmodule.i_val, dmodule.i_test
        )
        fid = self.fid(rgbs, targets, dmodule.i_train, dmodule.i_val, dmodule.i_test)

        self.log("test/psnr", psnr["test"], on_epoch=True)
        self.log("test/ssim", ssim["test"], on_epoch=True)
        self.log("test/msssim", msssim["test"], on_epoch=True)
        self.log("test/gmsd", gmsd["test"], on_epoch=True)
        self.log("test/msgmsd", msgmsd["test"], on_epoch=True)
        self.log("test/mdsi", mdsi["test"], on_epoch=True)
        self.log("test/haarpsi", haarpsi["test"], on_epoch=True)
        self.log("test/vsi", vsi["test"], on_epoch=True)
        self.log("test/fsim", fsim["test"], on_epoch=True)
        self.log("test/lpips", lpips["test"], on_epoch=True)
        self.log("test/fid", fid["test"], on_epoch=True)


        if self.compute_normal_metrics:

            mae_grad = torch.Tensor([m['mae_grad'] for m in outputs])
            mae_pred = torch.Tensor([m['mae_pred'] for m in outputs])
            sum_weights = torch.Tensor([m['sum_weights'] for m in outputs])

            self.log("test/normal_mae_grad", ((mae_grad * sum_weights).sum() / sum_weights.sum()), on_epoch=True)
            self.log("test/normal_mae_pred", ((mae_pred * sum_weights).sum() / sum_weights.sum()), on_epoch=True)


            mde = torch.Tensor([m['mde_no_back'] for m in outputs])
            self.log("test/mde_no_back", ((mde * sum_weights).sum() / sum_weights.sum()), on_epoch=True)
            mde = torch.Tensor([m['mde_back'] for m in outputs])
            self.log("test/mde_back", ((mde * sum_weights).sum() / sum_weights.sum()), on_epoch=True)

        return psnr, ssim, msssim, gmsd, msgmsd, mdsi, haarpsi, vsi, fsim, lpips, fid




    # ────────────────────────────────────────────────────────────────
    def predicted_normal_pred_loss(self, rendered_results):
        """Loss between gradient-based normals (n) and learned normals (n̂)."""
        rr   = rendered_results[-1]                       # final NeRF only
        w    = rr["weights"].detach()                    # stop grad via density
        n    = rr["normals"].detach()
        nhat = rr["normals_pred"]
        if n is None or nhat is None:
            raise ValueError("Normals cannot be None if predicted-normal loss is on.")

        loss_map = w * (1.0 - helper._angular_similarity_2(n, nhat))     # [B,H,W]
        loss     = loss_map.sum(dim=-1).mean()                  # average over rays

        return self._lambda_normals_pred() * loss

    # ────────────────────────────────────────────────────────────────
    def predicted_normal_gradient_loss(self, rendered_results):
        """Loss that treats learned normals as target, penalising gradient direction."""
        rr   = rendered_results[-1]
        w    = rr["weights"]
        n    = rr["normals"]                 # keep grad for density field
        nhat = rr["normals_pred"].detach()   # stop grad via normal-MLP branch
        if n is None or nhat is None:
            raise ValueError("Normals cannot be None if predicted-normal loss is on.")

        loss_map = w * (1.0 - helper._angular_similarity_2(n, nhat))
        loss     = loss_map.sum(dim=-1).mean()

        return self.predicted_normal_grad_loss_mult * loss

    # ────────────────────────────────────────────────────────────────
    def orientation_loss(self, rendered_results, viewdirs):
        """Encourages surface normals to face the camera (Ref-NeRF style)."""
        rr   = rendered_results[-1]
        w    = rr["weights"]
        nhat = rr["normals_pred"]
        if nhat is None:
            raise ValueError("Normals cannot be None if orientation loss is on.")

        v     = -viewdirs                                 # viewing direction
        n_dot = (nhat * v[..., None, :]).sum(-1)          # [B,H,W,S]
        loss_map = w * torch.clamp(n_dot, max=0.0) ** 2
        loss     = loss_map.sum(dim=-1).mean()
        
        return self._lambda_orientation() * loss
   

    def compute_normal_mae(self, rendered_results, normals, normals_pred=False, picture=False):
        normals_gt, alphas = torch.split(normals, [3, 1], dim=-1)
        weights = rendered_results["weights"] * alphas

        if picture:
            normalized_normals_gt = shiny_utils.l2_normalize(normals_gt)
        else:
            normalized_normals_gt = shiny_utils.l2_normalize(normals_gt[..., None, :])

        if normals_pred:
            normalized_normals = shiny_utils.l2_normalize(rendered_results["normals_pred"])
        else:
            normalized_normals = shiny_utils.l2_normalize(rendered_results["normals"])

        if picture:
            normal_mae = shiny_utils.compute_weighted_mae_picture_2(
                normalized_normals, normalized_normals_gt
            ) * alphas.squeeze()
        else:
            normal_mae = shiny_utils.compute_weighted_mae(
                weights, normalized_normals, normalized_normals_gt
            )

        return normal_mae

    def compute_bitangent_mae(self, rendered, bitangents, mask=None):
        eps = 1e-7
        # ---------- shared prep ----------
        btngs_gt, alphas = torch.split(bitangents, 3, -1)
        w   = rendered["weights"] * alphas          # (B,N)   or (R,S)
        pred = F.normalize(rendered["bitangents_pred"], dim=-1)
        gt   = F.normalize(btngs_gt[..., None, :],  dim=-1)
        gt_f = -gt

        # angle once
        ang_pos = torch.arccos((pred * gt     ).sum(-1).clamp(-1+1e-7, 1-1e-7))
        ang_neg = torch.arccos((pred * gt_f   ).sum(-1).clamp(-1+1e-7, 1-1e-7))
        ang     = torch.minimum(ang_pos, ang_neg)   # (B,N)

        # ---------- no-mask MAE ----------
        mae_all = (w * ang).sum() / (w.sum() + 1e-7)

        # ---------- eccentricity mask ----------
        mae_mask = None
        if mask is not None:
            # broadcast mask to (B,N) and put on same device/dtype
            m = mask
            if m.dim() < w.dim():
                m = m.expand_as(w)
            m = m.to(dtype=w.dtype, device=w.device)  # boolean->float for weighting

            w_m = w * m
            mae_mask = (w_m * ang).sum() / (w_m.sum() + eps)

        # return degrees
        return mae_all * (180.0 / torch.pi), (mae_mask * (180.0 / torch.pi) if mae_mask is not None else None)


    def compute_distance_mde(self, rendered_results, gt_distance, error_plus_background=False, picture=False):

        # Normalize results
        distance = rendered_results["distance"].cpu()
        if not picture:
            distance = distance[:, np.newaxis]

        # Normalize gt
        gt_dist_cpu_, alphas = torch.split(gt_distance, [3, 1], dim=-1)
        gt_dist_cpu_ = gt_dist_cpu_[...,0].unsqueeze(-1)
        gt_distance = gt_dist_cpu_


        weights = (rendered_results["weights"].cpu() * alphas).sum(-1).unsqueeze(-1)
        if not picture:
            gt_distance = gt_distance[:, np.newaxis]

        if torch.isnan(torch.abs(distance - gt_distance).mean()):
            print("MDE has nans")
            import pdb; pdb.set_trace()

        if error_plus_background:
            if picture:
                return torch.abs(distance - gt_distance)
            else:
                return (torch.abs(distance - gt_distance)).mean()
        else:
            if picture:
                return torch.abs(distance - gt_distance) * weights 
            else:
                return (torch.abs(distance - gt_distance) * weights).mean()


    def spline_interlevel_loss_torch(self, ray_history, *, mults, blurs=None,
                                    eps=1e-8):
        """
        PyTorch version of Zip-NeRF `spline_interlevel_loss`.

        https://github.com/jonbarron/camp_zipnerf/blob/8e6d57e3aee34235faf3ef99decca0994efe66c9/camp_zipnerf/internal/loss_utils.py#L25
        function spline_interlevel_loss


        Args
        ----
        ray_history : list of dicts, len == #levels
                    each dict must have 'sdist' (edges) and 'weights'.
                    last element is the final NeRF level.
        mults       : scalar or tuple, per-proposal loss weights.
        blurs       : None / scalar / tuple  (per proposal blur half-width)
        eps         : numeric stability term.

        Returns
        -------
        loss : scalar  (sum over proposal levels, already multiplied by mults)
        """

        num_rounds = len(ray_history) - 1
        mults = helper.ensure_tuple(mults, num_rounds)
        blurs = helper.ensure_tuple(blurs, num_rounds)

        if len(mults) != num_rounds or len(blurs) != num_rounds:
            raise ValueError("Length of mults/blurs must match #proposal levels.")

        # Final NeRF histogram (reference)
        c = ray_history[-1]["t_vals"]                  # [..., S]
        w = ray_history[-1]["weights"].detach()        # [..., S-1]

        losses = []
        for mult, blur, ray_res in zip(mults, blurs, ray_history[:-1]):
            if mult == 0.0:
                continue

            cp = ray_res["t_vals"]                     # proposal edges   [..., M]
            wp = ray_res["weights"]                    # proposal weights [..., M-1]

            # ---- choose half-width r ----------------------------------------
            blur_val = helper.auto_blur_halfwidth(c, cp) if blur is None else blur

            # ---- core resampling: NeRF → proposal bins ----------------------
            w_blur = helper.blur_and_resample_weights(
                cp, c, w, blur_halfwidth=blur_val
            )

            # truncated χ² divergence
            loss = torch.clamp(w_blur - wp, min=0.0) ** 2 / (wp + eps)
            losses.append(mult * loss.mean())

        return sum(losses)

    @staticmethod
    def proposal_overlap_loss(t, w, t_ref, w_ref, detach_ref=True):
        if detach_ref:
            t_ref, w_ref = t_ref.detach(), w_ref.detach()

        # interval start / end & interval weights
        start_i, end_i   = t[..., :-1], t[..., 1:]                  # [..., M]
        start_j, end_j   = t_ref[..., :-1], t_ref[..., 1:]          # [..., K]

        w_i = helper.to_interval_weights(t,     w)        # [..., M]
        w_j = helper.to_interval_weights(t_ref, w_ref)    # [..., K]

        # overlap mask (broadcasted)
        overlap = (start_i[..., :, None] < end_j[..., None, :]) & \
                (end_i  [..., :, None] > start_j[..., None, :])   # [..., M,K]

        # sum weights of reference intervals that intersect each proposal interval
        w_hat_sum = (overlap.float() * w_j[..., None, :]).sum(-1)    # [..., M]

        diff  = torch.clamp(w_i - w_hat_sum, min=0.0)
        loss  = (diff ** 2 / (w_i + 1e-8)).sum(-1)                   # [...]

        return loss.mean()                                           # scalar



    # ------------------------------------------------------------
    # distortion loss
    # ------------------------------------------------------------
    def distortion_loss(self, t_vals,                # [..., N]   (should already be in s-space)
                        weights,
                        weight=1.0,
                        detach=False):
        """
        Implements Eq.(15) from mip-NeRF 360.

        Args
        ----
        t_vals : tensor [..., N]       -- interval edges  (s-coordinates)
        weights: tensor [..., N] or [..., N-1]
        weight : float                 -- global scale (λ); paper uses 0.01
        detach : bool                  -- if True, detach both inputs
        """
        if detach:
            t_vals = t_vals.detach()
            weights = weights.detach()

        # one weight per interval  [..., I]  with I = N-1
        w_I = helper.to_interval_weights(t_vals, weights)          # [..., I]

        # interval centres and widths
        centres = helper._centres(t_vals)                           # [..., I]
        Δs      = t_vals[..., 1:] - t_vals[..., :-1]         # [..., I]

        # -------- pairwise spread term  Σ_i Σ_j  w_i w_j |c_i - c_j|
        ci = centres[..., :, None]                          # [..., I,1]
        cj = centres[..., None, :]                          # [..., 1,I]
        abs_d = (ci - cj).abs()                             # [..., I,I]

        wi = w_I[..., :, None]
        wj = w_I[..., None, :]
        term1 = (wi * wj * abs_d).sum(dim=(-2, -1))         # [...]

        # -------- width term  (1/3) Σ_i w_i² Δs_i
        term2 = (1.0 / 3.0) * (w_I ** 2 * Δs).sum(dim=-1)   # [...]

        # average over rays / batch and scale by λ
        return weight * (term1 + term2).mean()


    def _linear_warmup(self, start, end, warm_steps, step=None):
        """
        Linearly interpolate from `start` to `end` over `warm_steps` steps.
        After warmup, returns `end`.
        """
        if step is None:
            step = self.global_step

        if warm_steps <= 0:
            return end

        if step >= warm_steps:
            return end

        frac = step / float(warm_steps)
        return start + frac * (end - start)

    def _lambda_dist(self):
        return self._linear_warmup(
            start=self.dist_init_mult,
            end=self.dist_final_mult,
            warm_steps=6000,
        )


    def _lambda_orientation(self):
        return self._linear_warmup(
            start=self.orientation_loss_mult_s,
            end=self.orientation_loss_mult_f,
            warm_steps=6000,
        )


    def _lambda_normals_pred(self):
        return self._linear_warmup(
            start=self.predicted_normal_pred_loss_mult_s,
            end=self.predicted_normal_pred_loss_mult_f,
            warm_steps=30000,
        )

def norm8b(x):
    x = (x - x.min()) / (x.max() - x.min())
    return x

def norm8b255(x, mask = None, apply_colormap = False):

    if mask != None:
        x_masked = x[mask]
        x[mask] = (x_masked - x_masked.min()) / (x_masked.max() - x_masked.min())
    else:
        x = (x - x.min()) / (x.max() - x.min())

    device = x.device
    if apply_colormap:
        x = store_image.apply_colormap(x).to(device)

    if mask != None:
        x[~mask] = 0

    return (x * 255).to(torch.uint8)

