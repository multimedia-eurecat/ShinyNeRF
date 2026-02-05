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
import imageio

import gin
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.functional
import torch.nn.init as init
import math

import src.model.refnerf.helper as helper
import src.model.refnerf.ref_utils as ref_utils
import utils.store_image as store_image
from src.model.interface import LitModel

from skimage import io
import tifffile as tiff


@gin.configurable()
class RefNeRFMLP(nn.Module):
    def __init__(
        self,
        deg_view,
        min_deg_point: int = 0,
        max_deg_point: int = 16,
        netdepth: int = 8,
        netwidth: int = 256,
        bottleneck_width: int = 128,
        netdepth_viewdirs: int = 8,
        netwidth_viewdirs: int = 256,
        # net_activation: Callable[..., Any] = nn.ReLU(),
        skip_layer: int = 4,
        skip_layer_dir: int = 4,
        perturb: float = 1.0,
        input_ch: int = 3,
        input_ch_view: int = 3,
        num_rgb_channels: int = 3,
        num_density_channels: int = 1,
        num_roughness_channels: int = 1,
        # roughness_activation: Callable[..., Any] = nn.Softplus(),
        roughness_bias: float = -1.0,
        bottleneck_noise: float = 0.0,
        # density_activation: Callable[..., Any] = nn.Softplus(),
        density_bias: float = -1.0,
        density_noise: float = 0.0,
        rgb_premultiplier: float = 1.0,
        # rgb_activation: Callable[..., Any] = nn.Sigmoid(),
        rgb_bias: float = 0.0,
        rgb_padding: float = 0.001,
        num_normal_channels: int = 3,
        num_tint_channels: int = 3,
        # tint_activation: Callable[..., Any] = nn.Sigmoid(),
    ):
        for name, value in vars().items():
            if name not in ["self", "__class__"]:
                setattr(self, name, value)

        super(RefNeRFMLP, self).__init__()

        self.dir_enc_fn = ref_utils.generate_ide_fn(self.deg_view)
        self.net_activation = nn.ReLU()
        self.roughness_activation = nn.Softplus()
        self.density_activation = nn.Softplus()
        self.rgb_activation = nn.Sigmoid()
        self.tint_activation = nn.Sigmoid()

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

        self.views_linear = nn.ModuleList(views_linear)

        self.bottleneck_layer = nn.Linear(self.netwidth, self.bottleneck_width)
        self.density_layer = nn.Linear(self.netwidth, num_density_channels)
        self.rgb_layer = nn.Linear(self.netwidth_viewdirs, num_rgb_channels)

        self.normal_layer = nn.Linear(self.netwidth, self.num_normal_channels)
        self.rgb_diffuse_layer = nn.Linear(self.netwidth, self.num_rgb_channels)
        self.tint_layer = nn.Linear(self.netwidth, self.num_tint_channels)
        self.roughness_layer = nn.Linear(self.netwidth, self.num_roughness_channels)

        init.xavier_uniform_(self.bottleneck_layer.weight)
        init.xavier_uniform_(self.density_layer.weight)
        init.xavier_uniform_(self.normal_layer.weight)
        init.xavier_uniform_(self.rgb_diffuse_layer.weight)
        init.xavier_uniform_(self.tint_layer.weight)
        init.xavier_uniform_(self.roughness_layer.weight)

    """
    x: torch.Tensor, [batch, num_samples, feature]
    viewdirs: torch.Tensor, [batch, viewdirs]
    """

    def forward(self, samples, viewdirs, current_epoch):


        means, covs = samples

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

            raw_density = self.density_layer(x)

            raw_density_grad = torch.autograd.grad(
                outputs=raw_density.sum(), inputs=means, retain_graph=True
            )[0]

            raw_density_grad = raw_density_grad.reshape(
                -1, num_samples, self.num_normal_channels
            )

            normals = -ref_utils.l2_normalize(raw_density_grad)
            means.detach()
        density = self.density_activation(raw_density + self.density_bias)
        density = density.reshape(-1, num_samples, self.num_density_channels)


        grad_pred = self.normal_layer(x).reshape(
            -1, num_samples, self.num_normal_channels
        )
        normals_pred = -ref_utils.l2_normalize(grad_pred)
        normals_to_use = normals_pred

        raw_rgb_diffuse = self.rgb_diffuse_layer(x)

        tint = self.tint_layer(x)
        tint = self.tint_activation(tint)

        raw_roughness = self.roughness_layer(x)
        roughness = self.roughness_activation(raw_roughness + self.roughness_bias)
        roughness = roughness.reshape(-1, num_samples, self.num_roughness_channels)

        bottleneck = self.bottleneck_layer(x)

        bottleneck += self.bottleneck_noise * torch.randn_like(bottleneck)
        #bottleneck += exponential_decay(self.bottleneck_noise, current_epoch) * torch.randn_like(bottleneck)
        bottleneck = bottleneck.reshape(-1, num_samples, self.bottleneck_width)
        
        refdirs = ref_utils.reflect(-viewdirs[..., None, :], normals_to_use)
        dir_enc = self.dir_enc_fn(refdirs, roughness)

        dotprod = torch.sum(
            normals_to_use * viewdirs[..., None, :], dim=-1, keepdims=True
        )

        x = torch.cat([bottleneck, dir_enc, dotprod], dim=-1)
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

        return dict(
            rgb=rgb,
            density=density,
            normals=normals,
            normals_pred=normals_pred,
            roughness=roughness,
            diffuse=diffuse_linear.reshape(-1, num_samples, self.num_rgb_channels),
            specular=specular_linear.reshape(-1, num_samples, self.num_rgb_channels),
            tint=tint.reshape(-1, num_samples, self.num_rgb_channels),
            specular_rgb=rgb_spec.reshape(-1, num_samples, self.num_rgb_channels),
        )


@gin.configurable()
class RefNeRF(nn.Module):
    def __init__(
        self,
        num_samples: int = 128,
        num_levels: int = 2,
        resample_padding: float = 0.01,
        stop_level_grad: bool = True,
        use_viewdirs: bool = True,
        lindisp: bool = False,
        ray_shape: str = "cone",
        deg_view: int = 5,
        rgb_padding: float = 0.001,
    ):
        # Layers
        for name, value in vars().items():
            if name not in ["self", "__class__"]:
                setattr(self, name, value)

        super(RefNeRF, self).__init__()

        self.mlp = RefNeRFMLP(self.deg_view)

    def forward(self, rays, randomized, white_bkgd, near, far, radius, current_epoch):

        ret = []
        for i_level in range(self.num_levels):

            if near is None:
                distance_to_center = torch.sqrt(rays["rays_o"].T[0]**2 + rays["rays_o"].T[1]**2 + rays["rays_o"].T[2]**2)
                near = distance_to_center - radius + 1
                far = distance_to_center + 3

            if i_level == 0:
                t_vals, samples = helper.sample_along_rays(
                    rays_o=rays["rays_o"],
                    rays_d=rays["rays_d"],
                    radii=rays["radii"],
                    num_samples=self.num_samples,
                    near=near,
                    far=far,
                    randomized=randomized,
                    lindisp=self.lindisp,
                    ray_shape=self.ray_shape,
                )
            else:
                t_vals, samples = helper.resample_along_rays(
                    rays_o=rays["rays_o"],
                    rays_d=rays["rays_d"],
                    radii=rays["radii"],
                    t_vals=t_vals,
                    weights=weights,
                    randomized=randomized,
                    ray_shape=self.ray_shape,
                    stop_level_grad=self.stop_level_grad,
                    resample_padding=self.resample_padding,
                )


            ray_results = self.mlp(samples, rays["viewdirs"], current_epoch)
            comp_rgb, distance, distance_norm, acc, weights = helper.volumetric_rendering(
                ray_results["rgb"],
                ray_results["density"],
                t_vals,
                rays["rays_d"],
                white_bkgd=white_bkgd,
            )

            comp_diffuse, _, _, _, _ = helper.volumetric_rendering(
                ray_results["diffuse"],
                ray_results["density"],
                t_vals,
                rays["rays_d"],
                white_bkgd=white_bkgd,
            )

            comp_specular, _, _, _, _ = helper.volumetric_rendering(
                ray_results["specular"],
                ray_results["density"],
                t_vals,
                rays["rays_d"],
                white_bkgd=white_bkgd,
            )


            comp_tint, _, _, _, _ = helper.volumetric_rendering(
                ray_results["tint"],
                ray_results["density"],
                t_vals,
                rays["rays_d"],
                white_bkgd=white_bkgd,
            )


            comp_specular_rgb, _, _, _, _ = helper.volumetric_rendering(
                ray_results["specular_rgb"],
                ray_results["density"],
                t_vals,
                rays["rays_d"],
                white_bkgd=white_bkgd,
            )


            rendered_result = ray_results
            rendered_result["comp_rgb"] = comp_rgb
            rendered_result["distance"] = distance
            rendered_result["distance_norm"] = distance_norm
            rendered_result["acc"] = acc
            rendered_result["weights"] = weights
            rendered_result["t_vals"] = t_vals
            rendered_result["diffuse"] = comp_diffuse
            rendered_result["specular"] = comp_specular
            rendered_result["tint"] = comp_tint
            rendered_result["specular_rgb"] = comp_specular_rgb

            ret.append(rendered_result)

        return ret


@gin.configurable()
class LitRefNeRF(LitModel):
    def __init__(
        self,
        lr_init: float = 5.0e-4,
        lr_final: float = 5.0e-6,
        lr_delay_steps: int = 2500,
        lr_delay_mult: float = 0.01,
        coarse_loss_mult: float = 0.1,
        randomized: bool = True,
        orientation_loss_mult: float = 0.1,
        orientation_coarse_loss_mult: float = 0.01,
        predicted_normal_loss_mult: float = 3e-4,
        predicted_normal_coarse_loss_mult: float = 3e-5,
        compute_normal_metrics: bool = False,
        grad_max_norm: float = 0.001,
    ):
    
        for name, value in vars().items():
            if name not in ["self", "__class__"]:
                setattr(self, name, value)

        super(LitRefNeRF, self).__init__()
        self.model = RefNeRF()
        self.first_number_test = -1

    def setup(self, stage):
        if hasattr(self.trainer.datamodule, 'near'):
            print("near", self.trainer.datamodule.near)
            print("far", self.trainer.datamodule.far)
            self.near = self.trainer.datamodule.near
            self.far = self.trainer.datamodule.far
            self.radius = None
        else:
            self.radius = self.trainer.datamodule.radius
            self.near = None
            self.far = None

        self.white_bkgd = self.trainer.datamodule.white_bkgd


    def save_val_images(self, target_img_idx, val_dataloader, path):

        with torch.no_grad():

            # (1) Run model for each batch of rays corresponding to the target image
            output, image_indx = [], []
            
            for _, val_batch in enumerate(val_dataloader):
                if target_img_idx in torch.unique(val_batch['image_indx']):
                    for k in val_batch:
                        val_batch[k] = val_batch[k].to(self.device)
                    #_output = self.render_rays_one_pict(val_batch, 0, 'test')
                    rendered_results = self.model(
                        val_batch, False, self.white_bkgd, self.near, self.far, self.radius, 100000
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
            size_img = int(math.sqrt(x.shape[0])) # must be square
            if x.ndim == 3:
                joined[kk] = x.view(size_img,size_img,x.shape[-2], x.shape[-1]) # TODO change size_img
            else:
                joined[kk] = x.view(size_img,size_img,x.shape[-1]) # TODO change size_img

        # (3) Save output images
        self.save_img(joined, path)



    def calculate_var(self,rendered_results, batch_input, mode):
        ret = {}
  
        ret["rgb"] = rendered_results[1]["comp_rgb"]

        ret["diffuse"] = rendered_results[1]["diffuse"]
        ret["specular"] = rendered_results[1]["specular"]
        ret["tint"] = rendered_results[1]["tint"]
        ret["specular_rgb"] = rendered_results[1]["specular_rgb"]

        normals_deriv = (rendered_results[1]["weights"][..., None] * rendered_results[1]["normals"]).sum(1)
        ret["comp_normals"] = ref_utils.l2_normalize(normals_deriv)
        normals_pred_deriv = (rendered_results[1]["weights"][..., None] * rendered_results[1]["normals_pred"]).sum(1)
        ret["comp_normals_pred"] = ref_utils.l2_normalize(normals_pred_deriv)
        
        ret["roughness"] = (rendered_results[1]["weights"][..., None] * rendered_results[1]["roughness"]).sum(1)

        if mode == 'test':
            ret["distance"] = rendered_results[1]["distance"].cpu()[:, np.newaxis]
            ret["distance_norm"] = rendered_results[1]["distance_norm"].cpu()[:, np.newaxis]

        if self.compute_normal_metrics:
            ret["normals"] = rendered_results[1]["normals"]
            ret["normals_pred"] = rendered_results[1]["normals_pred"]
            ret["weights"] = rendered_results[1]["weights"]
            ret["gt_normals"] = batch_input['normals']
            ret["gt_distance"] = batch_input['distances']



        ret["weights"] = rendered_results[1]["weights"]

        ret["t_vals"] = rendered_results[1]["t_vals"]



        return ret

    def training_step(self, batch, batch_idx):

        rendered_results = self.model(
            batch, self.randomized, self.white_bkgd, self.near, self.far, self.radius, self.current_epoch
        )

        rgb_coarse = rendered_results[0]["comp_rgb"]
        rgb_fine = rendered_results[1]["comp_rgb"]
        target = batch["target"]

        loss0 = helper.img2mse(rgb_coarse, target)
        loss1 = helper.img2mse(rgb_fine, target)
        loss = loss1 + loss0 * self.coarse_loss_mult

        if self.compute_normal_metrics:
            normal_mae = self.compute_normal_mae(rendered_results[1], batch["normals"])
            self.log("train/normal_mae_grad", normal_mae, on_step=True)

            normal_mae = self.compute_normal_mae(rendered_results[1], batch["normals"], True)
            self.log("train/normal_mae_pred", normal_mae, on_step=True)

            self.log("train/mde_no_back", self.compute_distance_mde(rendered_results[1], batch["distances"].cpu()))
            self.log("train/mde_back", self.compute_distance_mde(rendered_results[1], batch["distances"].cpu(), error_plus_background=True))


        if self.orientation_coarse_loss_mult > 0 or self.orientation_loss_mult > 0:
            orientation_loss = self.orientation_loss(
                rendered_results, batch["viewdirs"]
            )
            self.log("train/orientation_loss", orientation_loss, on_step=True)
            loss += orientation_loss

        if (
            self.predicted_normal_coarse_loss_mult > 0
            or self.predicted_normal_loss_mult > 0
        ):
            pred_normal_loss = self.predicted_normal_loss(rendered_results)
            self.log("train/pred_normal_loss", pred_normal_loss, on_step=True)
            loss += pred_normal_loss

        psnr0 = helper.mse2psnr(loss0)
        psnr1 = helper.mse2psnr(loss1)

        self.log("train/psnr1", psnr1, on_step=True, prog_bar=True, logger=True)
        self.log("train/psnr0", psnr0, on_step=True, prog_bar=True, logger=True)
        self.log("train/loss", loss, on_step=True)

        return loss


    def render_rays_val(self, batch, batch_idx):
        ret = {}
        rendered_results = self.model(
            batch, False, self.white_bkgd, self.near, self.far, self.radius, self.current_epoch
        )
        rgb_fine = rendered_results[1]["comp_rgb"]
        target = batch["target"]
        ret["target"] = target
        ret["rgb"] = rgb_fine

        normals_deriv = (rendered_results[1]["weights"][..., None] * rendered_results[1]["normals"]).sum(1)
        ret["comp_normals"] = ref_utils.l2_normalize(normals_deriv)
        normals_pred_deriv = (rendered_results[1]["weights"][..., None] * rendered_results[1]["normals_pred"]).sum(1)
        ret["comp_normals_pred"] = ref_utils.l2_normalize(normals_pred_deriv)
        
    
        ret["distance_norm"] = rendered_results[1]["distance_norm"].cpu()[:, np.newaxis]

        if self.compute_normal_metrics:
            ret["sum_weights"] =  rendered_results[1]["weights"].sum()
            ret["mae_grad"] = torch.Tensor([self.compute_normal_mae(rendered_results[1], batch["normals"])])
            ret["mae_pred"] = torch.Tensor([self.compute_normal_mae(rendered_results[1], batch["normals"], True)])
            ret["mde_no_back"] =  self.compute_distance_mde(rendered_results[1], batch["distances"].cpu())
            ret["mde_back"] =  self.compute_distance_mde(rendered_results[1], batch["distances"].cpu(), error_plus_background=True)

        # if mode == 'test':
        #     ret["weights"] = rendered_results[1]["weights"]

        #     ret["t_vals"] = rendered_results[1]["t_vals"]

        ret["gt_normals"] =  batch["normals"]

        #ret["normals_pred"] = rendered_results[1]["normals_pred"]

        return ret


    def render_rays_test(self, batch, batch_idx):
        ret = {}
        rendered_results = self.model(
            batch, False, self.white_bkgd, self.near, self.far, self.radius, 100000
        )
        rgb_fine = rendered_results[1]["comp_rgb"]
        target = batch["target"]
        ret["target"] = target
        ret["rgb"] = rgb_fine

        ret["gt_normals"] =  batch["normals"]

        #self.point_cloud(batch, rendered_results)

        if self.compute_normal_metrics:
            ret["sum_weights"] =  rendered_results[1]["weights"].sum()
            ret["mae_grad"] = torch.Tensor([self.compute_normal_mae(rendered_results[1], batch["normals"])])
            ret["mae_pred"] = torch.Tensor([self.compute_normal_mae(rendered_results[1], batch["normals"], True)])
            ret["mde_no_back"] =  self.compute_distance_mde(rendered_results[1], batch["distances"].cpu())
            ret["mde_back"] =  self.compute_distance_mde(rendered_results[1], batch["distances"].cpu(), error_plus_background=True)

        #ret["normals_pred"] = rendered_results[1]["normals_pred"]

        return ret

    def point_cloud(self, batch, rendered_results):


        # Define the file path
        file_path = "./point_cloud.npz"

        import pdb; pdb.set_trace()
        points = batch

        normals_deriv = (rendered_results[1]["weights"][..., None] * rendered_results[1]["normals"]).sum(1)
        normals = ref_utils.l2_normalize(normals_deriv)


        # Convert PyTorch tensors to NumPy
        np_points = points.numpy()
        np_normals = normals.numpy()

        # Check if NPZ file exists
        if os.path.exists(file_path):
            # Load the existing NPZ file
            data = np.load(file_path)
            loaded_points = np.cat([data["points"], np_points], dim=-1)
            loaded_normals = np.cat([data["normals"], np_normals], dim=-1)



        # Save to NPZ
        np.savez(file_path, points=loaded_points, normals=loaded_normals)
        print(f"Saved point cloud to {file_path}")

        # Load to verify
        data = np.load(file_path)



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
            print("saving image", id_batch)
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
        path_norm = os.path.join(self.logdir, "val_norm_tiff")
        os.makedirs(path_rgb, exist_ok=True)
        os.makedirs(path_norm, exist_ok=True)

        val_image_sizes = self.trainer.datamodule.val_image_sizes
        rgbs = self.alter_gather_cat(outputs, "rgb", val_image_sizes)
        targets = self.alter_gather_cat(outputs, "target", val_image_sizes)
        
        comp_normals = self.alter_gather_cat(outputs, "comp_normals", val_image_sizes)
        comp_normals_pred = self.alter_gather_cat(outputs, "comp_normals_pred", val_image_sizes)
        
       
        psnr_mean = self.psnr_each(rgbs, targets).mean()
        ssim_mean = self.ssim_each(rgbs, targets).mean()
        msssim_mean = self.msssim_each(rgbs, targets).mean()
        #gmsd_mean = self.gmsd_each(rgbs, targets).mean()
        #msgmsd_mean = self.msgmsd_each(rgbs, targets).mean()
        #mdsi_mean = self.mdsi_each(rgbs, targets).mean()
        #haarpsi_mean = self.haarpsi_each(rgbs, targets).mean()
        #vsi_mean = self.vsi_each(rgbs, targets).mean()
        #fsim_mean = self.fsim_each(rgbs, targets).mean()
        lpips_mean = self.lpips_each(rgbs, targets).mean()
        #fid_mean = self.fid_each(rgbs, targets).mean()
        self.log("val/psnr", psnr_mean.item(), on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val/ssim", ssim_mean.item(), on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val/msssim", msssim_mean.item(), on_epoch=True,  prog_bar=True, sync_dist=True)
        #self.log("val/gmsd", gmsd_mean.item(), on_epoch=True,  prog_bar=True, sync_dist=True)
        #self.log("val/msgmsd", msgmsd_mean.item(), on_epoch=True,  prog_bar=True, sync_dist=True)
        #self.log("val/mdsi", mdsi_mean.item(), on_epoch=True,  prog_bar=True, sync_dist=True)
        #self.log("val/haarpsi", haarpsi_mean.item(), on_epoch=True,  prog_bar=True, sync_dist=True)
        #self.log("val/vsi", vsi_mean.item(), on_epoch=True,  prog_bar=True, sync_dist=True)
        #self.log("val/fsim", fsim_mean.item(), on_epoch=True,  prog_bar=True, sync_dist=True)
        self.log("val/lpips", lpips_mean.item(), on_epoch=True, prog_bar=True, sync_dist=True)
        #self.log("val/fid", fid_mean.item(), on_epoch=True,  prog_bar=True, sync_dist=True)

        rr = rgbs[0].permute(2,0,1)
        tt = targets[0].permute(2,0,1)
        stack_rgb =  torch.stack([rr, tt])

        gt_normals_aplha = self.alter_gather_cat(outputs, "gt_normals", val_image_sizes)
        
        gt_normals = [m[:,:,:3] for m in gt_normals_aplha]

        self.logger.experiment.add_images("val/images (Pred - GT)", stack_rgb, global_step=self.trainer.global_step)
        rr = rgbs[len(rgbs)//2].permute(2,0,1)
        tt = targets[len(rgbs)//2].permute(2,0,1)
        stack_rgb =  torch.stack([rr, tt])

        self.logger.experiment.add_images("val/images (Pred - GT) 2", stack_rgb, global_step=self.trainer.global_step)


        path = os.path.join(path_rgb , f"stack_rgb_{self.current_epoch}.tif")
        io.imsave(path, stack_rgb.cpu().numpy())

        norm = comp_normals[0].permute(2,0,1)
        norm_pred = comp_normals_pred[0].permute(2,0,1)
        gt_normal = gt_normals[0].permute(2,0,1)
        stack_norm =  torch.stack([norm_pred, norm, gt_normal])

        self.logger.experiment.add_images("val/norms (Pred - Density - GT)", norm8b(stack_norm), global_step=self.trainer.global_step)

        norm = comp_normals[len(comp_normals)//2].permute(2,0,1)
        norm_pred = comp_normals_pred[len(comp_normals)//2].permute(2,0,1)
        gt_normal = gt_normals[len(comp_normals)//2].permute(2,0,1)
        stack_norm =  torch.stack([norm_pred, norm, gt_normal])

        self.logger.experiment.add_images("val/norms (Pred - Density - GT) 2", norm8b(stack_norm), global_step=self.trainer.global_step)
        path = os.path.join(path_norm, f"stack_norm_{self.current_epoch}.tif")
        io.imsave(path, stack_norm.cpu().numpy())

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
        #msssim = self.msssim(rgbs, targets, dmodule.i_train, dmodule.i_val, dmodule.i_test)
        #gmsd = self.gmsd(rgbs, targets, dmodule.i_train, dmodule.i_val, dmodule.i_test)
        #msgmsd = self.msgmsd(rgbs, targets, dmodule.i_train, dmodule.i_val, dmodule.i_test)
        #mdsi = self.mdsi(rgbs, targets, dmodule.i_train, dmodule.i_val, dmodule.i_test)
        #haarpsi = self.haarpsi(rgbs, targets, dmodule.i_train, dmodule.i_val, dmodule.i_test)
        #vsi = self.vsi(rgbs, targets, dmodule.i_train, dmodule.i_val, dmodule.i_test)
        #fsim = self.fsim(rgbs, targets, dmodule.i_train, dmodule.i_val, dmodule.i_test)
        lpips = self.lpips(
            rgbs, targets, dmodule.i_train, dmodule.i_val, dmodule.i_test
        )
        #fid = self.fid(rgbs, targets, dmodule.i_train, dmodule.i_val, dmodule.i_test)

        self.log("test/psnr", psnr["test"], on_epoch=True)
        self.log("test/ssim", ssim["test"], on_epoch=True)
        #self.log("test/msssim", msssim["test"], on_epoch=True)
        #self.log("test/gmsd", gmsd["test"], on_epoch=True)
        #self.log("test/msgmsd", msgmsd["test"], on_epoch=True)
        #self.log("test/mdsi", mdsi["test"], on_epoch=True)
        #self.log("test/haarpsi", haarpsi["test"], on_epoch=True)
        #self.log("test/vsi", vsi["test"], on_epoch=True)
        #self.log("test/fsim", fsim["test"], on_epoch=True)
        self.log("test/lpips", lpips["test"], on_epoch=True)
        #self.log("test/fid", fid["test"], on_epoch=True)
  
        result_path = os.path.join(self.logdir, "results.json")
        self.write_stats(result_path, psnr, ssim, lpips) #msssim, gmsd, msgmsd, mdsi, haarpsi, vsi, fsim,  fid

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

        return psnr, ssim, lpips # msssim, gmsd, msgmsd, mdsi, haarpsi, vsi, fsim, fid



    def save_single_file(self, image_dir, folder,  i, img, colormap=False):
        full_path = os.path.join(image_dir, folder)
        os.makedirs(full_path, exist_ok=True)
        if not colormap:
            store_image.store_single_image(full_path, i, img)
        else:
            store_image.store_single_image_colormap(full_path, i, img)

    def save_single_file_tiff(self, image_dir, folder,  i, img):
        full_path = os.path.join(image_dir, folder)
        os.makedirs(full_path, exist_ok=True)
        img_array = img.cpu().numpy().astype(np.float32)
        tiff.imwrite(f'{full_path}/{i:03d}.tif', img_array)

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
        self.save_single_file_tiff(image_dir, 'specular_rgb_tif',  i, specular_rgb)
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


        distances = output['distance']
        self.save_single_file_tiff(image_dir, 'distances_tif',  i, distances)
        dist_3d = output['distance_norm'].repeat(1,1,3)
        self.save_single_file(image_dir, 'distance',  i, dist_3d, True)
        del distances
        del dist_3d

        roughness = output['roughness']
        self.save_single_file_tiff(image_dir, 'roughness_tif',  i, roughness)
        roughness = (roughness - roughness.min()) / (roughness.max() - roughness.min())
        roughness_3d = roughness.repeat(1,1,3)
        self.save_single_file(image_dir, 'roguhness',  i, roughness_3d, True)
        del roughness


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


        # weights = output['weights'].cpu()
        # t_vals = output['t_vals'].cpu()
        # self.save_single_file_np(image_dir, 'weights_tif',  i, weights)
        # self.save_single_file_np(image_dir, 't_vals_tif',  i, t_vals)



    def orientation_loss(self, rendered_results, viewdirs):
        total_loss = 0.0
        for i, rendered_result in enumerate(rendered_results):
            w = rendered_result["weights"]
            n = rendered_result["normals_pred"]
            if n is None:
                raise ValueError("Normals cannot be None if orientation loss is on.")
            v = -1.0 * viewdirs
            n_dot_v = (n * v[..., None, :]).sum(axis=-1)
            loss = torch.mean(
                (w * torch.fmin(torch.zeros_like(n_dot_v), n_dot_v) ** 2).sum(dim=-1)
            )
            if i < self.model.num_levels - 1:
                total_loss += self.orientation_coarse_loss_mult * loss
            else:
                total_loss += self.orientation_loss_mult * loss
        return total_loss

    def predicted_normal_loss(self, rendered_results):
        total_loss = 0.0
        for i, rendered_result in enumerate(rendered_results):
            w = rendered_result["weights"]
            n = rendered_result["normals"]
            n_pred = rendered_result["normals_pred"]
            if n is None or n_pred is None:
                raise ValueError(
                    "Predicted normals and gradient normals cannot be None if "
                    "predicted normal loss is on."
                )
            loss = torch.mean((w * (1.0 - torch.sum(n * n_pred, dim=-1))).sum(dim=-1))
            if i < self.model.num_levels - 1:
                total_loss += self.predicted_normal_coarse_loss_mult * loss
            else:
                total_loss += self.predicted_normal_loss_mult * loss
        return total_loss

    def compute_normal_mae(self, rendered_results, normals, normals_pred=False, picture=False):
        normals_gt, alphas = torch.split(normals, [3, 1], dim=-1)

        weights = rendered_results["weights"] * alphas

        if picture:
            normalized_normals_gt = ref_utils.l2_normalize(normals_gt)
        else:
            normalized_normals_gt = ref_utils.l2_normalize(normals_gt[..., None, :])

        if normals_pred:
            normalized_normals = ref_utils.l2_normalize(rendered_results["normals_pred"])
        else:
            normalized_normals = ref_utils.l2_normalize(rendered_results["normals"])

        #normalized_normals = (normalized_normals + 1) / 2
        #normalized_normals_gt = (normalized_normals_gt + 1) / 2
        if picture:
            normal_mae = ref_utils.compute_weighted_mae_picture_2(
                normalized_normals, normalized_normals_gt
            ) * alphas.squeeze()
        else:
            normal_mae = ref_utils.compute_weighted_mae(
                weights, normalized_normals, normalized_normals_gt
            )

        return normal_mae
    
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

    
def norm8b(x):
    x = (x - x.min()) / (x.max() - x.min())
    return x


import math

def exponential_decay(initial_value, current_epoch, decay_rate=0.3):
    return initial_value * math.exp(-decay_rate * current_epoch)
