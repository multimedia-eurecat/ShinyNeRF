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

import json
import os

import gdown
import imageio
import numpy as np
import torch
from torchvision import transforms as T
import skimage.transform as st
import OpenEXR
from imageio.core.util import Array


trans_t = lambda t: torch.tensor(
    [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, t], [0, 0, 0, 1]]
).float()

rot_phi = lambda phi: torch.tensor(
    [
        [1, 0, 0, 0],
        [0, np.cos(phi), -np.sin(phi), 0],
        [0, np.sin(phi), np.cos(phi), 0],
        [0, 0, 0, 1],
    ]
).float()

rot_theta = lambda th: torch.tensor(
    [
        [np.cos(th), 0, -np.sin(th), 0],
        [0, 1, 0, 0],
        [np.sin(th), 0, np.cos(th), 0],
        [0, 0, 0, 1],
    ]
).float()
tensor_transform = T.ToTensor()

def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi / 180.0 * np.pi) @ c2w
    c2w = rot_theta(theta / 180.0 * np.pi) @ c2w
    c2w = (
        torch.tensor([[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]).float()
        @ c2w
    )
    return c2w

def resize_images(
        img_downscale:int, 
        images:np.ndarray, 
        intrinsics: np.ndarray, 
        image_sizes: np.array
    ):
    for i in range(len(images)):
        
        img = images[i]
        intrinsics_img = intrinsics[i]
        [img_h, img_w] = image_sizes[i]

        # resize image
        img_w_n = img_w // img_downscale
        img_h_n = img_h // img_downscale


        img_new = st.resize(img, (img_w_n, img_h_n))
        #img_new = Image.fromarray(img).resize(size=(img_w_n, img_h_n))
        img_new = tensor_transform(img_new)

        # resize intrinsics
        K = np.zeros((3, 3), dtype=np.float32)
        img_w, img_h = int(intrinsics_img[0, 2]*2), int(intrinsics_img[1, 2]*2)
        img_w_, img_h_ = img_w//img_downscale, img_h//img_downscale
        K[0, 0] = intrinsics_img[0, 0]*img_w_/img_w # fx
        K[1, 1] = intrinsics_img[1, 1]*img_h_/img_h # fy
        K[0, 2] = intrinsics_img[0, 2]*img_w_/img_w # cx
        K[1, 2] = intrinsics_img[1, 2]*img_h_/img_h # cy
        K[2, 2] = 1

        images[i] = img_new
        intrinsics[i] = K
        image_sizes[i] = [img_h_n, img_w_n]

    return images, intrinsics, image_sizes

def resize(img_read, img_downscale):

    h, w = img_read.shape[:2]
    w = w // img_downscale
    h = h // img_downscale

    return st.resize(img_read, (h, w), anti_aliasing=True, preserve_range=True)

def load_shiny_blender_data(
    datadir: str,
    img_downscale: int,
    scene_name: str,
    train_skip: int,
    val_skip: int,
    test_skip: int,
    cam_scale_factor: float,
    white_bkgd: bool,
    radius: bool,
):
    basedir = os.path.join(datadir, scene_name).replace('\xa0', '')
    cam_trans = np.diag(np.array([1, -1, -1, 1], dtype=np.float32))
    splits = ["train", "val", "test"]
    metas = {}
    for s in splits:
        if s == "val":
            continue
        with open(os.path.join(basedir, "transforms_{}.json".format(s)), "r") as fp:
            metas[s] = json.load(fp)
            # Try test
            """
            if s == "train":
                metas[s]["frames"] = [metas[s]["frames"][0]]
            else:
                metas[s]["frames"] = metas[s]["frames"][:3]
            """
    metas["val"] = metas["test"]

    images = []
    normals = []
    distances = []
    extrinsics = []
    counts = [0]

    for s in splits:
        meta = metas[s]
        imgs = []
        poses = []
        norms = []
        alphas = []
        dists = []

        if s == "train":
            skip = train_skip
        elif s == "val":
            skip = val_skip
        elif s == "test":
            skip = test_skip

        for frame in meta["frames"][::skip]:
            img_fname = os.path.join(basedir, frame["file_path"] + ".png")
            norm_fname = os.path.join(basedir, frame["file_path"] + "_normal.png")
            dist_fname = os.path.join(basedir, frame["file_path"] + "_alpha.exr")
            if not os.path.exists(dist_fname):
                dist_fname = os.path.join(basedir, frame["file_path"] + "_alpha.png")


            img_read = imageio.imread(img_fname)
            imgs.append(resize(img_read, img_downscale))


            norm_read = imageio.imread(norm_fname)
            norms.append(resize(norm_read, img_downscale))


            if dist_fname[-3:] == "exr":
                dist_read = OpenEXR.InputFile(dist_fname)

                header = dist_read.header()
                dw = header['dataWindow']
                width = dw.max.x - dw.min.x + 1
                height = dw.max.y - dw.min.y + 1

                channels = ['R', 'G', 'B']

                # Read channel data and reshape it into an array
                pixels = [np.frombuffer(dist_read.channel(c), dtype=np.float32) for c in channels]
                reshaped_channels  = [p.reshape((height, width)) for p in pixels]

                # Combine channels into an RGB image
                dist_read = np.stack(reshaped_channels, axis=-1)
                dist_read = Array(dist_read)

            else:
                dist_read = imageio.imread(dist_fname)

                if len(dist_read.shape) == 3:
                    dist_read = dist_read[:,:,:-1]
                if len(dist_read.shape) == 2:
                    dist_read =  np.repeat(np.expand_dims(dist_read, axis=-1), repeats=3, axis=-1)


            dists.append(resize(dist_read, img_downscale))
            
            poses.append(np.array(frame["transform_matrix"]))
        
        imgs = (np.array(imgs) / 255.0).astype(np.float32)
        alphas = imgs[..., -1:]
        norms = np.array(norms).astype(np.float32)[..., :3] * 2.0 / 255.0 - 1.0

        # Concatenate normals and alphas for computing MAE
        norms = np.concatenate([norms, alphas], axis=-1)
        
        # Contatenate distance and alphas for computing error distance
        dists = np.concatenate([dists, alphas], axis=-1)
        if not os.path.exists(dist_fname):
            dists = dists / 255.0 * 8.0 # It has to match the max depth 
        
        poses = np.array(poses).astype(np.float32)
        counts.append(counts[-1] + imgs.shape[0])
        images.append(imgs)
        normals.append(norms)
        distances.append(dists)

        extrinsics.append(poses)

    i_split = [np.arange(counts[i], counts[i + 1]) for i in range(3)]

    images = np.concatenate(images, 0)
    normals = np.concatenate(normals, 0)
    distances = np.concatenate(distances, 0)

    extrinsics = np.concatenate(extrinsics, 0)

    extrinsics[:, :3, 3] *= cam_scale_factor
    extrinsics = extrinsics @ cam_trans

    h, w = imgs[0].shape[:2]
    num_frame = len(extrinsics)
    i_split += [np.arange(num_frame)]

    if "camera_angle_x" in meta:
        camera_angle_x = float(meta["camera_angle_x"])
        focal_x = 0.5 * w / np.tan(0.5 * camera_angle_x)
        focal_y = focal_x
    else: # for datasets created with nerfstudio
        focal_x = float(meta["fl_x"])
        focal_y = float(meta["fl_y"])
    
    intrinsics = np.array(
        [
            [[focal_x, 0.0, 0.5 * w], [0.0, focal_y, 0.5 * h], [0.0, 0.0, 1.0]]
            for _ in range(num_frame)
        ]
    )
    image_sizes = np.array([[h, w] for _ in range(num_frame)])

    render_poses = torch.stack(
        [
            pose_spherical(angle, -30.0, 4.0) @ cam_trans
            for angle in np.linspace(-180, 180, 40 + 1)[:-1]
        ],
        0,
    )
    render_poses[:, :3, 3] *= cam_scale_factor
    

    if white_bkgd:
        images = images[..., :3] * images[..., -1:] + (1.0 - images[..., -1:])
    else:
        images = images[..., :3]

    if radius:
        rad = 3
        return (
            images,
            intrinsics,
            extrinsics,
            image_sizes,
            rad,
            (-1, -1),
            i_split,
            render_poses,
            normals,
            distances,
            )
    else:
        near = 2.0 #2.0
        far = 6.0 #12.0 #6.0
        return (
            images,
            intrinsics,
            extrinsics,
            image_sizes,
            near,
            far,
            (-1, -1),
            i_split,
            render_poses,
            normals,
            distances,
        )
