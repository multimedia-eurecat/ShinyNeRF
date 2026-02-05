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
import cv2

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



def undistort_and_resize(image, camera_matrix, dist_coeffs, new_width, new_height):
    # Get the original image dimensions
    original_height, original_width = image.shape[:2]
    
    # Undistort the image
    new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(
        camera_matrix, dist_coeffs, (original_width, original_height), 1, (original_width, original_height)
    )
    undistorted_image = cv2.undistort(image, camera_matrix, dist_coeffs, None, new_camera_matrix)

    # Resize the undistorted image
    resized_image = resize(undistorted_image, 2)
    
    # Update the camera matrix for the resized image
    scale_x = new_width / original_width
    scale_y = new_height / original_height
    resized_camera_matrix = new_camera_matrix.copy()
    resized_camera_matrix[0, 0] *= scale_x  # Scale focal length x
    resized_camera_matrix[1, 1] *= scale_y  # Scale focal length y
    resized_camera_matrix[0, 2] *= scale_x  # Scale principal point x
    resized_camera_matrix[1, 2] *= scale_y  # Scale principal point y

    return resized_image, resized_camera_matrix




def load_shiny_blender_data_no_norm(
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
    metas["val"] = metas["test"]

    images = []
    mask_images = []
    normals = []
    distances = []
    extrinsics = []
    counts = [0]



    """
    meta = metas["train"]


    # Distortion coefficients
    k1 = meta["k1"]
    k2 = meta["k2"]
    p1 = meta["p1"]
    p2 = meta["p2"]

    dist_coeffs = np.array([k1, k2, p1, p2], dtype=np.float32)

    focal_x = float(meta["fl_x"]) * 1 / img_downscale
    focal_y = float(meta["fl_y"]) * 1 / img_downscale




    w = 800 * 1/ img_downscale
    h = 534 * 1/ img_downscale

    intrinsics = np.array([[focal_x, 0.0, 0.5 * w], [0.0, focal_y, 0.5 * h], [0.0, 0.0, 1.0]])
    """







    for s in splits:
        meta = metas[s]
        imgs = []
        img_masks = []
        poses = []

        if s == "train":
            skip = train_skip
        elif s == "val":
            skip = val_skip
        elif s == "test":
            skip = test_skip

        for frame in meta["frames"][::skip]:
            img_fname = os.path.join(basedir, frame["file_path"] + ".png")
            img_read = imageio.imread(img_fname)

            #undistorted_frame, resized_camera_matrix = undistort_and_resize(img_read, intrinsics, dist_coeffs, w, h)
            #imgs.append(undistorted_frame)

            imgs.append(resize(img_read, img_downscale))

            poses.append(np.array(frame["transform_matrix"]))


            img_fname = os.path.join(basedir, frame["file_path"] + "_mask.png")
            if os.path.exists(img_fname):
                mask_read = imageio.imread(img_fname)
                img_masks.append(resize(mask_read, img_downscale)[...,:1])


        
        imgs = (np.array(imgs) / 255.0).astype(np.float32)
        img_masks = (np.array(img_masks) / 255.0).astype(np.float32)
        img_masks = img_masks > 0.5

        # Concatenate normals and alphas for computing MAE
        poses = np.array(poses).astype(np.float32)
        counts.append(counts[-1] + imgs.shape[0])
        images.append(imgs)
        mask_images.append(img_masks)

        extrinsics.append(poses)

    i_split = [np.arange(counts[i], counts[i + 1]) for i in range(3)]

    images = np.concatenate(images, 0)
    if len(mask_images) > 0:
        mask_images = np.concatenate(mask_images, 0)
    

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
        focal_x = float(meta["fl_x"]) * 1 / img_downscale
        focal_y = float(meta["fl_y"]) * 1 / img_downscale

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

    distances = torch.ones((134, 270, 270, 4))

    if radius:
        rad = 3
        return (
            images,
            mask_images,
            intrinsics,
            extrinsics,
            image_sizes,
            rad,
            (-1, -1),
            i_split,
            render_poses,
            )
    else:
        near = 0.2 #0.2 #2.0 #0.5 #2.0 5.0
        far = 9.0 #9.0 #6.0 #4.5 #6.0 #12.0
        return (
            images,
            mask_images,
            intrinsics,
            extrinsics,
            image_sizes,
            near,
            far,
            (-1, -1),
            i_split,
            render_poses,
        )
