# ------------------------------------------------------------------------------------
# NeRF-Factory
# Copyright (c) 2022 POSTECH, KAIST, Kakao Brain Corp. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------

import os

import imageio
import numpy as np
from PIL import Image
import matplotlib as plt
import matplotlib.pyplot as plt2
import torch

def to8b(x):
    return (255 * np.clip(x, 0, 1)).astype(np.uint8)


def norm8b(x):
    x = (x - x.min()) / (x.max() - x.min())
    return to8b(x)


def apply_colormap(image, colormap_name='plasma'):

    if image.dim() == 3 and image.shape[2] == 3:
        # Convert RGB to grayscale by averaging channels
        grayscale = image.mean(dim=-1)
    else:
        raise ValueError("Image must be a 3-channel RGB image")


    #grayscale_normalized = (grayscale - grayscale.min()) / (grayscale.max() - grayscale.min())
    #mask = (grayscale_normalized > 0.1)

    #if mask.numel() > 0:
    #    grayscale_masked = grayscale_normalized[mask]
    #    grayscale_normalized[mask] = (grayscale_masked - grayscale_masked.min()) / (grayscale_masked.max() - grayscale_masked.min())

    cmap = plt.cm.get_cmap(colormap_name)
    
    # Apply the colormap by mapping the normalized grayscale values to colors+
    #colored_image = torch.zeros_like(grayscale_normalized)
    colored_image = cmap(grayscale.cpu().numpy())  # This returns RGBA values
    colored_image = torch.from_numpy(colored_image[..., :3])  # Drop the alpha channel and convert back to tensor
    #colored_image[~mask.unsqueeze(-1).repeat(1,1,3)] = 0

    return colored_image




def store_image_colormap(dirpath, rgbs):
    for (i, rgb) in enumerate(rgbs):
        imgname = f"image{str(i).zfill(3)}.png"

        rgb = apply_colormap(rgb)
        
        fig, ax = plt2.subplots(1, 1, figsize=(10,10))
        im=plt2.imshow(rgb, cmap='plasma')
        cbar = plt2.colorbar(im, ax=ax)
        imgpath = os.path.join(dirpath, imgname)

        plt2.savefig(imgpath)

def store_image_colormap(dirpath, rgbs):
    for (i, rgb) in enumerate(rgbs):
        store_single_image_colormap(dirpath, i, rgb)

def store_single_image_colormap(dirpath, i, rgb):

    imgname = f"image{str(i).zfill(3)}.png"

    rgb = apply_colormap(rgb)
    
    fig, ax = plt2.subplots(1, 1, figsize=(10,10))
    im=plt2.imshow(rgb, cmap='plasma')
    cbar = plt2.colorbar(im, ax=ax)
    imgpath = os.path.join(dirpath, imgname)

    plt2.savefig(imgpath)


def store_image(dirpath, rgbs, add_colormap = False):
    for (i, rgb) in enumerate(rgbs):
        store_single_image(dirpath, i, rgb, add_colormap)


def store_single_image(dirpath, i, rgb, add_colormap = False):

    imgname = f"image{str(i).zfill(3)}.png"
    if add_colormap:
        rgb = apply_colormap(rgb)

    rgbimg = Image.fromarray(to8b(rgb.detach().cpu().numpy()))

    imgpath = os.path.join(dirpath, imgname)
    rgbimg.save(imgpath)

def store_video(dirpath, rgbs, depths):
    rgbimgs = [to8b(rgb.cpu().detach().numpy()) for rgb in rgbs]
    video_dir = os.path.join(dirpath, "videos")
    os.makedirs(video_dir, exist_ok=True)
    imageio.mimwrite(os.path.join(video_dir, "images.mp4"), rgbimgs, fps=20, quality=8)
