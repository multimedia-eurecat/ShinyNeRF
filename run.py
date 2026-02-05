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

import argparse
import logging
import os
import shutil
from typing import *
from pathlib import Path

import gin
import torch
from pytorch_lightning import Trainer
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    TQDMProgressBar,
)
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.strategies.ddp import DDPStrategy
from utils.select_option import select_callback, select_dataset, select_model


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise Exception("Boolean value expected.")


@gin.configurable()
def run(
    ginc: str,
    ginb: str,
    resume_training: bool,
    ckpt_path: Optional[str],
    scene_name: Optional[str],
    datadir: Optional[str] = None,
    logbase: Optional[str] = None,
    model_name: Optional[str] = None,
    dataset_name: Optional[str] = None,
    img_downscale: Optional[int] = 1,
    img_downscale_dataset: Optional[int] = 2,
    postfix: Optional[str] = None,
    entity: Optional[str] = None,
    # Optimization
    max_steps: int = -1,
    max_epochs: int = -1,
    precision: int = 32,
    # Logging
    log_every_n_steps: int = 1000,
    progressbar_refresh_rate: int = 5,
    # Run Mode
    run_train: bool = True,
    run_eval: bool = False,
    run_render: bool = False, #deprecated
    num_devices: Optional[int] = None,
    num_sanity_val_steps: int = 0, 
    seed: int = 777,
    debug: bool = False,
    save_last: bool = True,
    grad_max_norm=10e-3,
    grad_clip_algorithm="norm",
):

    logging.getLogger("lightning").setLevel(logging.ERROR)
    datadir = datadir.rstrip("/")

    exp_name = (
        model_name + "_" + dataset_name + "_" + scene_name +  "_res" + str(img_downscale) + "_" + str(seed).zfill(3)
    )
    if postfix is not None:
        exp_name += "_" + postfix
    if debug:
        exp_name += "_debug"

    if num_devices is None:
        num_devices = torch.cuda.device_count()

    if model_name in ["plenoxel"]:
        num_devices = 1

    if logbase is None:
        logbase = "logs"

    if run_train or run_eval or run_render:
        img_downscale_dataset = img_downscale

    os.makedirs(logbase, exist_ok=True)
    logdir = os.path.join(logbase, exp_name)
    os.makedirs(logdir, exist_ok=True)
    os.makedirs(os.path.join(logdir, exp_name), exist_ok=True)

    version_num = len(os.listdir(os.path.join(logdir, exp_name)))
    
    logger = pl_loggers.TensorBoardLogger(
        save_dir=logdir,
        name=exp_name,
        #version= version_num-1 if resume_training else version_num,
    )

    # Logging all parameters
    if run_train:
        txt_path = os.path.join(logdir, "config.gin")
        with open(txt_path, "w") as fp_txt:
            for config_path in ginc:
                fp_txt.write(f"Config from {config_path}\n\n")
                with open(config_path, "r") as fp_config:
                    readlines = fp_config.readlines()
                for line in readlines:
                    fp_txt.write(line)
                fp_txt.write("\n")

            fp_txt.write("\n### Binded options\n")
            for line in ginb:
                fp_txt.write(line + "\n")

    seed_everything(seed, workers=True)

    lr_monitor = LearningRateMonitor(logging_interval="step")
    model_checkpoint = ModelCheckpoint(
        monitor="val/psnr",
        dirpath=logdir,
        filename="best",
        save_top_k=1,
        mode="max",
        save_last=save_last,
    )
    tqdm_progrss = TQDMProgressBar(refresh_rate=progressbar_refresh_rate)

    callbacks = []
    if not model_name in ["plenoxel"]:
        callbacks.append(lr_monitor)
    callbacks += [model_checkpoint, tqdm_progrss]
    callbacks += select_callback(model_name)

    ddp_plugin = DDPStrategy(find_unused_parameters=False) if num_devices > 1 else None

    data_module = select_dataset(
        dataset_name=dataset_name,
        img_downscale=img_downscale_dataset,
        scene_name=scene_name,
        datadir=datadir,
    )

    #@RM: better to use > 0 to ensure eval works OK before training starts
    #n_val_images = len(data_module.i_val)
    #epoch_len = (n_val_images * im_size) // 
    num_sanity_val_steps = 0 #len(data_module)
    os.environ["TORCH_SHOW_CPP_STACKTRACE"] = "1"   # C++ side
    trainer = Trainer(
        #detect_anomaly = True, # debugging
        logger=logger if run_train else None,
        log_every_n_steps=log_every_n_steps,
        devices=num_devices,
        max_epochs=max_epochs,
        max_steps=max_steps,
        accelerator="gpu",
        replace_sampler_ddp=False,
        strategy="ddp",
        check_val_every_n_epoch=1,
        precision=precision,
        num_sanity_val_steps=num_sanity_val_steps,
        callbacks=callbacks,
        gradient_clip_algorithm=grad_clip_algorithm,
        gradient_clip_val=grad_max_norm,
        inference_mode=False,
        #sync_batchnorm=True,
    )

    if resume_training:
        if ckpt_path is None:
            ckpt_path = f"{logdir}/last.ckpt"

    model = select_model(model_name=model_name)
    model.logdir = logdir
    if run_train:
        """best_ckpt = os.path.join(logdir, "best.ckpt")
        if os.path.exists(best_ckpt) and not resume_training :
            os.remove(best_ckpt)
        version0 = os.path.join(logdir, exp_name, "version_0")
        if os.path.exists(version0) and not resume_training:
            shutil.rmtree(version0, True)
        """

        source = Path("src/model/shinynerf/model.py")
        dest_dir = Path(logdir)
        dest_dir.mkdir(parents=True, exist_ok=True)   # create logdir if it doesn’t exist

        shutil.copy2(source, dest_dir / source.name)

        print(f"✓ Copied {source} to {dest_dir / source.name}")


        
        source = Path("src/model/shinynerf/shiny_utils.py")
        dest_dir = Path(logdir)
        dest_dir.mkdir(parents=True, exist_ok=True)

        shutil.copy2(source, dest_dir / source.name)

        print(f"✓ Copied {source} to {dest_dir / source.name}")


                
        source = Path("configs/shinynerf/shiny_blender_bitangent_radius.gin")
        dest_dir = Path(logdir)
        dest_dir.mkdir(parents=True, exist_ok=True)

        shutil.copy2(source, dest_dir / source.name)

        print(f"✓ Copied {source} to {dest_dir / source.name}")


        source = Path("src/model/shinynerf/helper.py")
        dest_dir = Path(logdir)
        dest_dir.mkdir(parents=True, exist_ok=True)

        shutil.copy2(source, dest_dir / source.name)

        print(f"✓ Copied {source} to {dest_dir / source.name}")


        source = Path("src/model/shinynerf/fb.py")
        dest_dir = Path(logdir)
        dest_dir.mkdir(parents=True, exist_ok=True)

        shutil.copy2(source, dest_dir / source.name)

        print(f"✓ Copied {source} to {dest_dir / source.name}")



        source = Path("src/model/shinynerf/asg2vmfs.py")
        dest_dir = Path(logdir)
        dest_dir.mkdir(parents=True, exist_ok=True)

        shutil.copy2(source, dest_dir / source.name)

        print(f"✓ Copied {source} to {dest_dir / source.name}")

        trainer.fit(model, data_module, ckpt_path=ckpt_path)






    if run_eval:
        ckpt_path = (
            f"{logdir}/best.ckpt"
            if model_name != "mipnerf360"
            else f"{logdir}/last.ckpt"
        )
        trainer.test(model, data_module, ckpt_path=ckpt_path)

    if run_render:
        ckpt_path = (
            f"{logdir}/best.ckpt"
            if model_name != "mipnerf360"
            else f"{logdir}/last.ckpt"
        )
        trainer.predict(model, data_module, ckpt_path=ckpt_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ginc",
        action="append",
        help="gin config file",
    )
    parser.add_argument(
        "--ginb",
        action="append",
        help="gin bindings",
    )
    parser.add_argument(
        "--resume_training",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="gin bindings",
    )
    parser.add_argument(
        "--ckpt_path", type=str, default=None, help="path to checkpoints"
    )
    parser.add_argument(
        "--scene_name", type=str, default=None, help="scene name to render"
    )
    parser.add_argument(
        "--downscale", type=int, default=1, help="resize of the dataset"
    )
    parser.add_argument(
        "--eval", action='store_true', help="use this flag to run the evaluation"
    )
    parser.add_argument("--seed", type=int, default=220901, help="seed to use")
    args = parser.parse_args()

    ginbs = []
    if args.ginb:
        ginbs.extend(args.ginb)

    logging.info(f"Gin configuration files: {args.ginc}")
    logging.info(f"Gin bindings: {ginbs}")

    gin.parse_config_files_and_bindings(args.ginc, ginbs)
    run(
        ginc=args.ginc,
        ginb=ginbs,
        img_downscale=args.downscale,
        scene_name=args.scene_name,
        resume_training=args.resume_training,
        ckpt_path=args.ckpt_path,
        seed=args.seed,
        run_train=not args.eval,
        run_eval=args.eval
    )
