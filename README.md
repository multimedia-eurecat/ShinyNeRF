# ShinyNeRF

![teaser](./assets/teaser_2.png)

ShinyNeRF models anisotropic reflections in Neural Radiance Fields, capturing view-dependent effects whose shape/orientation changes with surface rotation (e.g., brushed metals, satin/velvet, nacre).

- Built on top of NeRF-Factory (you can still run other methods via their `.gin` configs).
- Primary focus: anisotropic material modeling in NeRFs.

**Paper:** https://arxiv.org/abs/2512.21692  
**Project page:** https://multimedia-eurecat.github.io/ShinyNeRF/

The project was implemented by [Albert Barreiro](https://github.com/AlbertBarreiro).


## Installation

```bash
conda create -n shinynerf python=3.8.20 -y
conda activate shinynerf

# install torch + cuda (Conda)
conda install -c pytorch -c conda-forge \
  pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit==11.3.1 -y

# IMPORTANT: pin pip (some deps have metadata incompatibilities with newer pip)
python -m pip install "pip<24.1"

# install repo requirements
pip install -r requirements.txt
```

## Data format

Each scene directory must contain:
- `train/` images
- `test/` images
- `transforms_train.json`
- `transforms_test.json`

Recommended layout (matches `--scene teapot`):

```text
data/
  teapot/
    train/
      000.png
      001.png
      ...
    test/
      000.png
      001.png
      ...
    transforms_train.json
    transforms_test.json
```

## Training

```
python3 run.py --ginc configs/[model]/[data].gin --scene [scene] --downscale [value]

# Example:

python3 run.py \
  --ginc configs/shinynerf/shiny_blender_bitangent_radius.gin \
  --scene teapot \
  --downscale 1
```
Where:
- `--scene teapot` expects `data/teapot/` (see [Data format](#data-format)).
- `--downscale` controls resolution scaling (1 = full res, 2 = half, 4 = quarter, ...).


## Evaluation
Evaluation does not support multi-GPU (run it on a single GPU / single process).

Also note that --eval will save a lot of auxiliary outputs (not just the final RGB), e.g. RGB, depth, normals, tangents/bitangents, etc. This can quickly take significant disk space, especially for high resolutions / many views.
```
python3 run.py --ginc configs/[model]/[data].gin --scene [scene] --downscale [value] --eval

# Example:

python3 run.py \
  --ginc configs/shinynerf/shiny_blender_bitangent_radius.gin \
  --scene teapot \
  --downscale 1 \
  --eval

```

## Outputs

Runs will create an experiment log directory (exact naming depends on the NeRF-Factory logging setup + config).
To monitor training:
```
tensorboard --logdir logs --samples_per_plugin images=100
```

## Troubleshooting

### pip / dependency metadata errors

Make sure pip is:

```
python -m pip install "pip<24.1"
```

## License

This repository includes original code for ShinyNeRF:

Copyright (c) 2026 Albert Barreiro

and builds on top of **NeRF-Factory**, whose code and associated portions remain:

Copyright (c) 2022 POSTECH, KAIST, and Kakao Brain Corp.

Licensed under the **Apache License, Version 2.0** (Apache-2.0).  
See `LICENSE` for the full license text. If a `NOTICE` file is present, it must be preserved when redistributing.



## Citation

```
@Article{Barreiro2025ShinyNeRF,
  author    = {Barreiro, Albert and Mar\'{i}, Roger and Redondo, Rafael and Haro, Gloria and Bosch, Carles},
  title     = {{ShinyNeRF}: Modeling Anisotropic Reflections in Neural Radiance Fields},
  journal   = {The International Archives of the Photogrammetry, Remote Sensing and Spatial Information Sciences},
  year      = {2026}
}
```


## Acknowledgements

This project is built on top of NeRF-Factory and uses its training/evaluation pipeline and configuration system.