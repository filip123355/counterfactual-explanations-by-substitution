# counterfactual-explanations-by-substitution
Implementation of Counterfactual Explanations by Substitution, using Region Constrained Schrödinger Bridges (RCSB)

## Setup

To set up the project, firts create the `.env` file by copying the `.env.example` and adjusting the paths to your data. Then install `uv` if you don't have it and run `uv sync` to install the dependencies.

## Useful settngs

- `USE_HQ`: Set it to `True` if you want CelebA-HQ trained models, or `False` if you want CelebA trained models.
- `I2SB_IMAGE_SIZE`: Set it to power of 2 e.g. 128, 256, 512. Higher resolution will require more VRAM.
- `USE_FP16`: Set it to `True` to use half precision for the diffusion model. This will reduce VRAM usage but may cause instability.

### Environent variables:

1) `FACE_LANDMARK_MODEL_PATH`

Set it to model file for mediapipe face landmark detection. You can download it like this:

```bash
wget -O face_landmarker_v2_with_blendshapes.task -q https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
```

2) `I2SB_CELEB_MODEL_PATH`, `I2SB_CELEBHQ_MODEL_PATH`

Set it to inpainting model files from the RCSB repo. First download zip and extract to `data` directory. Set the variables to:

```
I2SB_CELEB_MODEL_PATH -> data/weights/inpainters/cddb/celeba/freeform_20_30/latest.pt
I2SB_CELEBHQ_MODEL_PATH -> data/weights/inpainters/cddb/celebahq/freeform_20_30/latest.pt
```

3) `CLASSIFIER_CELEB_MODEL_PATH`, `CLASSIFIER_CELEBHQ_MODEL_PATH`

Set it to densenet classifier model files from the RCSB repo:

```
CLASSIFIER_CELEB_MODEL_PATH -> data/weights/classifiers/celeba/densenet/ckpt.pt
CLASSIFIER_CELEBHQ_MODEL_PATH -> data/weights/classifiers/celebahq/densenet/ckpt.tar
```