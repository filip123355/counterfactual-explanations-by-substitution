# SubSHAP: Evaluating Geometric Substitution and Diffusion Models for Visual Feature Attribution

Feature attribution methods, such as Shapley values, are challenging to apply to images
because removing a visual feature often creates unnatural artifacts. This forces a choice between
interventional methods, which create out-of-distribution samples, and observational
methods, which keep the image on the true data manifold at the cost of feature correlation.
In this work we introduce a unified image substitution pipeline SubSHAP and use it to compare
different image alteration strategies: Black Fill, Thin-Plate Spline substitution, and
diffusion-based refinement (I2SB). We evaluate how true to the model those approaches
are using a bilinear approximation model and the ROAR benchmark (Hooker et al., 2019).
We observe that while diffusion methods produce more realistic images they suffer from
feature correlation leak that degrades the quality of model explanations. We find that simple
geometric substitution offers the best balance, maintaining feature diversity without
confusing the classifier, and ultimately resulting in the most reliable visual Shapley values

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

## Running scripts

To compute Shapley values for a specific config, run:

```bash
python scripts/run_shapley.py --config_path path/to/config.yaml
```

after adjusting the an appropriate config file in `configs` directory.

To build an aproximate bilinear model form 1 and 2-Shapley values and compute $R^2$ determinant, run:

```bash
python scripts/bilinear.py --config_path path/to/config.yaml
```

