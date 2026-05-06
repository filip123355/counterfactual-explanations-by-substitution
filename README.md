# counterfactual-explanations-by-substitution
Implementation of Counterfactual Explanations by Substitution, using Region Constrained Schrödinger Bridges (RCSB)

## Setup

To set up the project, firts create the `.env` file by copying the `.env.example` and adjusting the paths to your data. Then install `uv` if you don't have it and run `uv sync` to install the dependencies.

### Environent variables:

1) `FACE_LANDMARK_MODEL_PATH`

Set it to model file for mediapipe face landmark detection. You can download it like this:

```bash
wget -O face_landmarker_v2_with_blendshapes.task -q https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
```

2) `I2SB_MODEL_PATH`

Set it to the renamed model files for the I2SB model from the RCSB repo. First download zip and extract to `data` directory. Rename those two files:

```
data/weights/inpainters/cddb/celeba/freeform_20_30/latest.pt -> i2sb.pt
data/weights/inpainters/cddb/celebahq/freeform_20_30/latest.pt -> i2sb_hq.pt
```

Set `I2SB_MODEL_PATH` to the path of the one you want to use.