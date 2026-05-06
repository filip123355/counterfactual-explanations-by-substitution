# counterfactual-explanations-by-substitution
Implementation of Counterfactual Explanations by Substitution, using Region Constrained Schrödinger Bridges (RCSB)

## Setup

To set up the project, firts create the `.env` file by copying the `.env.example` and adjusting the paths to your data. Then install `uv` if you don't have it and run `uv sync` to install the dependencies.

Set `FACE_LANDMARK_MODEL_PATH` to model file for mediapipe face landmark detection. You can download it like this:

```bash
wget -O face_landmarker_v2_with_blendshapes.task -q https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
```

Set `I2SB_MODEL_PATH` to the model file for the I2SB model from the RCSB repo (download zip and extrac to `data` directory):

```
data/weights/inpainters/cddb/celeba/freeform_20_30/latest.pt
data/weights/inpainters/cddb/celebahq/freeform_20_30/latest.pt
```

Copy them to local repo and set the name to `i2sb.pt` and `i2sb_hq.pt` respectively (it's important that one has `hq` in the name to automatically set config).