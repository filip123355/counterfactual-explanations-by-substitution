from pathlib import Path

from decouple import Config, RepositoryEnv
from loguru import logger

# Data paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOTENV_PATH = PROJECT_ROOT / ".env"

try:
    config = Config(RepositoryEnv(str(DOTENV_PATH)))
except Exception as e:
    print(f"Error loading .env file: {e}")

DATASET: str = config("DATA_PATH")
FACE_LANDMARK_MODEL_PATH: str = config("FACE_LANDMARK_MODEL_PATH")

# Training parameters
BATCH_SIZE = 64

# Other
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# CLIP parameters
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"

# Model name and configuration
MODEL_NAME = "google/vit-base-patch16-224"
MODEL_PATH = "models/vit-base-patch16-224"

# I2SB
USE_HQ = True

I2SB_CELEB_MODEL_PATH = config("I2SB_CELEB_MODEL_PATH")
I2SB_CELEBHQ_MODEL_PATH = config("I2SB_CELEBHQ_MODEL_PATH")

I2SB_MODEL_PATH = I2SB_CELEBHQ_MODEL_PATH if USE_HQ else I2SB_CELEB_MODEL_PATH

CLASSIFIER_CELEB_MODEL_PATH = config("CLASSIFIER_CELEB_MODEL_PATH")
CLASSIFIER_CELEBHQ_MODEL_PATH = config("CLASSIFIER_CELEBHQ_MODEL_PATH")

CLASSIFIER_MODEL_PATH = (
    CLASSIFIER_CELEBHQ_MODEL_PATH if USE_HQ else CLASSIFIER_CELEB_MODEL_PATH
)

logger.info(
    f"Using I2SB model from {I2SB_MODEL_PATH}, classifier model from {CLASSIFIER_MODEL_PATH}."
)

I2SB_IMAGE_SIZE = 256

INTERVAL = 1000 if USE_HQ else 500
T = 1.0
T0 = 0.0001
OT_ODE = True
BETA_MAX = 1.0
USE_FP16 = False
EMA_DECAY = 0.99
CLIP_DENOISE = False
STEP_SIZE = 0.0
CLASSIFIER_SCALE = 1.0
UNET_CONDITIONING = False
CLASSIFIER_LABEL = "male"


MODEL_KWARGS = {
    "attention_resolutions": "32,16,8",
    "channel_mult": "",
    "class_cond": False,
    "dropout": 0.0,
    "image_size": 256 if USE_HQ else 128,
    "learn_sigma": False,
    "num_channels": 256 if USE_HQ else 128,
    "num_head_channels": 64,
    "num_heads": 4,
    "num_heads_upsample": -1,
    "num_res_blocks": 2,
    "resblock_updown": True,
    "use_checkpoint": False,
    "use_fp16": USE_FP16,
    "use_new_attention_order": False,
    "use_scale_shift_norm": True,
}
