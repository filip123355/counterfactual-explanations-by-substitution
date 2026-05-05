from pathlib import Path

from decouple import Config, RepositoryEnv

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

# I2SB

I2SB_MODEL_PATH = config("I2SB_MODEL_PATH")

INTERVAL = 1000
T = 1.0
T0 = 0.0001
OT_ODE = True
BETA_MAX = 1.0
USE_FP16 = True
EMA_DECAY = 0.99
CLIP_DENOISE = False

MODEL_KWARGS = {
    "attention_resolutions": "32,16,8",
    "channel_mult": "",
    "class_cond": False,
    "dropout": 0.0,
    "image_size": 256,
    "learn_sigma": False,
    "num_channels": 256,
    "num_head_channels": 64,
    "num_heads": 4,
    "num_heads_upsample": -1,
    "num_res_blocks": 2,
    "resblock_updown": True,
    "use_checkpoint": False,
    "use_fp16": False,
    "use_new_attention_order": False,
    "use_scale_shift_norm": True,
}
