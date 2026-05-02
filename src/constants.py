import os
from pathlib import Path
from decouple import Config, RepositoryEnv

# Data paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOTENV_PATH = PROJECT_ROOT / '.env'

try:
	config = Config(RepositoryEnv(str(DOTENV_PATH)))
except Exception as e:
	print(f"Error loading .env file: {e}")
	
DATASET = config("DATA_PATH")

# Training parameters
BATCH_SIZE = 64

# Other 
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Model name and configuration
MODEL_NAME = "google/vit-base-patch16-224"





