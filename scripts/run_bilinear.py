import os
import torch
import mlflow
from PIL import Image
from loguru import logger
from mlflow.tracking import MlflowClient

from src.bilinear_model import BilinearModel
from src.data_loading import CelebADataset, CompositeFeature, Feature
from src.inpainter.guidance.classifier import get_classifier
from src.keypoints import MediapipeFaceKeypointDetector
from src.inpainter.i2sb import I2SB
from src.utils import load_config, parse_args, log_config_params
from src.constants import TRACKING_URI
from src.inpainter.guidance import CLIPGuidance
from src.clip_inferance import load_clip


FEATURE_MAP = {
    "eyes": CompositeFeature.eyes,
    "nose": Feature.nose,
    "mouth": CompositeFeature.mouth,
    "hair": Feature.hair,
}


def download_shapley_values_artifact(
    client: MlflowClient,
    run_id: str,
    target_idx: int,
    n: int,
) -> str:
    artifact_path = f"shapley/values/target_{target_idx}_{n}_shapley_values.json"
    return client.download_artifacts(run_id, artifact_path)


def main():

    args = parse_args()
    config = load_config(args.config)

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(config["MLFLOW_EXPERIMENT_NAME"])

    run_name = config.get(
        "MLFLOW_RUN_NAME",
        f"target_{config['TARGET_INDEX']}",
    )

    with mlflow.start_run(run_name=run_name):

        log_config_params(config)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        mlflow.log_param("device", str(device))

        dataset = CelebADataset(split=config["DATASET_SPLIT"])
        face_keypoint_detector = MediapipeFaceKeypointDetector()
        model = get_classifier().to(device)
        features = [FEATURE_MAP[feature] for feature in config["FEATURES"]]
        client = MlflowClient(tracking_uri=TRACKING_URI)
        first_order_values_path = download_shapley_values_artifact(
            client=client,
            run_id=config["SHAPLEY_1_RUN_ID"],
            target_idx=config["TARGET_INDEX"],
            n=1,
        )
        second_order_values_path = download_shapley_values_artifact(
            client=client,
            run_id=config["SHAPLEY_2_RUN_ID"],
            target_idx=config["TARGET_INDEX"],
            n=2,
        )
        guidance = CLIPGuidance(load_clip(device=device))
        target_hq_idx = dataset.data.iloc[config["TARGET_INDEX"]]["idx"]
        target_image_path = os.path.join(dataset.img_dir, f"{target_hq_idx}.jpg")
        guidance.set_target(target_img=Image.open(target_image_path).convert("RGB"))
        inpainter = I2SB(device=device, guidance=guidance)
        bilinear_model = BilinearModel(
            features=features,
            target_idx=config["TARGET_INDEX"],
            first_order_values_path=first_order_values_path,
            second_order_values_path=second_order_values_path,
            dataset=dataset,
            inpainter=inpainter,
            face_keypoint_detector=face_keypoint_detector,
        )
        r_squared = bilinear_model.calculate_r_squared(
            ref_indices=list(range(config["REF_INDICES_RANGE"][0], config["REF_INDICES_RANGE"][1])), 
            model=model, 
            device=device,
            tau=config["TAU"],
            nfe=config["NFE"],
        )

        mlflow.log_metric("r_squared", r_squared)
        
        logger.info(f"Logged R-squared: {r_squared}")


if __name__ == "__main__":
    main()
