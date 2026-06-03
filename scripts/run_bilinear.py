import torch
import mlflow

from src.bilinear_model import BilinearModel
from src.data_loading import CelebADataset, CompositeFeature, Feature
from src.inpainter.guidance.classifier import get_classifier
from src.keypoints import MediapipeFaceKeypointDetector
from src.utils import load_config, parse_args, log_config_params
from src.constants import TRACKING_URI


FEATURE_MAP = {
    "eyes": CompositeFeature.eyes,
    "nose": Feature.nose,
    "mouth": CompositeFeature.mouth,
}


def main():

    args = parse_args()
    config = load_config(args.config)

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(config["MLFLOW_EXPERIMENT_NAME"])

    run_name = config.get(
        "MLFLOW_RUN_NAME",
        f"target_{config['TARGET_INDEX']}_n_{config['N']}",
    )

    with mlflow.start_run(run_name=run_name):

        log_config_params(config)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        mlflow.log_param("device", str(device))

        dataset = CelebADataset(split=config["DATASET_SPLIT"])
        face_keypoint_detector = MediapipeFaceKeypointDetector()
        model = get_classifier().to(device)
        features = [FEATURE_MAP[feature] for feature in config["FEATURES"]]
        bilinear_model = BilinearModel(
            features=features,
            target_idx=config["TARGET_INDEX"],
            dataset=dataset,
            face_keypoint_detector=face_keypoint_detector,
        )
        r_squared = bilinear_model.calculate_r_squared(
            ref_indices=list(range(config["REF_INDICES_RANGE"][0], config["REF_INDICES_RANGE"][1])), 
            model=model, 
            device=device,
        )

        mlflow.log_metric("r_squared", r_squared)
        
        print(f"R-squared: {r_squared:.4f}")


if __name__ == "__main__":
    main()