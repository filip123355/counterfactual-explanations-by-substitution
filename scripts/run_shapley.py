import os
import json
import tempfile
import torch
import mlflow
from loguru import logger
from PIL import Image

from src.constants import TRACKING_URI
from src.data_loading import CelebADataset, CompositeFeature, Feature
from src.inpainter.i2sb import I2SB
from src.keypoints import MediapipeFaceKeypointDetector
from src.inpainter.guidance import CLIPGuidance, get_classifier
from src.clip_inferance import load_clip
from src.shapley import NShapleyValueCalculator, _shapley_key_to_str
from src.substitution import Substitution
from src.visualize import show_shapley_values
from src.utils import log_config_params, parse_args, load_config

FEATURE_MAP = {
    "eyes": CompositeFeature.eyes,
    "nose": Feature.nose,
    "mouth": CompositeFeature.mouth,
    "hair": Feature.hair,
}


def save_shapley_values_json(shapely_values: dict, output_path: str) -> None:
    serializable_values = {
        _shapley_key_to_str(key): float(value)
        for key, value in shapely_values.items()
    }

    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(
            serializable_values,
            handle,
            indent=2,
            ensure_ascii=False,
        )


def save_shapley_features_json(shapely_values: dict, output_path: str) -> None:
    serializable_features = {
        _shapley_key_to_str(key): _shapley_key_to_str(key)
        for key in shapely_values.keys()
    }

    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(
            serializable_features,
            handle,
            indent=2,
            ensure_ascii=False,
        )


def main():
    args = parse_args()
    config = load_config(args.config)

    face_keypoint_detector = None

    ref_indices = list(
        range(
            config["REF_INDICES_RANGE"][0],
            config["REF_INDICES_RANGE"][1],
        )
    )

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(config["MLFLOW_EXPERIMENT_NAME"])

    run_name = config.get(
        "MLFLOW_RUN_NAME",
        f"target_{config['TARGET_INDEX']}_n_{config['N']}",
    )

    with mlflow.start_run(run_name=run_name):
        log_config_params(config)

        try:
            dataset = CelebADataset(split="test")
            face_keypoint_detector = MediapipeFaceKeypointDetector()

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            mlflow.log_param("device", str(device))

            guidance = CLIPGuidance(load_clip(device=device))

            target_hq_idx = dataset.data.iloc[config["TARGET_INDEX"]]["idx"]
            target_image_path = os.path.join(dataset.img_dir, f"{target_hq_idx}.jpg")

            guidance.set_target(
                target_img=Image.open(target_image_path).convert("RGB")
            )

            inpainter = I2SB(
                device=device,
                guidance=guidance,
            )

            shap_calculator = NShapleyValueCalculator(
                dataset=dataset,
                substitution=Substitution(dataset, face_keypoint_detector),
                inpainter=inpainter,
            )

            features  = [FEATURE_MAP.get(feature, feature) for feature in config["FEATURES"]]

            coalition_images = shap_calculator.prepare_coalitions_inpainting(
                target_idx=config["TARGET_INDEX"],
                ref_indices=ref_indices,
                features=features,
                tau=config["TAU"],
                nfe=config["NFE"],
            )

            model = get_classifier().to(device)

            shapely_values = shap_calculator.compute_n_shapley_values(
                n=config["N"],
                model=model,
                coalition_images=coalition_images,
                features=features,
                device=device,
                pred_prob=True,
            )

            for key, value in shapely_values.items():
                mlflow.log_metric(
                    "_".join(_shapley_key_to_str(key)[1:-1].split(", ")),
                    float(value),
                ) 

            print(f"{config['N']}-Shapley interaction values:", shapely_values)

            with tempfile.TemporaryDirectory() as tmpdir:
                value_file = os.path.join(
                    tmpdir,
                    f"target_{config['TARGET_INDEX']}_{config['N']}_shapley_values.json",
                )

                features_file = os.path.join(
                    tmpdir,
                    f"target_{config['TARGET_INDEX']}_{config['N']}_shapley_features.json",
                )

                plot_file = os.path.join(
                    tmpdir,
                    f"target_{config['TARGET_INDEX']}_{config['N']}_shapley_values.png",
                )

                save_shapley_values_json(
                    shapely_values=shapely_values,
                    output_path=value_file,
                )

                save_shapley_features_json(
                    shapely_values=shapely_values,
                    output_path=features_file,
                )

                show_shapley_values(
                    shapely_values,
                    save_path=plot_file,
                    title=f"{config['N']}-Shapley Interaction Values for Facial Features",
                )

                mlflow.log_artifact(
                    value_file,
                    artifact_path="shapley/values",
                )

                mlflow.log_artifact(
                    features_file,
                    artifact_path="shapley/features",
                )

                mlflow.log_artifact(
                    plot_file,
                    artifact_path="shapley/plots",
                )
                
                logger.info(f"Logged {config['N']}-Shapley values to MLflow")
                logger.info(f"Logged {config['N']}-Shapley features to MLflow")
                logger.info(f"Logged {config['N']}-Shapley plot to MLflow")

        finally:
            if face_keypoint_detector is not None:
                try:
                    face_keypoint_detector.close()
                except Exception:
                    pass


if __name__ == "__main__":
    main()
