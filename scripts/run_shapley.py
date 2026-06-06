from src.data.sampler import StratifiedSampler
import os
import json
import tempfile
import torch
import mlflow
from loguru import logger
from PIL import Image
import matplotlib.pyplot as plt

from src.constants import TRACKING_URI
from src.data import CelebADataset, CompositeFeature, Feature, FeatureType
from src.inpainter.i2sb import I2SB
from src.inpainter.guidance import CLIPGuidance, get_classifier
from src.interface import load_clip
from src.shapley import NShapleyValueCalculator, shapley_key_to_str
from src.substitution import Substitution, MediapipeFaceKeypointDetector
from src.visualize import render_shapley_values
from src.utils import log_config_params, parse_args, load_config

FEATURE_MAP = {
    "eyes": CompositeFeature.eyes,
    "nose": Feature.nose,
    "mouth": CompositeFeature.mouth,
    "hair": Feature.hair,
}


def save_shapley_values_json(shapely_values: dict, output_path: str) -> None:
    serializable_values = {
        shapley_key_to_str(key): float(value)
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
        shapley_key_to_str(key): shapley_key_to_str(key)
        for key in shapely_values.keys()
    }

    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(
            serializable_features,
            handle,
            indent=2,
            ensure_ascii=False,
        )

def calculate_shapley_difference(prev: dict, current: dict) -> float:
    max_diff = 0.0
    for key in current.keys():
        assert key in prev
        diff = abs(float(current[key]) - float(prev[key]))
        max_diff = max(max_diff, diff)
    return max_diff

def log_shapley_values(shapley_values: dict[tuple[FeatureType, ...], float], *, config: dict, i: int):
    for key, value in shapley_values.items():
        mlflow.log_metric(
            "_".join(shapley_key_to_str(key)[1:-1].split(", ")),
            float(value),
            step=int(i),
        )

    logger.info(f"{config['N']}-Shapley interaction values for prefix {i}: {shapley_values}")

    with tempfile.TemporaryDirectory() as tmpdir:
        value_file = os.path.join(
            tmpdir,
            f"target_{config['TARGET_INDEX']}_{config['N']}_shapley_values_{i:03d}.json"
        )

        features_file = os.path.join(
            tmpdir,
            f"target_{config['TARGET_INDEX']}_{config['N']}_shapley_features_{i:03d}.json"
        )

        plot_file = f"target_{config['TARGET_INDEX']}_{config['N']}_shapley_values_{i:03d}.png"

        save_shapley_values_json(
            shapely_values=shapley_values,
            output_path=value_file,
        )

        save_shapley_features_json(
            shapely_values=shapley_values,
            output_path=features_file,
        )

        fig, _ = render_shapley_values(
            shapley_values,
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

        mlflow.log_figure(
            fig,
            artifact_file=f"shapley/plots/{plot_file}",
        )
        plt.close(fig)

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

        try:
            dataset = CelebADataset(split="test")
            sampler = StratifiedSampler(dataset)
            face_keypoint_detector = MediapipeFaceKeypointDetector()

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            mlflow.log_param("device", str(device))

            target = dataset.get(config["TARGET_INDEX"])

            ref_indices = list(
                range(
                    config["REF_INDICES_RANGE"][0],
                    config["REF_INDICES_RANGE"][1],
                )
            ) if "REF_INDICES_RANGE" in config else sampler.sample(
                n_samples=config["N_SAMPLES"],
                label=config["CLASSIFIER_LABEL"].capitalize(),
                ratio=1.0 if target["label_value"] == 0 else 0.0,
            )

            logger.info(f"Selected reference indices: {ref_indices}")

            guidance = CLIPGuidance(load_clip(device=device))
            guidance.set_target(
                target_img=target["full_image"]
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
                keep_intermediate=config["KEEP_INTERMEDIATE_VALUES"],
            )

            model = get_classifier().to(device)

            shapely_values_batch = shap_calculator.compute_n_shapley_values(
                n=config["N"],
                model=model,
                coalition_images=coalition_images,
                features=features,
                device=device,
                pred_prob=config["PRED_PROB"],
            )

            prefixes = sorted(shapely_values_batch.keys())
            prev_shapley_values = None
            differences = []

            for i in prefixes:
                shapely_values = shapely_values_batch[i]
                log_shapley_values(shapely_values, config=config, i=i)
                
                if prev_shapley_values is not None:
                    diff = calculate_shapley_difference(prev_shapley_values, shapely_values)
                    differences.append((i, diff))

                prev_shapley_values = shapely_values
                
            logger.info(f"Logged {config['N']}-Shapley values to MLflow")
            logger.info(f"Logged {config['N']}-Shapley features to MLflow")
            logger.info(f"Logged {config['N']}-Shapley plot to MLflow")

            if config["KEEP_INTERMEDIATE_VALUES"]:
                fig, ax = plt.subplots(figsize=(10, 6))
                x, y = zip(*differences)
                ax.plot(x, y, marker="o")
                ax.set_title(f"Difference in {config['N']}-Shapley Values Between Prefixes")
                ax.set_xlabel("Prefix")
                ax.set_ylabel("Difference in Shapley Values")
                fig.tight_layout()

                diff_plot_file = f"target_{config['TARGET_INDEX']}_{config['N']}_shapley_difference.png"
                mlflow.log_figure(fig, artifact_file=f"shapley/plots/{diff_plot_file}")
                plt.close(fig)

                for i, diff in differences:
                    mlflow.log_metric(
                        "max_abs_shapley_difference",
                        diff,
                        step=int(i),
                    )

        finally:
            if face_keypoint_detector is not None:
                try:
                    face_keypoint_detector.close() # ty: ignore
                except Exception:
                    pass


if __name__ == "__main__":
    main()
