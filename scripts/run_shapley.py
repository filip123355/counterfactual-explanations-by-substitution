from PIL import Image
from src.data.sampler import StratifiedSampler
import os
import json
import tempfile
import torch
import mlflow
import lpips
from loguru import logger
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

from src.constants import TRACKING_URI, I2SB_IMAGE_SIZE
from src.data import CelebADataset, CompositeFeature, Feature, FeatureType
from src.inpainter.i2sb import I2SB
from src.inpainter.guidance import CLIPGuidance, get_classifier
from src.interface import load_clip
from src.shapley import NShapleyValueCalculator, shapley_key_to_str
from src.substitution import (
    ImageSubstitution,
    MediapipeFaceKeypointDetector,
    ColorFillSubstitution,
)
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
        shapley_key_to_str(key): float(value) for key, value in shapely_values.items()
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


def log_shapley_values(
    shapley_values: dict[tuple[FeatureType, ...], float], *, config: dict, i: int
):
    for key, value in shapley_values.items():
        mlflow.log_metric(
            "_".join(shapley_key_to_str(key)[1:-1].split(", ")),
            float(value),
            step=int(i),
        )

    logger.info(
        f"{config['N']}-Shapley interaction values for prefix {i}: {shapley_values}"
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        value_file = os.path.join(
            tmpdir,
            f"target_{config['TARGET_INDEX']}_{config['N']}_shapley_values_{i:03d}.json",
        )

        features_file = os.path.join(
            tmpdir,
            f"target_{config['TARGET_INDEX']}_{config['N']}_shapley_features_{i:03d}.json",
        )

        plot_file = (
            f"target_{config['TARGET_INDEX']}_{config['N']}_shapley_values_{i:03d}.png"
        )

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


def log_max_abs_shapley_difference(differences: list[tuple[int, float]], config: dict):
    fig, ax = plt.subplots(figsize=(10, 6))
    x, y = zip(*differences)
    ax.plot(x, y, marker="o")
    ax.set_title(f"Difference in {config['N']}-Shapley Values Between Prefixes")
    ax.set_xlabel("Prefix")
    ax.set_ylabel("Difference in Shapley Values")
    fig.tight_layout()

    diff_plot_file = (
        f"target_{config['TARGET_INDEX']}_{config['N']}_shapley_difference.png"
    )
    mlflow.log_figure(fig, artifact_file=f"shapley/plots/{diff_plot_file}")
    plt.close(fig)

    for i, diff in differences:
        mlflow.log_metric(
            "max_abs_shapley_difference",
            diff,
            step=int(i),
        )


def calc_lpips_values(
    target: Image.Image,
    target_pred: float,
    coalition_images: dict[tuple[FeatureType, ...], list[Image.Image]],
    preds: dict[tuple[FeatureType, ...], list[float]],
    device: torch.device,
) -> list[tuple[float, float]]:
    logger.info("Calculating LPIPS distances for individual images...")
    logger.debug(f"Target pred: {target_pred}")
    logger.debug(f"Preds for coalitions: {preds}")

    loss_fn_vgg = lpips.LPIPS(net="vgg").to(device).eval()

    transform = transforms.Compose(
        [
            transforms.Resize((I2SB_IMAGE_SIZE, I2SB_IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ]
    )

    target_tensor = transform(target).unsqueeze(0).to(device)

    results = []

    with torch.no_grad():
        for coalition, images in coalition_images.items():
            for img, pred_value in zip(images, preds[coalition]):
                img_tensor = transform(img).unsqueeze(0).to(device)
                score = loss_fn_vgg(target_tensor, img_tensor).item()

                results.append((score, float(pred_value - target_pred)))

    return results


def log_lpips(lpips_values: list[tuple[float, float]], config: dict):
    fig, ax = plt.subplots(figsize=(8, 6))

    x_lpips = [val[0] for val in lpips_values]
    y_preds = [val[1] for val in lpips_values]

    ax.scatter(x_lpips, y_preds, color="blue", alpha=0.5, edgecolors="none", s=20)

    ax.set_title(
        f"LPIPS Distance vs. Prediction Difference for {config['N']}-Shapley Coalitions"
    )
    ax.set_xlabel("LPIPS Distance to Target Image")
    ax.set_ylabel("Prediction Difference")
    ax.grid(True, linestyle="--", alpha=0.6)

    fig.tight_layout()

    plot_file = f"target_{config['TARGET_INDEX']}_{config['N']}_lpips_scatter.png"
    mlflow.log_figure(fig, artifact_file=f"lpips/plots/{plot_file}")
    plt.close(fig)

    with tempfile.TemporaryDirectory() as tmpdir:
        json_file = os.path.join(
            tmpdir, f"target_{config['TARGET_INDEX']}_{config['N']}_lpips_raw.json"
        )
        with open(json_file, "w") as f:
            json.dump({"lpips": x_lpips, "preds": y_preds}, f, indent=2)

        mlflow.log_artifact(json_file, artifact_path="lpips")


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

            ref_indices = (
                list(
                    range(
                        config["REF_INDICES_RANGE"][0],
                        config["REF_INDICES_RANGE"][1],
                    )
                )
                if "REF_INDICES_RANGE" in config
                else sampler.sample(
                    n_samples=config["N_SAMPLES"],
                    label=config["CLASSIFIER_LABEL"].capitalize(),
                    ratio=1.0 if target["label_value"] == 0 else 0.0,
                )
            )

            logger.info(f"Selected reference indices: {ref_indices}")

            guidance = CLIPGuidance(load_clip(device=device))
            guidance.set_target(target_img=target["full_image"])

            inpainter = (
                I2SB(
                    device=device,
                    guidance=guidance,
                )
                if config["TAU"] > 0
                else None
            )

            substitution = (
                ImageSubstitution(dataset, face_keypoint_detector)
                if config["SUBSTITUTION"] == "i2sb"
                else ColorFillSubstitution(dataset)
            )

            logger.debug(f"Inpainter initialized: {inpainter is not None}")
            logger.debug(f"Using substitution: {substitution.__class__.__name__}")

            shap_calculator = NShapleyValueCalculator(
                dataset=dataset,
                substitution=substitution,
                inpainter=inpainter,
            )

            features: list[FeatureType] = [
                FEATURE_MAP.get(feature, feature) for feature in config["FEATURES"]
            ]

            coalition_images = shap_calculator.prepare_coalitions_inpainting(
                target_idx=config["TARGET_INDEX"],
                ref_indices=ref_indices,
                features=features,
                tau=config["TAU"],
                nfe=config["NFE"],
                keep_intermediate=config["KEEP_INTERMEDIATE_VALUES"],
            )

            for feature_tuple, images in coalition_images[
                len(coalition_images)
            ].items():
                for j, img in enumerate(images):
                    img_file = f"coalition_{shapley_key_to_str(feature_tuple)}_{j}.png"
                    mlflow.log_image(
                        img,
                        artifact_file=f"shapley/coalitions/{img_file}",
                    )

            model = get_classifier().to(device)

            shapely_values_batch, preds_batch = (
                shap_calculator.compute_n_shapley_values(
                    n=config["N"],
                    model=model,
                    coalition_images=coalition_images,
                    features=features,
                    device=device,
                    pred_prob=config["PRED_PROB"],
                )
            )

            prefixes = sorted(shapely_values_batch.keys())
            prev_shapley_values = None
            differences = []

            for i in prefixes:
                shapely_values = shapely_values_batch[i]
                log_shapley_values(shapely_values, config=config, i=i)

                if prev_shapley_values is not None:
                    diff = calculate_shapley_difference(
                        prev_shapley_values, shapely_values
                    )
                    differences.append((i, diff))

                prev_shapley_values = shapely_values

            if config["KEEP_INTERMEDIATE_VALUES"]:
                log_max_abs_shapley_difference(differences, config)

            target_pred = preds_batch[len(coalition_images)][tuple(sorted(features))]
            target_pred = target_pred[0]

            lpips_values = calc_lpips_values(
                target=target["full_image"],
                target_pred=target_pred,
                coalition_images=coalition_images[len(coalition_images)],
                preds=preds_batch[len(preds_batch)],
                device=device,
            )
            log_lpips(lpips_values, config)

            logger.info(f"Logged {config['N']}-Shapley values to MLflow")
            logger.info(f"Logged {config['N']}-Shapley features to MLflow")
            logger.info(f"Logged {config['N']}-Shapley plot to MLflow")

        finally:
            if face_keypoint_detector is not None:
                try:
                    face_keypoint_detector.close()  # ty: ignore
                except Exception:
                    pass


if __name__ == "__main__":
    main()
