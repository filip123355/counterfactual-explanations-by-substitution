import json
import random
import tempfile
from collections.abc import Iterable

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import torch
from loguru import logger
from PIL import Image
from torchvision import transforms

from src.constants import CLASSIFIER_LABEL as DEFAULT_CLASSIFIER_LABEL
from src.constants import I2SB_IMAGE_SIZE
from src.constants import TRACKING_URI
from src.data import CelebADataset, CompositeFeature, Feature, FeatureType
from src.data.sampler import StratifiedSampler
from src.inpainter.guidance import CLIPGuidance
from src.inpainter.guidance.classifier import get_classifier
from src.inpainter.i2sb import I2SB, SampleType
from src.interface import load_clip
from src.substitution import (
    ColorFillSubstitution,
    ImageSubstitution,
    MediapipeFaceKeypointDetector,
    Substitution,
)
from src.utils import load_config, log_config_params, parse_args


FEATURE_MAP = {
    "eyes": CompositeFeature.eyes,
    "nose": Feature.nose,
    "mouth": CompositeFeature.mouth,
    "hair": Feature.hair,
}

IMAGE_SUBSTITUTION_NAMES = {"substitution", "i2sb"}


def parse_features(feature_names: Iterable[str]) -> list[FeatureType]:
    features: list[FeatureType] = []
    for feature_name in feature_names:
        features.append(
            FEATURE_MAP[feature_name] if feature_name in FEATURE_MAP else Feature(feature_name)
        )
    return features


def image_to_tensor(image: Image.Image, image_size: int) -> torch.Tensor:
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]
    )
    return transform(image)


def apply_prefix_substitution(
    dataset: CelebADataset,
    substitution: Substitution,
    dest_idx: int,
    features: list[FeatureType],
    src_idx: int | None = None,
    inpainter: I2SB | None = None,
    tau: float = 1.0,
    nfe: int = 100,
) -> Image.Image:
    image = dataset.get(dest_idx)["full_image"]
    masks = []
    for feature in features:
        if isinstance(substitution, ColorFillSubstitution):
            image = substitution.substitute(
                dest_idx=dest_idx,
                feature=feature,
                image=image,
                skip_missing=True,
            )
        else:
            if src_idx is None:
                raise ValueError("src_idx is required for ImageSubstitution.")
            image = substitution.substitute(
                dest_idx=dest_idx,
                feature=feature,
                image=image,
                src_idx=src_idx,
                skip_missing=True,
            )
        if inpainter is not None:
            mask_item = dataset.get(dest_idx, feature=feature, inflate_mask=10)
            if mask_item["mask"] is not None:
                masks.append(mask_item["mask"])

    if inpainter is not None and masks:
        if inpainter.guidance is not None:
            inpainter.guidance.set_target(target_img=dataset.get(dest_idx)["full_image"])
        image = inpainter.inpaint(
            image=image,
            mask=combine_masks(masks),
            tau=tau,
            sampler_type=SampleType.DDPM,
            nfe=nfe,
        )

    return image


def combine_masks(masks: list[np.ndarray]) -> np.ndarray:
    combined = masks[0].copy()
    for mask in masks[1:]:
        combined = np.logical_or(combined, mask).astype(np.uint8) * 255
    return combined


def evaluate_prefix(
    *,
    dataset: CelebADataset,
    substitution: Substitution,
    model: torch.nn.Module,
    device: torch.device,
    sample_indices: list[int],
    features: list[FeatureType],
    batch_size: int,
    image_size: int,
    source_indices: dict[int, int] | None = None,
    inpainter: I2SB | None = None,
    tau: float = 1.0,
    nfe: int = 100,
) -> dict[str, float]:
    y_true: list[int] = []
    y_pred: list[int] = []
    y_prob: list[float] = []

    model.eval()
    with torch.no_grad():
        for start in range(0, len(sample_indices), batch_size):
            batch_indices = sample_indices[start : start + batch_size]

            images = [
                apply_prefix_substitution(
                    dataset=dataset,
                    substitution=substitution,
                    dest_idx=idx,
                    features=features,
                    src_idx=source_indices[idx] if source_indices is not None else None,
                    inpainter=inpainter,
                    tau=tau,
                    nfe=nfe,
                )
                for idx in batch_indices
            ]
            labels = [int(dataset.get(idx)["label_value"]) for idx in batch_indices]

            image_tensor = torch.stack(
                [image_to_tensor(image, image_size=image_size) for image in images]
            ).to(device)
            logits = model(image_tensor)[:, 0]
            probs = torch.sigmoid(logits).detach().cpu()
            preds = (probs >= 0.5).long()

            y_true.extend(labels)
            y_pred.extend(preds.tolist())
            y_prob.extend(probs.tolist())

    correct = sum(int(pred == true) for pred, true in zip(y_pred, y_true))
    positives = [i for i, value in enumerate(y_true) if value == 1]
    negatives = [i for i, value in enumerate(y_true) if value == 0]

    pos_acc = (
        sum(int(y_pred[i] == 1) for i in positives) / len(positives)
        if positives
        else 0.0
    )
    neg_acc = (
        sum(int(y_pred[i] == 0) for i in negatives) / len(negatives)
        if negatives
        else 0.0
    )

    return {
        "accuracy": correct / len(y_true) if y_true else 0.0,
        "balanced_accuracy": (pos_acc + neg_acc) / 2.0,
        "positive_accuracy": pos_acc,
        "negative_accuracy": neg_acc,
        "mean_positive_probability": sum(y_prob) / len(y_prob) if y_prob else 0.0,
        "predicted_positive_rate": sum(y_pred) / len(y_pred) if y_pred else 0.0,
        "n_samples": float(len(y_true)),
    }


def log_results(results: list[dict[str, object]]) -> None:
    for result in results:
        step = int(result["n_removed_features"])
        metrics = result["metrics"]
        assert isinstance(metrics, dict)
        for name, value in metrics.items():
            mlflow.log_metric(name, float(value), step=step)

    with tempfile.TemporaryDirectory() as tmpdir:
        json_path = f"{tmpdir}/roar_eval_results.json"
        with open(json_path, "w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=2, ensure_ascii=False)
        mlflow.log_artifact(json_path, artifact_path="roar_eval")

        fig, ax = plt.subplots(figsize=(8, 5))
        x = [int(result["n_removed_features"]) for result in results]
        accuracy = [float(result["metrics"]["accuracy"]) for result in results]  # type: ignore[index]
        balanced_accuracy = [
            float(result["metrics"]["balanced_accuracy"]) for result in results  # type: ignore[index]
        ]
        ax.plot(x, accuracy, marker="o", label="Accuracy")
        ax.plot(x, balanced_accuracy, marker="o", label="Balanced accuracy")
        ax.set_xlabel("Number of removed prefix features")
        ax.set_ylabel("Score")
        ax.set_title("ROAR-like evaluation without retraining")
        ax.set_ylim(0.0, 1.0)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend()
        fig.tight_layout()
        mlflow.log_figure(fig, artifact_file="roar_eval/accuracy_curve.png")
        plt.close(fig)


def compute_clip_source_indices(
    *,
    dataset: CelebADataset,
    sample_indices: list[int],
    clip,
    batch_size: int,
) -> tuple[dict[int, int], dict[int, float]]:
    labels = {idx: int(dataset.get(idx)["label_value"]) for idx in sample_indices}
    if len(set(labels.values())) < 2:
        raise ValueError("ImageSubstitution pairing requires samples from both classes.")

    images = [dataset.get(idx)["full_image"] for idx in sample_indices]

    embeddings = []
    for start in range(0, len(images), batch_size):
        batch_images = images[start : start + batch_size]
        batch_embeddings = clip.compute_image_embeddings(batch_images, normalize=True)
        embeddings.append(batch_embeddings.float().cpu())

    embedding_tensor = torch.cat(embeddings, dim=0)
    similarities = embedding_tensor @ embedding_tensor.T

    source_indices: dict[int, int] = {}
    source_scores: dict[int, float] = {}
    for dest_pos, dest_idx in enumerate(sample_indices):
        candidate_positions = [
            pos
            for pos, candidate_idx in enumerate(sample_indices)
            if labels[candidate_idx] != labels[dest_idx]
        ]
        best_pos = max(
            candidate_positions,
            key=lambda pos: float(similarities[dest_pos, pos].item()),
        )
        source_indices[dest_idx] = sample_indices[best_pos]
        source_scores[dest_idx] = float(similarities[dest_pos, best_pos].item())

    return source_indices, source_scores


def log_source_indices(
    *,
    source_indices: dict[int, int],
    source_scores: dict[int, float],
    dataset: CelebADataset,
) -> None:
    rows = []
    for dest_idx, src_idx in source_indices.items():
        rows.append(
            {
                "dest_idx": dest_idx,
                "dest_label": int(dataset.get(dest_idx)["label_value"]),
                "src_idx": src_idx,
                "src_label": int(dataset.get(src_idx)["label_value"]),
                "clip_similarity": source_scores[dest_idx],
            }
        )

    with tempfile.TemporaryDirectory() as tmpdir:
        json_path = f"{tmpdir}/clip_source_indices.json"
        with open(json_path, "w", encoding="utf-8") as handle:
            json.dump(rows, handle, indent=2, ensure_ascii=False)
        mlflow.log_artifact(json_path, artifact_path="roar_eval")


def log_example_images(
    *,
    dataset: CelebADataset,
    substitution: Substitution,
    sample_indices: list[int],
    ordered_features: list[FeatureType],
    max_images: int,
    source_indices: dict[int, int] | None = None,
    inpainter: I2SB | None = None,
    tau: float = 1.0,
    nfe: int = 100,
) -> None:
    example_indices = select_balanced_example_indices(
        dataset=dataset,
        sample_indices=sample_indices,
        max_images=max_images,
    )
    for image_idx in example_indices:
        for n_removed in range(len(ordered_features) + 1):
            features = ordered_features[:n_removed]
            image = apply_prefix_substitution(
                dataset=dataset,
                substitution=substitution,
                dest_idx=image_idx,
                features=features,
                src_idx=source_indices[image_idx] if source_indices is not None else None,
                inpainter=inpainter,
                tau=tau,
                nfe=nfe,
            )
            feature_suffix = (
                "none" if not features else "_".join(str(feature.value) for feature in features)
            )
            mlflow.log_image(
                image,
                artifact_file=f"roar_eval/examples/{image_idx}_{n_removed}_{feature_suffix}.png",
            )


def select_balanced_example_indices(
    *,
    dataset: CelebADataset,
    sample_indices: list[int],
    max_images: int,
) -> list[int]:
    if max_images <= 0:
        return []

    positives = [idx for idx in sample_indices if int(dataset.get(idx)["label_value"]) == 1]
    negatives = [idx for idx in sample_indices if int(dataset.get(idx)["label_value"]) == 0]

    selected: list[int] = []
    for pos_idx, neg_idx in zip(positives, negatives):
        if len(selected) < max_images:
            selected.append(pos_idx)
        if len(selected) < max_images:
            selected.append(neg_idx)

    remaining = positives[len(selected) // 2 :] + negatives[len(selected) // 2 :]
    selected.extend(remaining[: max_images - len(selected)])
    return selected


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    classifier_label = config["CLASSIFIER_LABEL"].lower()
    if classifier_label != DEFAULT_CLASSIFIER_LABEL.lower():
        raise ValueError(
            "CLASSIFIER_LABEL in config must match src.constants.CLASSIFIER_LABEL "
            f"for this dataset loader. Got config={classifier_label!r}, "
            f"constants={DEFAULT_CLASSIFIER_LABEL.lower()!r}."
        )

    substitution_name = config["SUBSTITUTION"]
    if substitution_name != "color_fill" and substitution_name not in IMAGE_SUBSTITUTION_NAMES:
        raise ValueError(
            "run_roar_eval supports SUBSTITUTION: color_fill, substitution or i2sb."
        )

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(config["MLFLOW_EXPERIMENT_NAME"])
    run_name = config.get("MLFLOW_RUN_NAME", "roar_eval")

    with mlflow.start_run(run_name=run_name):
        log_config_params(config)

        dataset = CelebADataset(split=config["DATASET_SPLIT"])
        sampler = StratifiedSampler(dataset)
        sample_indices = sampler.sample(
            n_samples=config["N_SAMPLES"],
            label=config.get("BALANCE_LABEL", config["CLASSIFIER_LABEL"].capitalize()),
            ratio=0.5,
            seed=config.get("SEED", None),
        )
        random.Random(config.get("SEED", 42)).shuffle(sample_indices)
        mlflow.log_param("sample_indices", json.dumps(sample_indices))
        logger.info(f"Selected {len(sample_indices)} evaluation samples: {sample_indices}")
        logger.info(
            "Sample label counts: "
            f"positive={sum(int(dataset.get(idx)['label_value']) for idx in sample_indices)}, "
            f"negative={sum(1 - int(dataset.get(idx)['label_value']) for idx in sample_indices)}"
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        mlflow.log_param("device", str(device))

        model = get_classifier().to(device)
        ordered_features = parse_features(config["FEATURES"])
        source_indices = None
        source_scores = None
        face_keypoint_detector = None
        inpainter = None

        if substitution_name == "color_fill":
            color = tuple(int(value) for value in config.get("COLOR_FILL", [0, 0, 0]))
            substitution: Substitution = ColorFillSubstitution(dataset, color=color)  # type: ignore[arg-type]
        else:
            clip = load_clip(device=device)
            source_indices, source_scores = compute_clip_source_indices(
                dataset=dataset,
                sample_indices=sample_indices,
                clip=clip,
                batch_size=config.get("CLIP_BATCH_SIZE", config.get("BATCH_SIZE", 8)),
            )
            log_source_indices(
                source_indices=source_indices,
                source_scores=source_scores,
                dataset=dataset,
            )
            mlflow.log_param("source_indices", json.dumps(source_indices))
            face_keypoint_detector = MediapipeFaceKeypointDetector()
            substitution = ImageSubstitution(dataset, face_keypoint_detector)
            if substitution_name == "i2sb":
                guidance = CLIPGuidance(clip)
                inpainter = I2SB(device=device, guidance=guidance)

        try:
            results: list[dict[str, object]] = []
            for n_removed in range(len(ordered_features) + 1):
                prefix_features = ordered_features[:n_removed]
                metrics = evaluate_prefix(
                    dataset=dataset,
                    substitution=substitution,
                    model=model,
                    device=device,
                    sample_indices=sample_indices,
                    features=prefix_features,
                    batch_size=config.get("BATCH_SIZE", 8),
                    image_size=config.get("INPUT_IMAGE_SIZE", I2SB_IMAGE_SIZE),
                    source_indices=source_indices,
                    inpainter=inpainter,
                    tau=config.get("TAU", 1.0),
                    nfe=config.get("NFE", 100),
                )
                logger.info(f"Prefix {n_removed}, features={prefix_features}, metrics={metrics}")
                results.append(
                    {
                        "n_removed_features": n_removed,
                        "removed_features": [feature.value for feature in prefix_features],
                        "metrics": metrics,
                    }
                )

            log_results(results)
            log_example_images(
                dataset=dataset,
                substitution=substitution,
                sample_indices=sample_indices,
                ordered_features=ordered_features,
                max_images=config.get("LOG_IMAGES", 0),
                source_indices=source_indices,
                inpainter=inpainter,
                tau=config.get("TAU", 1.0),
                nfe=config.get("NFE", 100),
            )
        finally:
            if face_keypoint_detector is not None and hasattr(face_keypoint_detector, "close"):
                face_keypoint_detector.close()


if __name__ == "__main__":
    main()
