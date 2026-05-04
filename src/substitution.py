from typing import cast

import cv2
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from PIL import Image

from src.data_loading import (
    CELEB_HQ_SIZE,
    CelebADataset,
    CompositeFeature,
    Feature,
    FeatureType,
    get_base_features,
)
from src.utils import assert_not_none
from src.visualize import show_substitution


class Substitution:
    dataset: CelebADataset

    def __init__(self, dataset: CelebADataset):
        self.dataset = dataset

    def substitute(
        self,
        src_idx: int,
        dest_idx: int,
        feature: FeatureType,
        image: Image.Image | None = None,
        plot_points: bool = False,
    ) -> Image.Image:
        feature_parts = get_base_features(feature)
        substituted_image = image

        is_composite = isinstance(feature, CompositeFeature)

        for feature_part in feature_parts:
            src_item = self.dataset.get(src_idx, feature=feature_part, padding=0)
            dest_item = self.dataset.get(dest_idx, feature=feature_part, padding=0)

            src_mask, dest_mask = src_item["mask"], dest_item["mask"]
            if src_mask is None or dest_mask is None:
                if is_composite:
                    logger.warning(
                        f"Skipping feature part {feature_part} due to missing mask."
                    )
                    continue

                raise ValueError("One of the items does not have a mask.")

            src_bbox, dest_bbox = src_item["bbox"], dest_item["bbox"]
            if src_bbox is None or dest_bbox is None:
                if is_composite:
                    logger.warning(
                        f"Skipping feature part {feature_part} due to missing bounding box."
                    )
                    continue

                raise ValueError("One of the items does not have a bounding box.")

            src_image = src_item["full_image"]
            dest_image = (
                dest_item["full_image"]
                if substituted_image is None
                else substituted_image
            )

            substituted_image = self._substitute_feature(
                src_image=src_image,
                dest_image=dest_image,
                src_mask=src_mask,
                dest_mask=dest_mask,
                src_bbox=src_bbox,
                dest_bbox=dest_bbox,
                plot_points=plot_points,
            )

        return assert_not_none(substituted_image)

    def _substitute_feature(
        self,
        *,
        src_image: Image.Image,
        dest_image: Image.Image,
        src_mask: np.ndarray,
        dest_mask: np.ndarray,
        src_bbox: tuple[int, int, int, int],
        dest_bbox: tuple[int, int, int, int],
        plot_points: bool = False,
    ) -> Image.Image:
        assert src_mask.dtype == np.uint8 and dest_mask.dtype == np.uint8, (
            "Masks must be of type uint8."
        )

        src_keypoints, dest_keypoints = self._get_paired_keypoints(
            src_mask, dest_mask, src_bbox, dest_bbox
        )
        src_keypoints = src_keypoints.reshape(1, -1, 2)
        dest_keypoints = dest_keypoints.reshape(1, -1, 2)

        if plot_points:
            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            axes[0].imshow(src_image)
            axes[0].scatter(
                src_keypoints[0, :, 0], src_keypoints[0, :, 1], c="r", marker="x"
            )
            axes[0].set_title("Source Image with Keypoints")
            axes[1].imshow(dest_image)
            axes[1].scatter(
                dest_keypoints[0, :, 0], dest_keypoints[0, :, 1], c="r", marker="x"
            )
            axes[1].set_title("Destination Image with Keypoints")
            plt.show()

        tps = cv2.createThinPlateSplineShapeTransformer()
        matches = [cv2.DMatch(i, i, 0) for i in range(src_keypoints.shape[1])]

        tps.estimateTransformation(dest_keypoints, src_keypoints, matches)

        src_img_np = np.array(src_image)
        dest_img_np = np.array(dest_image)

        warped_src_img = tps.warpImage(src_img_np)
        warped_src_mask = tps.warpImage(src_mask)

        alpha = warped_src_mask.astype(np.float32) / 255.0
        alpha = np.stack([alpha] * 3, axis=-1)

        blended_np = (warped_src_img * alpha) + (dest_img_np * (1.0 - alpha))
        blended_np = np.clip(blended_np, 0, 255).astype(np.uint8)

        return Image.fromarray(blended_np)

    def _get_paired_keypoints(
        self,
        src_mask: np.ndarray,
        dest_mask: np.ndarray,
        src_bbox: tuple[int, int, int, int],
        dest_bbox: tuple[int, int, int, int],
    ) -> tuple[np.ndarray, np.ndarray]:
        def get_invalid_keypoints_indices(
            keypoints: list[tuple[float, float]],
        ) -> list[int]:
            seen = set()
            invalid_positions = []

            for idx, kp in enumerate(keypoints):
                if kp not in seen:
                    seen.add(kp)

                    if (
                        kp[0] < 0
                        or kp[1] < 0
                        or kp[0] >= CELEB_HQ_SIZE[0]
                        or kp[1] >= CELEB_HQ_SIZE[1]
                    ):
                        invalid_positions.append(idx)

                else:
                    invalid_positions.append(idx)

            return invalid_positions

        def extract_keypoints(
            mask: np.ndarray, bbox: tuple[int, int, int, int]
        ) -> list[tuple[float, float]]:
            mask_8u = mask.astype(np.uint8)

            contours, _ = cv2.findContours(
                mask_8u, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            c = max(contours, key=cv2.contourArea)  # ty: ignore

            ymin, ymax, xmin, xmax = bbox

            pt_bb_tl = (float(xmin), float(ymin))
            pt_bb_tr = (float(xmax), float(ymin))
            pt_bb_br = (float(xmax), float(ymax))
            pt_bb_bl = (float(xmin), float(ymax))

            M = cv2.moments(c)
            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]

            pt_centroid = (float(cx), float(cy))

            # Rotated bounding box
            # rect = cv2.minAreaRect(c)
            # box = cv2.boxPoints(rect).astype(np.float32)

            # s = box.sum(axis=1)
            # diff = np.diff(box, axis=1)

            # t = tuple[float, float]
            # pt_rbb_tl = cast(t, tuple(map(float, box[np.argmin(s)])))
            # pt_rbb_br = cast(t, tuple(map(float, box[np.argmax(s)])))
            # pt_rbb_tr = cast(t, tuple(map(float, box[np.argmin(diff)])))
            # pt_rbb_bl = cast(t, tuple(map(float, box[np.argmax(diff)])))

            return [
                pt_bb_tl,
                pt_bb_tr,
                pt_bb_br,
                pt_bb_bl,
                pt_centroid,
                # pt_rbb_tl,
                # pt_rbb_tr,
                # pt_rbb_br,
                # pt_rbb_bl,
            ]

        src_keypoints = extract_keypoints(src_mask, src_bbox)
        dest_keypoints = extract_keypoints(dest_mask, dest_bbox)

        src_duplicate_pos = get_invalid_keypoints_indices(src_keypoints)
        dest_duplicate_pos = get_invalid_keypoints_indices(dest_keypoints)

        unique_positions = (
            set(range(len(src_keypoints)))
            - set(src_duplicate_pos)
            - set(dest_duplicate_pos)
        )
        logger.debug(
            f"Found {len(unique_positions)} unique keypoint positions after removing duplicates."
        )

        src_keypoints = [src_keypoints[i] for i in unique_positions]
        dest_keypoints = [dest_keypoints[i] for i in unique_positions]

        return np.array(src_keypoints, dtype=np.float32), np.array(
            dest_keypoints, dtype=np.float32
        )


if __name__ == "__main__":
    dataset = CelebADataset(split="test")
    substitution = Substitution(dataset)

    src_idx = 1
    dest_idx = 3
    feature = Feature.nose

    src_img = dataset.get(src_idx, feature=feature)["full_image"]
    dest_img = dataset.get(dest_idx, feature=feature)["full_image"]

    result_image = substitution.substitute(
        src_idx, dest_idx, feature, plot_points=False
    )
    show_substitution(src_img, dest_img, result_image)
