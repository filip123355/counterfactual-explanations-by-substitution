import cv2
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from PIL import Image

from src.data_loading import (
    CelebADataset,
    CompositeFeature,
    Feature,
    FeatureType,
    get_base_features,
)
from src.keypoints import FaceKeypointDetector, MediapipeFaceKeypointDetector
from src.utils import assert_not_none
from src.visualize import show_substitution

FEATURE_INFLATION: dict[FeatureType, int] = {
    CompositeFeature.eyes: 10,
}


class Substitution:
    dataset: CelebADataset
    face_keypoint_detector: FaceKeypointDetector
    tps_padding: int

    def __init__(
        self,
        dataset: CelebADataset,
        face_keypoint_detector: FaceKeypointDetector,
        *,
        tps_padding: int = 20,
    ):
        self.dataset = dataset
        self.face_keypoint_detector = face_keypoint_detector
        self.tps_padding = tps_padding

    def substitute(
        self,
        src_idx: int,
        dest_idx: int,
        feature: FeatureType,
        image: Image.Image | None = None,
        plot_points: bool = False,
        skip_missing: bool = True,
    ) -> Image.Image:
        feature_parts = get_base_features(feature)
        substituted_image = image

        is_composite = isinstance(feature, CompositeFeature)
        tps: cv2.ThinPlateSplineShapeTransformer | None = None
        src_global_keypoints, dest_global_keypoints = None, None

        if self.face_keypoint_detector.is_global_detector:
            src_img = self.dataset.get(src_idx, feature=Feature.nose)["full_image"]
            dest_img = self.dataset.get(dest_idx, feature=Feature.nose)["full_image"]

            src_global_keypoints, dest_global_keypoints = (
                self.face_keypoint_detector.get_valid_paired_keypoints(
                    src_image=src_img, dest_image=dest_img
                )
            )

            tps = self._compute_warp(src_global_keypoints, dest_global_keypoints)

        for feature_part in feature_parts:
            inflation = FEATURE_INFLATION.get(
                feature_part, FEATURE_INFLATION.get(feature, 0)
            )

            src_item = self.dataset.get(
                src_idx, feature=feature_part, padding=0, inflate_mask=inflation
            )
            dest_item = self.dataset.get(
                dest_idx, feature=feature_part, padding=0, inflate_mask=inflation
            )

            src_mask, dest_mask = src_item["mask"], dest_item["mask"]
            if src_mask is None or dest_mask is None:
                if is_composite or skip_missing:
                    logger.warning(
                        f"Skipping feature part {feature_part} due to missing mask."
                    )
                    continue

                raise ValueError("One of the items does not have a mask.")

            src_bbox, dest_bbox = src_item["bbox"], dest_item["bbox"]
            if src_bbox is None or dest_bbox is None:
                if is_composite or skip_missing:
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

            src_keypoints, dest_keypoints = (
                self.face_keypoint_detector.get_valid_paired_keypoints(
                    src_image=src_image,
                    dest_image=dest_image,
                    src_mask=src_mask,
                    dest_mask=dest_mask,
                    src_bbox=src_bbox,
                    dest_bbox=dest_bbox,
                )
                if not self.face_keypoint_detector.is_global_detector
                else (src_global_keypoints, dest_global_keypoints)
            )

            if plot_points:
                assert src_keypoints is not None and dest_keypoints is not None

                fig, axes = plt.subplots(1, 2, figsize=(10, 5))
                axes[0].imshow(src_image)
                axes[0].scatter(
                    src_keypoints[:, 0], src_keypoints[:, 1], c="r", marker="x"
                )
                axes[0].set_title("Source Image with Keypoints")
                axes[1].imshow(dest_image)
                axes[1].scatter(
                    dest_keypoints[:, 0], dest_keypoints[:, 1], c="r", marker="x"
                )
                axes[1].set_title("Destination Image with Keypoints")
                plt.show()

            substituted_image = self._substitute_feature(
                src_image=src_image,
                dest_image=dest_image,
                src_mask=src_mask,
                dest_mask=dest_mask,
                dest_bbox=dest_bbox,
                src_keypoints=src_keypoints,
                dest_keypoints=dest_keypoints,
                tps=tps,
            )

        return assert_not_none(substituted_image)

    def _substitute_feature(
        self,
        *,
        src_image: Image.Image,
        dest_image: Image.Image,
        src_mask: np.ndarray,
        dest_mask: np.ndarray,
        dest_bbox: tuple[int, int, int, int],
        src_keypoints: np.ndarray | None = None,
        dest_keypoints: np.ndarray | None = None,
        tps: cv2.ThinPlateSplineShapeTransformer | None = None,
    ) -> Image.Image:
        assert src_mask.dtype == np.uint8 and dest_mask.dtype == np.uint8, (
            "Masks must be of type uint8."
        )

        if tps is None:
            assert src_keypoints is not None and dest_keypoints is not None
            tps = self._compute_warp(src_keypoints, dest_keypoints)

        src_img_np = np.array(src_image)
        dest_img_np = np.array(dest_image)

        src_img_np = self._transfer_color(
            src_img=src_img_np,
            dest_img=dest_img_np,
            src_mask=src_mask,
            dest_mask=dest_mask,
        )

        warped_src_img = tps.warpImage(src_img_np)
        warped_src_mask = tps.warpImage(src_mask)

        inflated_bbox = (
            max(dest_bbox[0] - self.tps_padding, 0),
            min(dest_bbox[1] + self.tps_padding, dest_img_np.shape[0]),
            max(dest_bbox[2] - self.tps_padding, 0),
            min(dest_bbox[3] + self.tps_padding, dest_img_np.shape[1]),
        )

        alpha = warped_src_mask.astype(np.float32) / 255.0

        # Sometimes mediapipe produces warps that stretch src feature across the whole image,
        # so this is a safeguard to ensure that only the area around the destination feature is affected.
        alpha[: inflated_bbox[0], :] = 0
        alpha[inflated_bbox[1] :, :] = 0
        alpha[:, : inflated_bbox[2]] = 0
        alpha[:, inflated_bbox[3] :] = 0

        alpha = np.stack([alpha] * 3, axis=-1)

        blended_np = (warped_src_img * alpha) + (dest_img_np * (1.0 - alpha))
        blended_np = np.clip(blended_np, 0, 255).astype(np.uint8)

        return Image.fromarray(blended_np)

    def _compute_warp(
        self,
        src_keypoints: np.ndarray,
        dest_keypoints: np.ndarray,
    ) -> cv2.ThinPlateSplineShapeTransformer:
        src_keypoints = src_keypoints.reshape(1, -1, 2)
        dest_keypoints = dest_keypoints.reshape(1, -1, 2)

        tps = cv2.createThinPlateSplineShapeTransformer()
        matches = [cv2.DMatch(i, i, 0) for i in range(src_keypoints.shape[1])]

        tps.estimateTransformation(dest_keypoints, src_keypoints, matches)

        return tps

    def _transfer_color(
        self,
        *,
        src_img: np.ndarray,
        dest_img: np.ndarray,
        src_mask: np.ndarray,
        dest_mask: np.ndarray,
    ) -> np.ndarray:
        src_lab = cv2.cvtColor(src_img, cv2.COLOR_RGB2LAB).astype(np.float32)
        dest_lab = cv2.cvtColor(dest_img, cv2.COLOR_RGB2LAB).astype(np.float32)

        src_mean, src_std = cv2.meanStdDev(src_lab, mask=src_mask)
        dest_mean, dest_std = cv2.meanStdDev(dest_lab, mask=dest_mask)

        src_mean = src_mean.reshape(1, 1, 3)
        src_std = src_std.reshape(1, 1, 3)
        dest_mean = dest_mean.reshape(1, 1, 3)
        dest_std = dest_std.reshape(1, 1, 3)

        result_lab = (src_lab - src_mean) * (dest_std / src_std) + dest_mean

        result_lab = np.clip(result_lab, 0, 255).astype(np.uint8)
        result_rgb = cv2.cvtColor(result_lab, cv2.COLOR_LAB2RGB)

        return result_rgb


if __name__ == "__main__":
    dataset = CelebADataset(split="test")

    face_keypoint_detector = MediapipeFaceKeypointDetector()
    substitution = Substitution(dataset, face_keypoint_detector)

    for src_idx in range(10):
        for dest_idx in range(10, 20):
            feature = CompositeFeature.eyes

            src_img = dataset.get(src_idx, feature=feature)["full_image"]
            dest_img = dataset.get(dest_idx, feature=feature)["full_image"]

            result_image = substitution.substitute(
                src_idx, dest_idx, feature, plot_points=False
            )
            show_substitution(
                src_img,
                dest_img,
                result_image,
                save_path=f"results/{src_idx}_{dest_idx}.png",
            )
