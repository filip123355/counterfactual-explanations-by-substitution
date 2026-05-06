from abc import abstractmethod

import cv2
import mediapipe as mp
import numpy as np
from loguru import logger
from PIL import Image

from src.constants import FACE_LANDMARK_MODEL_PATH
from src.data_loading import CELEB_HQ_SIZE


class FaceKeypointDetector:
    is_global_detector: bool
    keypoint_min_dist: float

    def __init__(self, keypoint_min_dist: float = 30.0):
        self.keypoint_min_dist = keypoint_min_dist

    @abstractmethod
    def get_keypoints(
        self,
        image: Image.Image,
        mask: np.ndarray | None = None,
        bbox: tuple[int, int, int, int] | None = None,
    ) -> list[tuple[float, float]]:
        pass

    def get_valid_paired_keypoints(
        self,
        *,
        src_image: Image.Image,
        dest_image: Image.Image,
        src_mask: np.ndarray | None = None,
        dest_mask: np.ndarray | None = None,
        src_bbox: tuple[int, int, int, int] | None = None,
        dest_bbox: tuple[int, int, int, int] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        src_keypoints = self.get_keypoints(src_image, src_mask, src_bbox)
        dest_keypoints = self.get_keypoints(dest_image, dest_mask, dest_bbox)

        src_duplicate_pos = self._get_invalid_keypoints_indices(src_keypoints)
        dest_duplicate_pos = self._get_invalid_keypoints_indices(dest_keypoints)

        unique_positions = (
            set(range(len(src_keypoints)))
            - set(src_duplicate_pos)
            - set(dest_duplicate_pos)
        )
        logger.debug(
            f"Found {len(unique_positions)} unique keypoint positions after removing invalid"
        )

        src_keypoints = [src_keypoints[i] for i in unique_positions]
        dest_keypoints = [dest_keypoints[i] for i in unique_positions]

        return np.array(src_keypoints, dtype=np.float32), np.array(
            dest_keypoints, dtype=np.float32
        )

    def _get_invalid_keypoints_indices(
        self, keypoints: list[tuple[float, float]]
    ) -> list[int]:
        invalid_positions = []
        accepted_keypoints = []

        for idx, kp in enumerate(keypoints):
            x, y = kp

            if x < 0 or y < 0 or x >= CELEB_HQ_SIZE[0] or y >= CELEB_HQ_SIZE[1]:
                invalid_positions.append(idx)
                continue

            is_too_close = False
            for akp in accepted_keypoints:
                dist_sq = (x - akp[0]) ** 2 + (y - akp[1]) ** 2
                if dist_sq < self.keypoint_min_dist**2:
                    is_too_close = True
                    break

            if is_too_close:
                invalid_positions.append(idx)
            else:
                accepted_keypoints.append(kp)

        return invalid_positions


class BoundingBoxKeypointDetector(FaceKeypointDetector):
    is_global_detector = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_keypoints(
        self,
        image: Image.Image,
        mask: np.ndarray | None = None,
        bbox: tuple[int, int, int, int] | None = None,
    ) -> list[tuple[float, float]]:
        assert mask is not None and bbox is not None, (
            "Mask and bounding box are required for BoundingBoxKeypointDetector."
        )

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


class MediapipeFaceKeypointDetector(FaceKeypointDetector):
    is_global_detector = True
    detector: mp.tasks.vision.FaceLandmarker

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        base_options = mp.tasks.BaseOptions(model_asset_path=FACE_LANDMARK_MODEL_PATH)
        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=base_options,
            num_faces=1,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        self.detector = mp.tasks.vision.FaceLandmarker.create_from_options(options)

    def get_keypoints(
        self,
        image: Image.Image,
        mask: np.ndarray | None = None,
        bbox: tuple[int, int, int, int] | None = None,
    ) -> list[tuple[float, float]]:
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.array(image))
        mp_results = self.detector.detect(mp_image)

        if not mp_results.face_landmarks:
            raise ValueError("No face landmarks detected.")

        keypoints = mp_results.face_landmarks[0]
        keypoints_array = [
            (kp.x * CELEB_HQ_SIZE[0], kp.y * CELEB_HQ_SIZE[1]) for kp in keypoints
        ]

        return keypoints_array
