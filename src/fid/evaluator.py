import random
import shutil
import tempfile
from pathlib import Path
from typing import Literal

import torch
from cleanfid import fid
from loguru import logger
from PIL import Image

from src.constants import I2SB_IMAGE_SIZE
from src.data import CelebADataset


class FIDEvaluator:
    def __init__(self, real_path: str, gen_path: str):
        self.real_path = Path(real_path)
        self.gen_path = Path(gen_path)

    def _get_image_files(self, path: Path) -> list[Path]:
        valid_exts = {".png", ".jpg", ".jpeg"}
        return [
            f for f in path.iterdir() if f.is_file() and f.suffix.lower() in valid_exts
        ]

    def _setup_temp_dir(
        self,
        files: list[Path],
        temp_dir: Path,
        resize_to: tuple[int, int] | None = None,
    ):
        for f in files:
            if resize_to is not None:
                with Image.open(f) as img:
                    img_resized = img.resize(resize_to, Image.Resampling.LANCZOS)
                    img_resized.save(temp_dir / f.name)
            else:
                shutil.copy2(f, temp_dir / f.name)

    def evaluate(self, metric: Literal["fid", "clip_fid", "kid"] = "fid") -> float:
        real_files = self._get_image_files(self.real_path)
        gen_files = self._get_image_files(self.gen_path)

        if not real_files or not gen_files:
            raise ValueError("One of the directories contains no valid images.")

        logger.info(
            f"Found {len(real_files)} real images and {len(gen_files)} generated images."
        )

        if metric in ["fid", "clip_fid"]:
            n_samples = min(len(real_files), len(gen_files))
            logger.info(f"Balancing datasets to {n_samples} samples.")

            sampled_real = random.sample(real_files, n_samples)
            sampled_gen = random.sample(gen_files, n_samples)

            with (
                tempfile.TemporaryDirectory() as tmp_real,
                tempfile.TemporaryDirectory() as tmp_gen,
            ):
                tmp_real_path = Path(tmp_real)
                tmp_gen_path = Path(tmp_gen)

                self._setup_temp_dir(
                    sampled_real,
                    tmp_real_path,
                    resize_to=(I2SB_IMAGE_SIZE, I2SB_IMAGE_SIZE),
                )
                self._setup_temp_dir(sampled_gen, tmp_gen_path)

                if metric == "fid":
                    score = fid.compute_fid(str(tmp_real_path), str(tmp_gen_path))
                else:
                    score = fid.compute_fid(
                        str(tmp_real_path),
                        str(tmp_gen_path),
                        model_name="clip_vit_b_32",
                    )

                return score

        elif metric == "kid":
            logger.info("Computing KID using all available images.")
            score = fid.compute_kid(str(self.real_path), str(self.gen_path))
            return score

        else:
            raise ValueError(f"Unknown metric: {metric}")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = CelebADataset(split="test")

    real_images_path = dataset.img_dir
    generated_images_path = "generated/fid_samples_color_fill_no_inpainter_no_guidance"

    evaluator = FIDEvaluator(real_path=real_images_path, gen_path=generated_images_path)

    fid_score = evaluator.evaluate(metric="fid")
    logger.success(f"FID Score: {fid_score:.4f}")

    clip_fid_score = evaluator.evaluate(metric="clip_fid")
    logger.success(f"CLIP-FID Score: {clip_fid_score:.4f}")

    kid_score = evaluator.evaluate(metric="kid")
    logger.success(f"KID Score: {kid_score:.6f}")
