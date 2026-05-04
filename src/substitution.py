from PIL import Image

from src.data_loading import CelebAFeatureDatasetFactory


class Substitution:
    dataset_factory: CelebAFeatureDatasetFactory
    split: str

    def __init__(self, split: str = "train"):
        self.split = split
        self.dataset_factory = CelebAFeatureDatasetFactory(transforms=None)

    def substitute(
        src_hq_idx: int,
        dest_hq_idx: int,
        feature: str,
        image: Image.Image | None = None,
    ) -> None:
        pass
