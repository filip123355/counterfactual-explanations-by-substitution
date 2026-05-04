from PIL import Image

from src.data_loading import CelebAFeatureDatasetFactory, Features


class Substitution:
    dataset_factory: CelebAFeatureDatasetFactory
    split: str

    def __init__(self, split: str = "train"):
        self.split = split
        self.dataset_factory = CelebAFeatureDatasetFactory(transforms=None)

    def substitute(
        self,
        src_hq_idx: int,
        dest_hq_idx: int,
        feature: Features,
        image: Image.Image | None = None,
    ) -> None:
        dataset = self.dataset_factory.get_dataset(self.split, feature)

        src_item = dataset[src_hq_idx]
        dest_item = dataset[dest_hq_idx]

        src_mask, dest_mask = src_item["mask"], dest_item["mask"]
        if src_mask is None or dest_mask is None:
            raise ValueError("One of the items does not have a mask.")

        src_bbox, dest_bbox = src_item["bbox"], dest_item["bbox"]
        if src_bbox is None or dest_bbox is None:
            raise ValueError("One of the items does not have a bounding box.")

        src_image = src_item["full_image"]
        dest_image = dest_item["full_image"] if image is None else image
