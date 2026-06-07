from pathlib import Path
from src.constants import CLASSIFIER_LABEL
from src.data import CelebADataset

from loguru import logger

class StratifiedSampler:
    def __init__(self, dataset: CelebADataset):
        self.dataset = dataset

    # Label must match the column name in CelebAMask-HQ-attribute-anno.txt
    def sample(
        self,
        n_samples: int,
        *,
        label: str = CLASSIFIER_LABEL.capitalize(),
        ratio: float = 0.5, # proportion of positive samples in the output
    ) -> list[int]:
        assert label in self.dataset.data.columns, f"Label '{label}' not found in dataset attributes."

        pos_samples = self.dataset.data[self.dataset.data[label] == 1]
        neg_samples = self.dataset.data[self.dataset.data[label] == 0]

        n_pos = int(n_samples * ratio)
        n_neg = n_samples - n_pos

        logger.info(f"Sampling {n_pos} positive and {n_neg} negative samples for label '{label}'.")

        sampled_pos = pos_samples.sample(n=n_pos, replace=False)
        sampled_neg = neg_samples.sample(n=n_neg, replace=False)

        pos_ilocs = self.dataset.data.index.get_indexer(sampled_pos.index).tolist()
        neg_ilocs = self.dataset.data.index.get_indexer(sampled_neg.index).tolist()

        return pos_ilocs + neg_ilocs

if __name__ == "__main__":
    dataset = CelebADataset()
    sampler = StratifiedSampler(dataset)
    samples = sampler.sample(20, ratio=0.0)

    save_dir = Path("results/sampler")
    save_dir.mkdir(exist_ok=True)
    for file in save_dir.iterdir():
        file.unlink()
    
    for idx in samples:
        img = dataset.get(idx)
        print(idx, img["hq_idx"])
        img["full_image"].save(f"results/sampler/{idx}.jpg")