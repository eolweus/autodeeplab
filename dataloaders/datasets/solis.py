#!/usr/bin/env python3
import pathlib
import random
from mypath import Path
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import DatasetFolder

import numpy as np
import rasterio
import torch
import warnings
from rasterio.errors import NotGeoreferencedWarning

import tracemalloc

class ChipFolderClassificationDataset(DatasetFolder):
    def __init__(self, root: str, transform=None, target_transform=None):
        super().__init__(
            root,
            loader=self.loader,
            extensions=(".jp2",),
            transform=transform,
            target_transform=target_transform)

    def find_classes(self, root):
        return ["negative", "positive"], {"negative": 0, "positive": 1}

    def loader(self, path):
        with rasterio.open(path) as dataset:
            data = torch.tensor(dataset.read().astype(np.float32))
            return data


class ChipFolderSegmentationDataset(Dataset):
    NUM_CLASSES = 2

    def __init__(self, args, root: Path.db_root_dir('solis'), transform=None) -> None:
        super().__init__()
        self.root = pathlib.Path(root)
        self.transform = transform
        self.args = args
        
        num_images = args.num_images or None
        if args.use_ab and num_images:
                num_images = int(num_images + num_images * 0.8)
        subset_ratio = args.subset_ratio or None

        positive_chips = []
        negative_chips = []

        for path in self.root.glob("positive/*.jp2"):
            target_path = self.root / "target" / f"{path.name}"
            if path.exists() and target_path.exists():
                positive_chips.append((path, target_path))

        for path in self.root.glob("negative/*.jp2"):
            if path.exists():
                negative_chips.append((path, None))

        # Calculate subset_ratio based on the provided number of images
        if num_images is not None:
            random.shuffle(positive_chips)
            random.shuffle(negative_chips)

            positive_chips = positive_chips[:num_images // 2]
            negative_chips = negative_chips[:num_images // 2]

        # Shuffle and take a random subset of the specified size for positive and negative chips
        elif subset_ratio is not None:
            pos_subset_size = int(len(positive_chips) * subset_ratio)
            neg_subset_size = int(len(negative_chips) * subset_ratio)

            random.shuffle(positive_chips)
            random.shuffle(negative_chips)

            positive_chips = positive_chips[:pos_subset_size]
            negative_chips = negative_chips[:neg_subset_size]

        self.chips = positive_chips + negative_chips
        random.shuffle(self.chips)

    def __len__(self):
        return len(self.chips)

    def __getitem__(self, key):
        if self.args.debug:
            tracemalloc.start()
            snapshot1 = tracemalloc.take_snapshot()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", NotGeoreferencedWarning)

        with rasterio.open(self.chips[key][0]) as dataset:
            data = torch.tensor(dataset.read().astype(np.float32))

        if self.chips[key][1]:
            with rasterio.open(self.chips[key][1]) as dataset:
                target = torch.tensor(dataset.read(
                    1).astype(np.float32))
                # TODO: This is a hack to make the target binary. We should probably fix the data itself instead.
                target = torch.clamp(target, 0, 1)
        else:
            target = torch.zeros(data.shape[1:], dtype=torch.float32)

        if self.transform:
            data, target = self.transform(data, target)

        # if self.args.num_bands == 3 only use band 3, 2 and 1 as red, green blue
        if self.args.num_bands == 3:
            data = data[[3, 2, 1], :, :]
        
        if self.args.debug:
            snapshot2 = tracemalloc.take_snapshot()
            top_stats = snapshot2.compare_to(snapshot1, 'lineno')
            with open(f'memory_log_loader.txt', 'w') as f:
                print("[ Top 10 differences for item {} ]".format(key), file=f)
                for stat in top_stats[:10]:
                    print(stat, file=f)
            tracemalloc.stop()

        return data, target


if __name__ == "__main__":
    dataset = ChipFolderSegmentationDataset(0, "/opt/data/copernicus")
    for data, target in DataLoader(dataset, batch_size=16, num_workers=4, shuffle=True):
        print(data.shape, target.shape)
