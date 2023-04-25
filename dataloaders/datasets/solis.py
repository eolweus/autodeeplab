#!/usr/bin/env python3
import pathlib
from mypath import Path
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import DatasetFolder

import numpy as np
import rasterio
import torch


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
        # def __init__(self, args, root: str, transform=None) -> None:
        super().__init__()
        self.root = pathlib.Path(root)
        self.transform = transform
        self.args = args

        self.chips = []
        for path in self.root.glob("positive/*.jp2"):
            target_path = self.root / "target" / f"{path.name}"
            if path.exists() and target_path.exists():
                self.chips.append((path, target_path))

        for path in self.root.glob("negative/*.jp2"):
            if path.exists():
                self.chips.append((path, None))

    def __len__(self):
        return len(self.chips)

    def __getitem__(self, key):
        with rasterio.open(self.chips[key][0]) as dataset:
            data = torch.tensor(dataset.read().astype(np.float32))

        if self.chips[key][1]:
            with rasterio.open(self.chips[key][1]) as dataset:
                target = torch.tensor(dataset.read(1).astype(np.float32))
        else:
            target = torch.zeros(data.shape[1:], dtype=torch.float32)

        if self.transform:
            data, target = self.transform(data, target)

        return data, target


if __name__ == "__main__":
    dataset = ChipFolderSegmentationDataset(0, "/opt/data/copernicus")
    for data, target in DataLoader(dataset, batch_size=16, num_workers=4, shuffle=True):
        print(data.shape, target.shape)
