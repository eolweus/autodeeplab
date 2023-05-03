#!/usr/bin/env python3
from mypath import Path
from .solis import ChipFolderClassificationDataset, ChipFolderSegmentationDataset
from ..custom_transforms import SolisCompose, SolisNormalize, SolisRandomHorizontalFlip, SolisRandomVerticalFlip
from torchvision import transforms

import numpy as np
import pytorch_lightning as pl
import torch


mean = (1517.43, 1571.67, 1697.46, 1763.70, 2012.83, 2499.75,
        2701.89, 2766.73, 2851.32, 2869.90, 2662.53, 2152.98)
std = (548.34, 565.12, 584.81, 691.14, 730.98, 816.28,
       904.67, 950.83, 962.56, 1134.48, 943.07, 779.01)


# Youre not using this one!
class ChipFolderClassificationDatamodule(pl.LightningDataModule):
    def __init__(self, root: Path.db_root_dir('solis'), batch_size: int = 128, num_workers: int = 4, num_images: int = None, subset_ratio: float = None):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers

        transform = transforms.SolisCompose([
            transforms.SolisNormalize(mean, std),
            transforms.SolisRandomHorizontalFlip(),
            transforms.SolisRandomVerticalFlip()
        ])

        self.dataset = ChipFolderClassificationDataset(
            root, transform=transform, subset_ratio=subset_ratio, num_images=num_images)

        train_dataset_size = int(0.8 * len(self.dataset))
        val_dataset_size = len(self.dataset) - train_dataset_size

        self.train_dataset, self.val_dataset = torch.utils.data.random_split(
            self.dataset,
            [train_dataset_size, val_dataset_size])

        print("Found %d %s images" % (train_dataset_size, "training"))
        print("Found %d %s images" % (val_dataset_size, "validation"))

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True)

    def predict_dataloader(self):
        return torch.utils.data.DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True)


class ChipFolderSegmentationDatamodule(pl.LightningDataModule):
    def __init__(self, args, root: str = Path.db_root_dir('solis')):
        super().__init__()
        self.batch_size = args.batch_size
        self.num_workers = args.workers

        transform = SolisCompose([
            SolisNormalize(mean, std),
            SolisRandomHorizontalFlip(),
            SolisRandomVerticalFlip()
        ])

        self.dataset = ChipFolderSegmentationDataset(
            args, root, transform=transform)

        train_dataset_size = int(0.8 * len(self.dataset))
        val_dataset_size = len(self.dataset) - train_dataset_size

        self.train_dataset, self.val_dataset = torch.utils.data.random_split(
            self.dataset,
            [train_dataset_size, val_dataset_size])

        print("Found %d %s images" % (train_dataset_size, "training"))
        print("Found %d %s images" % (val_dataset_size, "validation"))

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True)

    def predict_dataloader(self):
        return torch.utils.data.DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True)


def dataset_weights(dataset: torch.utils.data.Dataset):
    labels = np.fromiter(dataset.labels, int)
    positive = labels.sum()
    negative = labels.size - positive
    return labels / positive + (1 - labels) / negative
