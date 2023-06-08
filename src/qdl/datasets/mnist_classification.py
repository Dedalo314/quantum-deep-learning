import os
import logging
import random

from torchquantum.datasets import MNIST
import numpy as np
import torch
from einops import rearrange

class MNISTClassificationDataset(torch.utils.data.Dataset):
    """
    MNIST dataset for classification

    The labels are the classes of the images.
    """
    def __init__(self, cfg, split):
        super().__init__()

        self.logger = logging.getLogger(__name__)

        if split == "train":
            self.fashion_mnist = MNIST(
                root="./mnist_data",
                train_valid_split_ratio=cfg.train_valid_split_ratio,
                digits_of_interest=cfg.digits_of_interest,
                fashion=cfg.fashion
            )["train"]
        elif split == "validation":
            self.fashion_mnist = MNIST(
                root="./mnist_data",
                train_valid_split_ratio=cfg.train_valid_split_ratio,
                digits_of_interest=cfg.digits_of_interest,
                fashion=cfg.fashion
            )["valid"]
        elif split == "test":
            self.fashion_mnist = MNIST(
                root="./mnist_data",
                train_valid_split_ratio=cfg.train_valid_split_ratio,
                digits_of_interest=cfg.digits_of_interest,
                fashion=cfg.fashion
            )["test"]
        else:
            raise NotImplementedError()

    def __len__(self):
        return self.fashion_mnist.__len__()

    def __getitem__(self, idx):
        """
        Images in MNIST are already normalized
        """
        feed_dict = self.fashion_mnist.__getitem__(idx)
        image, idx_class = feed_dict["image"], feed_dict["digit"]
        image = rearrange(image, "1 h w -> h w", w=28, h=28).float()
        return image, idx_class
