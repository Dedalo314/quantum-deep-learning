import os
import logging
import random

from torchquantum.datasets import MNIST
import torchvision
import numpy as np
import torch
from einops import rearrange

class MNISTGenerationDataset(torch.utils.data.Dataset):
    """
    MNIST dataset for generation

    The labels are the values of the images displaced by one position.
    """
    def __init__(self, cfg, split):
        super().__init__()

        self.logger = logging.getLogger(__name__)

        if split == "train":
            self.fashion_mnist = MNIST(
                root="./mnist_data",
                train_valid_split_ratio=cfg.train_valid_split_ratio,
                digits_of_interest=cfg.digits_of_interest,
                fashion=cfg.fashion,
                binarize=cfg.binarize
            )["train"]
        else:
            self.fashion_mnist = MNIST(
                root="./mnist_data",
                train_valid_split_ratio=cfg.train_valid_split_ratio,
                digits_of_interest=cfg.digits_of_interest,
                fashion=cfg.fashion,
                binarize=cfg.binarize
            )["valid"]

        self.block_size = cfg.block_size
        self.num_classes = cfg.num_classes

    def __len__(self):
        return self.fashion_mnist.__len__()

    def __getitem__(self, idx):
        feed_dict = self.fashion_mnist.__getitem__(idx)
        image, idx_class = feed_dict["image"], feed_dict["digit"]
        image = 0 * (image <= 0) + 1 * (image > 0)
        image = rearrange(image, "1 h w -> (h w)", w=28, h=28)
        image = image.long()
        rand_start = random.randint(0, image.shape[-1] - 2 - self.block_size)
        input_ids = image[rand_start:rand_start + self.block_size] + self.num_classes
        idx_class = torch.tensor([idx_class])
        idx_class = 0 * (idx_class <= 0) + 1 * (idx_class > 0)
        input_ids = torch.cat([idx_class.long(), input_ids])
        labels = image[rand_start:rand_start + self.block_size + 1]
        return input_ids, labels
