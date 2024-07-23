# -*- coding:utf-8 -*-
import ray
import torch
import pickle
import random
import numpy as np
from typing import List
from dataset.augmentation import SignalAugmentation as SigAug
import warnings


warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)


def batch_dataloader(paths: List, batch_size: int, augmentations: List):
    augmentation = SigAug()
    np.random.shuffle(paths)

    def get_data(paths_: str) -> np.array:
        with open(paths_, 'rb') as fp:
            data = pickle.load(fp)
            x, info = data['x'], data['info']

        x = list(x)
        return x

    def convert_augmentation(x: np.array) -> (np.array, np.array):
        # selection augmentation name
        x = np.array(x)
        augmentation_1, augmentation_2 = random.sample(augmentations, 2)
        aug_name_1, aug_prob_1 = augmentation_1
        aug_name_2, aug_prob_2 = augmentation_2
        x1 = augmentation.process(x, aug_name=aug_name_1, p=aug_prob_1)
        x2 = augmentation.process(x, aug_name=aug_name_2, p=aug_prob_2)
        return x1, x2

    def convert_tensor(x1: np.array, x2: np.array) -> (torch.Tensor, torch.Tensor):
        x1 = torch.tensor(x1, dtype=torch.float32)
        x2 = torch.tensor(x2, dtype=torch.float32)
        return x1, x2

    it = (
        ray.util.iter.from_items(paths, num_shards=5)
                     .for_each(lambda x_: get_data(x_))
                     .flatten()
                     .local_shuffle(shuffle_buffer_size=2)
                     .batch(batch_size)
                     .for_each(lambda x_: convert_augmentation(x_))
                     .for_each(lambda x_: convert_tensor(x_[0], x_[1]))
    )
    return it

