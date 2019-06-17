# Author: Mikita Sazanovich

import os
import cv2
import numpy as np

from typing import List, Tuple
from enet.config import CFG


class Dataset:
    def __init__(self,
                 data_dir: str,
                 image_pair_paths: List[Tuple[str, str]],
                 img_height: int,
                 img_width: int,
                 no_of_classes: int):

        self.data_dir = data_dir
        self.image_pair_paths = image_pair_paths
        self.image_pair_index = 0
        self.reset()

        self.mean_channels = CFG.IMG_MEAN_CHANNELS

        self.img_height = img_height
        self.img_width = img_width
        self.no_of_classes = no_of_classes

    def reset(self):
        np.random.shuffle(self.image_pair_paths)
        self.image_pair_index = 0

    def convert_filename_to_path(self, filename: str) -> str:
        return os.path.join(self.data_dir, filename)

    def _get_next_image(self) -> Tuple[np.ndarray, np.ndarray]:
        if self.image_pair_index == len(self.image_pair_paths):
            self.reset()
            return self._get_next_image()
        else:
            image_path, label_path = self.image_pair_paths[self.image_pair_index]
            self.image_pair_index += 1

            # read the next img:
            img = cv2.imread(self.convert_filename_to_path(image_path), -1)
            img = img - self.mean_channels

            # read the next label:
            label = cv2.imread(self.convert_filename_to_path(label_path), -1)

            # convert the label to onehot:
            onehot_label = np.zeros(
                (self.img_height, self.img_width, self.no_of_classes), dtype=np.float32)
            layer_idx = np.arange(self.img_height).reshape(self.img_height, 1)
            component_idx = np.tile(np.arange(self.img_width), (self.img_height, 1))
            onehot_label[layer_idx, component_idx, label] = 1

            return img, onehot_label

    def get_next_batch(self, batch_size):
        assert batch_size <= len(self.image_pair_paths)

        batch_imgs = np.zeros((batch_size, self.img_height, self.img_width, 3), dtype=np.float32)
        batch_onehot_labels = np.zeros(
            (batch_size, self.img_height, self.img_width, self.no_of_classes), dtype=np.float32)

        for i in range(batch_size):
            batch_imgs[i], batch_onehot_labels[i] = self._get_next_image()
        return batch_imgs, batch_onehot_labels

    def get_batches_in_dataset(self, batch_size):
        return len(self.image_pair_paths) // batch_size
