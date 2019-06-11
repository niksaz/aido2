# Author: Mikita Sazanovich

import random
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
        random.shuffle(self.image_pair_paths)
        self.image_pair_index = 0

    def preprocess_path(self, path: str) -> str:
        return os.path.join(
            self.data_dir, path.replace('/Users/niksaz/4-JetBrains/generate-data/distscapes-4/', '',))

    def _get_next_image(self) -> Tuple[np.ndarray, np.ndarray]:
        if self.image_pair_index == len(self.image_pair_paths):
            self.reset()
            return self._get_next_image()
        else:
            image_path, trainId_label_path = self.image_pair_paths[self.image_pair_index]
            self.image_pair_index += 1

            # read the next img:
            img = cv2.imread(self.preprocess_path(image_path), -1)
            img = img - self.mean_channels

            # read the next label:
            trainId_label = cv2.imread(self.preprocess_path(trainId_label_path), -1)

            # convert the label to onehot:
            onehot_label = np.zeros(
                (self.img_height, self.img_width, self.no_of_classes), dtype=np.float32)
            layer_idx = np.arange(self.img_height).reshape(self.img_height, 1)
            component_idx = np.tile(np.arange(self.img_width), (self.img_height, 1))
            onehot_label[layer_idx, component_idx, trainId_label] = 1

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


if __name__ == '__main__':
    data_dir = '/Users/niksaz/4-JetBrains/aido2/segmentation/data'
    from pathlib import Path
    data_path = Path(data_dir)
    import numpy as np
    import re
    import pickle
    np.random.seed(64)
    img_paths = [str(path) for path in data_path.glob('*.png') if re.search(r'.*[0-9]+.png', str(path))]
    img_paths = np.random.choice(img_paths, 16, replace=False)
    label_paths = [img_path.replace('.png', '_label.png')for img_path in img_paths]
    pairs = list(zip(img_paths, label_paths))
    img_height = CFG.IMG_HEIGHT
    img_width = CFG.IMG_WIDTH
    no_of_classes = CFG.NUM_OF_CLASSES
    dataset = Dataset(data_dir, pairs, img_height, img_width, no_of_classes)
    for _ in range(dataset.get_batches_in_dataset(batch_size=4)):
        imgs, labels = dataset.get_next_batch(batch_size=4)
