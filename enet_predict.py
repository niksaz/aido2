# Author: Mikita Sazanovich

import numpy as np
import pickle
import tensorflow as tf
import cv2
import os
import re
import time
import matplotlib.pyplot as plt
from pathlib import Path

from enet.config import CFG
from enet.utilities import label_img_to_color
from enet.model import ENetModel


class EnetPredictor:

    def __init__(self, config_dir):
        mean_channels_path = os.path.join(config_dir, 'mean_channels.pkl')
        mean_channels = pickle.load(open(mean_channels_path, 'rb'))
        self.mean_channels = mean_channels
        self.batch_size = 1
        self.img_height = CFG.IMG_HEIGHT
        self.img_width = CFG.IMG_WIDTH
        self.no_of_classes = CFG.NUM_OF_CLASSES
        self.model = ENetModel(self.batch_size, self.img_height, self.img_width, self.no_of_classes)

        checkpoint_name = 'model_epoch_99.ckpt'
        checkpoint_path = os.path.join(config_dir, checkpoint_name)

        saver = tf.train.Saver(tf.trainable_variables(), write_version=tf.train.SaverDef.V2)

        self.sess = tf.Session()
        init = tf.global_variables_initializer()
        self.sess.run(init)

        saver.restore(self.sess, checkpoint_path)

    def infer_and_visualize_image(self, img: np.ndarray):
        img_read = img.copy()
        height_to_drop = img.shape[0] // 3
        img = img[height_to_drop:, :, :]
        img = cv2.resize(img, (self.img_width, self.img_height))
        img = img - self.mean_channels

        batch_imgs = np.zeros((1, self.img_height, self.img_width, 3), dtype=np.float32)
        batch_imgs[0] = img

        batch_feed_dict = self.model.create_feed_dict(
            imgs_batch=batch_imgs, early_drop_prob=0.0, late_drop_prob=0.0)

        t_start = time.time()
        logits = self.sess.run(self.model.logits, feed_dict=batch_feed_dict)
        t_cost = time.time() - t_start
        print('Inference took: {:.5f}s'.format(t_cost))

        predictions = np.argmax(logits, axis=3)
        pred_img = predictions[0]
        pred_img_color = label_img_to_color(pred_img)

        height_to_reshape_to = img_read.shape[0] - height_to_drop
        width_to_reshape_to = img_read.shape[1]
        pred_img_color = cv2.resize(
            pred_img_color, (width_to_reshape_to, height_to_reshape_to),
            interpolation=cv2.INTER_NEAREST)
        shaped_res = np.zeros_like(img_read)
        shaped_res[height_to_drop:, :, :] = pred_img_color

        images = [img_read, (0.1 * img_read + 0.9 * shaped_res).astype(np.uint8)]

        fig = plt.figure(figsize=(6 * len(images), 6))
        for i, image in enumerate(images):
            ax = fig.add_subplot(1, len(images), i + 1)
            ax.set_axis_off()
            plt.imshow(image[:, :, (2, 1, 0)])
        plt.tight_layout()
        plt.show()


def infer_and_visualize_subset_dir(
        dir_path_str: str,
        filename_pattern: str,
        predictor: EnetPredictor) -> None:
    np.random.seed(20)
    dir_path = Path(dir_path_str)
    image_paths = [str(path) for path in dir_path.glob('*.png') if re.search(filename_pattern, str(path))]
    image_paths = list(sorted(image_paths))
    image_paths_sub = np.random.choice(image_paths, 4, replace=False)
    for image_path in image_paths_sub:
        image = cv2.imread(str(image_path), -1)
        predictor.infer_and_visualize_image(image)


def main():
    predictor = EnetPredictor('/Users/niksaz/4-JetBrains/aido2/enet-config')

    infer_and_visualize_subset_dir(
        '/Users/niksaz/4-JetBrains/tiny/train/gt_image',
        '.*png',
        predictor)
    infer_and_visualize_subset_dir(
        '/Users/niksaz/4-JetBrains/tiny/val/gt_image',
        '.*png',
        predictor)
    infer_and_visualize_subset_dir(
        '/Users/niksaz/4-JetBrains/explore/data-logs',
        '.*_in\.png',
        predictor)
    infer_and_visualize_subset_dir(
        '/Users/niksaz/4-JetBrains/explore/data-jb/lane-data/rect',
        '.*\.png',
        predictor)


if __name__ == '__main__':
    main()
