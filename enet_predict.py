# Author: Mikita Sazanovich

import numpy as np
import pickle
import tensorflow as tf
import cv2
import os
import re
import time
import matplotlib.pyplot as plt
from collections import deque
from pathlib import Path
from typing import List, Tuple

from enet.config import CFG
from enet.model import ENetModel
from enet.preprocess_data import preprocess_image

import pydensecrf.densecrf as dcrf
from pydensecrf.utils import create_pairwise_bilateral, create_pairwise_gaussian, unary_from_softmax


class EnetPredictor:

    def __init__(self):
        data_dir = '/Users/niksaz/4-JetBrains/duckscapes-x4'
        mean_channels_path = os.path.join(data_dir, 'mean_channels.pkl')
        mean_channels = pickle.load(open(mean_channels_path, 'rb'))
        self.mean_channels = mean_channels
        self.batch_size = 1
        self.img_height = CFG.IMG_HEIGHT
        self.img_width = CFG.IMG_WIDTH
        self.no_of_classes = CFG.NUM_OF_CLASSES
        self.model = ENetModel(self.batch_size, self.img_height, self.img_width, self.no_of_classes)

        checkpoint_dir = '/Users/niksaz/4-JetBrains/aido2/logs/model_clipped/checkpoints'
        checkpoint_name = 'model_epoch_98.ckpt'
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)

        saver = tf.train.Saver(tf.trainable_variables(), write_version=tf.train.SaverDef.V2)

        self.sess = tf.Session()
        init = tf.global_variables_initializer()
        self.sess.run(init)

        saver.restore(self.sess, checkpoint_path)

    def infer_and_postprocess(self, img_read: np.ndarray) -> np.ndarray:
        img = preprocess_image(img_read)
        img = img - self.mean_channels

        batch_imgs = np.zeros((1, self.img_height, self.img_width, 3), dtype=np.float32)
        batch_imgs[0] = img

        batch_feed_dict = self.model.create_feed_dict(imgs_batch=batch_imgs,
                                                      early_drop_prob=0.0,
                                                      late_drop_prob=0.0)
        softmaxes = self.sess.run(self.model.softmax, feed_dict=batch_feed_dict)

        img_to_show = img + self.mean_channels
        predictions = np.argmax(softmaxes, axis=3)
        predictions = predictions.squeeze()

        softs = softmaxes.squeeze()
        softs = softs.transpose((2, 0, 1))

        unary = unary_from_softmax(softs)

        # The inputs should be C-continious -- we are using Cython wrapper
        unary = np.ascontiguousarray(unary)

        d = dcrf.DenseCRF(img.shape[0] * img.shape[1], 4)

        d.setUnaryEnergy(unary)

        # This potential penalizes small pieces of segmentation that are
        # spatially isolated -- enforces more spatially consistent segmentations
        feats = create_pairwise_gaussian(sdims=(50, 50), shape=img.shape[:2])

        d.addPairwiseEnergy(feats, compat=3,
                            kernel=dcrf.DIAG_KERNEL,
                            normalization=dcrf.NORMALIZE_SYMMETRIC)

        # This creates the color-dependent features --
        # because the segmentation that we get from CNN are too coarse
        # and we can use local color features to refine them
        feats = create_pairwise_bilateral(sdims=(50, 50), schan=(20, 20, 20), img=img, chdim=2)

        d.addPairwiseEnergy(feats, compat=10,
                            kernel=dcrf.DIAG_KERNEL,
                            normalization=dcrf.NORMALIZE_SYMMETRIC)
        Q = d.inference(5)

        res = np.argmax(Q, axis=0).reshape((img.shape[0], img.shape[1]))
        return img_to_show, predictions, res


def infer_and_postprocess_full(img_read: np.ndarray, predictor: EnetPredictor) -> None:
    t_start = time.time()
    img = preprocess_image(img_read)
    img = img - predictor.mean_channels

    batch_imgs = np.zeros((1, predictor.img_height, predictor.img_width, 3), dtype=np.float32)
    batch_imgs[0] = img

    batch_feed_dict = predictor.model.create_feed_dict(imgs_batch=batch_imgs, early_drop_prob=0.0,
                                                       late_drop_prob=0.0)
    softmaxes = predictor.sess.run(predictor.model.softmax, feed_dict=batch_feed_dict)

    img_to_show = img + predictor.mean_channels
    predictions = np.argmax(softmaxes, axis=3)
    predictions = predictions.squeeze()

    height_to_reshape_to = img_read.shape[0] - img_read.shape[0] // 3
    width_to_reshape_to = img_read.shape[1]
    predictions_to_show = np.zeros((img_read.shape[0], img_read.shape[1]))
    predictions_to_show[-height_to_reshape_to:, -width_to_reshape_to:] = cv2.resize(
        predictions, (width_to_reshape_to, height_to_reshape_to), interpolation=cv2.INTER_NEAREST)

    softs = softmaxes.squeeze()
    softs = cv2.resize(softs, (width_to_reshape_to, height_to_reshape_to),
                       interpolation=cv2.INTER_NEAREST)
    softs = softs.transpose((2, 0, 1))

    unary = unary_from_softmax(softs)

    # The inputs should be C-continious -- we are using Cython wrapper
    unary = np.ascontiguousarray(unary)

    d = dcrf.DenseCRF(height_to_reshape_to * width_to_reshape_to, 4)

    d.setUnaryEnergy(unary)

    # This potential penalizes small pieces of segmentation that are
    # spatially isolated -- enforces more spatially consistent segmentations
    feats = create_pairwise_gaussian(sdims=(10, 10),
                                     shape=(height_to_reshape_to, width_to_reshape_to))

    d.addPairwiseEnergy(feats, compat=3,
                        kernel=dcrf.DIAG_KERNEL,
                        normalization=dcrf.NORMALIZE_SYMMETRIC)

    # This creates the color-dependent features --
    # because the segmentation that we get from CNN are too coarse
    # and we can use local color features to refine them
    feats = create_pairwise_bilateral(
        sdims=(50, 50), schan=(20, 20, 20),
        img=img_read[-height_to_reshape_to:, -width_to_reshape_to:, :], chdim=2)

    d.addPairwiseEnergy(feats, compat=10,
                        kernel=dcrf.DIAG_KERNEL,
                        normalization=dcrf.NORMALIZE_SYMMETRIC)
    Q = d.inference(5)

    res = np.argmax(Q, axis=0).reshape((height_to_reshape_to, width_to_reshape_to))
    t_cost = time.time() - t_start
    print(f'Inference took: {t_cost}s')

    res_to_show = np.zeros((img_read.shape[0], img_read.shape[1]))
    res_to_show[-height_to_reshape_to:, -width_to_reshape_to:] = res

    f, (ax0, ax1, ax2) = plt.subplots(1, 3, sharey=True, figsize=(18, 8))
    ax0.set_title('Input image')
    ax0.imshow(img_read[:, :, (2, 1, 0)])
    ax1.set_title('Raw segmentation')
    ax1.imshow(predictions_to_show, vmin=0, vmax=CFG.NUM_OF_CLASSES - 1)
    ax2.set_title('Segmentation with CRF post-processing')
    ax2.imshow(res_to_show, vmin=0, vmax=CFG.NUM_OF_CLASSES - 1)
    plt.tight_layout()
    plt.show()


def find_border_lines(input_seg: np.ndarray) -> Tuple[List, List]:
    seg = input_seg.copy()
    label_name_to_id = {label.name: label.id for label in CFG.LABELS}
    background_id = label_name_to_id['background']
    road_id = label_name_to_id['road']
    white_marking_id = label_name_to_id['white marking']
    yellow_marking_id = label_name_to_id['yellow marking']
    N, M = seg.shape[0], seg.shape[1]
    x, y = None, None
    for dy in [0, 1]:
        for dx in [0, 1]:
            x = N - 1 + dx
            y = M // 2 - 1 + dy
            if seg[x, y] == road_id:
                break
    if seg[x, y] != road_id:
        return [], []

    q = deque()
    q.append((x, y))
    white_pixels = []
    yellow_pixels = []
    while len(q) > 0:
        x, y = q.pop()
        seg[x, y] = background_id
        for dx in [-1, +1]:
            for dy in [-1, +1]:
                nx = x + dx
                ny = y + dy
                if 0 <= nx < N and 0 <= ny < M:
                    if seg[nx, ny] == background_id:
                        continue
                    elif seg[nx, ny] == road_id:
                        q.append((nx, ny))
                    elif seg[nx, ny] == white_marking_id:
                        white_pixels.append((nx, ny))
                    elif seg[nx, ny] == yellow_marking_id:
                        yellow_pixels.append((nx, ny))
    return white_pixels, yellow_pixels


def infer_and_visualize_subset_dir(
        dir_path_str: str,
        filename_pattern: str,
        predictor: EnetPredictor,
        samples_to_take: int = 5,
        seed: int = 82) -> None:
    np.random.seed(seed)
    dir_path = Path(dir_path_str)
    image_paths = [str(path) for path in dir_path.glob('*.png') if re.search(filename_pattern, str(path))]
    image_paths = list(sorted(image_paths))
    image_paths_sub = np.random.choice(image_paths, samples_to_take, replace=False)
    for image_path in image_paths_sub:
        image = cv2.imread(str(image_path), -1)
        t_start = time.time()
        img_to_show, predictions, res = predictor.infer_and_postprocess(image)
        t_cost = time.time() - t_start
        print(f'Inference took: {t_cost}s')

        f, (ax0, ax1, ax2) = plt.subplots(1, 3, sharey=True, figsize=(18, 8))
        ax0.set_title('Input image')
        ax0.imshow(img_to_show[:, :, (2, 1, 0)].astype(np.uint8))
        ax1.set_title('Raw segmentation')
        ax1.imshow(predictions, vmin=0, vmax=CFG.NUM_OF_CLASSES - 1)
        ax2.set_title('Segmentation with CRF post-processing')
        ax2.imshow(res, vmin=0, vmax=CFG.NUM_OF_CLASSES - 1)
        plt.tight_layout()
        plt.show()

        white_pixels, yellow_pixels = find_border_lines(res)
        for x, y in white_pixels:
            img_to_show[x, y] = (255, 255, 255)
            # cv2.circle(img_to_show, (y, x), 1, (255, 255, 255))
        for x, y in yellow_pixels:
            img_to_show[x, y] = (0, 255, 255)
            # cv2.circle(img_to_show, (y, x), 1, (0, 255, 255))
        plt.figure(figsize=(18, 8))
        plt.imshow(img_to_show[:, :, (2, 1, 0)].astype(np.uint8))
        plt.axis('off')
        plt.show()


def main():
    predictor = EnetPredictor()

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
