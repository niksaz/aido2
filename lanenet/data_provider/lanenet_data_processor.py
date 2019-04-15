#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 18-5-11 下午4:58
# @Author  : Luo Yao
# @Site    : https://github.com/MaybeShewill-CV/lanenet-lane-detection
# @File    : lanenet_data_processor.py
# @IDE: PyCharm Community Edition
"""
实现LaneNet的数据解析类
"""
import os.path as ops

import cv2
import numpy as np

try:
    from cv2 import cv2
except ImportError:
    pass


class DataSet(object):
    """
    实现数据集类
    """

    def __init__(self, dataset_info_file, cfg):
        """

        :param dataset_info_file:
        """
        self._gt_img_list, self._gt_label_binary_list, self._gt_label_instance_list = \
            self._init_dataset(dataset_info_file)
        self._random_dataset()
        self._next_img_index = 0
        self._cfg = cfg

    def _init_dataset(self, dataset_info_file):
        """

        :param dataset_info_file:
        :return:
        """
        gt_img_list = []
        gt_label_binary_list = []
        gt_label_instance_list = []

        assert ops.exists(dataset_info_file), '{:s}　不存在'.format(dataset_info_file)

        with open(dataset_info_file, 'r') as file:
            for _info in file:
                info_tmp = _info.strip(' ').split()

                gt_img_list.append(info_tmp[0])
                gt_label_binary_list.append(info_tmp[1])
                gt_label_instance_list.append(info_tmp[2])

        return gt_img_list, gt_label_binary_list, gt_label_instance_list

    def _random_dataset(self):
        """

        :return:
        """
        assert len(self._gt_img_list) == len(self._gt_label_binary_list) == len(self._gt_label_instance_list)

        random_idx = np.random.permutation(len(self._gt_img_list))
        new_gt_img_list = []
        new_gt_label_binary_list = []
        new_gt_label_instance_list = []

        for index in random_idx:
            new_gt_img_list.append(self._gt_img_list[index])
            new_gt_label_binary_list.append(self._gt_label_binary_list[index])
            new_gt_label_instance_list.append(self._gt_label_instance_list[index])

        self._gt_img_list = new_gt_img_list
        self._gt_label_binary_list = new_gt_label_binary_list
        self._gt_label_instance_list = new_gt_label_instance_list

    def next_batch(self, batch_size):
        """

        :param batch_size:
        :return:
        """
        assert len(self._gt_img_list) == len(self._gt_label_binary_list) == len(self._gt_label_instance_list)

        if batch_size > len(self._gt_img_list):
            raise ValueError('Batch size不能大于样本的总数量')

        gt_imgs = []
        gt_labels_binary = []
        gt_labels_instance = []

        for max_attempts in range(4 * batch_size):
            if len(gt_imgs) == batch_size:
                break

            if self._next_img_index == len(self._gt_img_list):
                self._random_dataset()
                self._next_img_index = 0
                continue

            idx = self._next_img_index
            self._next_img_index += 1

            gt_img_path = self._gt_img_list[idx]
            gt_label_binary_path = self._gt_label_binary_list[idx]
            gt_label_instance_path = self._gt_label_instance_list[idx]

            gt_img = cv2.imread(gt_img_path, cv2.IMREAD_COLOR)

            label_img = cv2.imread(gt_label_binary_path, cv2.IMREAD_COLOR)
            label_binary = np.zeros([label_img.shape[0], label_img.shape[1]], dtype=np.uint8)
            idx = np.where((label_img[:, :, :] != [0, 0, 0]).all(axis=2))
            label_binary[idx] = 1
            gt_label_binary = label_binary

            label_img = cv2.imread(gt_label_instance_path, cv2.IMREAD_UNCHANGED)
            gt_label_instance = label_img

            if self._cfg.TRAIN.CROP_IMAGE:
                height_to_crop = gt_img.shape[0] // 3
                gt_img = gt_img[height_to_crop:, :]
                gt_label_binary = gt_label_binary[height_to_crop:, :]
                gt_label_instance = gt_label_instance[height_to_crop:, :]

            gt_img = cv2.resize(gt_img,
                                dsize=(self._cfg.TRAIN.IMG_WIDTH, self._cfg.TRAIN.IMG_HEIGHT),
                                dst=gt_img,
                                interpolation=cv2.INTER_LINEAR)
            gt_img = gt_img - self._cfg.TRAIN.CHANNEL_MEANS

            gt_label_binary = cv2.resize(gt_label_binary,
                                         dsize=(self._cfg.TRAIN.IMG_WIDTH, self._cfg.TRAIN.IMG_HEIGHT),
                                         dst=gt_label_binary,
                                         interpolation=cv2.INTER_NEAREST)
            gt_label_binary = np.expand_dims(gt_label_binary, axis=-1)
            gt_label_instance = cv2.resize(gt_label_instance,
                                           dsize=(self._cfg.TRAIN.IMG_WIDTH, self._cfg.TRAIN.IMG_HEIGHT),
                                           dst=gt_label_instance,
                                           interpolation=cv2.INTER_NEAREST)

            # Check that the ground truth segmentation contains at least 1 road-marking pixel
            idx = np.where(gt_label_binary[:, :, 0] == [1])
            if len(idx[0]) == 0:
                continue
            gt_imgs.append(gt_img)
            gt_labels_binary.append(gt_label_binary)
            gt_labels_instance.append(gt_label_instance)

        assert len(gt_imgs) == batch_size
        return gt_imgs, gt_labels_binary, gt_labels_instance


if __name__ == '__main__':
    from easydict import EasyDict as edict

    # Dataset config options
    cfg = edict()
    cfg.TRAIN = edict()
    cfg.TRAIN.IMG_WIDTH = 320
    cfg.TRAIN.IMG_HEIGHT = 160
    cfg.TRAIN.CROP_IMAGE = True
    cfg.TRAIN.CHANNEL_MEANS = [103.939, 116.779, 123.68]

    train = DataSet('/Users/niksaz/4-JetBrains/dataset-random/train.txt', cfg)
    b1, b2, b3 = train.next_batch(50)
    c1, c2, c3 = train.next_batch(50)
    dd, d2, d3 = train.next_batch(50)
