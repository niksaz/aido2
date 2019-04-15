# Author: Mikita Sazanovich

import time

import tensorflow as tf
import glog as log
import numpy as np
import matplotlib.pyplot as plt
import cv2

from lanenet.lanenet_model import lanenet_merge_model
from lanenet.lanenet_model import lanenet_cluster
from lanenet.lanenet_model import lanenet_postprocess
from lanenet.config.global_config import cfg


class LaneNetPredictor:
    def __init__(self, weights_path: str):
        self.cfg = {
            'IMG_HEIGHT': cfg.TRAIN.IMG_HEIGHT,
            'IMG_WIDTH': cfg.TRAIN.IMG_WIDTH,
            'CROP_IMAGE': cfg.TRAIN.CROP_IMAGE,
            'IMG_MEAN': cfg.TRAIN.CHANNEL_MEANS,
        }
        self.input_tensor = tf.placeholder(
            dtype=tf.float32,
            shape=[1, self.cfg['IMG_HEIGHT'], self.cfg['IMG_WIDTH'], 3],
            name='input_tensor')
        phase_tensor = tf.constant('test', tf.string)

        net = lanenet_merge_model.LaneNet(phase=phase_tensor, net_flag='vgg')
        self.binary_seg_ret, self.instance_seg_ret = \
            net.inference(input_tensor=self.input_tensor, name='lanenet_model')

        self.cluster = lanenet_cluster.LaneNetCluster()
        self.postprocessor = lanenet_postprocess.LaneNetPoseProcessor()

        sess_config = tf.ConfigProto(device_count={'CPU': 1})
        self.sess = tf.Session(config=sess_config)

        saver = tf.train.Saver()
        saver.restore(sess=self.sess, save_path=weights_path)

    def predict_lanes(self, image: np.ndarray):
        if self.cfg['CROP_IMAGE']:
            height_to_crop = image.shape[0] // 3
            image = image[height_to_crop:, :]
        image = cv2.resize(image, (self.cfg['IMG_WIDTH'], self.cfg['IMG_HEIGHT']), interpolation=cv2.INTER_LINEAR)
        image = image - self.cfg['IMG_MEAN']

        t_start = time.time()
        binary_seg_image, instance_seg_image = self.sess.run(
            [self.binary_seg_ret, self.instance_seg_ret],
            feed_dict={self.input_tensor: [image]})
        t_cost = time.time() - t_start
        log.info('Inference took: {:.5f}s'.format(t_cost))

        binary_seg_image[0] = self.postprocessor.postprocess(binary_seg_image[0])
        mask_image = self.cluster.get_lane_mask(binary_seg_ret=binary_seg_image[0],
                                                instance_seg_ret=instance_seg_image[0])

        return binary_seg_image[0] * 255, mask_image

    def predict_and_visualize_lanes(self, image: np.ndarray):
        mask_image, embedding_image = self.predict_lanes(image)
        images = [image, mask_image, embedding_image]
        fig = plt.figure(figsize=(15, 5))
        for i, image in enumerate(images):
            ax = fig.add_subplot(1, 3, i + 1)
            ax.set_axis_off()
            if i == 1:
                plt.imshow(image, cmap='gray')
            else:
                plt.imshow(image[:, :, (2, 1, 0)])
        plt.tight_layout()
        plt.show()

    @staticmethod
    def _minmax_scale(input_arr):
        min_val = np.min(input_arr)
        max_val = np.max(input_arr)

        output_arr = (input_arr - min_val) * 255.0
        if min_val < max_val:
            output_arr /= (max_val - min_val)
        return output_arr
