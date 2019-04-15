# Author: Mikita Sazanovich

import time

import tensorflow as tf
import glog as log
import numpy as np
import matplotlib.pyplot as plt
import cv2

from easydict import EasyDict as edict

from lanenet.lanenet_model import lanenet_merge_model
from lanenet.lanenet_model import lanenet_cluster
from lanenet.lanenet_model import lanenet_postprocess
from lanenet.config.global_config import cfg


class LaneNetPredictor:
    def __init__(self, weights_path: str):
        self.cfg = edict()
        self.cfg.img_height = cfg.TRAIN.IMG_HEIGHT
        self.cfg.img_width = cfg.TRAIN.IMG_WIDTH
        self.cfg.crop_image = cfg.TRAIN.CROP_IMAGE
        self.cfg.img_mean = cfg.TRAIN.CHANNEL_MEANS
        self.input_tensor = tf.placeholder(
            dtype=tf.float32,
            shape=[1, self.cfg.img_height, self.cfg.img_width, 3],
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

    def predict_lanes(self,
                      image: np.ndarray,
                      return_embedding: bool = False):
        # Crop the image if it is needed
        if self.cfg.crop_image:
            height_to_crop = image.shape[0] // 3
            image_cropped = image[height_to_crop:, :]
        else:
            height_to_crop = 0
            image_cropped = image

        # Downsample the image
        image = cv2.resize(
            image_cropped, (self.cfg.img_width, self.cfg.img_height), interpolation=cv2.INTER_LINEAR)
        image = image - self.cfg.img_mean

        # Infer the mask and the lane instances
        t_start = time.time()
        binary_seg_image, instance_seg_image = self.sess.run(
            [self.binary_seg_ret, self.instance_seg_ret], feed_dict={self.input_tensor: [image]})
        binary_seg_image = binary_seg_image[0]
        instance_seg_image = instance_seg_image[0]
        t_cost = time.time() - t_start
        log.info('Inference took: {:.5f}s'.format(t_cost))

        binary_seg_image = self.postprocessor.postprocess(binary_seg_image)
        mask_image = self.cluster.get_lane_mask(binary_seg_ret=binary_seg_image,
                                                instance_seg_ret=instance_seg_image)

        # Recover the inferred masks to the original size
        binary_seg_image = LaneNetPredictor._recover_mask(binary_seg_image, image_cropped.shape, height_to_crop)
        mask_image = LaneNetPredictor._recover_mask(mask_image, image_cropped.shape, height_to_crop)

        if return_embedding:
            instance_seg_image = LaneNetPredictor._recover_mask(instance_seg_image, image_cropped.shape, height_to_crop)
            for i in range(instance_seg_image.shape[2]):
                instance_seg_image[:, :, i] = LaneNetPredictor._minmax_scale(instance_seg_image[:, :, i])
            embedding_image = np.array(instance_seg_image, np.uint8)
            return embedding_image, binary_seg_image, mask_image
        else:
            return binary_seg_image, mask_image

    def predict_and_visualize_lanes(self, image: np.ndarray):
        embedding_image, binary_seg_image, mask_image = \
            self.predict_lanes(image, return_embedding=True)
        images = [image, embedding_image, binary_seg_image, mask_image]
        fig = plt.figure(figsize=(5 * len(images), 5))
        for i, image in enumerate(images):
            ax = fig.add_subplot(1, len(images), i + 1)
            ax.set_axis_off()
            if len(image.shape) == 2:
                plt.imshow(image, cmap='gray')
            else:
                plt.imshow(image[:, :, (2, 1, 0)])
        plt.tight_layout()
        plt.show()

    @staticmethod
    def _recover_mask(mask: np.ndarray, dst_shape: tuple, height_being_cropped: int):
        mask = cv2.resize(mask, dsize=(dst_shape[1], dst_shape[0]), dst=mask, interpolation=cv2.INTER_NEAREST)
        if len(mask.shape) == 2:
            res = np.zeros((mask.shape[0] + height_being_cropped, mask.shape[1]), mask.dtype)
            res[height_being_cropped:, :] = mask[:, :]
        else:
            res = np.zeros((mask.shape[0] + height_being_cropped, mask.shape[1], mask.shape[2]), mask.dtype)
            res[height_being_cropped:, :, :] = mask[:, :, :]
        return res

    @staticmethod
    def _minmax_scale(input_arr):
        min_val = np.min(input_arr)
        max_val = np.max(input_arr)

        output_arr = (input_arr - min_val) * 255.0
        if min_val < max_val:
            output_arr /= (max_val - min_val)
        return output_arr
