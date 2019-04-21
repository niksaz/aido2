# Author: Mikita Sazanovich

import argparse
import os
from pathlib import Path

import cv2
import numpy as np
from lanenet import LaneNetPredictor


def infer_and_visualize_image(image_path: str, lane_net_predictor: LaneNetPredictor):
    assert os.path.exists(image_path), '{:s} not exist'.format(image_path)
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    lane_net_predictor.predict_and_visualize_lanes(image)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('weights_path', type=str, help='Path to the model weights')
    return parser.parse_args()


def main():
    args = parse_args()
    lane_net_predictor = LaneNetPredictor(args.weights_path)

    IMAGE_DIR = Path('/Users/niksaz/4-JetBrains/dataset-random/gt_image')
    filenames = list(sorted(IMAGE_DIR.glob('*.png')))
    for filename in np.random.choice(filenames, 5, replace=False):
        infer_and_visualize_image(str(filename), lane_net_predictor)
    return

    IMAGE_DIR = Path('/Users/niksaz/4-JetBrains/dataset-val/gt_image')
    filenames = list(sorted(IMAGE_DIR.glob('*.png')))
    for filename in np.random.choice(filenames, 5, replace=False):
        image_path = os.path.join(IMAGE_DIR, filename)
        infer_and_visualize_image(image_path, lane_net_predictor)

    IMAGE_DIR = Path('/Users/niksaz/4-JetBrains/explore/data-logs')
    filenames = list(sorted(IMAGE_DIR.glob('*_in.png')))
    for filename in np.random.choice(filenames, 5, replace=False):
        image_path = os.path.join(IMAGE_DIR, filename)
        infer_and_visualize_image(image_path, lane_net_predictor)

    IMAGE_DIR = Path('/Users/niksaz/4-JetBrains/explore/data-jb/lane-data/rect')
    filenames = list(sorted(IMAGE_DIR.glob('*.png')))
    for filename in np.random.choice(filenames, 5, replace=False):
        image_path = os.path.join(IMAGE_DIR, filename)
        infer_and_visualize_image(image_path, lane_net_predictor)


if __name__ == '__main__':
    main()
