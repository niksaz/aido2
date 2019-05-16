# Author: Mikita Sazanovich

from easydict import EasyDict as edict
from collections import namedtuple
import numpy as np


CFG = edict()

Label = namedtuple('Label', [
    'name',         # The identifier of this label, e.g. 'car', 'person', ... .
                    # We use them to uniquely name a class
    'id',           # An integer ID that is associated with this label.
                    # The IDs are used to represent the label in ground truth images
                    # An ID of -1 means that this label does not have an ID and thus
                    # is ignored when creating ground truth images (e.g. license plate).
                    # Do not modify these IDs, since exactly these IDs are expected by the
                    # evaluation server.
    'trainId',      # Feel free to modify these IDs as suitable for your method. Then create
                    # ground truth images with train IDs, using the tools provided in the
                    # 'preparation' folder. However, make sure to validate or submit results
                    # to our evaluation server using the regular IDs above!
                    # For trainIds, multiple labels might have the same ID. Then, these labels
                    # are mapped to the same class in the ground truth images. For the inverse
                    # mapping, we use the label that is defined first in the list below.
                    # For example, mapping all void-type classes to the same ID in training,
                    # might make sense for some approaches.
                    # Max value is 255!
    'gray_color',   # The intensity of the label that simulator renders it to
    'bgr_color',    # The color of this label to display
])

CFG.LABELS = [
    #      name                 id      trainId     gray_color  bgr_color
    Label('background',         0,      0,          0,          (70,  70,  70 )),
    Label('road',               1,      1,          51,         (128, 64,  128)),
    Label('white marking',      2,      2,          255,        (230, 150, 140)),
    Label('yellow marking',     3,      3,          226,        (70,  130, 180)),
]

# The height all images will be resized to
CFG.TO_IMG_HEIGHT = 120
# The width all images will be resized to
CFG.TO_IMG_WIDTH = 160
# The height all images fed to the model will be cropped to
CFG.IMG_HEIGHT = 80
# The width all images fed to the model will be cropped to
CFG.IMG_WIDTH = 160
# Number of object classes (road, sidewalk, car etc.)
CFG.NUM_OF_CLASSES = len(CFG.LABELS)
# The mean channel values of non-randomized gym simulator
CFG.IMG_MEAN_CHANNELS = np.array([
    63.15628621537905,
    71.4558453671647,
    73.32099227860782,
])

# Number of epochs to train the model on
CFG.TRAIN_EPOCHS = 50
# The batch size used in the model
CFG.BATCH_SIZE = 8
# The weight decay that is applied to the model
CFG.WEIGHT_DECAY = 2e-4
# The learning rate for the training
CFG.LEARNING_RATE = 5e-3
# The maximum norm of a batch's gradient
CFG.GRAD_NORM_CLIP_VALUE = 2.0
