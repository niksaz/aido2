import cv2
import pickle
import os
import numpy as np
import random
import pathlib
import argparse

from enet.config import CFG


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('raw_dataset_dir', type=str, help='Dataset directory to process')
    parser.add_argument('dataset_dir', type=str, help='Directory to save the dataset to')
    return parser.parse_args()


def preprocess_image(image, interpolation=None):
    height_to_drop = image.shape[0] // 3
    if len(image.shape) == 2:
        image = image[height_to_drop:, :]
    else:
        image = image[height_to_drop:, :, :]
    if interpolation is None:
        image = cv2.resize(image, (CFG.IMG_WIDTH, CFG.IMG_HEIGHT))
    else:
        image = cv2.resize(image, (CFG.IMG_WIDTH, CFG.IMG_HEIGHT), interpolation=interpolation)
    return image


def process_dir(dirname, raw_dataset_dir, dataset_dir, id_to_trainId_map_func):
    proc_img_paths = []
    proc_trainId_label_paths = []

    dir_to_process = os.path.join(raw_dataset_dir, dirname)
    gt_image_dir = os.path.join(dir_to_process, 'gt_image')
    gt_seg_dir = os.path.join(dir_to_process, 'gt_seg')
    gt_image_path = pathlib.Path(gt_image_dir)
    img_paths = list(gt_image_path.glob('*.png'))
    for step, img_path in enumerate(img_paths):
        if step % 10 == 0:
            print(f'{dirname} step {step}/{len(img_paths) - 1}')

        img_name = img_path.name
        new_img_name = f'{dirname}_{img_name}'

        # read the image:
        img_path = os.path.join(gt_image_dir, img_name)
        img = cv2.imread(img_path, -1)

        # resize the image and save to project_data_dir:
        img_small = preprocess_image(img)
        img_small_path = os.path.join(dataset_dir, new_img_name)
        cv2.imwrite(img_small_path, img_small)
        assert os.path.exists(img_small_path)
        proc_img_paths.append(img_small_path)

        # read and resize the corresponding label image without interpolation
        # (want the resulting image to still only contain pixel values
        # corresponding to an object class):
        gt_img_path = os.path.join(gt_seg_dir, img_name)
        gt_img = cv2.imread(gt_img_path, -1)
        gt_img_small = preprocess_image(gt_img, interpolation=cv2.INTER_NEAREST)

        # convert the label image from id to trainId pixel values:
        id_label = gt_img_small
        trainId_label = id_to_trainId_map_func(id_label)

        # save the label image to project_data_dir:
        label_img_name = f'{new_img_name.split(".png")[0]}_label.png'
        trainId_label_path = os.path.join(dataset_dir, label_img_name)
        cv2.imwrite(trainId_label_path, trainId_label)
        assert os.path.exists(trainId_label_path)
        proc_trainId_label_paths.append(trainId_label_path)

    return proc_img_paths, proc_trainId_label_paths


def main():
    args = parse_args()
    raw_dataset_dir = args.raw_dataset_dir
    dataset_dir = args.dataset_dir
    os.makedirs(dataset_dir, exist_ok=True)

    # create a function mapping id to trainId:
    id_to_trainId = {label.gray_color: label.trainId for label in CFG.LABELS}
    id_to_trainId_map_func = np.vectorize(id_to_trainId.get)

    img_height = CFG.IMG_HEIGHT
    img_width = CFG.IMG_WIDTH
    no_of_classes = CFG.NUM_OF_CLASSES

    # get the path to all training images and their corresponding label image:
    train_img_paths, train_trainId_label_paths = process_dir(
        'train', raw_dataset_dir, dataset_dir, id_to_trainId_map_func)

    # compute the mean color channels of the train imgs:
    print("computing mean color channels of the train imgs")
    no_of_train_imgs = len(train_img_paths)
    mean_channels = np.zeros((3, ))
    for step, img_path in enumerate(train_img_paths):
        if step % 100 == 0:
            print(step)

        img = cv2.imread(img_path, -1)
        assert img is not None

        img_mean_channels = np.mean(img, axis=0)
        img_mean_channels = np.mean(img_mean_channels, axis=0)

        mean_channels += img_mean_channels

    mean_channels = mean_channels/float(no_of_train_imgs)

    # # save to disk:
    mean_channels_path = os.path.join(dataset_dir, 'mean_channels.pkl')
    pickle.dump(mean_channels, open(mean_channels_path, 'wb'))

    # compute the class weights:
    print("computing class weights")
    trainId_to_count = {}
    for trainId in range(no_of_classes):
        trainId_to_count[trainId] = 0

    # # get the total number of pixels in all train labels that are of each
    # # object class:
    for step, trainId_label_path in enumerate(train_trainId_label_paths):
        if step % 100 == 0:
            print(step)

        # read the label image:
        trainId_label = cv2.imread(trainId_label_path, -1)

        for trainId in range(no_of_classes):
            # count how many pixels in the label image are of object class trainId:
            trainId_mask = np.equal(trainId_label, trainId)
            label_trainId_count = np.sum(trainId_mask)

            # add to the total count:
            trainId_to_count[trainId] += label_trainId_count

    # # compute the class weights according to the paper:
    class_weights = []
    total_count = sum(trainId_to_count.values())
    for trainId, count in trainId_to_count.items():
        trainId_prob = float(count)/float(total_count)
        trainId_weight = 1/np.log(1.02 + trainId_prob)
        class_weights.append(trainId_weight)

    # # save to disk:
    class_weights_path = os.path.join(dataset_dir, 'class_weights.pkl')
    pickle.dump(class_weights, open(class_weights_path, 'wb'))

    # get the path to all validation images and their corresponding label image:
    val_img_paths, val_trainId_label_paths = process_dir(
        'val', raw_dataset_dir, dataset_dir, id_to_trainId_map_func)

    # # save the validation data to disk:
    val_trainId_labels_path = os.path.join(dataset_dir, 'val_trainId_label_paths.pkl')
    pickle.dump(val_trainId_label_paths, open(val_trainId_labels_path, 'wb'))
    val_imgs_path = os.path.join(dataset_dir, 'val_img_paths.pkl')
    pickle.dump(val_img_paths, open(val_imgs_path, 'wb'))

    # augment the train data by flipping all train imgs:
    no_of_train_imgs = len(train_img_paths)
    print(f'number of train imgs before augmentation: {no_of_train_imgs}')

    augmented_train_img_paths = []
    augmented_train_trainId_label_paths = []
    for step, (img_path, label_path) in enumerate(zip(train_img_paths, train_trainId_label_paths)):
        if step % 100 == 0:
            print(f'augmentation step {step}/{no_of_train_imgs - 1}')

        augmented_train_img_paths.append(img_path)
        augmented_train_trainId_label_paths.append(label_path)

        # # read the image:
        # img = cv2.imread(img_path, -1)
        #
        # # flip the image and save to project_data_dir:
        # img_flipped = cv2.flip(img, 1)
        # img_flipped_path = img_path.replace('.png', '_flipped.png')
        # cv2.imwrite(img_flipped_path, img_flipped)
        # augmented_train_img_paths.append(img_flipped_path)
        #
        # # read the corresponding label image:
        # label_img = cv2.imread(label_path, -1)
        #
        # # flip the label image and save to project_data_dir:
        # label_img_flipped = cv2.flip(label_img, 1)
        # label_img_flipped_path = label_path.replace('.png', '_flipped.png')
        # cv2.imwrite(label_img_flipped_path, label_img_flipped)
        # augmented_train_trainId_label_paths.append(label_img_flipped_path)

    # # randomly shuffle the augmented train data:
    augmented_train_data = zip(augmented_train_img_paths, augmented_train_trainId_label_paths)
    augmented_train_data = list(augmented_train_data)
    random.shuffle(augmented_train_data)

    # # save the augmented train data to disk:
    train_data = augmented_train_data
    train_img_paths, train_trainId_label_paths = zip(*train_data)
    train_img_paths = list(train_img_paths)
    train_trainId_label_paths = list(train_trainId_label_paths)

    train_img_paths_dump = os.path.join(dataset_dir, 'train_img_paths.pkl')
    pickle.dump(train_img_paths, open(train_img_paths_dump, 'wb'))
    train_trainId_label_paths_dump = os.path.join(dataset_dir, 'train_trainId_label_paths.pkl')
    pickle.dump(train_trainId_label_paths, open(train_trainId_label_paths_dump, 'wb'))

    no_of_train_imgs = len(train_img_paths)
    print(f'number of train imgs after augmentation: {no_of_train_imgs}')


if __name__ == '__main__':
    main()
