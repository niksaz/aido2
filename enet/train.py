# Author: Mikita Sazanovich

import numpy as np
import pickle
import tensorflow as tf
import cv2
import os
import argparse
import time
from typing import List, Tuple

from enet.config import CFG
from enet.dataset import Dataset
from enet.utilities import label_img_to_color
from enet.model import ENetModel


def do_training_epoch(
        model: ENetModel,
        epoch: int,
        number_of_epochs: int,
        dataset: Dataset,
        batch_size: int,
        sess: tf.Session) -> Tuple[float, float]:
    # run an epoch and get all batch losses:
    dataset.reset()
    number_of_batches = dataset.get_batches_in_dataset(batch_size)
    batch_losses = np.zeros(number_of_batches)
    pixels_correct = 0
    pixels_total = 0
    for batch_num in range(number_of_batches):
        imgs, onehot_labels = dataset.get_next_batch(batch_size)
        # create a feed dict containing the batch data:
        batch_feed_dict = model.create_feed_dict(imgs_batch=imgs,
                                                 early_drop_prob=0.01, late_drop_prob=0.1,
                                                 onehot_labels_batch=onehot_labels)

        # compute the batch loss and compute & apply all gradients w.r.t to
        # the batch loss (without model.train_op in the call, the network
        # would NOT train, we would only compute the batch loss):
        batch_loss, logits, grad_norm, _ = sess.run(
            [model.loss, model.logits, model.grad_norm, model.train_op], feed_dict=batch_feed_dict)
        batch_losses[batch_num] = batch_loss
        labels = np.argmax(onehot_labels, axis=3)
        predictions = np.argmax(logits, axis=3)
        pixels_correct += np.count_nonzero(labels == predictions)
        pixels_total += labels.size

        if number_of_batches >= 10 and batch_num % (number_of_batches // 10) == 0:
            print(f'epoch: {epoch + 1}/{number_of_epochs}, '
                  f'train batch_num: {batch_num}/{number_of_batches}, '
                  f'batch loss: {batch_loss}')
            print(f'gradients norm: {grad_norm}')

    # compute the train mean loss and accuracy
    return float(np.mean(batch_losses)), float(pixels_correct) / pixels_total


def do_evaluation(
        model: ENetModel,
        epoch: int,
        number_of_epochs: int,
        dataset: Dataset,
        batch_size: int,
        eval_img_dir: str,
        sess: tf.Session) -> Tuple[float, float]:
    # run an epoch and get all batch losses:
    dataset.reset()
    number_of_batches = dataset.get_batches_in_dataset(batch_size)
    batch_losses = np.zeros(number_of_batches)
    pixels_correct = 0
    pixels_total = 0
    for batch_num in range(number_of_batches):
        imgs, onehot_labels = dataset.get_next_batch(batch_size)
        # create a feed dict containing the batch data:
        batch_feed_dict = model.create_feed_dict(imgs_batch=imgs,
                                                 early_drop_prob=0.0, late_drop_prob=0.0,
                                                 onehot_labels_batch=onehot_labels)

        # run a forward pass, get the batch loss and the logits:
        batch_loss, logits = sess.run([model.loss, model.logits], feed_dict=batch_feed_dict)
        batch_losses[batch_num] = batch_loss
        labels = np.argmax(onehot_labels, axis=3)
        predictions = np.argmax(logits, axis=3)
        pixels_correct += np.count_nonzero(labels == predictions)
        pixels_total += labels.size

        if number_of_batches >= 10 and batch_num % (number_of_batches // 10) == 0:
            print(f'epoch: {epoch + 1}/{number_of_epochs}, '
                  f'val batch_num: {batch_num}/{number_of_batches}, '
                  f'batch loss: {batch_loss}')

        if batch_num == 0:
            # save the predicted label images to disk for debugging and qualitative evaluation:
            for i in range(batch_size):
                pred_img = predictions[i]
                label_img_color = label_img_to_color(pred_img)
                label_img_path = os.path.join(eval_img_dir, f'val_{epoch + 1}_{batch_num}_{i}.png')
                cv2.imwrite(label_img_path, label_img_color)

    # compute the val mean loss and accuracy
    return float(np.mean(batch_losses)), float(pixels_correct) / pixels_total


def read_dataset_image_paths(data_dir: str, data_part: str) -> List[Tuple[str, str]]:
    assert data_part in ['train', 'val']
    img_paths_dump = os.path.join(data_dir, f'{data_part}_img_paths.pkl')
    img_paths = pickle.load(open(img_paths_dump, 'rb'))
    trainId_label_paths_dump = os.path.join(data_dir, f'{data_part}_trainId_label_paths.pkl')
    trainId_label_paths = pickle.load(open(trainId_label_paths_dump, 'rb'))
    return list(zip(img_paths, trainId_label_paths))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--logs_dir', type=str, help='The directory to write logs to',
                        default='logs')
    parser.add_argument('--data_dir', type=str, help='The directory where the data is stored',
                        default='/data/sazanovich/aido2/duckscapes/')
    parser.add_argument('--seed', type=int, help='The randomness seed to use', default=2)
    return parser.parse_args()


def main() -> None:
    img_height = CFG.IMG_HEIGHT
    img_width = CFG.IMG_WIDTH
    no_of_classes = CFG.NUM_OF_CLASSES
    batch_size = CFG.BATCH_SIZE

    args = parse_args()
    np.random.seed(args.seed)
    tf.set_random_seed(args.seed)

    data_dir = args.data_dir
    assert os.path.exists(data_dir)
    class_weights_path = os.path.join(data_dir, 'class_weights.pkl')
    class_weights = pickle.load(open(class_weights_path, 'rb'))

    model = ENetModel(batch_size, img_height, img_width, CFG.NUM_OF_CLASSES, CFG.WEIGHT_DECAY)
    model.add_train_op(class_weights, CFG.LEARNING_RATE, CFG.GRAD_NORM_CLIP_VALUE)

    model_logs_dir = args.logs_dir
    os.makedirs(model_logs_dir, exist_ok=True)

    model_checkpoints_dir = os.path.join(model_logs_dir, 'checkpoints')
    model_debug_imgs_dir = os.path.join(model_logs_dir, 'imgs')
    model_summaries_dir = os.path.join(model_logs_dir, 'summaries')

    os.makedirs(model_checkpoints_dir, exist_ok=True)
    os.makedirs(model_debug_imgs_dir, exist_ok=True)
    os.makedirs(model_summaries_dir, exist_ok=True)

    # load the training data from disk:
    train_data = read_dataset_image_paths(data_dir, 'train')
    train_dataset = Dataset(data_dir, train_data, img_height, img_width, no_of_classes)

    # load the validation data from disk:
    val_data = read_dataset_image_paths(data_dir, 'val')
    val_dataset = Dataset(data_dir, val_data, img_height, img_width, no_of_classes)

    no_of_epochs = CFG.TRAIN_EPOCHS

    # create a saver for saving all model variables/parameters:
    saver = tf.train.Saver(tf.trainable_variables(), write_version=tf.train.SaverDef.V2)
    summary_writer = tf.summary.FileWriter(model_summaries_dir)

    # initialize a list containing the 5 best val losses (is used to tell when to
    # save a model checkpoint):
    best_epoch_losses = [1000, 1000, 1000, 1000, 1000]

    l_started = time.time()
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)

        for epoch in range(no_of_epochs):
            print("#"*45, "NEW EPOCH", "#"*45)

            time_checkpoint = time.time()
            train_epoch_loss, train_acc = do_training_epoch(
                model, epoch, no_of_epochs, train_dataset, batch_size, sess)
            print(f'training took: {time.time() - time_checkpoint}s')
            print(f'train_epoch_loss: {train_epoch_loss}, train_acc: {train_acc}')

            time_checkpoint = time.time()
            val_epoch_loss, val_acc = do_evaluation(
                model, epoch, no_of_epochs, val_dataset, batch_size, model_debug_imgs_dir, sess)
            print(f'validation took: {time.time() - time_checkpoint}s')
            print(f'val_epoch_loss: {val_epoch_loss}, val_acc: {val_acc}')

            # log the summary about the epoch
            summary = tf.Summary(
                value=[
                    tf.Summary.Value(tag='train_epoch_loss', simple_value=train_epoch_loss),
                    tf.Summary.Value(tag='train_acc', simple_value=train_acc),
                    tf.Summary.Value(tag='val_epoch_loss', simple_value=val_epoch_loss),
                    tf.Summary.Value(tag='val_acc', simple_value=val_acc)])
            summary_writer.add_summary(summary, epoch)

            if val_epoch_loss < max(best_epoch_losses):  # (if top 5 performance on val:)
                # save the model weights to disk:
                checkpoint_path = os.path.join(model_checkpoints_dir, f'model_epoch_{epoch+1}.ckpt')
                saver.save(sess, checkpoint_path)
                print(f'checkpoint saved in file: {checkpoint_path}')
                # update the top 5 val losses:
                index = best_epoch_losses.index(max(best_epoch_losses))
                best_epoch_losses[index] = val_epoch_loss
    l_finished = time.time()
    print(f'Learning started at {time.ctime(l_started)}')
    print(f'Learning finished at {time.ctime(l_finished)}')


if __name__ == '__main__':
    main()
