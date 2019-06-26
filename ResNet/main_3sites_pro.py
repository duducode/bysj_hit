import numpy as np
import tensorflow as tf
import math
import numpy as np
import tensorflow as tf
import random
import logging
import numpy as np
import argparse
import os
import sys
from datetime import datetime
from PIL import Image
import math
import time
from resnet_v2 import *
import pic_preprocessing as image_preprocess
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

# import cv2
# import pickle
slim = tf.contrib.slim

trunc_normal = lambda stddev: tf.truncated_normal_initializer(0.0, stddev)

logger = logging.getLogger('Training a chinese write char recognition')
logger.setLevel(logging.INFO)
# formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
logger.addHandler(ch)

tf.app.flags.DEFINE_boolean('random_flip_up_down', False, "Whether to random flip up down")
tf.app.flags.DEFINE_boolean('random_brightness', True, "whether to adjust brightness")
tf.app.flags.DEFINE_boolean('random_contrast', True, "whether to random constrast")
tf.app.flags.DEFINE_boolean('restore', True, 'whether to restore from checkpoint')
tf.app.flags.DEFINE_integer('charset_size', 67, "Choose the first `charset_size` character to conduct our experiment.")
tf.app.flags.DEFINE_integer('charset_size2', 71, "Choose the first `charset_size` character to conduct our experiment.")
tf.app.flags.DEFINE_integer('charset_size3', 73, "Choose the first `charset_size` character to conduct our experiment.")
tf.app.flags.DEFINE_integer('image_size', 112, "Needs to provide same value as in training.")
tf.app.flags.DEFINE_boolean('gray', True, "whether to change the rbg to gray")
tf.app.flags.DEFINE_integer('max_steps', 3000000000, 'the max training steps ')
tf.app.flags.DEFINE_integer('eval_steps', 10625, "the step num to eval")
tf.app.flags.DEFINE_integer('save_steps', 10625, "the steps to save")
tf.app.flags.DEFINE_integer('batch_size', 256, '')
tf.app.flags.DEFINE_string('checkpoint_dir1', './checkpoint3755_67/', 'the checkpoint dir')
tf.app.flags.DEFINE_string('checkpoint_dir2', './checkpoint3755_71/', 'the checkpoint dir')
tf.app.flags.DEFINE_string('checkpoint_dir3', './checkpoint3755_73/', 'the checkpoint dir')
tf.app.flags.DEFINE_string('log_dir1', './log3755_67', 'the logging dir')
tf.app.flags.DEFINE_string('log_dir2', './log3755_71', 'the logging dir')
tf.app.flags.DEFINE_string('log_dir3', './log3755_73', 'the logging dir')
tf.app.flags.DEFINE_string('train_data_dir', '/home/why2/dataset/3755/train/', 'the train dataset dir')
tf.app.flags.DEFINE_string('test_data_dir', '/home/why2/dataset/3755/test/', 'the test dataset dir')
tf.app.flags.DEFINE_string('val_data_dir', '/home/why2/dataset/3755/val/', 'the val dataset dir')
tf.app.flags.DEFINE_string('mode', 'test', 'Running mode. One of {"test", "inference"}')
FLAGS = tf.app.flags.FLAGS
print("-----------------------------main.py start--------------------------")

def file_name(file_dir, is_train='', lb=''):
    L = []
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            if lb == 'train':
                if os.path.splitext(file)[0] == 'train_data':
                    L.append(os.path.join(root, file))
            elif lb == 'test':
                if os.path.splitext(file)[0] == 'test_data':
                    L.append(os.path.join(root, file))
            else:
                if os.path.splitext(file)[0] == 'val_data':
                    L.append(os.path.join(root, file))
    return L

def get_batch_val(dirpath =  '',is_train='', lb = ''):
    filenames = file_name(dirpath, is_train, lb)
    #print(filenames)
    dataset = tf.data.TFRecordDataset(filenames)
    def parser(record):
        features = {
                'image': tf.FixedLenFeature([], tf.string, default_value=""),
                'label': tf.FixedLenFeature((), tf.int64, default_value=tf.zeros([], dtype=tf.int64)),
            }
        parsed = tf.parse_single_example(record, features)
        image = tf.decode_raw(parsed["image"], tf.uint8)
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)  # convert to [0,1]
        image = tf.subtract(1.0, image)
        #image = tf.reshape(image, [112, 112, 1])
        image = tf.reshape(image, [FLAGS.image_size, FLAGS.image_size, 1])
        #new_image = 255-tf.image.resize_images(image, (FLAGS.image_size, FLAGS.image_size))
        #new_image = tf.reshape(new_image, [224, 224])#for matplot
        label = tf.cast(parsed["label"], tf.int32)
        return image, label
    dataset = dataset.map(parser)
    dataset = dataset.repeat(1).batch(FLAGS.batch_size)
    iterator = dataset.make_one_shot_iterator()
    image_batch, label_batch = iterator.get_next()
    return image_batch, label_batch


def build_graph(num_classes=FLAGS.charset_size, top_k=5, is_train=True, is_test=False):
    images = tf.placeholder(dtype=tf.float32, shape=[None, FLAGS.image_size, FLAGS.image_size, 1], name='image_batch')
    labels = tf.placeholder(dtype=tf.int64, shape=[None], name='label_batch')
    # is_training = tf.placeholder(dtype=tf.bool, name='training_mode')

    if is_train:
        net, end_points = resnet_v2_50(images, num_classes=num_classes,
                                       is_training=True)  # (images, num_classes=num_classes, is_training=is_training)
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=net, labels=labels))
        pre_label = tf.argmax(net, 1)
        accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(net, 1), labels), tf.float32))
        probabilities = tf.nn.softmax(net)
        predicted_val_top_k, predicted_index_top_k = tf.nn.top_k(probabilities, k=top_k)
        accuracy_in_top_k = tf.reduce_mean(tf.cast(tf.nn.in_top_k(probabilities, labels, top_k), tf.float32))
    else:
        # arg_scope = resnet_arg_scope()
        # with slim.arg_scope(arg_scope):
        if is_test == False:
            vali_net, vali_end_points = resnet_v2_50(images, num_classes=num_classes, is_training=False, reuse=True)  #
        else:
            vali_net, vali_end_points = resnet_v2_50(images, num_classes=num_classes, is_training=False, reuse=False)
        vali_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=vali_net, labels=labels))
        vali_pre_label = tf.argmax(vali_net, 1)
        vali_accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(vali_net, 1), labels), tf.float32))
        vali_probabilities = tf.nn.softmax(vali_net)
        vali_predicted_val_top_k, vali_predicted_index_top_k = tf.nn.top_k(vali_probabilities, k=top_k)
        vali_accuracy_in_top_k = tf.reduce_mean(tf.cast(tf.nn.in_top_k(vali_probabilities, labels, top_k), tf.float32))
        return {'images': images,
                'labels': labels,
                'logits': vali_net,
                'top_k': top_k,
                'loss': vali_loss,
                'accuracy': vali_accuracy,
                'predicted': vali_pre_label,
                'accuracy_top_k': vali_accuracy_in_top_k,
                'predicted_distribution': vali_probabilities,
                'predicted_index_top_k': vali_predicted_index_top_k,
                'predicted_val_top_k': vali_predicted_val_top_k}

    global_step = tf.get_variable("step", [], initializer=tf.constant_initializer(0.0), trainable=False)
    rate = tf.train.exponential_decay(0.001, global_step, decay_steps=FLAGS.decay_steps, decay_rate=0.97,
                                      staircase=True)
    opt = tf.train.AdamOptimizer(learning_rate=rate)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='resnet_v2_50')
    with tf.control_dependencies([tf.group(*update_ops)]):
        train_op = opt.minimize(loss, global_step=global_step)
    tf.summary.scalar('loss', loss)
    tf.summary.scalar('accuracy', accuracy)
    merged_summary_op = tf.summary.merge_all()

    return {'images': images,
            'labels': labels,
            'logits': net,
            'top_k': top_k,
            'global_step': global_step,
            'train_op': train_op,
            'loss': loss,
            'accuracy': accuracy,
            'accuracy_top_k': accuracy_in_top_k,
            'merged_summary_op': merged_summary_op,
            'predicted': pre_label,
            'predicted_distribution': probabilities,
            'predicted_index_top_k': predicted_index_top_k,
            'predicted_val_top_k': predicted_val_top_k}


def test():
    g1 = tf.Graph()
    g2 = tf.Graph()
    g3 = tf.Graph()
    print('validation67')
    with tf.Session(graph = g1) as sess:
        val_images, val_labels = get_batch_val(dirpath=FLAGS.test_data_dir, lb='test')
        graph = build_graph(num_classes=FLAGS.charset_size, top_k=67, is_train=False, is_test=True)
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        if FLAGS.restore:
            ckpt = tf.train.latest_checkpoint(FLAGS.checkpoint_dir1)
            if ckpt:
                saver.restore(sess, ckpt)
                print("restore from the checkpoint {0}".format(ckpt))
        coord = tf.train.Coordinator()
        logger.info(':::Start validation1:::')
        while True:
            try:
                i = 0
                acc_top_1, acc_top_k = 0.0, 0.0
                sum_index_top_k = np.zeros([8000000, 68])
                sum_pre_val = np.zeros([8000000, 67])
                while not coord.should_stop():
                    i += 1
                    val_images_batch, val_labels_batch = sess.run([val_images, val_labels])
                    val_labels_batch = val_labels_batch % 67
                    feed_dict = {graph['images']: val_images_batch,
                                 graph['labels']: val_labels_batch,
                                 }
                    act_label, acc_1, acc_k, pre_label, pre_val, index_top_k = sess.run([graph['labels'],
                                                                                         graph['accuracy'],
                                                                                         graph['accuracy_top_k'],
                                                                                         graph['predicted'],
                                                                                         graph['predicted_val_top_k'],
                                                                                         graph[
                                                                                             'predicted_index_top_k']],
                                                                                        feed_dict=feed_dict)
                    acc_top_1 += acc_1
                    acc_top_k += acc_k
                    for n in range(0, len(act_label)):
                        sum_index_top_k[n + (i - 1) * 256] = np.append(act_label[n], index_top_k[n])
                        sum_pre_val[n + (i - 1) * 256] = pre_val[n]
                    if i % 100 == 0:
                        logger.info("the batch {0} takes x seconds, accuracy = {1}(top_1) {2}(top_k)"
                                    .format(i, acc_1, acc_k))
            except tf.errors.OutOfRangeError:
                logger.info('==================Validation Finished================')
                acc_top_1 = acc_top_1 / (i - 1)
                acc_top_k = acc_top_k / (i - 1)
                logger.info('top 1 accuracy {0}, top k accuracy {1}'.format(acc_top_1, acc_top_k))
                break
    count = 0
    for k in range(0,8000000):
        count += 1
        if (sum_index_top_k[k][1]==0.0000)&(sum_index_top_k[k][2]==0.0000):
            break
    print('validation71')
    with tf.Session(graph = g2) as sess:
        val_images2, val_labels2 = get_batch_val(dirpath=FLAGS.test_data_dir, lb='test')
        graph2 = build_graph(num_classes=FLAGS.charset_size2, top_k=71, is_train=False, is_test=True)
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        if FLAGS.restore:
            ckpt = tf.train.latest_checkpoint(FLAGS.checkpoint_dir2)
            if ckpt:
                saver.restore(sess, ckpt)
                print("restore from the checkpoint {0}".format(ckpt))
        coord = tf.train.Coordinator()
        logger.info(':::Start validation2:::')
        while True:
            try:
                i = 0
                acc_top_12, acc_top_k2 = 0.0, 0.0
                sum_index_top_k2 = np.zeros([8000000, 72])
                sum_pre_val2 = np.zeros([8000000, 71])
                while not coord.should_stop():
                    i += 1
                    val_images_batch, val_labels_batch = sess.run([val_images2, val_labels2])
                    val_labels_batch = val_labels_batch % 71
                    feed_dict = {graph2['images']: val_images_batch,
                                 graph2['labels']: val_labels_batch,
                                 }
                    act_label2, acc_12, acc_k2, pre_label2, pre_val2, index_top_k2 = sess.run([graph2['labels'],
                                                                                         graph2['accuracy'],
                                                                                         graph2['accuracy_top_k'],
                                                                                         graph2['predicted'],
                                                                                         graph2['predicted_val_top_k'],
                                                                                         graph2[
                                                                                             'predicted_index_top_k']],
                                                                                        feed_dict=feed_dict)
                    acc_top_12 += acc_12
                    acc_top_k2 += acc_k2
                    for n in range(0, len(act_label2)):
                        sum_index_top_k2[n + (i - 1) * 256] = np.append(act_label2[n], index_top_k2[n])
                        sum_pre_val2[n + (i - 1) * 256] = pre_val2[n]
                    if i % 100 == 0:
                        logger.info("the batch {0} takes x seconds, accuracy = {1}(top_1) {2}(top_k)"
                                    .format(i, acc_12, acc_k2))
            except tf.errors.OutOfRangeError:
                logger.info('==================Validation Finished================')
                acc_top_12 = acc_top_12 / (i - 1)
                acc_top_k2 = acc_top_k2 / (i - 1)
                logger.info('top 1 accuracy {0}, top k accuracy {1}'.format(acc_top_12, acc_top_k2))
                break

    print('validation73')
    with tf.Session(graph=g3) as sess:
        val_images3, val_labels3 = get_batch_val(dirpath=FLAGS.test_data_dir, lb='test')
        graph3 = build_graph(num_classes=FLAGS.charset_size3, top_k=73, is_train=False, is_test=True)
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        if FLAGS.restore:
            ckpt = tf.train.latest_checkpoint(FLAGS.checkpoint_dir3)
            if ckpt:
                saver.restore(sess, ckpt)
                print("restore from the checkpoint {0}".format(ckpt))
        coord = tf.train.Coordinator()
        logger.info(':::Start validation3:::')
        while True:
            try:
                i = 0
                acc_top_13, acc_top_k3 = 0.0, 0.0
                sum_index_top_k3 = np.zeros([8000000, 74])
                sum_pre_val3 = np.zeros([8000000, 73])
                while not coord.should_stop():
                    i += 1
                    val_images_batch, val_labels_batch = sess.run([val_images3, val_labels3])
                    val_labels_batch = val_labels_batch % 73
                    feed_dict = {graph3['images']: val_images_batch,
                                 graph3['labels']: val_labels_batch,
                                 }
                    act_label3, acc_13, acc_k3, pre_label3, pre_val3, index_top_k3 = sess.run([graph3['labels'],
                                                                                               graph3['accuracy'],
                                                                                               graph3['accuracy_top_k'],
                                                                                               graph3['predicted'],
                                                                                               graph3[
                                                                                                   'predicted_val_top_k'],
                                                                                               graph3[
                                                                                                   'predicted_index_top_k']],
                                                                                              feed_dict=feed_dict)
                    acc_top_13 += acc_13
                    acc_top_k3 += acc_k3
                    for n in range(0, len(act_label3)):
                        sum_index_top_k3[n + (i - 1) * 256] = np.append(act_label3[n], index_top_k3[n])
                        sum_pre_val3[n + (i - 1) * 256] = pre_val3[n]
                    if i % 100 == 0:
                        logger.info("the batch {0} takes x seconds, accuracy = {1}(top_1) {2}(top_k)"
                                    .format(i, acc_13, acc_k3))
            except tf.errors.OutOfRangeError:
                logger.info('==================Validation Finished================')
                acc_top_13 = acc_top_13 / (i - 1)
                acc_top_k3 = acc_top_k3 / (i - 1)
                logger.info('top 1 accuracy {0}, top k accuracy {1}'.format(acc_top_13, acc_top_k3))
                break
    index_67, result_67 = sum_index_top_k, sum_pre_val
    index_71, result_71 = sum_index_top_k2, sum_pre_val2
    index_73, result_73 = sum_index_top_k3, sum_pre_val3
    sum_acc = 0.0
    print('count = ', count)
    for i in range(0, count):
        pro = np.zeros([3755])
        for j in range(0, 3755):
            if(len(np.argwhere(index_67[i] == j % 67)) == 2):
                a = np.argwhere(index_67[i] == j % 67)[1][0]
            else:
                a = np.argwhere(index_67[i] == j % 67)[0][0]

            if (len(np.argwhere(index_71[i] == j % 71)) == 2):
                b = np.argwhere(index_71[i] == j % 71)[1][0]
            else:
                b = np.argwhere(index_71[i] == j % 71)[0][0]

            if (len(np.argwhere(index_73[i] == j % 73)) == 2):
                c = np.argwhere(index_73[i] == j % 73)[1][0]
            else:
                c = np.argwhere(index_73[i] == j % 73)[0][0]
            pro[j] = math.log(result_67[i][a-1]+1e-50) + math.log(result_71[i][b-1]+1e-50) + math.log(result_73[i][c-1]+1e-50)
        num = np.where(pro == np.max(pro))[0][0]
        if (num % 67 ==index_67[i][0])&(num % 71 ==index_71[i][0])&(num % 73 ==index_73[i][0]):
            sum_acc += 1.0
        else:
            print('{0} error'.format(i))
        if i % 200 == 0:
            print('{0} current acc: {1}'.format(i, sum_acc / (i+1)))
    print('acc = ', sum_acc / count)

def main(_):
    if FLAGS.mode == "test":
        test()

if __name__ == "__main__":
    tf.app.run()