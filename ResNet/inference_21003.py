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
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

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

tf.app.flags.DEFINE_boolean('restore', True, 'whether to restore from checkpoint')
tf.app.flags.DEFINE_integer('charset_size', 149, "Choose the first `charset_size` character to conduct our experiment.")
tf.app.flags.DEFINE_integer('charset_size2', 151, "Choose the first `charset_size` character to conduct our experiment.")
tf.app.flags.DEFINE_integer('charset_size3', 157, "Choose the first `charset_size` character to conduct our experiment.")
tf.app.flags.DEFINE_integer('charset_size4', 163, "Choose the first `charset_size` character to conduct our experiment.")
tf.app.flags.DEFINE_integer('image_size', 112, "Needs to provide same value as in training.")
tf.app.flags.DEFINE_boolean('gray', True, "whether to change the rbg to gray")
tf.app.flags.DEFINE_string('checkpoint_dir1', './checkpoint21003_149/', 'the checkpoint dir')
tf.app.flags.DEFINE_string('checkpoint_dir2', './checkpoint21003_151/', 'the checkpoint dir')
tf.app.flags.DEFINE_string('checkpoint_dir3', './checkpoint21003_157/', 'the checkpoint dir')
tf.app.flags.DEFINE_string('checkpoint_dir4', './checkpoint21003_163/', 'the checkpoint dir')
tf.app.flags.DEFINE_string('log_dir1', './log21003_149', 'the logging dir')
tf.app.flags.DEFINE_string('log_dir2', './log21003_151', 'the logging dir')
tf.app.flags.DEFINE_string('log_dir3', './log21003_157', 'the logging dir')
tf.app.flags.DEFINE_string('log_dir4', './log21003_163', 'the logging dir')
tf.app.flags.DEFINE_string('png_dir', 'H:/苏统华的空间/3755类汉字样本for荟俨/2/1.png', 'the train dataset dir')
tf.app.flags.DEFINE_string('mode', 'test', 'Running mode. One of {"test", "inference"}')
FLAGS = tf.app.flags.FLAGS
print("-----------------------------main.py start--------------------------")
def build_graph(num_classes=FLAGS.charset_size, top_k=5, is_train=True, is_test=False):
    images = tf.placeholder(dtype=tf.float32, shape=[None, FLAGS.image_size, FLAGS.image_size, 1], name='image_batch')
    labels = tf.placeholder(dtype=tf.int64, shape=[None], name='label_batch')
    if is_train:
        net, end_points = resnet_v2_50(images, num_classes=num_classes,
                                       is_training=True)  # (images, num_classes=num_classes, is_training=is_training)
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=net, labels=labels))
        pre_label = tf.argmax(net, 1)
        accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(net, 1), labels), tf.float32))
        probabilities = tf.nn.softmax(net, name='probabilities')
        predicted_val_top_k, predicted_index_top_k = tf.nn.top_k(probabilities, k=top_k)
        accuracy_in_top_k = tf.reduce_mean(tf.cast(tf.nn.in_top_k(probabilities, labels, top_k), tf.float32))
    else:
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
    g4 = tf.Graph()
    print('inference149')
    with tf.Session(graph=g1) as sess:
        temp_image = Image.open(FLAGS.png_dir).convert('L')
        temp_image = temp_image.resize((FLAGS.image_size, FLAGS.image_size), Image.ANTIALIAS)
        temp_image = np.asarray(temp_image) / 255.0
        temp_image = tf.subtract(1.0, temp_image)
        temp_image = tf.reshape(temp_image, shape=[-1, 112, 112, 1])
        graph = build_graph(num_classes=FLAGS.charset_size, top_k=149, is_train=False, is_test=True)
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        #print(temp_image)
        temp_image2 = sess.run(temp_image)
        if FLAGS.restore:
            ckpt = tf.train.latest_checkpoint(FLAGS.checkpoint_dir1)
            if ckpt:
                saver.restore(sess, ckpt)
                print("restore from the checkpoint {0}".format(ckpt))
        predict_val1, predict_index1 = sess.run([graph['predicted_val_top_k'], graph['predicted_index_top_k']],
                                              feed_dict={graph['images']: temp_image2})
        print(predict_index1)
        #print(predict_val)
    print('inference151')
    with tf.Session(graph=g2) as sess:
        temp_image = Image.open(FLAGS.png_dir).convert('L')
        temp_image = temp_image.resize((FLAGS.image_size, FLAGS.image_size), Image.ANTIALIAS)
        temp_image = np.asarray(temp_image) / 255.0
        temp_image = tf.subtract(1.0, temp_image)
        temp_image = tf.reshape(temp_image, shape=[-1, 112, 112, 1])
        graph2 = build_graph(num_classes=FLAGS.charset_size2, top_k=151, is_train=False, is_test=True)
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        #print(temp_image)
        temp_image2 = sess.run(temp_image)
        if FLAGS.restore:
            ckpt = tf.train.latest_checkpoint(FLAGS.checkpoint_dir2)
            if ckpt:
                saver.restore(sess, ckpt)
                print("restore from the checkpoint {0}".format(ckpt))
        predict_val2, predict_index2 = sess.run([graph2['predicted_val_top_k'], graph2['predicted_index_top_k']],
                                              feed_dict={graph2['images']: temp_image2})
        print(predict_index2)
        #print(predict_val2)

    print('inference157')
    with tf.Session(graph=g3) as sess:
        temp_image = Image.open(FLAGS.png_dir).convert('L')
        temp_image = temp_image.resize((FLAGS.image_size, FLAGS.image_size), Image.ANTIALIAS)
        temp_image = np.asarray(temp_image) / 255.0
        temp_image = tf.subtract(1.0, temp_image)
        temp_image = tf.reshape(temp_image, shape=[-1, 112, 112, 1])
        graph3 = build_graph(num_classes=FLAGS.charset_size3, top_k=157, is_train=False, is_test=True)
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        # print(temp_image)
        temp_image2 = sess.run(temp_image)
        if FLAGS.restore:
            ckpt = tf.train.latest_checkpoint(FLAGS.checkpoint_dir3)
            if ckpt:
                saver.restore(sess, ckpt)
                print("restore from the checkpoint {0}".format(ckpt))
        predict_val3, predict_index3 = sess.run([graph3['predicted_val_top_k'], graph3['predicted_index_top_k']],
                                                feed_dict={graph3['images']: temp_image2})
        print(predict_index3)
        #print(predict_val3)

    print('inference163')
    with tf.Session(graph=g4) as sess:
        temp_image = Image.open(FLAGS.png_dir).convert('L')
        temp_image = temp_image.resize((FLAGS.image_size, FLAGS.image_size), Image.ANTIALIAS)
        temp_image = np.asarray(temp_image) / 255.0
        temp_image = tf.subtract(1.0, temp_image)
        temp_image = tf.reshape(temp_image, shape=[-1, 112, 112, 1])
        graph4 = build_graph(num_classes=FLAGS.charset_size4, top_k=163, is_train=False, is_test=True)
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        # print(temp_image)
        temp_image2 = sess.run(temp_image)
        if FLAGS.restore:
            ckpt = tf.train.latest_checkpoint(FLAGS.checkpoint_dir4)
            if ckpt:
                saver.restore(sess, ckpt)
                print("restore from the checkpoint {0}".format(ckpt))
        predict_val4, predict_index4 = sess.run([graph4['predicted_val_top_k'], graph4['predicted_index_top_k']],
                                                feed_dict={graph4['images']: temp_image2})
        print(predict_index4)
        #print(predict_val4)
    pro2 = np.zeros([21003])
    pro3 = np.zeros([21003])
    pro4 = np.zeros([21003])
    for j in range(0, 21003):
        a = np.argwhere(predict_index1[0] == j % 149)[0][0]
        #print(a)
        b = np.argwhere(predict_index2[0] == j % 151)[0][0]
        #print(b)
        c = np.argwhere(predict_index3[0] == j % 157)[0][0]
        #print(c)
        d = np.argwhere(predict_index4[0] == j % 163)[0][0]
        #print(d)
        pro2[j] = math.log(predict_val1[0][a] + 1e-50) + math.log(predict_val2[0][b] + 1e-50)
        pro3[j] = math.log(predict_val1[0][a] + 1e-50) + math.log(predict_val2[0][b] + 1e-50) + math.log(predict_val3[0][c] + 1e-50)
        pro4[j] = math.log(predict_val1[0][a] + 1e-50) + math.log(predict_val2[0][b] + 1e-50) + math.log(predict_val3[0][c] + 1e-50) + math.log(predict_val4[0][d] + 1e-50)
    num2 = np.where(pro2 == np.max(pro2))[0][0]
    num3 = np.where(pro3 == np.max(pro3))[0][0]
    num4 = np.where(pro4 == np.max(pro4))[0][0]
    print('acc2 = ', num2)
    print('acc3 = ', num3)
    print('acc4 = ', num4)


def main(_):
    if FLAGS.mode == "test":
        test()

if __name__ == "__main__":
    tf.app.run()