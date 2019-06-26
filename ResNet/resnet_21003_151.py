import numpy as np
import tensorflow as tf
import random
import logging
import numpy as np
import argparse
import os
import sys
from datetime import datetime
import math
import time
from resnet_v2 import *
import pic_preprocessing as image_preprocess
os.environ["CUDA_VISIBLE_DEVICES"] = '2'


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

tf.app.flags.DEFINE_integer('charset_size', 151, "Choose the first `charset_size` character to conduct our experiment.")
tf.app.flags.DEFINE_integer('image_size', 112, "Needs to provide same value as in training.")
tf.app.flags.DEFINE_boolean('gray', True, "whether to change the rbg to gray")
tf.app.flags.DEFINE_integer('max_steps', 3000000000, 'the max training steps ')
tf.app.flags.DEFINE_integer('eval_steps', 34573, "the step num to eval")
tf.app.flags.DEFINE_integer('save_steps', 34573, "the steps to save")

tf.app.flags.DEFINE_string('checkpoint_dir', './checkpoint21003_151/', 'the checkpoint dir')
tf.app.flags.DEFINE_string('train_data_dir', '/home/why2/dataset/21003/train/', 'the train dataset dir')
tf.app.flags.DEFINE_string('test_data_dir', '/home/why2/dataset/21003/test/', 'the test dataset dir')
tf.app.flags.DEFINE_string('val_data_dir', '/home/why2/dataset/21003/val/', 'the val dataset dir')
tf.app.flags.DEFINE_string('log_dir', './log21003_151', 'the logging dir')

tf.app.flags.DEFINE_boolean('restore', True, 'whether to restore from checkpoint')
tf.app.flags.DEFINE_integer('epoch', 30, 'Number of epoches')
tf.app.flags.DEFINE_integer('decay_steps', 34573, '')
tf.app.flags.DEFINE_integer('batch_size', 256, '')
tf.app.flags.DEFINE_string('mode', 'train', 'Running mode. One of {"train", "validation", "inference"}')
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


def get_batch_train(dirpath =  '',is_train='', lb = ''):
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
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)# convert to [0,1]
        image = image_preprocess.preprocess_image(image,FLAGS.image_size, FLAGS.image_size)

        #image = tf.reshape(image, [112, 112, 1])
        #image = tf.reshape(image, [112, 112, 1])
        #new_image = 255-tf.image.resize_images(image, (FLAGS.image_size, FLAGS.image_size))
        #new_image = tf.reshape(new_image, [224, 224])#for matplot
        label = tf.cast(parsed["label"], tf.int32)
        return image, label
    dataset = dataset.map(parser)
    dataset = dataset.shuffle(buffer_size=5000)
    dataset = dataset.repeat(FLAGS.epoch)
    dataset = dataset.batch(FLAGS.batch_size)
    iterator = dataset.make_one_shot_iterator()
    image_batch, label_batch = iterator.get_next()
    return image_batch, label_batch


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


acc = []
result = []
step = 0

def train():
    global step
    print('Begin training')
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.per_process_gpu_memory_fraction = 0.8
    sess = tf.Session(config=tf_config)
    # feed_images = tf.placeholder(dtype=tf.float32, shape=[None, FLAGS.image_size, FLAGS.image_size, 1], name='image_batch')
    # feed_labels = tf.placeholder(dtype=tf.int64, shape=[None], name='label_batch')
    # is_training = tf.placeholder(dtype=tf.bool, shape=[None], name='training_mode')
    #
    # net, end_points = resnet_v2_50(feed_images, num_classes=FLAGS.charset_size, is_training=is_training)
    # loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=net, labels=feed_labels))
    # accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(net, 1), feed_labels), tf.float32))
    # global_step = tf.get_variable("step", [], initializer=tf.constant_initializer(0.0), trainable=False)
    # rate = tf.train.exponential_decay(0.0001, global_step, decay_steps=FLAGS.decay_steps, decay_rate=0.997, staircase=True)
    # train_op = tf.train.AdamOptimizer(learning_rate=rate).minimize(loss, global_step=global_step)
    # probabilities = tf.nn.softmax(net)
    # tf.summary.scalar('loss', loss)
    # tf.summary.scalar('accuracy', accuracy)
    # merged_summary_op = tf.summary.merge_all()
    # predicted_val_top_k, predicted_index_top_k = tf.nn.top_k(probabilities, k=10)
    # accuracy_in_top_k = tf.reduce_mean(tf.cast(tf.nn.in_top_k(probabilities, feed_labels, 10), tf.float32))

    with sess:
        train_images, train_labels = get_batch_train(dirpath=FLAGS.train_data_dir, is_train=True, lb='train')
        #train_images = tf.layers.batch_normalization(train_images, training=True)

        graph = build_graph(num_classes=FLAGS.charset_size, top_k=5, is_train=True, is_test=False)
        vali_graph = build_graph(is_train=False, num_classes=FLAGS.charset_size, top_k=5)
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        saver = tf.train.Saver()
        train_writer = tf.summary.FileWriter(FLAGS.log_dir + '/train', sess.graph)
        # val_writer = tf.summary.FileWriter(FLAGS.log_dir + '/val')
        start_step = 0
        if FLAGS.restore:
            ckpt = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
            if ckpt:
                saver.restore(sess, ckpt)
                print("Resnet21003_151 restore from the checkpoint {0}".format(ckpt))
                start_step += int(ckpt.split('-')[-1])
        # logger.info("@ Train begin time: {0}".format(datetime.now()))
        num = 0;  # compute epoch
        count = 0;  # early stop num
        maxAcc = 0  # best accuracy
        allAcc1Train = 0.0
        allAcc10Train = 0.0
        trainIter = 0;
        while True:
            try:
                # start_time = time.time()
                train_images_batch, train_labels_batch = sess.run([train_images, train_labels])
                train_labels_batch = train_labels_batch % 151
                #print(train_labels_batch)
                #train_images = tf.layers.batch_normalization(train_images_batch, training=True)
                # tmpimage = tf.reshape(train_images_batch[0], [225, 225])
                # plt.imshow(tmpimage, cmap=plt.cm.gray)
                # plt.show()
                # value = sess.run(label)
                # print(train_labels_batch[0])
                feed_dict = {graph['images']: train_images_batch,
                             graph['labels']: train_labels_batch
                             # graph['is_training']: True
                             }
                _, loss_train, acc1_train, acc10_train, train_summary, step, logit = sess.run(
                    [graph['train_op'], graph['loss'], graph['accuracy'], graph['accuracy_top_k'],
                     graph['merged_summary_op'],
                     graph['global_step'], graph['logits']], feed_dict=feed_dict)

                # if trainIter > 100:
                #     logger.info("the batch {0}, accuracy = {1}(top_1)"
                #                 .format(step, acc_train))
                #     logger.info("truth: {0}".format(train_labels_batch))
                #     logger.info("preed: {0}".format(_labels))
                # feed_dict = {
                #     feed_images:train_images_batch,
                #     feed_labels:train_labels_batch,
                #     is_training:True
                # }
                # _, loss_train, acc_train,train_summary,step = sess.run([train_op,loss,accuracy,merged_summary_op,global_step],feed_dict = feed_dict)
                #     [graph['train_op'], graph['loss'], graph['accuracy'],graph['merged_summary_op'],
                #      graph['global_step'], graph['logits'],graph['labels']], feed_dict=feed_dict)
                trainIter = trainIter + 1
                allAcc1Train = allAcc1Train + acc1_train
                allAcc10Train = allAcc10Train + acc10_train
                # logger.info('step loss: {}'.format(loss_val))
                # end_time = time.time()
                train_writer.add_summary(train_summary, step)
                '''//Print images
                print(train_images_batch.shape)
                for i in range(FLAGS.batch_size):
                    images = train_images_batch[i]
                    #h, w, c = images.shape
                    #assert c == 1
                    images = images.reshape(224, 224)
                    print(images)
                    plt.imshow(images)
                    plt.show()
                '''
                # logger.info("the step {0} takes {1} acc1 {2}".format(step, end_time - start_time, acc1_train))

                # if step == 1:
                #    logger.info('Save the ckpt of {0}'.format(step))
                #    saver.save(sess, os.path.join(FLAGS.checkpoint_dir, 'my-model'),
                #               global_step=graph['global_step'])
                if step % FLAGS.eval_steps == 0:
                    val_images, val_labels = get_batch_val(dirpath=FLAGS.val_data_dir, lb='val')

                    allAcc1Train = allAcc1Train / trainIter
                    allAcc10Train = allAcc10Train / trainIter
                    logger.info("@Resnet21003_151 Accuracy of train set util {0}: {1}/{2}".format(step, allAcc1Train, allAcc10Train))
                    allAcc1Train = 0.0
                    allAcc10Train = 0.0
                    trainIter = 0.0

                    # val_images, val_labels = get_batch2(dirpath=FLAGS.val_data_dir, is_train='val', is_val=True)
                    # val_images_batch, val_labels_batch = sess.run([val_images, val_labels])
                    # feed_dict = {vali_graph['images']: val_images_batch,
                    #              vali_graph['labels']: val_labels_batch,
                    #              }
                    # accuracy_test, test_summary = sess.run(
                    #     [vali_graph['accuracy'], vali_graph['merged_summary_op']],
                    #     feed_dict=feed_dict)
                    # # test_writer.add_summary(test_summary, step)
                    # # #logger.info('===============Eval a batch=======================')
                    # logger.info('the epoch {0} val accuracy: {1}'
                    #             .format(eval_count, accuracy_test))
                    # # #logger.info('===============Eval a batch=======================')
                    coord = tf.train.Coordinator()
                    #logger.info("@Resnet21003-151 Validation start time: {0}".format(datetime.now()))
                    while True:
                        try:
                            i = 0
                            acc_top_1 = 0.0
                            acc_top_10 = 0.0
                            while not coord.should_stop():
                                i += 1
                                val_images_batch, val_labels_batch = sess.run([val_images, val_labels])
                                val_labels_batch = val_labels_batch % 151
                                feed_dict = {vali_graph['images']: val_images_batch,
                                             vali_graph['labels']: val_labels_batch
                                             # graph['is_training']: False
                                             }
                                acc_1, acc_10, vali_predicted = sess.run([vali_graph['accuracy'],
                                                                          vali_graph['accuracy_top_k'],
                                                                          vali_graph['predicted']],
                                                                         feed_dict=feed_dict)
                                #             #logger.info("truth: {0} . pred: {1}"
                                #             #            .format(val_labels_batch, vali_predicted))
                                #             # feed_dict = {
                                #             #     feed_images: val_images_batch,
                                #             #     feed_labels: val_labels_batch,
                                #             #     is_training: False
                                #             # }
                                #             # acc_1, acc_10, step,test_summary = sess.run(
                                #             #     [accuracy, accuracy_in_top_k, global_step, merged_summary_op],
                                #             #                                              feed_dict=feed_dict)
                                #
                                acc_top_1 += acc_1
                                acc_top_10 += acc_10
                                # logger.info("the batch {0}, accuracy = {1}(top_1), accuracy = {2}(top_3)"
                                #             .format(i, acc_1, acc_10))
                        except tf.errors.OutOfRangeError:
                            acc_top_1 = acc_top_1 / (i - 1)
                            acc_top_10 = acc_top_10 / (i - 1)
                            logger.info('Resnet21003_151 epoch # {0}, top 1 accuracy: {1}, top 5 accuracy: {2}'.format(num, acc_top_1,
                                                                                                        acc_top_10))
                            # logger.info("@ Validation end time: {0},i={1}".format(datetime.now(),i))
                            break
                    acc.append(acc_top_1)
                    # print('the {0} epoch\'s accuracy is {1}'.format(num, acc[num]))
                    # print('epoch = ', num)
                    # if num != 0:
                    if maxAcc < acc[num]:
                        print('up, count=', count)
                        maxAcc = acc[num]
                        count = 0
                        # logger.info('@ Save the ckpt of {0} step(s)'.format(step))
                        logger.info('@Resnet21003_151 Save the ckpt of {0} step(s)/{1} epoch(es)'.format(step, num))
                        saver.save(sess, os.path.join(FLAGS.checkpoint_dir, 'my-model'),
                                   global_step=graph['global_step'])
                    else:
                        count = count + 1
                        print('count=', count)
                        logger.info('@Resnet21003_151 Early stop num: {0}'.format(count))
                    # logger.info('@ Save the ckpt of {0} step(s)/{1} epoch(es)'.format(step, num ))
                    num = num + 1
                    # saver.save(sess, os.path.join(FLAGS.checkpoint_dir, 'my-model'),
                    #            global_step=graph['global_step'])
                #     if step % FLAGS.save_steps == 1:
                #     logger.info('Save the ckpt of {0}'.format(step))
                #     saver.save(sess, os.path.join(FLAGS.checkpoint_dir, 'my-model'),
                #                global_step=graph['global_step'])
                if step > FLAGS.max_steps:
                    logger.info('@Resnet21003_151 Best Accuracy: {0} (Step over)'.format(maxAcc))
                    break
                if count == 10:
                    logger.info('@Resnet21003_151 Best Accuracy: {0} (Early stopped)'.format(maxAcc))
                    break
            except tf.errors.OutOfRangeError:
                logger.info('@Resnet21003_151 Best Accuracy: {0} (Out of range) step : {1}'.format(maxAcc, step))
                break
    train_writer.close()


def validation():
    print('validation')
    # tf_config = tf.ConfigProto()
    # tf_config.gpu_options.per_process_gpu_memory_fraction = 0.8
    # sess = tf.Session(config=tf_config)
    with tf.Session() as sess:
        val_images, val_labels = get_batch_val(dirpath=FLAGS.test_data_dir, lb='test')
        graph = build_graph(num_classes=FLAGS.charset_size, top_k=5, is_train=False, is_test=True)
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        # test_writer = tf.summary.FileWriter(FLAGS.log_dir + '/val_all')
        if FLAGS.restore:
            ckpt = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
            if ckpt:
                saver.restore(sess, ckpt)
                print("restore from the checkpoint {0}".format(ckpt))
        coord = tf.train.Coordinator()
        logger.info(':::Start validation:::')
        while True:
            try:
                i = 0
                acc_top_1, acc_top_k = 0.0, 0.0
                while not coord.should_stop():
                    i += 1
                    val_images_batch, val_labels_batch = sess.run([val_images, val_labels])
                    #val_labels_batch = val_labels_batch % 151
                    # print(val_labels_batch)
                    '''
                    print(val_images_batch.shape)
                    for i in range(FLAGS.batch_size):
                        images = val_images_batch[i]
                        #h, w, c = images.shape
                        #assert c == 1
                        images = images.reshape(32, 32)
                        print(images)
                        plt.imshow(images)
                        plt.show()
                    '''
                    # numSamp = len(val_labels_batch)
                    # print(numSamp)
                    # if numSamp<FLAGS.batch_size:
                    #    logger.info('num of samples {0}'.format(numSamp))
                    feed_dict = {graph['images']: val_images_batch,
                                 graph['labels']: val_labels_batch,
                                 }
                    # print(i, '-real:', val_labels_batch)
                    # acc_1, acc_k, step, test_summary = sess.run([graph['accuracy'],
                    acc_1, acc_k, pre_label, index_top_k = sess.run([graph['accuracy'],
                                                                     graph['accuracy_top_k'],
                                                                     graph['predicted'],
                                                                     graph['predicted_index_top_k']],
                                                                    feed_dict=feed_dict)
                    # graph['global_step'],
                    # graph['merged_summary_op']], feed_dict=feed_dict)
                    #print(i, '-pre:', pre_label)
                    acc_top_1 += acc_1
                    acc_top_k += acc_k
                    # logger.info("the batch {0} takes x seconds, accuracy = {1}(top_1) {2}(top_k)"
                    #             .format(i, acc_1, acc_k))
            except tf.errors.OutOfRangeError:
                logger.info('==================Validation Finished================')
                acc_top_1 = acc_top_1 / (i - 1)
                acc_top_k = acc_top_k / (i - 1)
                logger.info('top 1 accuracy {0}, top k accuracy {1}'.format(acc_top_1, acc_top_k))
                # a = np.array([0.0, 0.0])
                # a[0] = acc_top_1
                # a[1] = acc_top_k
                # np.savetxt('151_val_result.txt', a, fmt='%1.4e')
                # print(i)
                break
    return {'acc_top_1': acc_top_1}


def inference(image):
    print('inference')
    temp_image = Image.open(image).convert('L')
    temp_image = temp_image.resize((FLAGS.image_size, FLAGS.image_size), Image.ANTIALIAS)
    temp_image = np.asarray(temp_image) / 255.0
    temp_image = tf.subtract(1.0, temp_image)
    #temp_image = temp_image.reshape([-1, 112, 112, 1])
    temp_image = tf.reshape(temp_image, shape=[-1, 112, 112, 1])

    with tf.Session() as sess:
        logger.info('========start inference============')
        # images = tf.placeholder(dtype=tf.float32, shape=[None, 64, 64, 1])
        # Pass a shadow label 0. This label will not affect the computation graph.
        #start_time = time.time()
        graph = build_graph(num_classes=FLAGS.charset_size, top_k=5, is_train=False, is_test=True)
        saver = tf.train.Saver()
        ckpt = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
        temp_image = sess.run(temp_image)
        if ckpt:
            saver.restore(sess, ckpt)
        predict_val, predict_index = sess.run([graph['predicted_val_top_k'], graph['predicted_index_top_k']],
                                              feed_dict={graph['images']: temp_image})
        #end_time = time.time()
        #print('time :', end_time - start_time)
    return predict_val, predict_index


def main(_):
    #FLAGS.mode = sys.argv[2]
    if FLAGS.mode == "train":
        train()
    elif FLAGS.mode == "validation":
        validation()
    elif FLAGS.mode == "inference":
        image_path = FLAGS.png_dir
        #print(sys.argv[1])
        #final_predict_val, final_predict_index = inference(image_path)
        final_predict_val, final_predict_index = inference(sys.argv[1])
        logger.info('the result info label {0} predict index {1} predict_val {2}'.format(00000, final_predict_index, final_predict_val))
        #np.savetxt('151_index.txt', final_predict_index, fmt='%d')
        #np.savetxt('151_result.txt', final_predict_val, fmt='%1.4e')

if __name__ == "__main__":
    tf.app.run()
