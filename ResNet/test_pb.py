import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import tensorflow as tf
from tensorflow.python.platform import gfile
import numpy as np
import matplotlib.pylab as plt
import matplotlib.image as mpimg # mpimg 用于读取图片
from PIL import Image

pb_file_path = "./pb/frozen_model_21003_163.pb"


#im = Image.open('D:/py_code/zheng.jpeg')
# im = mpimg.imread('H:/苏统华的空间/3755类汉字样本for荟俨/2/1.png', 'utf-8')
# im = np.reshape(im,[1,112,112,1])
# im = im/255.0
# print(im)
temp_image = Image.open('H:/苏统华的空间/3755类汉字样本for荟俨/2/1.png').convert('L')
temp_image = temp_image.resize((112, 112), Image.ANTIALIAS)
temp_image = np.asarray(temp_image) / 255.0
temp_image = tf.subtract(1.0, temp_image)
temp_image = tf.reshape(temp_image, shape=[-1, 112, 112, 1])

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.9 # 占用GPU90%的显存
sess = tf.Session(config=config)
#sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
with gfile.FastGFile(pb_file_path, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    sess.graph.as_default()
    tf.import_graph_def(graph_def, name='') # 导入计算图
    input_x = sess.graph.get_tensor_by_name('image_batch:0')
    softmax = sess.graph.get_tensor_by_name('probabilities:0')
    print(softmax.shape)
    #index = tf.argmax(softmax,axis=1)
    val, index = tf.nn.top_k(softmax, k=163)
    #print(index)
    val, index = sess.run([val,index],{input_x: sess.run(temp_image)})
    print(val)
    print(index)
    #print(index)