import os
import numpy as np
import tensorflow as tf
import math

tf.app.flags.DEFINE_string('mode', 'test', 'Running mode. One of {"test", "inference"}')
FLAGS = tf.app.flags.FLAGS


def test():
    result_67 = np.loadtxt('67_val_result.txt')
    result_71 = np.loadtxt('71_val_result.txt')
    result_73 = np.loadtxt('73_val_result.txt')
    index_67 = np.loadtxt('67_val_index.txt')
    index_71 = np.loadtxt('71_val_index.txt')
    index_73 = np.loadtxt('73_val_index.txt')
    sum_acc = 0.0
    for i in range(0, 760319):
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
            pro[j] = math.log(result_67[i][a-1]) + math.log(result_71[i][b-1]) + math.log(result_73[i][c-1])
        num = np.where(pro == np.max(pro))[0][0]
        if (num % 67 ==index_67[i][0])&(num % 71 ==index_71[i][0])&(num % 73 ==index_73[i][0]):
            sum_acc += 1.0
        else:
            print('{0} error'.format(i))
        if i % 200 == 0:
            print('{0} current acc: {1}'.format(i, sum_acc / (i+1)))
    print('acc = ', sum_acc / 760319)


def inference():
    os.system("C:\WinPython-64bit-3.6.3.0Qt5\python-3.6.3.amd64\python.exe resnet_3755_67.py " + png_path + " inference")
    os.system("C:\WinPython-64bit-3.6.3.0Qt5\python-3.6.3.amd64\python.exe resnet_3755_71.py " + png_path + " inference")
    a = np.zeros((71, 71))
    for i in range(0, 71):
        for j in range(0, 71):
            a[i][j] = -1
    for i in range(0, 3755):
        a[i % 67][i % 71] = i
    index67 = np.loadtxt('67_index.txt')
    result67 = np.loadtxt('67_result.txt')
    index71 = np.loadtxt('71_index.txt')
    result71 = np.loadtxt('71_result.txt')
    for i in range(0, 3):
        x = int(index67[i])
        for j in range(0, 3):
            y = int(index71[j])
            if a[x][y] != -1:
                print('label:{0}, accuracy: {1}'.format(a[x][y], (result67[i] + result71[j]) / 2.0))


def main(_):
    if FLAGS.mode == "test":
        test()
    elif FLAGS.mode == "inference":
        inference()

if __name__ == "__main__":
    tf.app.run()