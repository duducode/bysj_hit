import os
import numpy as np
import tensorflow as tf
import math

tf.app.flags.DEFINE_string('mode', 'test', 'Running mode. One of {"test", "inference"}')
FLAGS = tf.app.flags.FLAGS
png_path = 'H:/苏统华的空间/3755类汉字样本for荟俨/2/247.png'

def test():
    #os.system("C:\WinPython-64bit-3.6.3.0Qt5\python-3.6.3.amd64\python.exe resnet_3755_67.py " + png_path + " validation")
    #os.system("C:\WinPython-64bit-3.6.3.0Qt5\python-3.6.3.amd64\python.exe resnet_3755_71.py " + png_path + " validation")
    result_67 = np.loadtxt('67_val_result.txt')
    result_71 = np.loadtxt('71_val_result.txt')
    index_67 = np.loadtxt('67_val_index.txt')
    index_71 = np.loadtxt('71_val_index.txt')
    sum_acc = 0.0
    for i in range(0, 760319):
    #for i in range(0, 2):
        #单行计算
        pro = np.zeros([3755])
        for j in range(0, 3755):
            #print(index_67[2])
            if(len(np.argwhere(index_67[i] == j % 67)) == 2):
                a = np.argwhere(index_67[i] == j % 67)[1][0]
                #print('a1 =',a)
            else:
                a = np.argwhere(index_67[i] == j % 67)[0][0]
                #print('a2 =', a)
            if (len(np.argwhere(index_71[i] == j % 71)) == 2):
                #print(np.argwhere(index_71[i] == j % 71))
                b = np.argwhere(index_71[i] == j % 71)[1][0]
                #print('b1 =', b)
            else:
                b = np.argwhere(index_71[i] == j % 71)[0][0]
                #print('b2 =', b)
            # print('********j= ', j, '*********')
            #print('a=', a)
            #print('b=', b)
            pro[j] = math.log(result_67[i][a-1]) + math.log(result_71[i][b-1])
            # if j < 3:
            #     print(result_67[i][a-1])
            #     print(result_71[i][b-1])
            # print(pro[j])
        #print(np.max(pro))
        # print(pro[0])
        # print(pro[1])
        # print(pro[3])
        # print(pro[803])
        # print(pro[804])
        num = np.where(pro == np.max(pro))[0][0]
        #print(num)
        if (num % 67 ==index_67[i][0])&(num % 71 ==index_71[i][0]):
            sum_acc += 1.0
            #print('第{0}行 正确'.format(i))
        else:
            print('第{0}行 错误'.format(i))
        if i % 200 == 0:
            print('{0} current acc: {1}'.format(i, sum_acc / (i+1)))
    print('acc = ', sum_acc / 760319)
    #print('acc = ', sum_acc / 2)
    #print('top 1 accuracy {0}, top 5 accuracy {1}'.format((result_67[0]+result_71[0])/2.0, (result_67[1]+result_71[1])/2.0))

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