# from PIL import Image
#
# img = Image.open("H:/why_workspace/Dataset_MLT/ImagesPart1/tr_img_00001.jpg")
# print(img.size)
# #cropped = img.crop((487, 576, 705, 506),)  # (left, upper, right, lower)
# cropped = img.crop((58, 507, 1435, 876),)
#
# cropped.save("H:/why_workspace/3.jpg")

import numpy as np
with open("wubi21003+.txt", "r", encoding='utf-8') as file:
    line = file.readlines()

file.close()
dict = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g': 6, 'h': 7, 'i': 8, 'j': 9, 'k': 10, 'l': 11,
        'm': 12, 'n': 13, 'o': 14, 'p': 15, 'q': 16, 'r': 17, 's': 18, 't': 19, 'u': 20, 'v': 21, 'w': 22,
        'x': 23, 'y': 24, '\n': 25}

# pro4 = np.zeros([10000, 21003])
# for i in range(0,10000):
#     pro4[i][1] = 1
# pro4[2][2] = 2
# #print(line.shape)
# num4 = np.argmax(pro4, axis=1)
# for i in range(0, 10000):
#     print(dict[line[num4[i] + 1][2:3]])
print(dict[line[0 + 1][5:6]])