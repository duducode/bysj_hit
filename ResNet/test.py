# import numpy as np
# import math
#
# a = np.zeros((11, 11))
# for i in range(0, 11):
#     for j in range(0, 11):
#         a[i][j] = -1
# b = np.zeros(94)
# for i in range(0,94):
#     b[i] = i
# print(b)
# print('********')
# b = ((b%11)+np.ceil(b/11)*2)%11
# print(b)
#
# for i in range(0, 94):
#     c1 = ((i % 11) + math.ceil(i / 11) * 1) % 11
#     c2 = ((i % 11) + math.ceil(i / 11) * 2) % 11
#     a[c1][c2] = i
# #print(a)
# count = 0
# for i in range(0, 11):
#     for j in range(0, 11):
#         if a[i][j] != -1:
#             #print(a[i][j])
#             count += 1
#
# #print('count = ', count)
#
#
#
#
#
#
#
# import numpy as np
# import math
#
# #print(math.log(1e-1000))
# result_73 = np.loadtxt('73_val_result.txt')
# result_79 = np.loadtxt('79_val_result.txt')
# print('finished')
# for i in range(3460, 3600):
#     for j in range(0, 73):
#         print(i, j)
#         print(math.log(result_73[i][j]))
#
# print('**********************')
# for i in range(3460, 3600):
#     for j in range(0, 79):
#         print(i, j)
#         print(math.log(result_79[i][j]))


# with open("lexicon21003.txt", "r", encoding='utf-8') as file1:
#     with open("wubi21003.txt", "r", encoding='utf-8') as file2:
#         with open("wubi21003+.txt", "w", encoding='utf-8') as file3:
#             i = 0
#             j = 0
#             line1 = file1.readlines()
#             line2 = file2.readlines()
#             # print(line1[1][0])
#             # print(line1[1][2:6])
#             # print(line2[1][0])
#             for i in range(1,21004):
#                 if i % 1000 == 0:
#                     print(i)
#                 for j in range(1,21004):
#                     if line1[i][0] == line2[j][0]:
#                         file3.write(line2[j])
#                         #print('i=', i)
#                         #print('j=', j)
#             file1.close()
#             file2.close()
#             file3.close()

#
# with open("wubi21003.txt", "r", encoding='utf-8') as file3:
# #     line3 = file3.readlines()
# # print(line3[1])
# # file3.close()
import time
import numpy as np
import datetime
# index1 = np.array([[ 0, 1],
#                 [0, 2],
#                 [3, 0]])
# result1 = np.array([[0.89, 0.01, 0.01],
#                 [0.99, 1e-10, 0.01],
#                 [0.98, 1e-5, 0.01]])
# index2 = np.array([[0, 1],
#                 [2, 0],
#                 [4, 0]])
# result2 = np.array([[0.99, 0.01, 0.01],
#                 [0.89, 1e-10, 0.01],
#                 [0.58, 1e-5, 0.01]])
# #print(result1[[0,1,2],[1,0,1]])
# a = np.array([0,1,2])
# b = np.array([1,0,1])
# c = np.array([0,0,1])
# e = []
#
# result1 = np.zeros(2000000*8).reshape(2000000, 8)
# result2 = np.zeros(2000000*8).reshape(2000000, 8)
# result3 = np.zeros(2000000*8).reshape(2000000, 8)
# time1 = time.clock()
# for i in range(0,2000000):
#     result3[i, 0]= result1[i, 0]+result2[i, 0]
# time2 = time.clock()
# print(time2-time1)
#
# time1 = time.clock()
# d = result1[:,0]+result2[:,0]
# time2 = time.clock()
# print(time2-time1)

#print(result1[a, b])
# print(a)
# print(result1[0, a[0,1]])
# print(b)
# print(result1+result2)
# print(result2[:,a[:,1]])
# print(a)

pro2 = np.array([[0,1,2,3,4],
 [0,2,3,4,5],
 [0,3,4,5,6],
 [0,1,1,1,1],
 [0,2,3,4,5]])
pro4 = np.zeros([2, 3])
print(pro4.shape)
print(np.argwhere(pro2 == (0 % 71))[:,1])




