import numpy as np
import cv2
import heapq
import numpy as np
import time
import matplotlib.pyplot as plt


'''
矩阵取点，拟合直线实验，效果不好，有改进空间，算法部分明确约束条件
'''
import random
def Least_squares(x,y):
    x_ = np.mean(x)
    y_ = np.mean(y)
    m = np.zeros(1)
    n = np.zeros(1)
    k = np.zeros(1)
    p = np.zeros(1)
    for i in np.arange(50):
        k = (x[i]-x_)* (y[i]-y_)
        m += k
        p = np.square( x[i]-x_ )
        n = n + p
    a = m/n
    b = y_ - a* x_
    return a,b
# a = np.random.randint(50,size= (4,5))
# a = np.array(a)
# print(a)
# lists = [[] for i in range(4)]
# for i in range(len(a)):
# #print(heapq.nlargest(3, range(len(a[i])), a[i].take))
#     lists[i].append(heapq.nlargest(3, range(len(a[i])), a[i].take))
# print(lists)
img= cv2.imread("C:/Users/Yu/Desktop/res_rotated.jpg")
t0=time.time()
gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
a= cv2.Canny(gray,100,200)
h_max=np.max(a,axis=1)
w_max=np.max(a,axis=0)
c1=np.flatnonzero(h_max)
c2=np.flatnonzero(w_max)
# print(c1[0],c1[-1],c2[0],c2[-1])
b = a[c1[0]:c1[-1],c2[0]:c2[-1]]
line_res = b.copy()
cv2.imwrite("C:/Users/Yu/Desktop/roi.jpg",b)

h_roi,w_roi = b.shape
x1_point = []
y1_point = []
x2_point = []
y2_point = []
loc=[]
for i1 in range (0,h_roi-2,1):
    for j1 in range (0,int(w_roi/5),1):
        if b[i1][j1] == 0 and b[i1][j1+1]>0:
            # x1_point.append(j1)
            # y1_point.append(-i1)
            loc.append([j1,-1*i1])
# print(x1_point)
# print(y1_point)
loc = np.array(loc)
line = cv2.fitLine(loc, cv2.DIST_L2, 0, 0.01, 0.01)
print(line)
cos_theta = line[0]
sin_theta = line[1]
x0 ,y0= line[2],line[3]
k = sin_theta / cos_theta
b = y0 - k * x0
print(k,b)
x1=0
y1 = 1*k * x1 +b
print(y1)
x2=100
y2 = k * x2 +b
print(y2)
cv2.line(line_res, (x0, -y0), (x1, -y1), (255,255,255), 10)
# cv2.imwrite("C:/Users/Yu/Desktop/line_res.jpg",line_res)
# k = output[1] / output[0]
# a1,b1=Least_squares(x1_point,y1_point)
# colors1 = '#00CED1' #点的颜色
# colors2 = '#DC143C'
# area = np.pi * 4**2  # 点面积
# x = np.linspace(0, h_roi-2, num=h_roi-2)
# y1 = a1*x + b1
# plt.scatter(x1_point, y1_point, s=area, c=colors1, alpha=0.4, label='类别A')
# plt.plot(x, y1, 'ro', lw=2, markersize=6)
#
# plt.show()

# print(a1,b1)
# t1=time.time()
# print(t1-t0)
#
#
loc1=[]
for j2 in range (w_roi,0,-2):
    for i2 in range(h_roi ,0,-2):
        if b[i2][j2]>0 :
            x2_point.append(j2)
            y2_point.append(-i2)
            loc1.append([j1,-1*i1])
            break

# print(x1_point)
# print(y1_point)
loc1 = np.array(loc1)
line1 = cv2.fitLine(loc1, cv2.DIST_L2, 0, 0.01, 0.01)
print(line1)
cos_theta1 = line[0]
sin_theta1 = line[1]
x0 ,y0= line[2],line[3]
k1 = sin_theta1 / cos_theta1
b1 = y0 - k1 * x0
print(k1,b1)
x11=0
y11 = 1*k1 * x1 +b1
print(y1)
x21=100
y21 = k1 * x2 +b1
print(y2)
cv2.line(line_res, (x0, -y0), (x11, -y11), (255,255,255), 10)
cv2.imwrite("C:/Users/Yu/Desktop/line_res.jpg",line_res)
# colors1 = '#00CED1' #点的颜色
# colors2 = '#DC143C'
# a2,b2=Least_squares(x2_point,y2_point)
# area = np.pi * 4**2  # 点面积
# x = np.linspace(0, h_roi-2, num=h_roi-2)
# y2 = a2*x + b2
# plt.scatter(x2_point, y2_point, s=area, c=colors1, alpha=0.4, label='类别A')
# plt.plot(x, y2, 'ro', lw=2, markersize=6)
# plt.show()
#
# y2 = a2*x + b2
# plt.figure(figsize=(10, 5), facecolor='w')
# plt.plot(x, y1, 'ro', lw=2, markersize=6)
# # plt.plot(x, y2, 'ro', lw=2, markersize=6)
# plt.grid(b=True, ls=':')
# plt.xlabel(u'X', fontsize=16)
# plt.ylabel(u'Y', fontsize=16)
# plt.show()
# print(a1,b1,a2,b2)
# print(b.shape)
# h_max_idx = np.where(b==np.max(b,axis=1))
# print(h_max_idx)
# idx = np.argmax(a, axis=1)
# pro = np.amax(a, axis=1)
# b = sorted(idx)[len(idx)//2]
# c=np.flatnonzero(h_max)
# print(idx)
# print("换行")
# print(pro)
# print("非零值的行")
# print(c)
# # print(idx[c[0]:c[-1]])