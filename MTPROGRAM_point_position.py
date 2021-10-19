import numpy as np
import cv2
import math
import operator
def rotate_bound(image, angle):
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # perform the actual rotation and return the image
    shuchu = cv2.warpAffine(image, M, (nW, nH))
    return shuchu
    # while (1):
    #     cv2.imshow('shuchu', shuchu)
    #     if cv2.waitKey(1) & 0xFF == 27:
    #         break
def find_thresholdedLines(arrDistance, arrPoints):
    counter = 0
    index1 = 0
    maxDistance1 = 0
    for i in arrDistance:
        if (i > maxDistance1):
            maxDistance1 = i
            index1 = counter
        counter = counter + 1
    line1 = arrPoints[index1]
    return line1
def fill_lty(img, num):
    '''
    img:输入,二值化图像
    num:按面积排序需要保留的连通域数量，例如面积最大的前10个，剩下的连通域被填充
    '''
    lty = img.copy()
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(lty, connectivity=8)
    id = stats[np.lexsort(-stats.T)]
    for istat in id[num:-1]:
        if istat[3] > istat[4]:
            r = istat[3]
        else:
            r = istat[4]
        cv2.rectangle(lty, tuple(istat[0:2]), tuple(istat[0:2] + istat[2:4]), (0, 0, 0), thickness=-1)
    return lty
    # output = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
    # for i in range(1, num_labels):
    #     mask = labels == i
    #     output[:, :, 0][mask] = np.random.randint(0, 255)
    #     output[:, :, 1][mask] = np.random.randint(0, 255)
    #     output[:, :, 2][mask] = np.random.randint(0, 255)

def MITIANPRO_point_position(inputimg, cv_TH=115,cv_C=10, conn_num=10,
                             line_threshold=142, line_minLineLength=200, line_maxLineGap=327, angel_th=41):
    '''
    :param inputimg: 输入图像，彩图（BGR）
    :param cv_TH: cv2.adaptive threshould,default:115,数值大，突出整体；数值小，突出细节
    :param cv_C: cv2.adaptive C,default:10，数值越大，图像越亮（白）
    :param conn_num: 你需要保留多少个连通域，按面积降序
    :param line_threshold:霍夫直线检测，多少个点组成一条线
    :param line_minLineLength:霍夫直线检测，最小直线长度，小于该值的直线被剔除
    :param line_maxLineGap:霍夫直线检测，最大直线间隔，小于该值的间隔，将会连成一条直线
    :param angel_th:图像目标倾斜角度的范围，用于筛选，只显示目标水平的直线
    :return:输出图像，彩图（BGR）
    '''

    gray = cv2.cvtColor(inputimg, cv2.COLOR_BGR2GRAY) #灰度变换
    th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, cv_TH, cv_C)#阈值分割
    lty = fill_lty(th, num=conn_num) #连通域提取
    retval1 = cv2.getStructuringElement(cv2.MORPH_RECT, ksize=(15, 65))  # 开运算的核
    retval2 = cv2.getStructuringElement(cv2.MORPH_RECT, ksize=(15, 55))  # 闭运算的核
    close = cv2.morphologyEx(lty, cv2.MORPH_CLOSE, retval2)  # 先闭运算
    open = cv2.morphologyEx(close, cv2.MORPH_OPEN, retval1)  # 再开运算
    h_max = np.max(open, axis=1)
    w_max = np.max(open, axis=0) #提取roi区域
    c1 = np.flatnonzero(h_max)
    c2 = np.flatnonzero(w_max)
    # print(c1[0],c1[-1],c2[0],c2[-1])
    roi_img = open[c1[0]:c1[-1], c2[0]:c2[-1]]
    output1 = np.zeros((roi_img.shape[0], roi_img.shape[1], 3), np.uint8)
    output1 = cv2.cvtColor(output1, cv2.COLOR_BGR2GRAY)
    output2 = inputimg[c1[0]:c1[-1], c2[0]:c2[-1]]
    edges = cv2.Canny(roi_img, 150, 300)#边缘检测
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180,
                            threshold=142, minLineLength=200, maxLineGap=327)#霍夫直线检测
    arrDistance1 = []
    arrPoints1 = []
    for line in lines:
        x1 = line[0][0]
        y1 = line[0][1]
        x2 = line[0][2]
        y2 = line[0][3]
        dx1 = x2 - x1
        dy1 = y2 - y1
        angle1 = math.atan2(dy1, dx1)#计算每条直线的斜率，角度
        angle1 = angle1 * 180 / math.pi
        if -1*angel_th < angle1 < angel_th:
            arrDistance1.append(max(y1, y2))
            arrPoints1.append([x1, y1, x2, y2])
    line1 = find_thresholdedLines(arrDistance1, arrPoints1)#约束条件，选出霍夫直线检测中y坐标最大的直线，即最下方的直线
    dx1 = line1[2] - line1[0]
    dy1 = line1[3] - line1[1]
    angle_res = math.atan2(dy1, dx1)
    k_res = math.tan(angle_res)
    angle_res = angle_res * 180 / math.pi
    y_p = line1[3] - k_res * line1[2]
    cv2.line(output1, (0, int(y_p)), (line1[2], line1[3]), (255, 255, 255), 3)#在空白图上画出该直线
    open_dst = rotate_bound(roi_img, -1 * angle_res)
    edges_dst = rotate_bound(edges, -1 * angle_res)
    output2 = rotate_bound(output2, -1 * angle_res)
    output1 = rotate_bound(output1, -1 * angle_res)#图像矫正，此时直线水平
    res_l = output1.sum(axis=1)
    max_index0, max_number0 = max(enumerate(res_l), key=operator.itemgetter(1))#筛选：每行求和，行之和最大的索引，即水平直线所在位置，对应y
    res_open = open_dst.sum(axis=0)
    max_index1, max_number1 = max(enumerate(res_open), key=operator.itemgetter(1))#对矫正后的open图像，筛选：每列求和，列之和最大的索引，即纵向直线的位置，对应x
    cv2.line(output2, (max_index1, 0), (max_index1, edges_dst.shape[1]), (255, 0, 0), 3)
    cv2.line(output2, (0, max_index0), (edges_dst.shape[1], max_index0), (0, 255, 0), 3)
    cv2.circle(output2, (max_index1, max_index0), 3, (0, 0, 255), 5)
    return output2, (max_index1,max_index0), angle_res

filePath = "images/com.jpg"
img1 = cv2.imread(filePath)
h, w, d = img1.shape
img1 = img1[0:h, 0:2 * h]
res, (x,y), angel = MITIANPRO_point_position(img1)
cv2.imwrite("C:/Users/Yu/Desktop/mitianres.jpg", res)