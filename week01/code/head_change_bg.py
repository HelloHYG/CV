import cv2
import matplotlib.pyplot as plt
import numpy as np
from removebg import RemoveBg
import os

"""证件背景替换
    方法一
    代码实现思路：
    1.腐蚀 + 高斯模糊：图像与背景交汇处高斯模糊化
    说明：腐蚀和膨胀阶段会影响图像的精度，即不能完美的剪切；只能对单一的背景(蓝底或红底)进行修改
"""
img = cv2.imread('../res/Totoro.jpg')

# 尺寸调节
img = cv2.resize(img, (300, 300))
rows, cols, channels = img.shape
cv2.imshow('img', img)

# 转换hsv
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
lower_blue = np.array([78, 43, 46])
upper_blue = np.array([110, 255, 255])
mask = cv2.inRange(hsv, lower_blue, upper_blue)
cv2.imshow('Mask', mask)

# 腐蚀和膨胀
erode = cv2.erode(mask, None, iterations=1)
cv2.imshow('erode', erode)
dilate = cv2.dilate(erode, None, iterations=1)
cv2.imshow('dilate', dilate)

# 遍历替换
for i in range(rows):
    for j in range(cols):
        if dilate[i, j] == 255:
            img[i, j] = (85, 20, 156) # 此处替换颜色，为BGR通道
cv2.imshow('res', img)
key = cv2.waitKey(0)
if key == 27:
    cv2.destroyAllWindows()


"""方法二
    HSV色系对用户来说是一种直观的颜色模型，每一种颜色都是由色相（Hue，简H），饱和度（Saturation，简S）和色明度（Value，简V）所表示的。
这个模型中颜色的参数分别是：色调（H），饱和度（S），亮度（V）
其中h的取值范围为0-180；s和v的范围都是0-255"""
"""
# 这种方法是对图像的颜色进行划分，主要依据h的取值筛选出相应的颜色，然后对其进行赋值以改变颜色。因为它是对整个图进行筛选与赋值
# 因此这种方法不能做到真正的对图像进行换背景或颜色。
def img_show(img, trans_model):
    plt.imshow(cv2.cvtColor(img, trans_model))
    plt.show()

img = cv2.imread('../res/wangwang.jpg')
img_show(img, trans_model=cv2.COLOR_BGR2RGB)
img_h, img_s, img_v = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
img_h[((img_h >= 0) & (img_h <= 10)) | ((img_h >= 155) & (img_h <= 180))] = 120  # 红色转蓝色
# img_h[(img_h >= 95) & (img_h <= 124)] = 5  # 蓝色转红色
img_show(cv2.merge((img_h, img_s, img_v)), trans_model=cv2.COLOR_HSV2RGB)
"""

