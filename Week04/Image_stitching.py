# 调用说明
# 在cmd中转到此文件所在位置，然后输入：
# python image_stitching.py --images images/scottsdale --output output.png --crop 1
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2


# 构造参数解析器并解析参数
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images", type=str, required=True,
    help="path to input directory of images to stitch")
ap.add_argument("-o", "--output", type=str, required=True,
    help="path to the output image")
ap.add_argument("-c", "--crop", type=int, default=0,
    help="whether to crop out largest rectangular region")
args = vars(ap.parse_args())

# 获取输入图像的路径并初始化图像列表
print("[INFO] loading images...")
imagePaths = sorted(list(paths.list_images(args["images"])))
images = []

# 遍历图像路径，并将它们添加到图像列表
for imagePath in imagePaths:
    image = cv2.imread(imagePath)
    images.append(image)

# 初始化OpenCV的图像拼接器(stitcher)对象，然后执行图像拼接打印
print("[INFO] stitching images...")
stitcher = cv2.createStitcher() if imutils.is_cv3() else cv2.Stitcher_create()
(status, stitched) = stitcher.stitch(images)

# if the status is '0', 说明OpenCV成功的进行了图像拼接
if status == 0:
    # 看看我们是否应该从拼接图像中裁剪出最大的矩形
    if args["crop"] > 0:
        # 在拼接后的图像周围创建一个10像素的边框
        print("[INFO] cropping...")
        stitched_b = cv2.copyMakeBorder(stitched, 10, 10, 10, 10,
            cv2.BORDER_CONSTANT, (0, 0, 0))

        # 将拼接后的图像转换为灰度并设置阈值
        # 使所有大于0的像素都设置为255
        gray = cv2.cvtColor(stitched_b, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)[1]

        # 找到阈值图像中所有的外部轮廓
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        c = max(cnts, key=cv2.contourArea)
        # 为掩码分配内存，掩码将包含拼接图像区域的矩形边框
        mask = np.zeros(thresh.shape, dtype="uint8")
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)

        # 创建2个mask副本：一个作为我们实际的最小矩形区域，
        # 另一个作为需要移除多少像素才能形成最小矩形区域的计数器
        minRect = mask.copy()
        sub = mask.copy()

        # 继续循环，直到减去的图像中没有非零像素
        while cv2.countNonZero(sub) > 0:
            # 侵蚀最小矩形的mask，然后从最小矩形的mask中减去阈值图像，
            # 这样我们就可以计算是否还有非零像素
            minRect = cv2.erode(minRect, None)
            sub = cv2.subtract(minRect, thresh)

            # 在最小矩形的mask中找到等值线，然后提取边界框(x, y)坐标
        cnts = cv2.findContours(minRect.copy(), cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        c = max(cnts, key=cv2.contourArea)
        (x, y, w, h) = cv2.boundingRect(c)

        # 使用边界框坐标提取我们的最终拼接的图像
        stitched_b = stitched_b[y:y + h, x:x + w]

    # 将输出拼接的图像写到磁盘
    cv2.imwrite(args["output"], stitched_b)
    # 在屏幕上显示拼接图像
    cv2.imshow("Stitched_simple", stitched)
    cv2.imshow("Stitched", stitched_b)
    cv2.waitKey(20000)

# 否则拼接失败(可能是由于没有足够的关键点)被检测到
else:
    print("[INFO] image stitching failed ({})".format(status))