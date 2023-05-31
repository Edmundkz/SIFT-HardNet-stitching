import logging

import cv2
import numpy
import numpy as np

from sp_extractor import SuperPointFrontend




class ImageStitcher:

    def __init__(self, min_num: int = 4, lowe: float = 0.7, knn_clusters: int = 2):

        self.min_num = min_num
        self.lowe = lowe
        self.knn_clusters = knn_clusters

        self.flann = cv2.FlannBasedMatcher({'algorithm': 0, 'trees': 5}, {'checks': 50})
        self.sift = cv2.SIFT_create()
        self.detector = SuperPointFrontend(weights_path="superpoint_v1.pth", nms_dist=4, conf_thresh=0.015, nn_thresh=0.7)

        self.result_image = None
        self.result_image_gray = None

    def getpoint(self, im):
        pts, desc, heatmap = self.detector.run(im)
        pts = np.delete(pts, 2, axis=0)
        desc = np.delete(desc, 2, axis=0)
        pts = np.transpose(pts)
        desc = np.transpose(desc)
        pts = pts.tolist()
        desc = desc.tolist()
        return {'kp': pts, 'des': desc}


    def add_image(self, image: numpy.ndarray, blending=True):
        '''
        添加新的图片拼接
        '''

        assert image.ndim == 3, '必须是图片'
        assert image.shape[-1] == 3, '图片格式必须为RGB'
        assert image.dtype == numpy.uint8, '必须类型为uint8'

        image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        if self.result_image is None:
            self.result_image = image
            self.result_image_gray = image_gray
            return

        # todo(will.brennan) - stop computing features on the results image each time!
        result_features = self.getpoint(self.result_image_gray)
        image_features = self.getpoint(image_gray)


        matches = self.flann.knnMatch(
            np.asarray(image_features['des'], np.float32),
            np.asarray(result_features['des'], np.float32),
            k=2
        )



        good = []
        for i, (m, n) in enumerate(matches):
            if m.distance < 0.7 * n.distance:
                good.append((m.trainIdx, m.queryIdx))
                
        if len(good) < self.min_num:
            logging.warning('too few correspondences to add image to stitched image')
            return        
        
        #pointsCurrent = result_features['kp']  # 图2的关键点合集
        pointsCurrent = image_features['kp']
        #pointsPrevious =image_features['kp']  # 图1的关键点合集
        pointsPrevious = result_features['kp']

        matches_dst = np.float32(
            [pointsCurrent[i] for (__, i) in good]
        )
        matches_src = np.float32(
            [pointsPrevious[i] for (i, __) in good]
        )

        #print(pointsCurrent)
        homography, _ = cv2.findHomography(matches_src, matches_dst, cv2.RANSAC, 5.0)


        logging.debug('正在拼接')
        self.result_image = self.combine_images(image, self.result_image, homography)
        self.result_image_gray = cv2.cvtColor(self.result_image, cv2.COLOR_RGB2GRAY)



    def image(self):
        '''class for fetching the stitched image'''
        return self.result_image




    def combine_images(self, img0, img1, h_matrix):
        '''
        this takes two images and the homography matrix from 0 to 1 and combines the images together!
        the logic is convoluted here and needs to be simplified!
        '''

        global left, right
        points0 = numpy.array(
            [[0, 0], [0, img0.shape[0]], [img0.shape[1], img0.shape[0]], [img0.shape[1], 0]],
            dtype=numpy.float32)   #图像1的四个边界点的坐标，最后展现出的是图像的一整个矩形


        points0 = points0.reshape((-1, 1, 2))  #即n个一行两列的坐标矩阵，两列代表横纵坐标，等价于将坐标齐次化后与H矩阵乘法运算

        points1 = numpy.array(
            [[0, 0], [0, img1.shape[0]], [img1.shape[1], img1.shape[0]], [img1.shape[1], 0]],
            dtype=numpy.float32)
        points1 = points1.reshape((-1, 1, 2))

        points2 = cv2.perspectiveTransform(points1, h_matrix)


        points = numpy.concatenate((points0, points2), axis=0)  #变换后的points1与points2点集进行拼接

        [x_min, y_min] = (points.min(axis=0).ravel() - 0.5).astype(numpy.int32)  #axis=0代表垂直方向（行的概念），然后按着每一列或者行标签向下执行
        [x_max, y_max] = (points.max(axis=0).ravel() + 0.5).astype(numpy.int32)  #ravel代表拉成一维数组,这个0.5我觉得应该是凑整用的

        #总之就是找到两张图拼接后的点里面的最大最小点


        h_translation = numpy.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]])



        logging.debug('warping previous image...')

        output_img = cv2.warpPerspective(img1, h_translation.dot(h_matrix),
                                         (x_max - x_min, y_max - y_min))

        #output_img = cv2.warpPerspective(img1, np.linalg.inv(h_matrix), (img0.shape[1] + img1.shape[1], img1.shape[0]))
        #output_img[0:img0.shape[0], 0:img0.shape[1]] = img0
        output_img[-y_min:img0.shape[0] - y_min, -x_min:img0.shape[1] - x_min] = img0

        '''rows, cols = img0.shape[:2]

        for col in range(0, cols):
            if img1[:, col].any() and output_img[:, col].any():  # 开始重叠的最左端
                left = col

        for col in range(cols - 1, 0, -1):
            if img1[:, col].any() and output_img[:, col].any():  # 重叠的最右一列
                right = col

        res = numpy.zeros([rows, cols, 3], numpy.uint8)

        for row in range(0, rows):
            for col in range(0, cols):
                if not img1[row, col].any():  # 如果没有原图，用旋转的填充
                    res[row, col] = output_img[row, col]
                elif not output_img[row, col].any():
                    res[row, col] = img1[row, col]
                else:
                    srcImgLen = float(abs(col - left))
                    testImgLen = float(abs(col - right))
                    alpha = srcImgLen / (srcImgLen + testImgLen)
                    res[row, col] = numpy.clip(img0[row, col] * (1 - alpha) + output_img[row, col] * alpha, 0, 255)

            output_img[0:img0.shape[0], 0:img0.shape[1]] = res'''



        return output_img









    def alpha_blend(self, row1, row2, seam_x, window, direction='left'):
        if direction == 'right':
            row1, row2 = row2, row1

        new_row = np.zeros(shape=row1.shape, dtype=np.uint8)

        for x in range(len(row1)):
            color1 = row1[x]
            color2 = row2[x]
            if x < seam_x - window:
                new_row[x] = color2
            elif x > seam_x + window:
                new_row[x] = color1
            else:
                ratio = (x - seam_x + window) / (window * 2)
                new_row[x] = (1 - ratio) * color2 + ratio * color1

        return new_row


