import logging

import cv2
import numpy







class ImageStitcher:

    def __init__(self, min_num: int = 4, lowe: float = 0.7, knn_clusters: int = 2):

        self.min_num = min_num
        self.lowe = lowe
        self.knn_clusters = knn_clusters

        self.flann = cv2.FlannBasedMatcher({'algorithm': 0, 'trees': 5}, {'checks': 50})
        self.sift = cv2.SIFT_create()

        self.result_image = None
        self.result_image_gray = None

    def add_image(self, image: numpy.ndarray):
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
        '''kp1, descrip1 = self.sift.detectAndCompute(self.result_image_gray, None)
        kp2, descrip2 = self.sift.detectAndCompute(image_gray, None)

        matches = self.flann.knnMatch(descrip1, descrip2, k=2)
        good = []
        for i, (m, n) in enumerate(matches):
            if (m.distance < 0.75 * n.distance):
                good.append(m)
        matches_src = numpy.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        matches_dst = numpy.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)'''

        result_features = self.sift.detectAndCompute(self.result_image_gray, None)
        image_features = self.sift.detectAndCompute(image_gray, None)

        matches_src, matches_dst, n_matches = self.compute_matches(result_features,
                                                              image_features,
                                                              matcher=self.flann,
                                                              knn=self.knn_clusters,
                                                              lowe=self.lowe)

        if n_matches < self.min_num:
            logging.warning('too few correspondences to add image to stitched image')
            return

        logging.debug('计算H矩阵')
        homography, _ = cv2.findHomography(matches_src, matches_dst, cv2.RANSAC, 5.0)

        logging.debug('正在拼接')
        self.result_image = self.combine_images(image, self.result_image, homography)
        self.result_image_gray = cv2.cvtColor(self.result_image, cv2.COLOR_RGB2GRAY)

    def image(self):
        '''class for fetching the stitched image'''
        return self.result_image


    def compute_matches(self, features0, features1, matcher, knn=5, lowe=0.7):
        '''
        this applies lowe-ratio feature matching between feature0 an dfeature 1 using flann
        '''
        keypoints0, descriptors0 = features0
        keypoints1, descriptors1 = features1

        logging.debug('finding correspondence')

        matches = matcher.knnMatch(descriptors0, descriptors1, k=knn)

        logging.debug("filtering matches with lowe test")

        positive = []
        for match0, match1 in matches:
            if match0.distance < lowe * match1.distance:
                positive.append(match0)

        src_pts = numpy.array([keypoints0[good_match.queryIdx].pt for good_match in positive],
                              dtype=numpy.float32)
        src_pts = src_pts.reshape((-1, 1, 2))
        dst_pts = numpy.array([keypoints1[good_match.trainIdx].pt for good_match in positive],
                              dtype=numpy.float32)
        dst_pts = dst_pts.reshape((-1, 1, 2))

        return src_pts, dst_pts, len(positive)

    def combine_images(self, img0, img1, h_matrix):
        '''
        this takes two images and the homography matrix from 0 to 1 and combines the images together!
        the logic is convoluted here and needs to be simplified!
        '''
        global left, right
        logging.debug('combining images... ')

        points0 = numpy.array(
            [[0, 0], [0, img0.shape[0]], [img0.shape[1], img0.shape[0]], [img0.shape[1], 0]],
            dtype=numpy.float32)
        points0 = points0.reshape((-1, 1, 2))
        points1 = numpy.array(
            [[0, 0], [0, img1.shape[0]], [img1.shape[1], img1.shape[0]], [img1.shape[1], 0]],
            dtype=numpy.float32)
        points1 = points1.reshape((-1, 1, 2))

        points2 = cv2.perspectiveTransform(points1, h_matrix)
        points = numpy.concatenate((points0, points2), axis=0)

        [x_min, y_min] = (points.min(axis=0).ravel() - 0.5).astype(numpy.int32)
        [x_max, y_max] = (points.max(axis=0).ravel() + 0.5).astype(numpy.int32)

        h_translation = numpy.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]])

        logging.debug('warping previous image...')
        output_img = cv2.warpPerspective(img1, h_translation.dot(h_matrix),
                                         (x_max - x_min, y_max - y_min))
        output_img[-y_min:img0.shape[0] - y_min, -x_min:img0.shape[1] - x_min] = img0



        '''rows, cols = img0.shape[:2]

        for col in range(0, cols):
            if img0[:, col].any() and output_img[:, col].any():  # 开始重叠的最左端
                left = col

        for col in range(cols - 1, 0, -1):
            if img0[:, col].any() and output_img[:, col].any():  # 重叠的最右一列
                right = col

        res = numpy.zeros([rows, cols, 3], numpy.uint8)

        for row in range(0, rows):
            for col in range(0, cols):
                if not img0[row, col].any():  # 如果没有原图，用旋转的填充
                    res[row, col] = output_img[row, col]
                elif not output_img[row, col].any():
                    res[row, col] = img0[row, col]
                else:
                    srcImgLen = float(abs(col - left))
                    testImgLen = float(abs(col - right))
                    alpha = srcImgLen / (srcImgLen + testImgLen)
                    res[row, col] = numpy.clip(img0[row, col] * (1 - alpha) + output_img[row, col] * alpha, 0, 255)

            #output_img[0:img0.shape[0], 0:img0.shape[1]] = res
            output_img[-y_min:img0.shape[0]- y_min , -x_min:img0.shape[1]- x_min ] = res'''

        return output_img
