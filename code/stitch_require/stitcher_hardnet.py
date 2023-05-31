#!/usr/bin/python2 -utt
# -*- coding: utf-8 -*-
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import time
import os
import cv2
import math
import numpy
import numpy as np
import logging
import matplotlib.pyplot as plt



class L2Norm(nn.Module):
    def __init__(self):
        super(L2Norm,self).__init__()
        self.eps = 1e-10
    def forward(self, x):
        norm = torch.sqrt(torch.sum(x * x, dim = 1) + self.eps)
        x= x / norm.unsqueeze(-1).expand_as(x)
        return x

class L1Norm(nn.Module):
    def __init__(self):
        super(L1Norm,self).__init__()
        self.eps = 1e-10
    def forward(self, x):
        norm = torch.sum(torch.abs(x), dim = 1) + self.eps
        x= x / norm.expand_as(x)
        return x

class HardNet(nn.Module):
    """HardNet model definition
    """
    def __init__(self):
        super(HardNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32, affine=False),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32, affine=False),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128, affine=False),
            nn.ReLU(),

            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128, affine=False),
            nn.ReLU(),

            nn.Dropout(0.3),
            nn.Conv2d(128, 128, kernel_size=8, bias=False),
            nn.BatchNorm2d(128, affine=False),
        )
        # self.features.apply(weights_init)

    def input_norm(self,x):
        flat = x.view(x.size(0), -1)
        mp = torch.mean(flat, dim=1)
        sp = torch.std(flat, dim=1) + 1e-7
        return (x - mp.detach().unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand_as(x)) / sp.detach().unsqueeze(-1).unsqueeze(-1).unsqueeze(1).expand_as(x)

    def forward(self, input):
        x_features = self.features(self.input_norm(input))
        x = x_features.view(x_features.size(0), -1)
        return L2Norm()(x)



class ImageStitcher:

    def __init__(self, min_num: int = 4, lowe: float = 0.7, knn_clusters: int = 2):

        self.min_num = min_num
        self.lowe = lowe
        self.knn_clusters = knn_clusters

        self.sift = cv2.SIFT_create()
        self.flann = cv2.FlannBasedMatcher({'algorithm': 0, 'trees': 5}, {'checks': 50})

        self.result_image = None
        self.result_image_gray = None

        self.DO_CUDA = False
        model_weights = 'D:/pycharm project/test/Python-Multiple-Image-Stitching/code/stitch_require/hardnet_pretrained/train_liberty_with_aug/checkpoint_liberty_with_aug.pth'
        self.model = HardNet()
        checkpoint = torch.load(model_weights, map_location='cpu')
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.eval()

        if self.DO_CUDA:
            self.model.cuda()
        else:
            self.model = self.model.cpu()




    def descriptor(self, kp, gray, num):
        size = 64
        coor = np.zeros((num, 2))
        patches = np.ndarray((num, 1, 32, 32), dtype=np.float32)
        row, col = gray.shape[:2]
        for j in range(num):
            (x, y) = kp[j].pt
            # -----------------加旋转角----------------------------
            M2 = cv2.getRotationMatrix2D((x, y), kp[j].angle, 1)
            gray1 = cv2.warpAffine(gray, M2, (col, row))
            patch = gray1[int(y - size / 2): int(y + size / 2), int(x - size / 2): int(x + size / 2)]  # 截取patch
            # --------------------不旋转------------------------
            # patch = gray[int(y - size / 2): int(y + size / 2), int(x - size / 2): int(x + size / 2)]

            patches[j, 0, :, :] = cv2.resize(patch, (32, 32)) / 255.
            coor[j, :] = [x, y]
        bs = 150
        descriptors = np.zeros((len(patches), 128))
        for i in range(0, len(patches), bs):
            data_a = patches[i: i + bs, :, :, :].astype(np.float32)
            data_a = torch.from_numpy(data_a)
            if self.DO_CUDA:
                data_a = data_a.cuda()
            data_a = Variable(data_a)
            with torch.no_grad():
                out_a = self.model(data_a)
            descriptors[i: i + bs, :] = out_a.data.cpu().numpy().reshape(-1, 128)
        return descriptors, coor






    def getpoint_sift(self, im):
        kps = self.sift.detect(im, None)
        return kps



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
        kp11 = self.getpoint_sift(self.result_image_gray)
        kp22 = self.getpoint_sift(image_gray)



        kp1 = [k for k in kp11 if
               k.pt[0] >= 32 and self.result_image_gray.shape[0] - k.pt[1] >= 32 and k.pt[1] >= 32 and self.result_image_gray.shape[1] - k.pt[
                   0] >= 32]  # 去除边缘点

        kp2 = [k for k in kp22 if
               k.pt[0] >= 32 and image_gray.shape[0] - k.pt[1] >= 32 and k.pt[1] >= 32 and image_gray.shape[1] - k.pt[
                   0] >= 32]  # 去除边缘点

        num1 = len(kp1)
        num2 = len(kp2)

        descriptors1, coor1 = self.descriptor(kp1, self.result_image_gray, num1)
        descriptors2, coor2 = self.descriptor(kp2, image_gray, num2)





        matches = self.flann.knnMatch(
            np.asarray(descriptors1, np.float32),
            np.asarray(descriptors2, np.float32),
            k=2
        )

        positive = []
        for match0, match1 in matches:
            if match0.distance < 0.65 * match1.distance:
                positive.append(match0)

        src_pts = numpy.array([kp1[good_match.queryIdx].pt for good_match in positive],
                              dtype=numpy.float32)
        src_pts = src_pts.reshape((-1, 1, 2))
        dst_pts = numpy.array([kp2[good_match.trainIdx].pt for good_match in positive],
                            dtype=numpy.float32)
        dst_pts = dst_pts.reshape((-1, 1, 2))

        self.homography, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)




        logging.debug('正在拼接')
        self.result_image = self.combine_images(image, self.result_image, self.homography)
        self.result_image_gray = cv2.cvtColor(self.result_image, cv2.COLOR_RGB2GRAY)



    def image(self):
        '''class for fetching the stitched image'''
        return self.result_image






