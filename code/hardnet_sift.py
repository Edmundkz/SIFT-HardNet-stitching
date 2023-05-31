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
import numpy as np
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

def find_min(arr):
    in1=arr.index(min(arr))
    isn1=min(arr)
    arr[arr.index(min(arr))]=2
    isn2 = min(arr)
    if isn1 < 0.6 *isn2:
        x1=in1
    else:
        x1=-1
    return x1

def drawMatches(img1,  img2,matches):
    rows1 = img1.shape[0]
    cols1 = img1.shape[1]
    rows2 = img2.shape[0]
    cols2 = img2.shape[1]
    out = np.zeros((max([rows1, rows2]), cols1 + cols2, 3), dtype='uint8')
    out[:rows1, :cols1] = img1
    out[:rows2, cols1:] = img2
    for mat in matches:
        (x1,y1,x2,y2)=mat
        cv2.line(out, (int(np.round(x1)), int(np.round(y1))), (int(np.round(x2) + cols1), int(np.round(y2))), (255, 255, 255),
                 1, lineType=cv2.LINE_AA, shift=0)  # 画线，cv2.line()参考官方文档
    return out
def drawpoint(point,image):
    num=len(point)
    for i in range(num):
        cv2.circle(image, (int(point[i].pt[0]), int(point[i].pt[1])), 4, (255, 255, 255), 1)
    return image


def descriptor(kp,gray,num):
    coor = np.zeros((num, 2))  #生成num行2列的零矩阵
    patches = np.ndarray((num, 1, 32, 32), dtype=np.float32)
    row, col = gray.shape[:2]
    for j in range(num):
        (x, y) = kp[j].pt
        #-----------------加旋转角----------------------------
        M2 = cv2.getRotationMatrix2D((x, y), kp[j].angle, 1)
        gray1 = cv2.warpAffine(gray, M2, (col, row))
        patch = gray1[int(y - size / 2): int(y + size / 2), int(x - size / 2): int(x + size / 2)]  # 截取patch
        #--------------------不旋转------------------------
        # patch = gray[int(y - size / 2): int(y + size / 2), int(x - size / 2): int(x + size / 2)]

        patches[j, 0, :, :] = cv2.resize(patch, (32, 32)) / 255.
        coor[j, :] = [x, y]
    bs = 150
    descriptors = np.zeros((len(patches), 128))
    for i in range(0, len(patches), bs):
        data_a = patches[i: i + bs, :, :, :].astype(np.float32)
        data_a = torch.from_numpy(data_a)
        if DO_CUDA:
            data_a = data_a.cuda()
        data_a = Variable(data_a)
        with torch.no_grad():
            out_a = model(data_a)
        descriptors[i: i + bs, :] = out_a.data.cpu().numpy().reshape(-1, 128)
    return descriptors, coor


if __name__ == '__main__':
    DO_CUDA = False


    input_img_fname1 = 'D:/pycharm project/test/Python-Multiple-Image-Stitching/img_cut/3414c.bmp'
    input_img_fname2 = 'D:/pycharm project/test/Python-Multiple-Image-Stitching/img_cut/3415c.bmp'
    model_weights = '../pretrained/train_liberty_with_aug/checkpoint_liberty_with_aug.pth'
    model = HardNet()
    checkpoint = torch.load(model_weights, map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    if DO_CUDA:
        model.cuda()
        print('Extracting on GPU')
    else:
        print('Extracting on CPU')
        model = model.cpu()
    sift = cv2.SIFT_create(0, 3, 0.04, 10, 1.6)

    image1 = cv2.imread(input_img_fname1)
    image2 = cv2.imread(input_img_fname2)

    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)  # 灰度处理图像
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)  # 灰度处理图像
    # -------------------特征检测------------------------
    t1 = time.time()
    kp11 = sift.detect(gray1, None)



    for m in range(len(kp11) - 2, -1, -1):  #去除重复点
        '''if (kp11[m].pt[0] == kp11[m + 1].pt[0]):
            del kp11[m + 1]'''
    kp1 = [k for k in kp11 if
           k.pt[0] >= 32 and gray1.shape[0] - k.pt[1] >= 32 and k.pt[1] >= 32 and gray1.shape[1] - k.pt[0] >= 32]  #去除边缘点

    kp22 = sift.detect(gray2, None)
    for m in range(len(kp22) - 2, -1, -1):  #去除重复点
        '''if (kp22[m].pt[0] == kp22[m + 1].pt[0]):
            del kp22[m + 1]'''
    kp2 = [k for k in kp22 if
           k.pt[0] >= 32 and gray2.shape[0] - k.pt[1] >= 32 and k.pt[1] >= 32 and gray2.shape[1] - k.pt[0] >= 32]  #去除边缘点



    num1 = len(kp1)
    #num1 = len(kp11)
    num2 = len(kp2)
    #num2 = len(kp22)
    size = 64
    print('Amount of patches: ', len(kp1), len(kp2))
    #print('Amount of patches: ', len(kp11), len(kp22))
    # -------------------特征描述------------------------
    descriptors1, coor1 = descriptor(kp1, gray1, num1)
    descriptors2, coor2 = descriptor(kp2, gray2, num2)

    #-------------------特征匹配------------------------
    dis = np.zeros(num2)


    matches=[]
    for d1 in range(num1):
        for d2 in range(num2):
            dis[d2] = np.linalg.norm(descriptors1[d1, :]-descriptors2[d2, :]) #距离
        dis = list(dis)
        ind = find_min(dis)
        if ind != -1:  #非0
            (x2, y2) = coor2[ind]
            (x1, y1) = coor1[d1]
            matches.append([x1, y1, x2, y2])







    print('Amount of matches: ', len(matches))

    t2 = time.time()
    match1 = matches
    print('processing', t2-t1)

    # ------------------- 显示图像 ----------------------------
    image1 = drawpoint(kp1, image1)
    #image1 = drawpoint(kp11, image1)
    image2 = drawpoint(kp2, image2)
    #image2 = drawpoint(kp22, image2)
    imgout = drawMatches(image1, image2, matches)
    plt.imshow(imgout), plt.title('stitch_hn'), plt.xticks([]), plt.yticks([]), plt.show()


    '''
    cv2.namedWindow("image", cv2.WINDOW_NORMAL)
    cv2.imshow('image', imgout)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    '''

















