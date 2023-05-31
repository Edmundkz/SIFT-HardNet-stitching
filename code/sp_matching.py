import numpy as np
import cv2
import matplotlib.pyplot as plt
from sp_extractor import SuperPointFrontend
from sp_extractor import PointTracker


#因为提取出的关键点是坐标，那就将这些坐标每一个点都画成点，表示成点的形式显示在图像上。
def showpoint(img, ptx):
    for i in range(ptx.shape[1]): #矩阵列的长度
        x = int(round(ptx[0, i])) #行
        y = int(round(ptx[1, i])) #列
        # if x>20 and y>20 and x<640 and y <450:
        #   None
        cv2.circle(img, (x, y), 1, color=(255, 0, 0), thickness=2)
        #cv::circle(InputOutputArray img, Point center, int radius, const Scalar & color, int thickness = 1, int lineType = LINE_8, int shift = 0 )
        #输入图片，中心坐标，半径大小，颜色，厚度等等
    return img



def drawMatches(img1, kp1, img2, kp2, matches):
    """
    My own implementation of cv2.drawMatches as OpenCV 2.4.9
    does not have this function available but it's supported in
    OpenCV 3.0.0
    This function takes in two images with their associated
    keypoints, as well as a list of DMatch data structure (matches)
    that contains which keypoints matched in which images.
    An image will be produced where a montage is shown with
    the first image followed by the second image beside it.
    Keypoints are delineated with circles, while lines are connected
    between matching keypoints.
    img1,img2 - Grayscale images
    kp1,kp2 - Detected list of keypoints through any of the OpenCV keypoint
              detection algorithms
    matches - A list of matches of corresponding keypoints through any
              OpenCV keypoint matching algorithm
    """

    # Create a new output image that concatenates the two images together
    # (a.k.a) a montage
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    rows1 = img1.shape[0]
    cols1 = img1.shape[1]
    rows2 = img2.shape[0]
    cols2 = img2.shape[1]

    # out = np.zeros((max([rows1,rows2]),cols1+cols2,3), dtype='uint8')
    # Place the first image to the left
    i1 = np.dstack([img1, img1, img1])
    i2 = np.dstack([img2, img2, img2])
    #cv2.imshow("sdd", i1)
    #cv2.imshow("sd", i1)
    out = np.hstack([i1, i2])
    #print("sdsdsd", out.shape)
    # Place the next image to the right of it
    # out[0:480,640:1280] = np.dstack([img2, img2, img2])

    # For each pair of points we have between both images
    # draw circles, then connect a line between them
    for i in range(matches.shape[1]):
        # Get the matching keypoints for each of the images

        img1_idx = matches[0, i]
        img2_idx = matches[1, i]
        x11 = int(img1_idx)
        y11 = int(img1_idx)
        x22 = int(img2_idx)
        y22 = int(img2_idx)

        # x - columns
        # y - rows
        x1 = kp1[0, x11]
        y1 = kp1[1, y11]
        x2 = kp2[0, x22]
        y2 = kp2[1, y22]

        # Draw a small circle at both co-ordinates
        # radius 4
        # colour blue
        # thickness = 1
        a = np.random.randint(0, 256)
        b = np.random.randint(0, 256)
        c = np.random.randint(0, 256)

        cv2.circle(out, (int(np.round(x1)), int(np.round(y1))), 2, (a, b, c), 2)  # 画圆，cv2.circle()参考官方文档
        cv2.circle(out, (int(np.round(x2) + cols1), int(np.round(y2))), 2, (a, b, c), 2)

        # Draw a line in between the two points
        # thickness = 1
        # colour blue
        cv2.line(out, (int(np.round(x1)), int(np.round(y1))), (int(np.round(x2) + cols1), int(np.round(y2))), (0, 255, 0),
                 2, shift=0)  # 画线，cv2.line()参考官方文档

    # Also return the image if you'd like a copy
    return out



'''读取图片部分'''
img1 = cv2.imread(r'D:/pycharm project/test/Python-Multiple-Image-Stitching/images/1Hill.jpg') #3350-
img2 = cv2.imread(r'D:/pycharm project/test/Python-Multiple-Image-Stitching/images/2Hill.jpg')

'''采用的特征提取方法'''


detector = SuperPointFrontend(weights_path="superpoint_v1.pth", nms_dist=4, conf_thresh=0.015, nn_thresh=0.7)

'''Superpoint提取的关键点和描述子'''
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
pts1, desc1, heatmap1 = detector.run(gray1)

gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
pts2, desc2, heatmap2 = detector.run(gray2)

print(pts1.shape[1])

'''Superpoint展示提取的特征'''

#print(pts1.shape[1])#特征点的数目,shape[1]表示的应该是列数
imgx1 = img1.copy()#复制为新的图片
imgsu1 = showpoint(imgx1, pts1) #显示特征点
plt.imshow(imgsu1),plt.title('superpoint_img1'),plt.xticks([]),plt.yticks([]),plt.show()
cv2.imwrite("D:/pycharm project/test/Python-Multiple-Image-Stitching/1.jpg", imgsu1)


#print(pts2.shape[1])#特征点的数目
imgx2 = img2.copy()#复制为新的图片
imgsu2 = showpoint(imgx2, pts2) #显示特征点
#plt.imshow(imgsu2),plt.title('superpoint_img2'),plt.xticks([]),plt.yticks([]),plt.show()

'''Superpoint展示匹配对数'''
match_superpoint = PointTracker.nn_match_two_way(desc1, desc2, 0.3) #这个里面的0.7表示的是描述子之间的距离，在什么距离下算是好的匹配对
#print("图1与图2匹配对数", match_superpoint.shape[1]) #这里展示的是这个矩阵的列，根据官方说明是匹配的对


'''Superpoint展示匹配效果'''
out_superpoint = drawMatches(img1, pts1, img2, pts2, match_superpoint)
#out_superpoint = drawMatches(img1, img2, match_superpoint)
#plt.imshow(out_superpoint),plt.title('Superpoint_match'),plt.xticks([]),plt.yticks([]),plt.show()



#cv2.imwrite("D:/pycharm project/test/Python-Multiple-Image-Stitching/compare/3354c-3362c/match/superpoint/match/3357-3358match.jpg", out_superpoint)
#cv2.imwrite("D:/pycharm project/test/Python-Multiple-Image-Stitching/compare/3354c-3362c/match/superpoint/3361point.jpg", imgsu1)
#cv2.imwrite("D:/pycharm project/test/Python-Multiple-Image-Stitching/compare/3354c-3362c/match/superpoint/3362point.jpg", imgsu2)



