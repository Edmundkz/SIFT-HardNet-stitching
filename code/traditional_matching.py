import numpy
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sp_extractor import SuperPointFrontend
from sp_extractor import PointTracker

def getpoint_sift(im):
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    kps, dec = sift.detectAndCompute(gray, None)
    return {'kp': kps, 'des': dec}



'''读取图片部分'''
img1 = cv2.imread(r'D:/pycharm project/test/Python-Multiple-Image-Stitching/img_cut/3355c.bmp')
img2 = cv2.imread(r'D:/pycharm project/test/Python-Multiple-Image-Stitching/grail/grail01.jpg')

'''采用的特征提取方法'''
sift = cv2.SIFT_create()


gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
#gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)


'''SIFT特征关键点和描述子'''
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

#print(len(kp1))



FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5) #代表选择的FLANN中随机kd树算法，不同参数有不同的算法
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params) #Flann匹配器的初始化设置




'''image1 = getpoint_sift(img1)
image2 = getpoint_sift(img2)

match_sift = flann.knnMatch(image1['des'], image2['des'], k=2)

good_sift=[]

for match0, match1 in match_sift:
    if match0.distance < 0.7 *match1.distance:
        good_sift.append(match0)
print(match_sift)


if len(good_sift) > 4:
    points2 = image2['kp']
    points1 = image1['kp']
    src_pts = numpy.array([points1[good_match.queryIdx].pt for good_match in good_sift],
                          dtype=numpy.float32)
    src_pts = src_pts.reshape((-1, 1, 2))

    dst_pts = numpy.array([points2[good_match.trainIdx].pt for good_match in good_sift],
                          dtype=numpy.float32)
    dst_pts = dst_pts.reshape((-1, 1, 2))

    H, s = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 4)


    

    for m in range(len(good_sift) - 1, -1, -1):
        if (s[m] == 0):
            del good_sift[m]








out_sift = cv2.drawMatches(gray1, kp1, gray2, kp2, good_sift, None, flags=2)'''
#plt.imshow(out_sift),plt.title('sift_match'),plt.xticks([]),plt.yticks([]),plt.show()





#SIFT展示提取的特征
imgz1 = img1.copy()#复制为新的图片
imgsi1=cv2.drawKeypoints(imgz1,kp1,None,(255,0,0))
plt.imshow(imgsi1),plt.title('sift_img1'),plt.xticks([]),plt.yticks([]),plt.show()
cv2.imwrite("D:/pycharm project/test/Python-Multiple-Image-Stitching/1.jpg", imgsi1)


imgz2 = img2.copy()#复制为新的图片
imgsi2=cv2.drawKeypoints(imgz2,kp2,None,(255,0,0))
#plt.imshow(imgsi2),plt.title('sift_img2'),plt.xticks([]),plt.yticks([]),plt.show()


'''
cv2.imwrite("D:/pycharm project/test/Python-Multiple-Image-Stitching/compare/3354c-3362c/match/sift/3361-3362match.jpg", out_sift)
cv2.imwrite("D:/pycharm project/test/Python-Multiple-Image-Stitching/compare/3354c-3362c/match/sift/3361point.jpg", imgsi1)
cv2.imwrite("D:/pycharm project/test/Python-Multiple-Image-Stitching/compare/3354c-3362c/match/sift/3362point.jpg", imgsi2)

'''












'''
#ORB特征关键点和描述子
kps1, descr1 = orb.detectAndCompute(img1, None)
kps2, descr2 = orb.detectAndCompute(img2, None)


#ORB等传统提取方法配备的匹配器
bf=cv2.BFMatcher(cv2.NORM_HAMMING,crossCheck=True)

#ORB展示提取的特征
imgy1 = img1.copy()#复制为新的图片
#imgsi1=cv2.drawKeypoints(imgz1,kp1,None,(255,0,0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
imgo1=cv2.drawKeypoints(imgy1,kp1,None,(255,0,0))
plt.imshow(imgo1),plt.title('orb_img1'),plt.xticks([]),plt.yticks([]),plt.show()

imgy2 = img2.copy()#复制为新的图片
imgo2=cv2.drawKeypoints(imgy2,kp2,None,(255,0,0))
plt.imshow(imgo2),plt.title('orb_img2'),plt.xticks([]),plt.yticks([]),plt.show()

#ORB匹配效
match_orb = bf.match(descr1, descr2)
#match_orb=sorted(match_orb, key=lambda x:x.distance)
out_orb = cv2.drawMatches(gray1, kps1, gray2, kps2, match_orb[:70], None, (0, 255, 0), flags=2)
#out_sift = drawMatches(gray1, kp1, gray2, kp2, match_sift)
plt.imshow(out_orb),plt.title('orb_match'),plt.xticks([]),plt.yticks([]),plt.show()

'''

