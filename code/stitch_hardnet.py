

from stitch_require import stitcher_hardnet



import matplotlib.pyplot as plt
import utils
import time
import cv2



if __name__ == '__main__':

    stitch_hn = stitcher_hardnet.ImageStitcher()


    images = utils.parse("./txtlists/test.txt")
    no_of_images = len(images)

    t1 = time.time()
    for i in range(no_of_images):
        stitch_hn.add_image(images[i])

    result_hn = stitch_hn.image()
    t2 = time.time()

    print('processing', t2 - t1)


    plt.imshow(result_hn), plt.title('stitch_hardnet'), plt.xticks([]), plt.yticks([]), plt.show()
    cv2.imwrite("D:/pycharm project/test/Python-Multiple-Image-Stitching/code/evaluate_methods/compare/hardnet.jpg",result_hn)