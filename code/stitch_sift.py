

from stitch_require import stitcher_sift

import matplotlib.pyplot as plt
import utils
import time
import cv2




if __name__ == '__main__':

    stitch_si = stitcher_sift.ImageStitcher()


    images = utils.parse("./txtlists/test.txt")
    no_of_images = len(images)

    t1 = time.time()
    for i in range(no_of_images):
        stitch_si.add_image(images[i])

    result_si = stitch_si.image()
    t2 = time.time()



    print('processing', t2 - t1)
    plt.imshow(result_si), plt.title('stitch_sift'), plt.xticks([]), plt.yticks([]), plt.show()
    cv2.imwrite("D:/pycharm project/test/Python-Multiple-Image-Stitching/code/evaluate_methods/compare/sift.jpg",result_si)