import imutils

from stitch_require import stitcher_sp


import matplotlib.pyplot as plt
import utils
import time
import cv2
import numpy as np

if __name__ == '__main__':

    stitch_sp = stitcher_sp.ImageStitcher()


    images = utils.parse("./txtlists/test.txt")
    no_of_images = len(images)

    t1 = time.time()
    for i in range(no_of_images):
        stitch_sp.add_image(images[i])

    result_sp = stitch_sp.image()
    t2 = time.time()

    print('processing', t2 - t1)


    plt.imshow(result_sp), plt.title('stitch_sp'), plt.xticks([]), plt.yticks([]), plt.show()







    '''result_sp = cv2.copyMakeBorder(result_sp, 2, 2, 2, 2,
                                      cv2.BORDER_CONSTANT, (0, 0, 0))

        # convert the stitched image to grayscale and threshold it
        # such that all pixels greater than zero are set to 255
        # (foreground) while all others remain 0 (background)
    gray = cv2.cvtColor(result_sp, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)[1]

        # find all external contours in the threshold image then find
        # the *largest* contour which will be the contour/outline of
        # the stitched image
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key=cv2.contourArea)

        # allocate memory for the mask which will contain the
        # rectangular bounding box of the stitched image region
    mask = np.zeros(thresh.shape, dtype="uint8")
    (x, y, w, h) = cv2.boundingRect(c)
    cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)

        # create two copies of the mask: one to serve as our actual
        # minimum rectangular region and another to serve as a counter
        # for how many pixels need to be removed to form the minimum
        # rectangular region
    minRect = mask.copy()
    sub = mask.copy()

        # keep looping until there are no non-zero pixels left in the
        # subtracted image
    while cv2.countNonZero(sub) > 0:
            # erode the minimum rectangular mask and then subtract
            # the thresholded image from the minimum rectangular mask
            # so we can count if there are any non-zero pixels left
        minRect = cv2.erode(minRect, None)
        sub = cv2.subtract(minRect, thresh)

        # find contours in the minimum rectangular mask and then
        # extract the bounding box (x, y)-coordinates
    cnts = cv2.findContours(minRect.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key=cv2.contourArea)
    (x, y, w, h) = cv2.boundingRect(c)

        # use the bounding box coordinates to extract the our final
        # stitched image
    result_sp = result_sp[y:y + h, x:x + w]

    # write the output stitched image to disk


    # display the output stitched image to our screen
    plt.imshow(result_sp), plt.title('stitch_image'), plt.xticks([]), plt.yticks([]), plt.show()
    cv2.imwrite("D:/pycharm project/test/Python-Multiple-Image-Stitching/code/evaluate_methods/compare/sp.jpg",result_sp)'''




