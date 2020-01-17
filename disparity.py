import os
import sys

import numpy as np
import cv2

from time import sleep






if __name__ == '__main__':

    video_filename = sys.argv[1]
    cap = cv2.VideoCapture(video_filename)


    # SGBM Parameters -----------------
    window_size = 3                     # wsize default 3; 5; 7 for SGBM reduced size image; 15 for SGBM full size image (1300px and above); 5 Works nicely

    """
    # ORIGINAL INTERNET
    left_matcher = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=160,             # max_disp has to be dividable by 16 f. E. HH 192, 256
        blockSize=5,
        # wsize default 3; 5; 7 for SGBM reduced size image;
        # 15 for SGBM full size image (1300px and above); 5 Works nicely
        P1=8 * 3 * window_size ** 2,    
        P2=32 * 3 * window_size ** 2,
        disp12MaxDiff=1,
        uniquenessRatio=15,
        speckleWindowSize=0,
        speckleRange=2,
        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )
    right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)
    """

    left_matcher = cv2.StereoSGBM_create(
        minDisparity=0,
        # max_disp has to be dividable by 16 f. E. HH 192, 256
        numDisparities=16,
        blockSize=3,
        # wsize default 3; 5; 7 for SGBM reduced size image;
        # 15 for SGBM full size image (1300px and above); 5 Works nicely
        P1=8 * 1 * window_size ** 2,    
        P2=32 * 1 * window_size ** 2,
        disp12MaxDiff=1,
        uniquenessRatio=30,
        speckleWindowSize=100,
        speckleRange=32,
        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )
    right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)


    # FILTER Parameters
    lmbda = 80000
    sigma = 1.2
    visual_multiplier = 1.0
     
    wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
    wls_filter.setLambda(lmbda)
    wls_filter.setSigmaColor(sigma)


    while True:

        # print('reding images: ')
        # print(images_left[i])
        # print(images_right[i])
        # print()

        # img_l = cv2.imread(images_left[i])
        # img_r = cv2.imread(images_right[i])
        # outL, outR = stereo_manager.calc_stereo(img_l, img_r)
        # cv2.imshow("imageL", outL)
        # cv2.imshow("imageR", outR)

        # cv2.imshow("Original Image L", img_l)

        """"
        based by: http://timosam.com/python_opencv_depthimage
        """

        ret, frame = cap.read()

        if not ret:
            print("NO DATA")
            break

        h, w = frame.shape[:2]

        imgL = frame[:, 0:w//2]
        imgR = frame[:, w//2:w]

        #   print('computing disparity...')
        displ = left_matcher.compute(imgL, imgR)  # .astype(np.float32)/16
        dispr = right_matcher.compute(imgR, imgL)  # .astype(np.float32)/16
        displ = np.int16(displ)
        dispr = np.int16(dispr)
        filteredImg = wls_filter.filter(displ, imgL, None, dispr)  # important to put "imgL" here!!!

        cv2.imshow('Disparity Map', filteredImg)

        filteredImg = cv2.normalize(src=filteredImg, dst=filteredImg, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX);
        filteredImg = np.uint8(filteredImg)
        cv2.imshow('Disparity Map', filteredImg)
        cv2.imshow('original', imgL)

        sw = 1
        ikey = cv2.waitKey(1)
        if ikey == 27:
            break
        elif ikey == ord('q'):
            sw = 0 if sw == 1 else 1
        # sleep(1)

    cv2.destroyAllWindows()

