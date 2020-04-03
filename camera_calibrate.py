# python3

import numpy as np
import cv2
import glob
import argparse


class StereoCalibration(object):

    def __init__(self, filepath):
        # termination criteria
        self.criteria = (cv2.TERM_CRITERIA_EPS +
                         cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        self.criteria_cal = (cv2.TERM_CRITERIA_EPS +
                             cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)

        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        self.objp = np.zeros((9*6, 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

        # Arrays to store object points and image points from all the images.
        self.objpoints = []  # 3d point in real world space
        self.imgpoints_l = []  # 2d points in image plane.
        self.imgpoints_r = []  # 2d points in image plane.

        self.camera_model = None
        self.cal_path = filepath
        self.read_images(self.cal_path)

    def read_images(self, cal_path):

        if cal_path[-1] != '/':
            cal_path = cal_path + '/'

        # print('main path')
        # print(cal_path)
        # print()

        images_right = glob.glob(cal_path + 'right/*.png')
        images_left = glob.glob(cal_path + 'left/*.png')
        images_left.sort()
        images_right.sort()

        self.images_left = images_left
        self.images_right = images_right

        for i, fname in enumerate(images_right):
            # print('reding images: ')
            # print(images_left[i])
            # print(images_right[i])
            # print()

            img_l = cv2.imread(images_left[i])
            img_r = cv2.imread(images_right[i])

            gray_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
            gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)

            # Find the chess board corners
            ret_l, corners_l = cv2.findChessboardCorners(gray_l, (9, 6), None)
            ret_r, corners_r = cv2.findChessboardCorners(gray_r, (9, 6), None)

            if (not ret_l is True) or (not ret_r is True):
                # print('CONTINUE')
                continue

            # If found, add object points, image points (after refining them)
            self.objpoints.append(self.objp)

            if ret_l is True:
                rt = cv2.cornerSubPix(gray_l, corners_l, (11, 11),
                                      (-1, -1), self.criteria)
                self.imgpoints_l.append(corners_l)

                # Draw and display the corners
                ret_l = cv2.drawChessboardCorners(img_l, (9, 6),
                                                  corners_l, ret_l)
                # cv2.imshow(images_left[i], img_l)
                # cv2.waitKey(500)

            if ret_r is True:
                rt = cv2.cornerSubPix(gray_r, corners_r, (11, 11),
                                      (-1, -1), self.criteria)
                self.imgpoints_r.append(corners_r)

                # Draw and display the corners
                ret_r = cv2.drawChessboardCorners(img_r, (9, 6),
                                                  corners_r, ret_r)
                # cv2.imshow(images_right[i], img_r)
                # cv2.waitKey(500)
            img_shape = gray_l.shape[::-1]

        # print('img_shape: ')
        # print(img_shape)

        # print('OBJ POINTS')
        # print(len(self.objpoints))

        # print('IMG_PPOINTS LEFT')
        # print(len(self.imgpoints_l))
        # print('IMG_PPOINTS RIGHT')
        # print(len(self.imgpoints_r))

        rt, self.M1, self.d1, self.r1, self.t1 = cv2.calibrateCamera(
            self.objpoints, self.imgpoints_l, img_shape, None, None)
        rt, self.M2, self.d2, self.r2, self.t2 = cv2.calibrateCamera(
            self.objpoints, self.imgpoints_r, img_shape, None, None)

        self.camera_model = self.stereo_calibrate(img_shape)

    def stereo_calibrate(self, dims):
        flags = 0
        # flags |= cv2.CALIB_FIX_INTRINSIC
        #flags |= cv2.CALIB_FIX_PRINCIPAL_POINT
        flags |= cv2.CALIB_USE_INTRINSIC_GUESS
        flags |= cv2.CALIB_FIX_FOCAL_LENGTH
        flags |= cv2.CALIB_FIX_ASPECT_RATIO
        flags |= cv2.CALIB_ZERO_TANGENT_DIST
        flags |= cv2.CALIB_RATIONAL_MODEL
        # flags |= cv2.CALIB_SAME_FOCAL_LENGTH
        flags |= cv2.CALIB_FIX_K3
        flags |= cv2.CALIB_FIX_K4
        flags |= cv2.CALIB_FIX_K5

        stereocalib_criteria = (cv2.TERM_CRITERIA_MAX_ITER +
                                cv2.TERM_CRITERIA_EPS, 100, 1e-5)
        ret, M1, d1, M2, d2, R, T, E, F = cv2.stereoCalibrate(
            self.objpoints, self.imgpoints_l,
            self.imgpoints_r, self.M1, self.d1, self.M2,
            self.d2, dims,
            criteria=stereocalib_criteria, flags=flags)

        # print('Intrinsic_mtx_1', M1)
        # print('dist_1', d1)
        # print('Intrinsic_mtx_2', M2)
        # print('dist_2', d2)
        # print('R', R)
        # print('T', T)
        # print('E', E)
        # print('F', F)

        # for i in range(len(self.r1)):
        #     print("--- pose[", i+1, "] ---")
        #     self.ext1, _ = cv2.Rodrigues(self.r1[i])
        #     self.ext2, _ = cv2.Rodrigues(self.r2[i])
        #     print('Ext1', self.ext1)
        #     print('Ext2', self.ext2)

        # print('')

        camera_model = dict([('M1', M1), ('M2', M2), ('D1', d1),
                            ('D2', d2), ('rvecs1', self.r1),
                            ('rvecs2', self.r2), ('R', R), ('T', T),
                            ('E', E), ('F', F)])

        # print(camera_model)

        cv2.destroyAllWindows()
        return camera_model

    def getModelCamera(self):
        return self.camera_model


class StereoRectifier(object):

    def __init__(self, _stereo_calibration):
        self._stereo_model = None
        pass

    def getStereoModel(self):
        return self._stereo_model

    def calcRectifier(self):
        pass


class IntrinsicGenerator(object):

    def __init__(self, camera_model):
        self._model_camera = camera_model

    def save(self, filename):
        # help(FileStorage)
        f = cv2.FileStorage(filename, 1)
        f.write(name='M1',val=self._model_camera['M1'])
        f.write(name='D1',val=self._model_camera['D1'])
        f.write(name='M2',val=self._model_camera['M2'])
        f.write(name='D2',val=self._model_camera['D1'])
        f.release()


class ExtrinsicGenerator(object):

    def __init__(self, camera_model):
        self._model_camera = camera_model

    def save(self, filename):
        # help(FileStorage)
        f = cv2.FileStorage(filename, 1)
        f.write(name='R',val=self._model_camera['R'])
        f.write(name='T',val=self._model_camera['T'])
        f.release()


class StereoManager(object):

    def __init__(self, camera_model):
        self._model_camera = camera_model
        self.mapx = None
        self.mapy = None
        self.width, self.height = 640, 480
        self._init_calc()

    def _init_calc(self):
        # auxiliar internal method
        def stereo_rectify(R, T):
            R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
                                self._model_camera['M1'],
                                self._model_camera['D1'],
                                self._model_camera['M2'],
                                self._model_camera['D2'],
                                (640, 480), R, T, alpha=0)
            return R1, R2, P1, P2
        # -------------------------------------------------
        R1, R2, P1, P2 = stereo_rectify(self._model_camera['R'], self._model_camera['T'])

        self.R1 = R1
        self.P1 = P1
        self.mapx_l = np.ndarray(shape=(self.height, self.width, 1), dtype='float32')
        self.mapy_l = np.ndarray(shape=(self.height, self.width, 1), dtype='float32')
        cv2.initUndistortRectifyMap(self._model_camera["M1"], self._model_camera["D1"], self.R1, self.P1,
                (self.width, self.height), cv2.CV_32FC1, self.mapx_l, self.mapy_l)

        self.R2 = R2
        self.P2 = P2
        self.mapx_r = np.ndarray(shape=(self.height, self.width, 1), dtype='float32')
        self.mapy_r = np.ndarray(shape=(self.height, self.width, 1), dtype='float32')
        cv2.initUndistortRectifyMap(self._model_camera["M2"], self._model_camera["D2"], self.R2, self.P2,
                (self.width, self.height), cv2.CV_32FC1, self.mapx_r, self.mapy_r)

    def calc_stereo(self, imageL, imageR, interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_REFLECT_101):
        # output_image = None
        left_output_image = cv2.remap(imageL, self.mapx_l, self.mapy_l, interpolation=interpolation, borderMode=border_mode)
        right_output_image = cv2.remap(imageL, self.mapx_r, self.mapy_r, interpolation=interpolation, borderMode=border_mode)
        # return img 
        return left_output_image, right_output_image



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('filepath', help='String Filepath')
    args = parser.parse_args()
    cal_data = StereoCalibration(args.filepath)

    # _stereo_rect = StereoRectifier(cal_data)
    # print(cal_data.getModelCamera())

    # UNCOMMENT
    # _intrinsic_generator = IntrinsicGenerator(cal_data.getModelCamera())
    # _intrinsic_generator.save("intrinsic.yml")
    
    # _extrinsic_generator = ExtrinsicGenerator(cal_data.getModelCamera())
    # _extrinsic_generator.save("extrinsic.yml")

    stereo_manager =  StereoManager(cal_data.getModelCamera())

    # filenameimageL = "/media/ebenezerpdi/EBBACKUP/JELAMBRAR/images_3d/DUAL_MATT_IMAGES/left/left_00.png"
    # imageL = cv2.imread(filenameimageL, 1)

    # filenameimageR = "/media/ebenezerpdi/EBBACKUP/JELAMBRAR/images_3d/DUAL_MATT_IMAGES/right/right_00.png"
    # imageR = cv2.imread(filenameimageR, 1)

    cal_path= args.filepath

    cal_path = args.filepath
    images_right = glob.glob(cal_path + '/right/*.png')
    images_left = glob.glob(cal_path + '/left/*.png')
    images_left.sort()
    images_right.sort()

    # print(images_right)
    # print(images_left)

    for i, fname in enumerate(images_right):

        print('reding images: ')
        print(images_left[i])
        print(images_right[i])
        print()

        img_l = cv2.imread(images_left[i])
        img_r = cv2.imread(images_right[i])

        outL, outR = stereo_manager.calc_stereo(img_l, img_r)

        cv2.imshow("imageL", outL)
        cv2.imshow("imageR", outR)

        cv2.imshow("Original Image L", img_l)

        """"
        based by: http://timosam.com/python_opencv_depthimage
        """

        imgL = img_l
        imgR = img_r

        # SGBM Parameters -----------------
        window_size = 3                     # wsize default 3; 5; 7 for SGBM reduced size image; 15 for SGBM full size image (1300px and above); 5 Works nicely

        left_matcher = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=16,             # max_disp has to be dividable by 16 f. E. HH 192, 256
            blockSize=5,
            P1=8 * 3 * window_size ** 2,    # wsize default 3; 5; 7 for SGBM reduced size image; 15 for SGBM full size image (1300px and above); 5 Works nicely
            P2=32 * 3 * window_size ** 2,
            disp12MaxDiff=1,
            uniquenessRatio=15,
            speckleWindowSize=0,
            speckleRange=2,
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


        print('computing disparity...')
        displ = left_matcher.compute(imgL, imgR)  # .astype(np.float32)/16
        dispr = right_matcher.compute(imgR, imgL)  # .astype(np.float32)/16
        displ = np.int16(displ)
        dispr = np.int16(dispr)
        filteredImg = wls_filter.filter(displ, imgL, None, dispr)  # important to put "imgL" here!!!

        cv2.imshow('Disparity Map', filteredImg)

        filteredImg = cv2.normalize(src=filteredImg, dst=filteredImg, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX);
        filteredImg = np.uint8(filteredImg)
        cv2.imshow('Disparity Map', filteredImg)

        cv2.waitKey()

    cv2.destroyAllWindows()
    print("FINAL")

