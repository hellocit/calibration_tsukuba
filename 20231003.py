#!/usr/bin/env python3
import cv2
import numpy as np

class Correction():
    def __init__(self, mtx, dist, formatType=None):

        if ( formatType == "csv" ):
            self.mtx = np.loadtxt(mtx, delimiter=",")
            self.dist= np.loadtxt(dist, delimiter=",")
        else:
            self.mtx = mtx
            self.dist= dist

    def __call__(self, image):
        cv2.CALIB_FIX_PRINCIPAL_POINT
#        print(image.shape)
        h, w = image.shape[:2]
        newcameramtx, roi=cv2.getOptimalNewCameraMatrix(self.mtx, self.dist, (w, h), 1, (w, h))
#        print(roi)

        # undistort
        dst = cv2.undistort(image, self.mtx, self.dist, None, newcameramtx)


        # crop the image
        x,y,w,h = roi
        dst = dst[y:y+h, x:x+w]

        return dst

if __name__ ==  "__main__":
    corr = Correction("img_20231005T184557_mtx.csv", "img_20231005T184557_dist.csv", formatType="csv")
    img = corr(cv2.imread("img_20231005T184557.jpg"))
    cv2.imwrite( "img_20231005T184557.jpg", img)